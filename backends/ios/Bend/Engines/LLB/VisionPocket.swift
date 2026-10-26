import CoreGraphics
import CoreML
import Foundation
import ImageIO
import Vision

/// G27: the pocket skeleton: one photo → the detector's boxes (RF-DETR-S int8) → the segmenter's mask for the top box
/// (EdgeTAM, or SAM 2.1 tiny, prompted with the box's two corners) → the mask + one sentence to speak. Off unless asked:
///   -visionPocket | env BEND_VISION_POCKET=1 | UserDefaults bend.vision.pocket   POST /v1/bend/pocket answers (else 404, as before)
///   -pocketDir <p>        packages (relative to Documents; default `ml`, the G11 staging dir, env BEND_VISION_DIR)
///   -pocketDet <name>     detector package (default rfdetr-small-int8)
///   -pocketSeg <subdir>   segmenter subdirectory (default edgetam-box: EdgeTAM's encoder + decoder with G27's box-safe prompt
///                         encoder re-export; `edgetam` = the official export (box refused → the box's centre as one point);
///                         sam21-tiny = the SAM 2.1 Apple port; `none` = boxes only)
///   -pocketUnits ane|all|gpu|cpu   compute units for every package (default ane = CPU_AND_NE, VISION-CROWN's pin)
///   -pocketMinScore x     the detector's floor (default 0.35)
///   G38: a detector staged as `<-pocketDet>.onnx` (e.g. `-pocketDet dfine-n`: D-FINE-N, 80 COCO classes) runs on
///   onnxruntime's CoreML execution provider instead (OrtDetector, DetectORT.swift: -pocketEP coreml|cpu, -pocketOrtLog).
/// Loaded once, on the first request, and kept (it is ~240 MB peak on the M3U proxy, G11). The sentence is a stub: the
/// label, where the mask sits, how much of the photo it covers; the web speaks it through /voice/speak.
/// Core only (CoreML + CoreGraphics), so it builds standalone with EngineCheckVision.swift (-D VISION_STANDALONE).
final class VisionPocket: @unchecked Sendable {
    static let enabled: Bool = {
        let a = ProcessInfo.processInfo.arguments
        if a.contains("-visionPocket") || ProcessInfo.processInfo.environment["BEND_VISION_POCKET"] == "1" { return true }
        if let i = a.firstIndex(of: "-engineCheck"), i + 1 < a.count, a[i + 1] == "pocket" { return true }
        return UserDefaults.standard.bool(forKey: "bend.vision.pocket")
    }()

    struct Options: Sendable {
        var dir: URL
        var cache: URL
        var det = "rfdetr-small-int8"
        var seg = "edgetam-box"
        var unitsName = "ane"
        var minScore: Float = 0.35
        var ep = "coreml"  // G38: an .onnx detector's execution provider (coreml | cpu)
        var ortLog = false

        static func parse(_ args: [String], documents: URL, cache: URL) -> Options {
            func value(_ flag: String) -> String? {
                guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
                return args[i + 1]
            }
            let env = ProcessInfo.processInfo.environment
            let dirArg = value("-pocketDir") ?? env["BEND_VISION_DIR"] ?? "ml"
            var o = Options(dir: URL(fileURLWithPath: dirArg, relativeTo: documents).standardizedFileURL, cache: cache)
            if let v = value("-pocketDet") { o.det = v }
            if let v = value("-pocketSeg") ?? env["BEND_POCKET_SEG"] { o.seg = v }
            if let v = value("-pocketUnits"), VisionCheck.Options.units(v) != nil { o.unitsName = v }
            if let v = value("-pocketMinScore").flatMap(Float.init) { o.minScore = v }
            if let v = value("-pocketEP"), ["coreml", "cpu"].contains(v) { o.ep = v }
            o.ortLog = args.contains("-pocketOrtLog")
            return o
        }
    }

    struct Box: Sendable {
        var cls: Int, label: String, score: Float
        var x0: Float, y0: Float, x1: Float, y1: Float  // normalized to the photo, 0…1
        var json: [String: Any] { ["label": label, "class": cls, "score": Double(score), "box": [x0, y0, x1, y1].map(Double.init)] }
    }

    struct Mask: Sendable {
        var side: Int, bits: [UInt8]  // side×side over the photo (stretched), 1 = the object
        var score: Float, area: Float  // the decoder's IoU guess; the share of the photo
        var x0: Float, y0: Float, x1: Float, y1: Float, cx: Float, cy: Float  // its extent and centroid, normalized
    }

    struct Answer: Sendable {
        var boxes: [Box], mask: Mask?, sentence: String
        var prompt: String  // box | point | none (no mask asked)
        var ms: [(String, Double)]  // pre, det, enc, prompt, dec, post
        var total: Double { ms.reduce(0) { $0 + $1.1 } }
    }

    /// The detector: a Core ML package (RF-DETR-S int8, G27) or an ONNX file on ORT's CoreML EP (G38).
    enum Detector {
        case coreml(MLModel)
        #if !VISION_STANDALONE
        case ort(OrtDetector)
        #endif
    }

    let options: Options
    let det: Detector
    let seg: [MLModel]  // encoder, prompt encoder, mask decoder (empty: boxes only)
    let segName: String
    /// How the box reaches the prompt encoder: `box` (corners labelled 2/3) or `point` (the box's centre, labelled 1), for
    /// an export that only takes one positive point (EdgeTAM's: its traced label masks refuse 2/3 with CoreML -5).
    private(set) var prompt = "box"
    private let lock = NSLock()

    /// `rfdetr-small-int8 coreml(ane)` / `dfine-n ort-coreml(MLProgram,CPUAndNeuralEngine)`: who found the boxes.
    var detName: String {
        switch det {
        case .coreml: return "\(options.det) coreml(\(options.unitsName))"
        #if !VISION_STANDALONE
        case .ort(let o): return "\(o.name) ort-\(o.ep)"
        #endif
        }
    }

    private init(options: Options, det: Detector, seg: [MLModel]) {
        self.options = options; self.det = det; self.seg = seg
        segName = seg.isEmpty ? "none" : options.seg
    }

    // MARK: load

    private static func load(_ src: URL, _ o: Options, log: (String) -> Void) async throws -> MLModel {
        let name = src.deletingPathExtension().lastPathComponent
        var url = src, compileMs = 0.0
        if src.pathExtension == "mlpackage" {
            let fm = FileManager.default
            // Keyed by the package's directory too: edgetam/ and edgetam-box/ both hold an edgetam_prompt_encoder (a
            // name-only key would hand one the other's compiled model), and G11's check shares this cache dir.
            let dst = o.cache.appendingPathComponent("pocket-\(src.deletingLastPathComponent().lastPathComponent)-\(name).mlmodelc")
            let srcDate = (try? fm.attributesOfItem(atPath: src.appendingPathComponent("Manifest.json").path)[.modificationDate] as? Date) ?? .distantFuture
            if let d = try? fm.attributesOfItem(atPath: dst.path)[.modificationDate] as? Date, d >= srcDate {
                url = dst
            } else {
                let t0 = Date()
                let tmp = try await MLModel.compileModel(at: src)
                try? fm.createDirectory(at: o.cache, withIntermediateDirectories: true)
                try? fm.removeItem(at: dst)
                try fm.moveItem(at: tmp, to: dst)
                url = dst
                compileMs = Date().timeIntervalSince(t0) * 1000
            }
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = VisionCheck.Options.units(o.unitsName) ?? .cpuAndNeuralEngine
        let before = VisionCheck.footprint(), t0 = Date()
        let m = try await MLModel.load(contentsOf: url, configuration: cfg)
        log(String(format: "pocket load %@ units=%@ compile_ms=%.0f load_ms=%.0f delta_mb=%d %@", name, o.unitsName, compileMs,
                   Date().timeIntervalSince(t0) * 1000, VisionCheck.footprint() - before, VisionCheck.mem))
        return m
    }

    /// The packages from `o.dir`; nil (with the reason logged) when one is missing or does not load.
    static func open(_ o: Options, log: (String) -> Void) async -> VisionPocket? {
        let fm = FileManager.default
        let dir = o.dir.resolvingSymlinksInPath()
        let detURL = ["mlmodelc", "mlpackage", "onnx"].map { dir.appendingPathComponent("\(o.det).\($0)") }.first { fm.fileExists(atPath: $0.path) }
        guard let detURL else { log("pocket: no detector \(o.det) in \(dir.path)"); return nil }
        var segURLs: [URL] = []
        if o.seg != "none" {
            let sd = dir.appendingPathComponent(o.seg).resolvingSymlinksInPath()
            let all = ((try? fm.contentsOfDirectory(at: sd, includingPropertiesForKeys: nil)) ?? []).filter { ["mlpackage", "mlmodelc"].contains($0.pathExtension) }
            func pick(_ f: (String) -> Bool) -> URL? { all.first { f($0.lastPathComponent.lowercased()) } }
            guard let enc = pick({ $0.contains("image") && $0.contains("encoder") }), let pro = pick({ $0.contains("prompt") }),
                  let dec = pick({ $0.contains("decoder") }) else {
                log("pocket: segmenter \(o.seg) needs image encoder + prompt encoder + mask decoder in \(sd.path)"); return nil
            }
            segURLs = [enc, pro, dec]
        }
        do {
            let d: Detector
            if detURL.pathExtension == "onnx" {
                #if VISION_STANDALONE
                log("pocket: \(detURL.lastPathComponent) needs onnxruntime (the app build)"); return nil
                #else
                let before = VisionCheck.footprint(), t0 = Date()
                let ort = try OrtDetector(url: detURL, ep: o.ep, units: o.unitsName, cache: o.cache.deletingLastPathComponent().appendingPathComponent("ort-coreml"),
                                          verbose: o.ortLog)
                log(String(format: "pocket load %@ ep=%@ side=%d load_ms=%.0f delta_mb=%d %@", ort.name, ort.ep, ort.side,
                           Date().timeIntervalSince(t0) * 1000, VisionCheck.footprint() - before, VisionCheck.mem))
                d = .ort(ort)
                #endif
            } else {
                d = .coreml(try await load(detURL, o, log: log))
            }
            var s: [MLModel] = []
            for u in segURLs { s.append(try await load(u, o, log: log)) }
            return VisionPocket(options: o, det: d, seg: s)
        } catch {
            log("pocket load FAILED: \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: run

    struct Failure: Error, LocalizedError { let errorDescription: String?; init(_ s: String) { errorDescription = s } }

    /// One photo, start to end (serialized: one pocket at a time).
    func run(_ image: CGImage) throws -> Answer {
        lock.lock(); defer { lock.unlock() }
        var ms: [(String, Double)] = []
        var t = Date()
        func lap(_ k: String) { ms.append((k, Date().timeIntervalSince(t) * 1000)); t = Date() }

        let found: [Box]
        switch det {
        case .coreml(let det):
            let detIn = det.modelDescription.inputDescriptionsByName.values.first { $0.type == .image }
            guard let detIn, let dc = detIn.imageConstraint else { throw Failure("detector has no image input") }
            let scale = [MLFeatureValue.ImageOption.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue]
            let detImage = try MLFeatureValue(cgImage: image, constraint: dc, options: scale)
            lap("pre")
            let dout = try det.prediction(from: MLDictionaryFeatureProvider(dictionary: [detIn.name: detImage]))
            lap("det")
            guard let logits = dout.featureValue(for: "logits")?.multiArrayValue, let boxes = dout.featureValue(for: "boxes")?.multiArrayValue else {
                throw Failure("detector outputs: want logits + boxes, got \(dout.featureNames.sorted())")
            }
            found = Self.decode(logits: logits, boxes: boxes, minScore: options.minScore)
        #if !VISION_STANDALONE
        case .ort(let ort):
            let px = try ort.pixels(image)
            lap("pre")
            if ort.coco {
                let out = try ort.runCoco(px, width: image.width, height: image.height)
                lap("det")
                found = Self.decodeCoco(labels: out.labels, boxes: out.boxes, scores: out.scores,
                                        w: Float(image.width), h: Float(image.height), minScore: options.minScore)
            } else {
                let out = try ort.run(px)
                lap("det")
                found = Self.decode(l: out.logits, b: out.boxes, q: out.q, c: out.c, minScore: options.minScore)
            }
        #endif
        }
        var mask: Mask?
        if let top = found.first, seg.count == 3 {
            mask = try segment(image, top, lap: lap)
        }
        let sentence = Self.sentence(found, mask)
        lap("post")
        return Answer(boxes: found, mask: mask, sentence: sentence, prompt: mask == nil ? "none" : prompt, ms: ms)
    }

    /// The 3-head COCO export: `labels` (0…79 contiguous COCO), `boxes` [Q,4] xyxy in the target's pixels,
    /// `scores` already sigmoid'd. Normalized to the photo here so every consumer sees one box space.
    static func decodeCoco(labels: [Int64], boxes: [Float], scores: [Float], w: Float, h: Float,
                           minScore: Float, keep: Int = 8) -> [Box] {
        var out: [Box] = []
        for i in 0..<scores.count where i * 4 + 3 < boxes.count && i < labels.count {
            let s = scores[i]
            guard s >= minScore else { continue }
            let cls = Int(labels[i])
            out.append(Box(cls: cls, label: cls >= 0 && cls < coco80.count ? coco80[cls] : "thing", score: s,
                           x0: max(0, min(1, boxes[i * 4] / w)), y0: max(0, min(1, boxes[i * 4 + 1] / h)),
                           x1: max(0, min(1, boxes[i * 4 + 2] / w)), y1: max(0, min(1, boxes[i * 4 + 3] / h))))
        }
        return Array(out.sorted { $0.score > $1.score }.prefix(keep))
    }

    /// RF-DETR's raw heads: logits [1,Q,91] (COCO category ids, sigmoid), boxes [1,Q,4] cxcywh normalized.
    static func decode(logits: MLMultiArray, boxes: MLMultiArray, minScore: Float, keep: Int = 8) -> [Box] {
        decode(l: floats(logits), b: floats(boxes), q: logits.shape[1].intValue, c: logits.shape[2].intValue, minScore: minScore, keep: keep)
    }

    /// The same over plain arrays. C = 91: COCO category ids (RF-DETR); C = 80: the 80 contiguous COCO classes (D-FINE).
    static func decode(l: [Float], b: [Float], q: Int, c: Int, minScore: Float, keep: Int = 8) -> [Box] {
        let names = c == 80 ? coco80 : coco
        var out: [Box] = []
        for i in 0..<q {
            var best = 0, bv = -Float.infinity
            for k in 0..<c where l[i * c + k] > bv { bv = l[i * c + k]; best = k }
            let s = 1 / (1 + exp(-bv))
            guard s >= minScore else { continue }
            let cx = b[i * 4], cy = b[i * 4 + 1], w = b[i * 4 + 2], h = b[i * 4 + 3]
            out.append(Box(cls: best, label: best < names.count ? names[best] : "thing", score: s,
                           x0: max(0, cx - w / 2), y0: max(0, cy - h / 2), x1: min(1, cx + w / 2), y1: min(1, cy + h / 2)))
        }
        return Array(out.sorted { $0.score > $1.score }.prefix(keep))
    }

    /// The photo's prompt-free features (side = 1024: the prompt's coordinate space; SAM2Transforms stretches to it).
    private func encode(_ image: CGImage, lap: (String) -> Void) throws -> MLFeatureProvider {
        let enc = seg[0]
        guard let encIn = enc.modelDescription.inputDescriptionsByName.values.first(where: { $0.type == .image }), let ec = encIn.imageConstraint else {
            throw Failure("segmenter encoder has no image input")
        }
        let encImage = try MLFeatureValue(cgImage: image, constraint: ec, options: [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue])
        let eout = try enc.prediction(from: MLDictionaryFeatureProvider(dictionary: [encIn.name: encImage]))
        lap("enc")
        return eout
    }

    private func segment(_ image: CGImage, _ box: Box, encoded: MLFeatureProvider? = nil, lap: (String) -> Void) throws -> Mask {
        let (enc, pro, dec) = (seg[0], seg[1], seg[2])
        let eout = try encoded ?? encode(image, lap: lap)
        let side = Float(enc.modelDescription.inputDescriptionsByName.values.first { $0.type == .image }?.imageConstraint?.pixelsWide ?? 1024)

        // The box as SAM2ImagePredictor gives it: its two corners as points labelled 2 and 3 (EdgeTAM's exported prompt
        // encoder drops `boxes`; the Apple SAM 2.1 port has no box input).
        let corners: [Float] = prompt == "box" ? [box.x0 * side, box.y0 * side, box.x1 * side, box.y1 * side]
            : [(box.x0 + box.x1) / 2 * side, (box.y0 + box.y1) / 2 * side]
        let labels: [Float] = prompt == "box" ? [2, 3] : [1]
        var pin: [String: MLFeatureValue] = [:]
        for (name, d) in pro.modelDescription.inputDescriptionsByName {
            guard let c = d.multiArrayConstraint else { continue }
            let n = name.lowercased()
            var shape = Self.shape(c)
            if c.shapeConstraint.type == .range, shape.count >= 2 { shape[1] = labels.count }  // the corners, or the centre
            let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: c.dataType)
            var v = [Float](repeating: 0, count: a.count)
            if n.contains("label") {
                for i in 0..<v.count { v[i] = i < labels.count ? labels[i] : -1 }
            } else if n.contains("coord") || n == "points" {
                for i in 0..<min(corners.count, v.count) { v[i] = corners[i] }
            }
            Self.fill(a, v)
            pin[name] = MLFeatureValue(multiArray: a)
        }
        let pout: MLFeatureProvider
        do { pout = try pro.prediction(from: MLDictionaryFeatureProvider(dictionary: pin)) } catch where prompt == "box" {
            prompt = "point"  // from now on; this run again with the centre
            return try segment(image, box, encoded: eout, lap: lap)
        }
        lap("prompt")

        var din: [String: MLFeatureValue] = [:]
        for (name, d) in dec.modelDescription.inputDescriptionsByName {
            if let v = ([name] + (VisionCheck.aliases[name] ?? [])).lazy.compactMap({ eout.featureValue(for: $0) ?? pout.featureValue(for: $0) }).first {
                din[name] = v
                continue
            }
            guard let c = d.multiArrayConstraint else { throw Failure("mask decoder input \(name): not an array") }
            let a = try MLMultiArray(shape: Self.shape(c).map { NSNumber(value: $0) }, dataType: c.dataType)
            Self.fill(a, [Float](repeating: name.lowercased().contains("multimask") ? 1 : 0, count: a.count))  // image_pe: baked in (EdgeTAM export)
            din[name] = MLFeatureValue(multiArray: a)
        }
        let mout = try dec.prediction(from: MLDictionaryFeatureProvider(dictionary: din))
        lap("dec")
        guard let masks = (mout.featureValue(for: "masks") ?? mout.featureValue(for: "low_res_masks"))?.multiArrayValue,
              let scores = (mout.featureValue(for: "iou_pred") ?? mout.featureValue(for: "scores"))?.multiArrayValue else {
            throw Failure("mask decoder outputs: want masks/low_res_masks + iou_pred/scores, got \(mout.featureNames.sorted())")
        }
        let m = Self.floats(masks), s = Self.floats(scores)
        let k = s.indices.max { s[$0] < s[$1] } ?? 0
        let ms = masks.shape[masks.shape.count - 1].intValue, plane = ms * ms
        var bits = [UInt8](repeating: 0, count: plane)
        var n = 0, sx = 0, sy = 0, x0 = ms, y0 = ms, x1 = -1, y1 = -1
        for y in 0..<ms {
            for x in 0..<ms where m[k * plane + y * ms + x] > 0 {
                bits[y * ms + x] = 1
                n += 1; sx += x; sy += y
                x0 = min(x0, x); y0 = min(y0, y); x1 = max(x1, x); y1 = max(y1, y)
            }
        }
        let f = Float(ms)
        return Mask(side: ms, bits: bits, score: s[k], area: Float(n) / Float(plane),
                    x0: n > 0 ? Float(x0) / f : 0, y0: n > 0 ? Float(y0) / f : 0, x1: n > 0 ? Float(x1 + 1) / f : 0, y1: n > 0 ? Float(y1 + 1) / f : 0,
                    cx: n > 0 ? (Float(sx) / Float(n) + 0.5) / f : 0, cy: n > 0 ? (Float(sy) / Float(n) + 0.5) / f : 0)
    }

    // MARK: the sentence (stub)

    static func sentence(_ boxes: [Box], _ mask: Mask?) -> String {
        guard let top = boxes.first else { return "I don't see anything I recognise here." }
        let cx = mask.map(\.cx) ?? (top.x0 + top.x1) / 2
        let area = mask.map(\.area) ?? (top.x1 - top.x0) * (top.y1 - top.y0)
        let place = cx < 0.34 ? "on the left" : cx > 0.66 ? "on the right" : "in the middle"
        let pct = max(1, Int((area * 100).rounded()))
        let article = "aeiou".contains(top.label.first ?? "x") ? "an" : "a"
        let more = boxes.count - 1
        return "I see \(article) \(top.label) \(place), taking up about \(pct)% of the photo"
            + (more > 0 ? ", and \(more) other thing\(more == 1 ? "" : "s") nearby." : ".")
    }

    // MARK: arrays

    static func shape(_ c: MLMultiArrayConstraint) -> [Int] {
        if c.shapeConstraint.type == .enumerated, let first = c.shapeConstraint.enumeratedShapes.first { return first.map(\.intValue) }
        if c.shapeConstraint.type == .range { return c.shapeConstraint.sizeRangeForDimension.map { max(1, $0.rangeValue.location) } }
        return c.shape.map(\.intValue)
    }

    /// Row-major values of any array (fp16 / fp32 / double), honouring its strides (ANE outputs can be padded).
    static func floats(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map(\.intValue), strides = a.strides.map(\.intValue), n = a.count
        var out = [Float](repeating: 0, count: n)
        a.withUnsafeBytes { raw in
            var idx = [Int](repeating: 0, count: shape.count)
            for i in 0..<n {
                var off = 0
                for d in 0..<shape.count { off += idx[d] * strides[d] }
                switch a.dataType {
                case .float16: out[i] = Float(raw.load(fromByteOffset: off * 2, as: Float16.self))
                case .float32: out[i] = raw.load(fromByteOffset: off * 4, as: Float.self)
                case .double: out[i] = Float(raw.load(fromByteOffset: off * 8, as: Double.self))
                default: out[i] = a[i].floatValue
                }
                var d = shape.count - 1
                while d >= 0 { idx[d] += 1; if idx[d] < shape[d] { break }; idx[d] = 0; d -= 1 }
            }
        }
        return out
    }

    /// A freshly made (contiguous) array set from row-major values.
    static func fill(_ a: MLMultiArray, _ v: [Float]) {
        a.withUnsafeMutableBytes { raw, _ in
            for i in 0..<min(v.count, a.count) {
                switch a.dataType {
                case .float16: raw.storeBytes(of: Float16(v[i]), toByteOffset: i * 2, as: Float16.self)
                case .float32: raw.storeBytes(of: v[i], toByteOffset: i * 4, as: Float.self)
                case .double: raw.storeBytes(of: Double(v[i]), toByteOffset: i * 8, as: Double.self)
                default: a[i] = NSNumber(value: v[i])
                }
            }
        }
    }

    // MARK: images

    static func decode(_ data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }

    static func png(_ image: CGImage) -> Data {
        let d = NSMutableData()
        guard let dst = CGImageDestinationCreateWithData(d, "public.png" as CFString, 1, nil) else { return Data() }
        CGImageDestinationAddImage(dst, image, nil)
        CGImageDestinationFinalize(dst)
        return d as Data
    }

    /// The mask as an 8-bit grey PNG (255 = the object), side×side over the stretched photo.
    static func maskPNG(_ m: Mask) -> Data {
        let px = m.bits.map { $0 * 255 }
        guard let provider = CGDataProvider(data: Data(px) as CFData),
              let img = CGImage(width: m.side, height: m.side, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: m.side,
                                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider,
                                decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return Data() }
        return png(img)
    }

    /// For receipts: the photo (longest side `maxSide`), the mask in red, every box outlined (the top one thicker).
    static func overlayPNG(_ image: CGImage, _ a: Answer, maxSide: Int = 768) -> Data {
        let s = Double(maxSide) / Double(max(image.width, image.height))
        let w = Int(Double(image.width) * min(1, s)), h = Int(Double(image.height) * min(1, s))
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return Data() }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let W = CGFloat(w), H = CGFloat(h)
        if let m = a.mask {
            let cell = CGSize(width: W / CGFloat(m.side), height: H / CGFloat(m.side))
            ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 0.45)
            for y in 0..<m.side {
                for x in 0..<m.side where m.bits[y * m.side + x] == 1 {
                    ctx.fill(CGRect(x: CGFloat(x) * cell.width, y: H - CGFloat(y + 1) * cell.height, width: cell.width, height: cell.height))
                }
            }
        }
        for (i, b) in a.boxes.enumerated() {
            ctx.setStrokeColor(red: i == 0 ? 0 : 1, green: 1, blue: i == 0 ? 0 : 1, alpha: 1)
            ctx.setLineWidth(i == 0 ? 4 : 1.5)
            ctx.stroke(CGRect(x: CGFloat(b.x0) * W, y: H - CGFloat(b.y1) * H, width: CGFloat(b.x1 - b.x0) * W, height: CGFloat(b.y1 - b.y0) * H))
        }
        return ctx.makeImage().map(png) ?? Data()
    }

    /// The JSON the route answers (and the Mac CLI writes).
    static func json(_ a: Answer, segName: String, detName: String = "", withMask: Bool) -> [String: Any] {
        var o: [String: Any] = ["boxes": a.boxes.map(\.json), "say": a.sentence, "seg": segName, "prompt": a.prompt, "det": detName,
                                "ms": Dictionary(uniqueKeysWithValues: a.ms.map { ($0.0, ($0.1 * 10).rounded() / 10) })
                                    .merging(["total": (a.total * 10).rounded() / 10]) { x, _ in x },
                                "footprint_mb": VisionCheck.footprint(), "available_mb": VisionCheck.available()]
        if let m = a.mask {
            var mj: [String: Any] = ["side": m.side, "score": Double(m.score), "area": Double(m.area),
                                     "box": [m.x0, m.y0, m.x1, m.y1].map(Double.init), "centroid": [Double(m.cx), Double(m.cy)]]
            if withMask { mj["png"] = maskPNG(m).base64EncodedString() }
            o["mask"] = mj
        }
        return o
    }

    /// The 80 contiguous COCO classes, in D-FINE's head order (its HF config's id2label, with COCO's own names).
    static let coco80 = Array(coco.filter { $0 != "thing" })

    /// COCO category ids as RF-DETR's 91-wide head numbers them (0 and the retired ids are "thing").
    static let coco = ["thing", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "thing", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "thing", "backpack", "umbrella", "thing", "thing", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "thing", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "thing", "dining table", "thing", "thing", "toilet", "thing", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "thing", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]
}

/// The app's one pocket: opened on the first request (packages from Documents, compiled into Caches/vision-mlmodelc as
/// G11's check does) and kept; a failed open is retried on the next request (the packages may be staged since).
actor PocketHost {
    static let shared = PocketHost()
    private var loading: Task<VisionPocket?, Never>?

    func get(log: @escaping @Sendable (String) -> Void) async -> VisionPocket? {
        if let loading, let p = await loading.value { return p }
        let task = Task<VisionPocket?, Never> {
            let fm = FileManager.default
            let o = VisionPocket.Options.parse(ProcessInfo.processInfo.arguments,
                documents: fm.urls(for: .documentDirectory, in: .userDomainMask)[0],
                cache: fm.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("vision-mlmodelc"))
            return await VisionPocket.open(o, log: log)
        }
        loading = task
        return await task.value
    }
}
