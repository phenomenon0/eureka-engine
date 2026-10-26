import CoreML
import Foundation
import Vision
#if os(iOS)
import os
#endif

/// G11: `-engineCheck vision`: the phone's vision numbers (VISION-CROWN §6). Loads CoreML packages from a directory
/// (`-visionDir <path>`, relative to Documents; default Documents/ml), then runs `-visionIters N` (default 30) frames of
/// the detector and, when present, the segmenter (encoder → prompt → decoder, chained by output name). One line per
/// frame, the soak's `<check> answered …` shape (pocketd soak-iphone.sh with CHECK=vision):
///   BEND check vision answered iter=3 det_ms=18.4 seg_ms=21.0 ms=39.4 footprint_mb=412 available_mb=5731
/// footprint_mb is phys_footprint, available_mb os_proc_available_memory (PHONE-CEILING's two columns: fp + avail = the
/// limit). Packages are compiled once into Caches/vision-mlmodelc (a relaunch reuses them).
///   Documents/ml/<detector>.mlpackage|.mlmodelc   the first top-level package (sorted), or `-visionDet <name>`
///   Documents/ml/<segmenter>/*.mlpackage           the first subdirectory holding packages, or `-visionSeg <dir>`; `-visionSeg none` skips it
///   -visionUnits ane|all|gpu|cpu                   MLModelConfiguration.computeUnits (default ane = CPU_AND_NE, VISION-CROWN's pin)
///   -visionUnitsFor decoder=all[,prompt=cpu]       per package (name contains the key): VISION-CROWN ran the EdgeTAM decoder on ALL
///   -visionWithChat                                warm chat first (a p2 turn): the co-resident row; without it the
///                                                  launch lifecycle leaves chat cold (EngineManager.launchLifecycle)
/// Inputs are the bundled engine-check/look-news-1280.jpg for images and in-range fills for arrays; the numbers are
/// latency + memory, not quality. PASS: every model loaded and every frame answered.
enum VisionCheck {
    struct Options {
        var dir: URL
        var cache: URL
        var photo: URL?
        var iters = 30
        var units: MLComputeUnits = .cpuAndNeuralEngine
        var unitsName = "ane"
        var det: String?
        var seg: String?
        var unitsFor: [(key: String, name: String)] = []

        static func units(_ name: String) -> MLComputeUnits? {
            ["ane": .cpuAndNeuralEngine, "all": .all, "gpu": .cpuAndGPU, "cpu": .cpuOnly][name]
        }

        /// The units for one package: the first `-visionUnitsFor` key its name contains, else `-visionUnits`.
        func units(for package: String) -> (MLComputeUnits, String) {
            if let o = unitsFor.first(where: { package.lowercased().contains($0.key.lowercased()) }), let u = Options.units(o.name) { return (u, o.name) }
            return (units, unitsName)
        }

        static func parse(_ args: [String], documents: URL, cache: URL, photo: URL?) -> Options {
            func value(_ flag: String) -> String? {
                guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
                return args[i + 1]
            }
            let dirArg = value("-visionDir") ?? ProcessInfo.processInfo.environment["BEND_VISION_DIR"] ?? "ml"
            var o = Options(dir: URL(fileURLWithPath: dirArg, relativeTo: documents).standardizedFileURL, cache: cache, photo: photo)
            if let n = value("-visionIters").flatMap(Int.init), n > 0 { o.iters = n }
            if let n = value("-visionUnits"), let u = units(n) { o.units = u; o.unitsName = n }
            o.unitsFor = (value("-visionUnitsFor") ?? "").split(separator: ",").compactMap {
                let kv = $0.split(separator: "=", maxSplits: 1).map(String.init)
                return kv.count == 2 && units(kv[1]) != nil ? (kv[0], kv[1]) : nil
            }
            o.det = value("-visionDet")
            o.seg = value("-visionSeg")
            return o
        }
    }

    final class Stage {
        let role: String, name: String, model: MLModel
        var fixed: [String: MLFeatureValue] = [:]
        var chained = true
        init(role: String, name: String, model: MLModel) { self.role = role; self.name = name; self.model = model }
    }

    /// EngineManager.footprint's, so VisionCheck also builds standalone (-D VISION_STANDALONE, the Mac CLI test).
    static func footprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint / 1_048_576) : -1
    }

    static func available() -> Int {
        #if os(iOS)
        return Int(os_proc_available_memory() / 1_048_576)
        #else
        return -1
        #endif
    }

    static var mem: String { "footprint_mb=\(footprint()) available_mb=\(available())" }

    private static func isPackage(_ u: URL) -> Bool { ["mlpackage", "mlmodelc"].contains(u.pathExtension) }

    private static func list(_ dir: URL) -> [URL] {
        ((try? FileManager.default.contentsOfDirectory(at: dir.resolvingSymlinksInPath(), includingPropertiesForKeys: nil)) ?? [])
            .map { $0.resolvingSymlinksInPath() }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Segmenter parts in pipeline order: image encoder, prompt encoder, mask decoder (anything else after).
    private static func segOrder(_ u: URL) -> Int {
        let n = u.lastPathComponent.lowercased()
        if n.contains("prompt") { return 1 }
        if n.contains("decoder") { return 2 }
        if n.contains("encoder") { return 0 }
        return 3
    }

    /// The compiled model, from the cache when the package is not newer than it.
    private static func compiled(_ src: URL, cache: URL) async throws -> (URL, Double) {
        if src.pathExtension == "mlmodelc" { return (src, 0) }
        let fm = FileManager.default
        let dst = cache.appendingPathComponent(src.deletingPathExtension().lastPathComponent + ".mlmodelc")
        let srcDate = (try? fm.attributesOfItem(atPath: src.appendingPathComponent("Manifest.json").path)[.modificationDate] as? Date) ?? .distantFuture
        if let d = try? fm.attributesOfItem(atPath: dst.path)[.modificationDate] as? Date, d >= srcDate { return (dst, 0) }
        let t0 = Date()
        let tmp = try await MLModel.compileModel(at: src)
        try? fm.createDirectory(at: cache, withIntermediateDirectories: true)
        try? fm.removeItem(at: dst)
        try fm.moveItem(at: tmp, to: dst)
        return (dst, Date().timeIntervalSince(t0) * 1000)
    }

    private static func load(_ role: String, _ src: URL, _ o: Options, log: (String) -> Void) async -> Stage? {
        let name = src.deletingPathExtension().lastPathComponent
        do {
            let (url, compileMs) = try await compiled(src, cache: o.cache)
            let cfg = MLModelConfiguration()
            let (units, unitsName) = o.units(for: name)
            cfg.computeUnits = units
            let before = footprint()
            let t0 = Date()
            let model = try await MLModel.load(contentsOf: url, configuration: cfg)
            let loadMs = Date().timeIntervalSince(t0) * 1000
            let ins = model.modelDescription.inputDescriptionsByName.keys.sorted().joined(separator: ",")
            log(String(format: "vision load %@ %@ units=%@ compile_ms=%.0f load_ms=%.0f delta_mb=%d %@ inputs=%@",
                       role, name, unitsName, compileMs, loadMs, footprint() - before, mem, ins))
            return Stage(role: role, name: name, model: model)
        } catch {
            log("vision load \(role) \(name) FAILED: \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: inputs

    /// Upstream outputs that feed a downstream input under another name (EdgeTAM, SAM 2.1 Apple port).
    static let aliases: [String: [String]] = [
        "image_embeddings": ["vision_features", "image_embedding"],
        "image_embedding": ["vision_features", "image_embeddings"],
        "sparse_prompt_embeddings": ["sparse_embeddings"], "sparse_embedding": ["sparse_embeddings"],
        "dense_prompt_embeddings": ["dense_embeddings"], "dense_embedding": ["dense_embeddings"],
    ]

    private static func shape(_ c: MLMultiArrayConstraint) -> [Int] {
        // The default shape; for a flexible one, the lower bounds (enumerated: its first shape).
        if let first = c.shapeConstraint.enumeratedShapes.first, c.shapeConstraint.type == .enumerated { return first.map(\.intValue) }
        if c.shapeConstraint.type == .range { return c.shapeConstraint.sizeRangeForDimension.map { max(1, $0.rangeValue.location) } }
        return c.shape.map(\.intValue)
    }

    /// A plausible value by name: a point in the middle of a 1024 canvas, a foreground label, a centred box.
    private static func fill(_ name: String, index i: Int) -> Double? {
        let n = name.lowercased()
        if n.contains("label") { return i == 0 ? 1 : -1 }
        if n.contains("coord") || n.contains("point") { return 512 }
        if n.contains("box") { return [256, 256, 768, 768][i % 4] }
        if n.contains("multimask") { return 1 }
        return nil // zeros
    }

    private static func synthetic(_ d: MLFeatureDescription, photo: URL?) throws -> MLFeatureValue {
        switch d.type {
        case .image:
            let c = d.imageConstraint!
            if let photo, let v = try? MLFeatureValue(imageAt: photo, constraint: c, options: [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue]) { return v }
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(nil, c.pixelsWide, c.pixelsHigh, kCVPixelFormatType_32BGRA, nil, &pb)
            if let pb {
                CVPixelBufferLockBaseAddress(pb, [])
                memset(CVPixelBufferGetBaseAddress(pb), 0x80, CVPixelBufferGetDataSize(pb))
                CVPixelBufferUnlockBaseAddress(pb, [])
                return MLFeatureValue(pixelBuffer: pb)
            }
            throw NSError(domain: "vision", code: 1, userInfo: [NSLocalizedDescriptionKey: "no pixel buffer \(c.pixelsWide)x\(c.pixelsHigh)"])
        case .multiArray:
            let c = d.multiArrayConstraint!
            let s = shape(c)
            let a = try MLMultiArray(shape: s.map { NSNumber(value: $0) }, dataType: c.dataType)
            a.withUnsafeMutableBytes { p, _ in _ = memset(p.baseAddress, 0, p.count) }
            if fill(d.name, index: 0) != nil { // the small prompt arrays only; the feature maps stay zero
                let last = max(1, s.last ?? 1)
                for i in 0..<a.count { a[i] = NSNumber(value: fill(d.name, index: i % last) ?? 0) }
            }
            return MLFeatureValue(multiArray: a)
        case .double:
            let n = d.name.lowercased()
            return MLFeatureValue(double: n.contains("iou") ? 0.45 : n.contains("conf") ? 0.25 : 0)
        case .int64: return MLFeatureValue(int64: 0)
        default:
            throw NSError(domain: "vision", code: 2, userInfo: [NSLocalizedDescriptionKey: "input \(d.name): unsupported type \(d.type.rawValue)"])
        }
    }

    private static func inputs(_ s: Stage, upstream: [String: MLFeatureValue]) throws -> MLFeatureProvider {
        var dict = s.fixed
        if s.chained {
            for name in s.model.modelDescription.inputDescriptionsByName.keys {
                if let v = ([name] + (aliases[name] ?? [])).lazy.compactMap({ upstream[$0] }).first { dict[name] = v }
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: dict)
    }

    // MARK: run

    static func percentile(_ xs: [Double], _ p: Double) -> Double {
        guard !xs.isEmpty else { return -1 }
        let s = xs.sorted()
        return s[min(s.count - 1, Int((Double(s.count - 1) * p).rounded()))]
    }

    /// The whole check; every line goes to `log` (the app prefixes `BEND check `).
    static func run(_ o: Options, log: (String) -> Void) async -> Bool {
        let base = footprint()
        let per = o.unitsFor.map { "\($0.key)=\($0.name)" }.joined(separator: ",")
        log("vision dir=\(o.dir.path) units=\(o.unitsName)\(per.isEmpty ? "" : " units_for=\(per)") iters=\(o.iters) baseline \(mem)")
        let top = list(o.dir)
        guard !top.isEmpty else { log("vision: no packages in \(o.dir.path) (stage them: scripts/vision-stage.sh)"); return false }
        let detURL = o.det.map { n in top.first { $0.deletingPathExtension().lastPathComponent == n || $0.lastPathComponent == n } } ?? top.first(where: isPackage)
        var segURLs: [URL] = []
        if o.seg != "none", let segDir = o.seg.map({ o.dir.appendingPathComponent($0) })
            ?? top.first(where: { !isPackage($0) && list($0).contains(where: isPackage) }) {
            segURLs = list(segDir).filter(isPackage).sorted { segOrder($0) < segOrder($1) }
            guard !segURLs.isEmpty else { log("vision: no segmenter packages in \(segDir.path)"); return false }
        }
        guard let detURL else { log("vision: no detector package (\(o.det ?? "first top-level") not in \(o.dir.path))"); return false }
        log("vision models det=\(detURL.lastPathComponent) seg=\(segURLs.isEmpty ? "none" : segURLs.map(\.lastPathComponent).joined(separator: ","))")

        var stages: [Stage] = []
        guard let det = await load("det", detURL, o, log: log) else { return false }
        stages.append(det)
        for u in segURLs {
            guard let s = await load("seg", u, o, log: log) else { return false }
            stages.append(s)
        }
        // Fixed inputs once per stage (the photo, the fills); chained ones replace them per frame.
        for s in stages {
            for (_, d) in s.model.modelDescription.inputDescriptionsByName {
                do { s.fixed[d.name] = try synthetic(d, photo: o.photo) }
                catch { log("vision input \(s.name).\(d.name) FAILED: \(error.localizedDescription)"); return false }
            }
        }
        let loaded = footprint()
        log("vision loaded \(stages.count) model(s) delta_mb=\(loaded - base) \(mem)")

        var det_ms: [Double] = [], seg_ms: [Double] = [], all_ms: [Double] = []
        var peak = loaded, minAvail = available(), answered = 0
        for iter in 1...o.iters {
            var dMs = 0.0, sMs = 0.0, parts: [String] = [], ok = true
            autoreleasepool {
                var upstream: [String: MLFeatureValue] = [:]
                for s in stages {
                    var t0 = Date()
                    var result: Result<MLFeatureProvider, Error>
                    do { result = .success(try s.model.prediction(from: inputs(s, upstream: upstream))) } catch { result = .failure(error) }
                    if case .failure(let e) = result, s.chained, s.role == "seg" {
                        // A chained shape the model refuses: its own fills from now on.
                        s.chained = false
                        log("vision \(s.name): chained inputs refused (\(e.localizedDescription)), using fills")
                        t0 = Date()
                        do { result = .success(try s.model.prediction(from: inputs(s, upstream: [:]))) } catch { result = .failure(error) }
                    }
                    let ms = Date().timeIntervalSince(t0) * 1000
                    switch result {
                    case .failure(let e):
                        log("vision \(s.name) prediction FAILED at iter \(iter): \(e.localizedDescription)")
                        ok = false
                        return
                    case .success(let out):
                        // Chain the segmenter's own outputs only (the detector's `boxes` is not the prompt encoder's).
                        if s.role == "seg" { for n in out.featureNames { if let v = out.featureValue(for: n) { upstream[n] = v } } }
                    }
                    if s.role == "det" { dMs += ms } else { sMs += ms }
                    parts.append(String(format: "%@:%.1f", s.name, ms))
                }
            }
            let fp = footprint(), av = available()
            peak = max(peak, fp)
            if av >= 0 { minAvail = min(minAvail, av) }
            guard ok else { return false }
            answered += 1
            det_ms.append(dMs); seg_ms.append(sMs); all_ms.append(dMs + sMs)
            log(String(format: "vision answered iter=%d det_ms=%.1f seg_ms=%.1f ms=%.1f footprint_mb=%d available_mb=%d parts=%@",
                       iter, dMs, sMs, dMs + sMs, fp, av, parts.joined(separator: ",")))
        }
        // iter 1 is the cold frame (first ANE dispatch); the percentiles are over the warm ones when there are any.
        let warm = { (xs: [Double]) in xs.count > 1 ? Array(xs.dropFirst()) : xs }
        log(String(format: "vision summary iters=%d cold_ms=%.1f det_ms_p50=%.1f det_ms_p90=%.1f seg_ms_p50=%.1f seg_ms_p90=%.1f ms_p50=%.1f ms_p90=%.1f footprint_mb_base=%d footprint_mb_loaded=%d footprint_mb_peak=%d delta_mb=%d available_mb_min=%d units=%@",
                   answered, all_ms.first ?? -1, percentile(warm(det_ms), 0.5), percentile(warm(det_ms), 0.9),
                   percentile(warm(seg_ms), 0.5), percentile(warm(seg_ms), 0.9), percentile(warm(all_ms), 0.5), percentile(warm(all_ms), 0.9),
                   base, loaded, peak, peak - base, minAvail, o.unitsName))
        return answered == o.iters
    }
}

#if !VISION_STANDALONE // the Mac CLI test builds VisionCheck alone
extension EngineCheck {
    /// `-engineCheck vision` (or G27's `pocket`) without `-visionWithChat`: the launch lifecycle leaves chat cold, so the
    /// footprint is the app + the vision stack only (PHONE-CEILING's swap-mode base).
    static let visionOnly: Bool = {
        let a = ProcessInfo.processInfo.arguments
        guard let i = a.firstIndex(of: "-engineCheck"), i + 1 < a.count else { return false }
        return ["vision", "pocket", "ears"].contains(a[i + 1]) && !a.contains("-visionWithChat")  // G38: ears too
    }()

    static func vision(key: String) async -> Bool {
        let args = ProcessInfo.processInfo.arguments
        if args.contains("-visionWithChat") {
            let warmed = await chat(key: key)
            while LLBEngine.shared.runningBy != nil { try? await Task.sleep(for: .milliseconds(20)) }
            say("vision with chat warm (p2 \(warmed ? "answered" : "FAILED")) footprint_mb=\(EngineManager.footprint())")
            guard warmed else { return false }
        }
        let fm = FileManager.default
        let o = VisionCheck.Options.parse(args,
            documents: fm.urls(for: .documentDirectory, in: .userDomainMask)[0],
            cache: fm.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("vision-mlmodelc"),
            photo: fixtures.appendingPathComponent("look-news-1280.jpg"))
        return await VisionCheck.run(o) { say($0) }
    }
}
#endif
