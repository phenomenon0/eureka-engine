import CryptoKit
import ImageIO
import Foundation
import SwiftUI

// PHONECAM's Mac harness (scripts/cam-mac.sh): the app's camera code (ios/Bend/Camera/*.swift, minus the iOS capture
// view) + its engine layer (Engines/*.swift) over the vendored macOS slice of LLBEngine — the same Swift the phone
// runs, in process, so the pipeline's bits and timings are measured where the simulator can't run the engine.
//   cam-mac fixtures <list.tsv> <out.jsonl>       every battery canvas through CameraPipeline.check (the tap path)
//   cam-mac timing <list.tsv> <n> <gap_ms> <out>  n warm taps per item after 3 warm-ups, gap_ms idle before each;
//                                                  + the camera-size path: a 1280×720 frame letterboxed in the timed span
//   cam-mac replay <clip.rgb> <frames> <out>      raw 320×320 rgb24 frames (15 fps) through the LIVE path: gate + engine
//   cam-mac synthetic <k> <out>                    k generated poses (vlive1 G4's recipe) through the LIVE path
//   cam-mac shots <list.tsv> <outdir>              the surface rendered (390×844 @3x) on real verdicts
//   cam-mac caption <list.tsv> <outdir> [n]        CAMCAP: the caption card on real captions (VisionCaptioner, the
//                                                  chat look path): loading / streaming / done shots, + failed and
//                                                  unavailable cards; n runs each (the text must repeat: greedy)
// Env as engine-mac.sh: BEND_MODELS_DIR (qwen35-4b-q4km/, kev-4b-q4km/, qwen35-4b-mmproj/), BEND_CACHE_DIR.

let args = CommandLine.arguments
@MainActor func say(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
func emit(_ h: FileHandle, _ obj: [String: Any]) {
    let d = try! JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
    h.write(d); h.write("\n".data(using: .utf8)!)
}
func hex(_ v: UInt32) -> String { String(format: "%08x", v) }
func sha(_ b: [UInt8]) -> String { SHA256.hash(data: Data(b)).map { String(format: "%02x", $0) }.joined() }
func out(_ path: String) -> FileHandle {
    FileManager.default.createFile(atPath: path, contents: nil)
    return FileHandle(forWritingAtPath: path)!
}
func list(_ path: String) -> [(id: String, path: String, q: String)] {
    (try! String(contentsOfFile: path, encoding: .utf8)).split(separator: "\n").compactMap {
        let p = $0.split(separator: "\t").map(String.init)
        return p.count == 3 ? (p[0], p[1], p[2]) : nil
    }
}
func canvasOf(_ path: String) -> Canvas { Canvas.ppm(FileManager.default.contents(atPath: path)!)! }

/// One verdict through the pipeline's tap path, waited for.
final class Waiter: @unchecked Sendable {
    let sem = DispatchSemaphore(value: 0)
    var got: [(CameraVerdict, CameraPipeline.Origin)] = []
    let lock = NSLock()
}

func run() throws {
    let mode = args[1]
    let engine = CameraEngine.shared
    let pipe = CameraPipeline(engine: engine)
    let w = Waiter()
    pipe.onVerdict = { v, o, _ in w.lock.withLock { w.got.append((v, o)) }; w.sem.signal() }
    let t0 = Date()
    // as the app: CameraPipeline.prepare (the engine up, then one discarded warm-up request; keep-warm armed)
    let readySem = DispatchSemaphore(value: 0)
    final class Box: @unchecked Sendable { var r: Result<Void, Error>? }
    let box = Box()
    pipe.onReady = { r in box.r = r; readySem.signal() }
    pipe.prepare()
    readySem.wait()
    try box.r!.get()
    let prep = Date().timeIntervalSince(t0) * 1000
    let footprint = EngineManager.footprint()
    func tap(_ c: Canvas, _ q: String) -> CameraVerdict {
        pipe.setQuestion(q)
        pipe.check(c)
        w.sem.wait()
        return w.lock.withLock { w.got.removeLast().0 }
    }
    switch mode {
    case "fixtures":
        let h = out(args[3])
        emit(h, ["prepare_ms": prep, "footprint_mb": footprint, "resident": LLBEngine.residentReport, "prefix_ids": CameraConfig.vlive1Prefix.count])
        for it in list(args[2]) {
            let c = canvasOf(it.path)
            let v = tap(c, it.q)
            EngineManager.log(v.logLine("tap"))
            emit(h, ["id": it.id, "canvas_sha256": sha(c.rgb), "yes_bits": hex(v.yesBits), "no_bits": hex(v.noBits),
                     "score": v.score, "verdict": v.yes ? "yes" : "no", "confidence": v.confidence, "first_id": v.firstID,
                     "frame_ms": v.frameMs, "engine_ms": v.engineMs, "tower_ms": v.towerMs, "llm_ms": v.llmMs])
        }
        emit(h, ["resident": LLBEngine.residentReport, "footprint_mb": EngineManager.footprint()])
    case "timing":
        let items = list(args[2]), n = Int(args[3])!, gap = Double(args[4])!
        let h = out(args[5])
        emit(h, ["prepare_ms": prep, "footprint_mb": footprint, "gap_ms": gap])
        for it in items {
            let c = canvasOf(it.path)
            for _ in 0..<3 { _ = tap(c, it.q) }
            for r in 0..<n {
                if gap > 0 { usleep(useconds_t(gap * 1000)) }
                let v = tap(c, it.q)
                emit(h, ["id": it.id, "path": "canvas", "rep": r, "frame_ms": v.frameMs, "engine_ms": v.engineMs,
                         "tower_ms": v.towerMs, "llm_ms": v.llmMs, "yes_bits": hex(v.yesBits), "no_bits": hex(v.noBits)])
            }
            // the camera-size path: a 1280×720 frame (the canvas upscaled into a 16:9 frame) → letterbox inside the timing
            var big = [UInt8](repeating: 0, count: 1280 * 720 * 3)
            for y in 0..<720 { for x in 0..<1280 {
                let sx = min(319, x * 320 / 1280), sy = min(319, y * 320 / 720)
                for k in 0..<3 { big[(y * 1280 + x) * 3 + k] = c.rgb[(sy * 320 + sx) * 3 + k] }
            } }
            for r in 0..<n {
                if gap > 0 { usleep(useconds_t(gap * 1000)) }
                let f0 = Date()
                let lb = Canvas.letterbox(rgb: big, width: 1280, height: 720)
                let lbMs = Date().timeIntervalSince(f0) * 1000
                let v = tap(lb, it.q)
                emit(h, ["id": it.id, "path": "frame1280x720", "rep": r, "letterbox_ms": lbMs, "frame_ms": lbMs + v.frameMs,
                         "engine_ms": v.engineMs, "verdict": v.yes ? "yes" : "no"])
            }
        }
        emit(h, ["resident": LLBEngine.residentReport, "footprint_mb": EngineManager.footprint()])
    case "replay", "synthetic":
        let h = out(args[mode == "replay" ? 4 : 3])
        var frames: [(Canvas, Double, Int?)] = []
        if mode == "replay" {
            let d = FileManager.default.contents(atPath: args[2])!, nF = Int(args[3])!, fb = 320 * 320 * 3
            for i in 0..<nF { frames.append((Canvas(side: 320, rgb: [UInt8](d[(i * fb)..<((i + 1) * fb)])), Double(i) * 1000 / 15, nil)) }
        } else {
            let seq = SyntheticSequence(poses: SyntheticSequence.patterns(Int(args[2])!))
            for i in 0..<seq.count { let f = seq.frame(i); frames.append((f.canvas, f.tMs, f.pose)) }
        }
        pipe.setQuestion("Is there a person in this image?")
        pipe.setLive(true)
        // the gate's own decisions, frame by frame (a second gate on the same frames: what the pipeline's gate sees)
        var shadow = PixelGate()
        var t = Date()
        for (i, f) in frames.enumerated() {
            let g0 = Date()
            let st = f.0.rgb.withUnsafeBufferPointer { shadow.step($0, width: 320, height: 320, tMs: f.1) }
            let gateUs = Date().timeIntervalSince(g0) * 1e6
            pipe.frame(f.0, tMs: f.1)
            var row: [String: Any] = ["frame": i, "t_ms": f.1, "fire": st.fire, "state": st.state.rawValue,
                                      "d_anchor": st.dAnchor, "gate_us": gateUs]
            if let p = f.2 { row["pose"] = p }
            if let r = st.dRef { row["d_ref"] = r }
            emit(h, row)
            // real time: 15 fps
            let next = t.addingTimeInterval(1.0 / 15)
            let wait = next.timeIntervalSinceNow
            if wait > 0 { usleep(useconds_t(wait * 1e6)) }
            t = next
        }
        pipe.drain()
        for (v, o) in w.lock.withLock({ w.got }) {
            emit(h, ["verdict": v.yes ? "yes" : "no", "origin": o.rawValue, "score": v.score, "frame_ms": v.frameMs,
                     "engine_ms": v.engineMs, "yes_bits": hex(v.yesBits), "no_bits": hex(v.noBits)])
        }
        let stats: CameraPipeline.Stats? = pipe.currentStats()
        if let s = stats {
            emit(h, ["frames": s.frames, "fires": s.fires, "skipped": s.skipped, "warms": s.warms, "verdicts": s.verdicts,
                     "errors": s.errors, "duty_last10s": s.duty, "footprint_mb": EngineManager.footprint()])
        }
    case "look":
        // ATTACHVIS: chat looks (VisionLook) on images (PNG/JPEG via ImageIO, or a canvas PPM); n runs each (bits: the
        // ids must repeat exactly, greedy). list: id \t image path \t question ("-" = the default question)
        let items = list(args[2]), h = out(args[3]), n = args.count > 4 ? Int(args[4])! : 1
        emit(h, ["prepare_ms": prep, "footprint_mb": footprint, "canvas_side": VisionLook.canvasSide])
        for it in items {
            let image: CGImage = it.path.hasSuffix(".ppm") ? canvasOf(it.path).cgImage()!
                : try VisionLook.decode(FileManager.default.contents(atPath: it.path)!.base64EncodedString())
            let q = it.q == "-" ? "" : it.q
            var first: [UInt32]?
            for r in 0..<n {
                let res = try VisionLook.shared.look(image, question: q)
                let same = first.map { $0 == res.ids } ?? true
                if first == nil { first = res.ids }
                FileHandle.standardError.write(Data(("look \(it.id) r\(r) \(String(format: "%.0f", res.totalMs)) ms, \(res.ids.count) ids: \(res.text)\n").utf8))
                emit(h, ["id": it.id, "rep": r, "question": q.isEmpty ? VisionLook.defaultQuestion : q, "image": "\(image.width)x\(image.height)",
                         "image_tokens": res.imageTokens, "prompt_ids": res.promptIds, "new_ids": res.ids.count, "finish": res.finish,
                         "first_ms": res.firstMs, "total_ms": res.totalMs, "ms_per_id": res.totalMs / Double(max(1, res.ids.count)),
                         "tower_ms_sum": res.towerMsSum, "llm_ms_sum": res.llmMsSum, "ids_same_as_r0": same,
                         "ids": res.ids.map(String.init).joined(separator: " "), "text": res.text])
            }
        }
        emit(h, ["resident": LLBEngine.residentReport, "footprint_mb": EngineManager.footprint()])
    case "gen":
        // IMGPERF: the image generate's byte gate (llb_generate_rgb on the resident session) against ENGINE-IPHONEVIS's
        // Q3 goldens: a PNG's raw pixels (ImageIO's decoded bytes, no colour conversion: what the CLI's decoder gives)
        // at its own size, the golden prompt (template.ids: 248056 = the image's pads), greedy n, ids compared.
        // args: <png> <template.ids> <expect.ids> <n> <out.jsonl> [reps]
        let png = args[2], h = out(args[6]), n = Int(args[5])!, reps = args.count > 7 ? Int(args[7])! : 1
        let tmpl = (try! String(contentsOfFile: args[3], encoding: .utf8)).split(whereSeparator: \.isWhitespace).compactMap { UInt32($0) }
        // the golden .out: one id per line, then a "wall_ms N" line (not an id)
        let want = (try! String(contentsOfFile: args[4], encoding: .utf8)).split(separator: "\n").filter { !$0.hasPrefix("wall_ms") }
            .compactMap { UInt32($0.trimmingCharacters(in: .whitespaces)) }
        let cut = tmpl.firstIndex(of: 248056)!
        let pre = Array(tmpl[..<cut]), post = Array(tmpl[(cut + 1)...])
        let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: png) as CFURL, nil)!
        let img = CGImageSourceCreateImageAtIndex(src, 0, nil)!
        let raw = [UInt8](img.dataProvider!.data! as Data)
        let bpp = img.bitsPerPixel / 8, iw = img.width, ih = img.height
        var rgb = [UInt8](repeating: 0, count: iw * ih * 3)
        for y in 0..<ih { for x in 0..<iw { for c in 0..<3 { rgb[(y * iw + x) * 3 + c] = raw[y * img.bytesPerRow + x * bpp + c] } } }
        // vpre's grid: each side to the nearest multiple of 32 (277 x 211 → 9 x 7, 640 x 488 → 20 x 15)
        let tokens = Int((Double(iw) / 32).rounded()) * Int((Double(ih) / 32).rounded())
        for r in 0..<reps {
            let t = Date()
            let res = try LLBEngine.shared.run("chat") {
                try LLBEngine.shared.generateImageLocked(rgb: rgb, side: iw, height: ih, pre: pre, post: post, imageTokens: tokens,
                                                         nNew: n) { _, _ in true }
            }
            let got = Array(res.ids.prefix(n))
            let k = zip(got, want).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset
            let same = got.count >= want.count && k == nil
            FileHandle.standardError.write(Data(("gen \(png) r\(r): \(same ? "PASS (exact) \(want.count)/\(want.count) ids identical" : "DIFF at \(k ?? got.count)") rows_kept=\(res.stats.rows_kept) n_img=\(res.stats.n_img) tower_ms=\(String(format: "%.0f", res.stats.tower_ms)) prefill_ms=\(String(format: "%.0f", res.stats.prefill_ms)) decode_ms_per_id=\(String(format: "%.1f", res.stats.steps > 0 ? res.stats.step_ms_sum / Double(res.stats.steps) : 0)) wall_ms=\(String(format: "%.0f", Date().timeIntervalSince(t) * 1000)) stream=\(res.mismatch ?? "ok \(res.streamed)")\n").utf8))
            emit(h, ["png": png, "rep": r, "pass": same, "first_diff": k as Any, "ids": got.map(String.init).joined(separator: " "),
                     "want": want.map(String.init).joined(separator: " "), "rows_kept": res.stats.rows_kept, "n_img": res.stats.n_img,
                     "n_ids": res.stats.n_ids, "tower_ms": res.stats.tower_ms, "prefill_ms": res.stats.prefill_ms,
                     "step_ms_sum": res.stats.step_ms_sum, "steps": res.stats.steps, "streamed": res.streamed, "mismatch": res.mismatch ?? ""])
        }
    case "shots":
        let items = list(args[2]), dir = args[3]
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        var shots: [(String, Canvas, CameraVerdict, CameraPipeline.Origin, Bool)] = []
        for it in items.prefix(3) {
            let c = canvasOf(it.path)
            let v = tap(c, it.q)
            shots.append((it.id, c, v, .tap, false))
        }
        let s = DispatchSemaphore(value: 0)
        Task { @MainActor in
            for (id, c, v, o, live) in shots {
                let m = CameraSurfaceModel(engine: engine)
                m.phase = .ready
                m.question = v.question
                m.frozen = c
                m.verdict = v
                m.verdictOrigin = o
                m.live = live
                m.sourceLabel = "fixture \(id)"
                m.stats = CameraPipeline.Stats(frames: 1, fires: 0, skipped: 0, warms: 0, verdicts: 1)
                let view = CameraSurface(model: m, preview: Color.black, onClose: {}, editableQuestion: false).frame(width: 390, height: 844)
                let r = ImageRenderer(content: view)
                r.scale = 3
                if let img = r.cgImage {
                    let url = URL(fileURLWithPath: "\(dir)/mac-surface-\(id)-390x844.png")
                    let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
                    CGImageDestinationAddImage(dest, img, nil)
                    CGImageDestinationFinalize(dest)
                    say("shot \(url.path)")
                }
            }
            s.signal()
        }
        s.wait()
    case "caption":
        let items = list(args[2]), dir = args[3], n = args.count > 4 ? Int(args[4])! : 2
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let jl = out("\(dir)/mac-caption.jsonl")
        let s = DispatchSemaphore(value: 0)
        final class Down: Captioner, @unchecked Sendable {
            var unavailableReason: String? { nil }
            func caption(_ image: CGImage, onPiece: @escaping @Sendable (String) -> Bool) async throws -> CaptionResult {
                throw CameraEngine.Failure("vision: the engine went away (simulated)")
            }
        }
        final class Missing: Captioner, @unchecked Sendable {
            var unavailableReason: String? { "no vision tower: Documents/models/qwen35-4b-mmproj/mmproj-F16.gguf" }
            func caption(_ image: CGImage, onPiece: @escaping @Sendable (String) -> Bool) async throws -> CaptionResult { fatalError() }
        }
        Task { @MainActor in
            @MainActor func render(_ m: CameraSurfaceModel, _ name: String) {
                let view = CameraSurface(model: m, preview: Color.black, onClose: {}, editableQuestion: false, onSendToChat: {}).frame(width: 390, height: 844)
                let r = ImageRenderer(content: view)
                r.scale = 3
                guard let img = r.cgImage else { return }
                let url = URL(fileURLWithPath: "\(dir)/mac-caption-\(name)-390x844.png")
                let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
                CGImageDestinationAddImage(dest, img, nil)
                CGImageDestinationFinalize(dest)
                say("shot \(url.path)")
            }
            @MainActor func surface(_ it: (id: String, path: String, q: String), _ captioner: Captioner) -> (CameraSurfaceModel, CGImage) {
                let image: CGImage = it.path.hasSuffix(".ppm") ? canvasOf(it.path).cgImage()!
                    : try! VisionLook.decode(FileManager.default.contents(atPath: it.path)!.base64EncodedString())
                let m = CameraSurfaceModel(engine: engine, captioner: captioner)
                m.phase = .ready
                m.question = it.q
                m.frozen = Canvas.letterbox(image)
                m.sourceLabel = "fixture \(it.id)"
                return (m, image)
            }
            @MainActor func waitFor(_ m: CameraSurfaceModel, _ what: (CaptionController.State) -> Bool) async {
                for _ in 0..<12000 where !what(m.caption.state) { try? await Task.sleep(for: .milliseconds(5)) }
            }
            let real = VisionCaptioner()
            for it in items.prefix(3) {
                var texts: [String] = []
                for r in 0..<n {
                    let (m, image) = surface(it, real)
                    m.caption.ask(image)
                    if r == 0 { render(m, "\(it.id)-1-loading") }
                    var sawStreaming = false
                    while true {
                        if case .streaming(let t) = m.caption.state, !sawStreaming, r == 0, t.split(separator: " ").count >= 3 {
                            sawStreaming = true
                            render(m, "\(it.id)-2-streaming")
                        }
                        if case .done = m.caption.state { break }
                        if case .failed = m.caption.state { break }
                        try? await Task.sleep(for: .milliseconds(5))
                    }
                    if r == 0 { render(m, "\(it.id)-3-done") }
                    if case .done(let text, let ms) = m.caption.state {
                        texts.append(text)
                        emit(jl, ["id": it.id, "rep": r, "image": "\(image.width)x\(image.height)", "text": text, "total_ms": ms, "streamed_card": sawStreaming])
                        say("caption \(it.id) r\(r) \(String(format: "%.0f", ms)) ms: \(text)")
                    } else {
                        emit(jl, ["id": it.id, "rep": r, "state": "\(m.caption.state)"])
                    }
                }
                emit(jl, ["id": it.id, "same_text_every_run": Set(texts).count == 1, "runs": texts.count])
            }
            if let it = items.first {
                let (m, image) = surface(it, Down())
                m.caption.ask(image)
                await waitFor(m) { if case .failed = $0 { return true } else { return false } }
                render(m, "\(it.id)-4-failed")
                let (u, uimage) = surface(it, Missing())
                u.caption.ask(uimage)
                render(u, "\(it.id)-5-unavailable")
            }
            s.signal()
        }
        s.wait()
    default:
        throw CameraEngine.Failure("unknown mode \(mode)")
    }
}

MainActor.assumeIsolated { EngineManager.shared.start() }
Thread.detachNewThread {
    do { try run(); exit(0) } catch { FileHandle.standardError.write("cam-mac: \(error.localizedDescription)\n".data(using: .utf8)!); exit(1) }
}
RunLoop.main.run()
