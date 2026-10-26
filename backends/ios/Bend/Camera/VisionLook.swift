import CoreGraphics
import Foundation
import ImageIO

/// ATTACHVIS → IMGPERF → IMAGETRUE: a photo → a free-text answer grounded in it, on the camera's in-process vision path
/// (CameraEngine: the chat model's resident session + the 4B Metal tower). One look is ONE resident job
/// (llb_generate_rgb): the tower once (or not at all: the engine keeps the last canvases' rows by their bytes), one
/// prefill of prefix + pads + the turn, then decode at chat's speed, each id streamed as it lands. Every caller rides
/// it: chat photo looks, the camera's caption, and the Look surface's conversation (IMAGETRUE: `earlier` turns render
/// after the image, so the photo stays in context and the prefix snapshot means a follow-up prefills only its turn).
/// `-lookLoop` keeps ATTACHVIS's greedy per-id loop over llb_classify_rgb (one tower + prefill per id) for the A/B.
final class VisionLook: @unchecked Sendable {
    static let shared = VisionLook()

    /// The look's canvas: side × side, letterboxed (CameraConfig.canvasSide's rule). 448 → 196 image tokens, inside
    /// the session image area (1024) and vlive1 G0's measured sizes (224 / 320 / 448). BEND_LOOK_CANVAS (a multiple of
    /// 32, ≤ 1024; the Mac harnesses' A/B) overrides it.
    static let canvasSide: Int = {
        guard let v = ProcessInfo.processInfo.environment["BEND_LOOK_CANVAS"].flatMap(Int.init), v % 32 == 0, (224...1024).contains(v) else { return 448 }
        return v
    }()
    /// The answer's id budget: a look is one to three sentences (spoken on the Look surface: short reads better aloud).
    static let defaultMaxNew = 64
    static let maxNew = 160

    static let system = "You are Eureka, an assistant running entirely on the person's phone. You can see the photo the person sent. Answer their question about it plainly and briefly, in one to three sentences, from what is in the photo."
    /// The question when the message has a photo and no words (bend-app R19's LOOK_PROMPT).
    static let defaultQuestion = "Describe this photo in one or two sentences."

    static func prefixText() -> String { "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n<|vision_start|>" }
    static func suffixText(_ question: String) -> String {
        "<|vision_end|>\(question)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// IMAGETRUE: one earlier turn of a conversation about the photo (EngineManager.swift: it builds without the camera).
    typealias Exchange = VisionLookExchange

    /// IMAGETRUE: the text after the image when the photo has a conversation: the image turn's question, then every
    /// earlier exchange as chat renders history (LLBTokenizer.render: past assistant turns without a think block), then
    /// the new question with the no-think opener. The image stays where it was (the first user turn), so `pre` and the
    /// image rows are the same for every turn: the engine's prefix snapshot (pre + image) is restored and only this
    /// text is prefilled. No earlier turns: exactly `suffixText` (the single look, byte for byte).
    static func conversationText(_ earlier: [Exchange], question: String) -> String {
        guard let first = earlier.first else { return suffixText(question) }
        let trim = { (s: String) in s.trimmingCharacters(in: .whitespacesAndNewlines) }
        var out = "<|vision_end|>\(trim(first.asked))<|im_end|>\n<|im_start|>assistant\n\(trim(first.answered))<|im_end|>\n"
        for e in earlier.dropFirst() {
            out += "<|im_start|>user\n\(trim(e.asked))<|im_end|>\n<|im_start|>assistant\n\(trim(e.answered))<|im_end|>\n"
        }
        return out + "<|im_start|>user\n\(question)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ m: String) { errorDescription = m }
    }

    struct Result: Sendable {
        var text: String
        var ids: [UInt32]
        var promptIds: Int
        var imageTokens: Int
        var finish: String
        var firstMs: Double
        var totalMs: Double
        var towerMsSum: Double
        var llmMsSum: Double
        /// IMGPERF: ensureReady (chat warm + the tower open, if either was cold), the letterbox, and how many times
        /// the tower ran for this one look.
        var readyMs: Double = 0
        var letterboxMs: Double = 0
        var towerCalls: Int = 0
        /// The one job's prefill (ms) and decode (ms per id), and whether the photo's rows were kept (no tower ran).
        var prefillMs: Double = 0
        var decodeMsPerId: Double = 0
        var rowsKept = false
        /// IMAGETRUE: prompt ids restored from the engine's prefix snapshot (not prefilled; 0: none).
        var prefixIds = 0
    }

    private let lock = NSLock()
    private var tokenizer: LLBTokenizer?
    private var lastUse = Date.distantPast
    private var closer: DispatchWorkItem?
    /// Looks running now: neither the idle close nor the camera surface's close takes the tower from under one.
    private var running = 0
    var inFlight: Bool { lock.withLock { running > 0 } }
    /// The tower closes this long after the last look (the camera surface closes it on its own when it goes).
    /// BEND_LOOK_IDLE_S overrides it (IMGPERF's measurement: "the person came back after the close", in seconds).
    static let idleClose: TimeInterval = ProcessInfo.processInfo.environment["BEND_LOOK_IDLE_S"].flatMap(Double.init) ?? 90
    /// True while the camera surface is up (it owns the tower then; the idle close leaves it alone). Set by CameraView.
    static var cameraHeld: Bool {
        get { heldLock.withLock { held } }
        set { heldLock.withLock { held = newValue } }
    }
    private static let heldLock = NSLock()
    nonisolated(unsafe) private static var held = false

    /// Why this build can't look, or nil: the camera's reasons (simulator Metal, -kevChat, the model, the tower).
    static var unavailableReason: String? { CameraEngine.unavailableReason }

    /// A data URL or base64 image (JPEG / PNG / HEIC: whatever ImageIO decodes) → its CGImage.
    static func decode(_ source: String) throws -> CGImage {
        var b64 = Substring(source)
        if source.hasPrefix("data:") {
            guard let comma = source.firstIndex(of: ",") else { throw Failure("image: a data URL without data") }
            b64 = source[source.index(after: comma)...]
        }
        guard let data = Data(base64Encoded: String(b64), options: .ignoreUnknownCharacters), !data.isEmpty else {
            throw Failure("image: not base64")
        }
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary) else {
            throw Failure("image: couldn't decode it (\(data.count) bytes)")
        }
        return image
    }

    /// IMGPERF: `-lookLoop` (or BEND_LOOK_LOOP=1) keeps ATTACHVIS's per-id loop over llb_classify_rgb (the before path,
    /// for the A/B). Off: one llb_generate_rgb per look.
    static let loop: Bool = ProcessInfo.processInfo.arguments.contains("-lookLoop")
        || ProcessInfo.processInfo.environment["BEND_LOOK_LOOP"] == "1"

    /// One look: `image` + `question` → the answer, each piece through `onPiece` as it lands (false: stop now).
    /// IMGPERF: the tower runs once on the canvas, or not at all when the engine keeps this photo's rows (a look at the
    /// same photo again: a follow-up question, "Ask about this photo", a look after the tower's idle close), then one
    /// resident job prefills once and decodes at chat's speed (llb_generate_rgb). The ids are the same greedy decode.
    func look(_ image: CGImage, question: String, maxNew: Int = defaultMaxNew,
              onPiece: @escaping (String) -> Bool = { _ in true }) throws -> Result {
        if Self.loop { return try loopLook(image, question: question, maxNew: maxNew, onPiece: onPiece) }
        let tl = Date()
        let canvas = Self.canvas(image)
        return try look(canvas: canvas, question: question, maxNew: maxNew, letterboxMs: Date().timeIntervalSince(tl) * 1000,
                        onPiece: onPiece)
    }

    /// The photo as the tower sees it: letterboxed into the look's canvas.
    static func canvas(_ image: CGImage) -> Canvas { Canvas.letterbox(image, side: canvasSide) }

    /// A look at a canvas already letterboxed (a photo the web handed over ahead: send-once).
    func look(canvas: Canvas, question: String, earlier: [Exchange] = [], maxNew: Int = defaultMaxNew, letterboxMs: Double = 0,
              onPiece: @escaping (String) -> Bool = { _ in true }) throws -> Result {
        if Self.loop, earlier.isEmpty, let image = canvas.cgImage() { return try loopLook(image, question: question, maxNew: maxNew, onPiece: onPiece) }
        let t0 = Date()
        if let why = Self.unavailableReason { throw Failure("no-vision: \(why)") }
        lock.withLock { running += 1; lastUse = Date() }
        defer { lock.withLock { running -= 1 }; touch() }
        let kept = LLBEngine.shared.run("camera") { LLBEngine.shared.hasImageLocked(rgb: canvas.rgb, side: canvas.side) }
        let tr = Date()
        try CameraEngine.shared.ensureReady(tower: !kept)
        let readyMs = Date().timeIntervalSince(tr) * 1000
        let tok = try tokenizerLocked()
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        let pre = tok.encode(Self.prefixText())
        let post = tok.encode(Self.conversationText(earlier, question: q.isEmpty ? Self.defaultQuestion : q))
        let budget = min(max(maxNew, 1), Self.maxNew)
        let rows = (canvas.side / 32) * (canvas.side / 32)
        // The pieces land on the poller's thread, one id at a time and in order.
        let state = PieceState()
        let r = try LLBEngine.shared.run("chat") {
            try LLBEngine.shared.generateImageLocked(rgb: canvas.rgb, side: canvas.side, pre: pre, post: post, imageTokens: rows,
                                                     nNew: budget) { _, id in
                if id == tok.imEnd || id == tok.endOfText { state.finish = "stop"; return false }
                if state.firstMs < 0 { state.firstMs = Date().timeIntervalSince(t0) * 1000 }
                state.ids.append(id)
                let piece = state.joiner.push(tok.piece(id))
                state.text += piece
                if !piece.isEmpty, !onPiece(piece) { state.finish = "aborted"; return false }
                return true
            }
        }
        if let m = r.mismatch { EngineManager.log("look stream \(m)") }
        let rest = state.joiner.flush()
        if !rest.isEmpty, state.finish != "aborted" { state.text += rest; _ = onPiece(rest) }
        let st = r.stats
        return Result(text: state.text, ids: state.ids, promptIds: Int(st.n_ids), imageTokens: Int(st.n_img), finish: state.finish,
                      firstMs: state.firstMs, totalMs: Date().timeIntervalSince(t0) * 1000, towerMsSum: st.tower_ms,
                      llmMsSum: st.llm_ms, readyMs: readyMs, letterboxMs: letterboxMs, towerCalls: st.rows_kept != 0 ? 0 : 1,
                      prefillMs: st.prefill_ms, decodeMsPerId: st.steps > 0 ? st.step_ms_sum / Double(st.steps) : 0,
                      rowsKept: st.rows_kept != 0, prefixIds: Int(st.prefix_ids))
    }

    // MARK: native aspect (BENDHARVEST H2: the crew's R19 vision gate)

    /// A photo at its own aspect, as the tower takes it: both sides multiples of 32 (one image token per 32×32 after
    /// the merge), at most `maxTokens` tokens (the session's image area, CameraConfig.imageArea), never upscaled.
    struct NativeCanvas: Sendable {
        let width: Int
        let height: Int
        let rgb: [UInt8]
        var tokens: Int { (width / 32) * (height / 32) }
    }

    /// The canvas for a native-aspect look. A photo already on the 32 grid and within the area passes through pixel for
    /// pixel (its decoded bytes, no colour conversion: the invoice golden's 1024×640 PNG is exactly what R19's CLI read);
    /// anything else is scaled down onto the grid (aspect kept to within a patch).
    static func nativeCanvas(_ image: CGImage, maxTokens: Int = Int(CameraConfig.imageArea)) -> NativeCanvas {
        let (w, h) = (image.width, image.height)
        if w % 32 == 0, h % 32 == 0, (w / 32) * (h / 32) <= maxTokens, let rgb = rawRGB(image) {
            return NativeCanvas(width: w, height: h, rgb: rgb)
        }
        let s = min(1, (Double(maxTokens * 1024) / Double(w * h)).squareRoot())
        var gw = max(1, Int(Double(w) * s / 32)), gh = max(1, Int(Double(h) * s / 32))
        while gw * gh > maxTokens { if gw >= gh { gw -= 1 } else { gh -= 1 } }
        let (cw, ch) = (gw * 32, gh * 32)
        var rgba = [UInt8](repeating: 255, count: cw * ch * 4)
        rgba.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: cw, height: ch, bitsPerComponent: 8, bytesPerRow: cw * 4,
                                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: cw, height: ch))
        }
        var rgb = [UInt8](repeating: 0, count: cw * ch * 3)
        for p in 0..<(cw * ch) { rgb[p * 3] = rgba[p * 4]; rgb[p * 3 + 1] = rgba[p * 4 + 1]; rgb[p * 3 + 2] = rgba[p * 4 + 2] }
        return NativeCanvas(width: cw, height: ch, rgb: rgb)
    }

    /// The image's own 8-bit RGB bytes (RGB, RGBA/RGBX in that order), or nil for any other layout (then it's drawn).
    private static func rawRGB(_ image: CGImage) -> [UInt8]? {
        guard image.bitsPerComponent == 8, let data = image.dataProvider?.data as Data? else { return nil }
        let bpp = image.bitsPerPixel / 8, row = image.bytesPerRow, (w, h) = (image.width, image.height)
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        guard order == [] || order == .byteOrder32Big || (bpp == 3 && order == .byteOrderDefault) else { return nil }
        switch (bpp, image.alphaInfo) {
        case (3, .none), (4, .noneSkipLast), (4, .last), (4, .premultipliedLast): break
        default: return nil
        }
        guard data.count >= row * h else { return nil }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            for y in 0..<h {
                for x in 0..<w {
                    let i = y * row + x * bpp, o = (y * w + x) * 3
                    rgb[o] = src[i]; rgb[o + 1] = src[i + 1]; rgb[o + 2] = src[i + 2]
                }
            }
        }
        return rgb
    }

    /// The crew's R19 prompt, split at the image: Qwen3.5's no-think user turn with the image ahead of the text and no
    /// system turn, rendered by the same template R19's shell route used (LLBTokenizer.render).
    static func r19Prompt(_ prompt: String) throws -> (pre: String, post: String) {
        let rendered = try LLBTokenizer.render([.init(role: "user", content: "<|vision_start|><|image_pad|><|vision_end|>" + prompt)],
                                               enableThinking: false)
        guard let pad = rendered.range(of: "<|image_pad|>") else { throw Failure("look: no image pad in the rendered prompt") }
        return (String(rendered[..<pad.lowerBound]), String(rendered[pad.upperBound...]))
    }

    /// BENDHARVEST H2: a look at a photo's native aspect (no letterbox) with R19's prompt, through the same one resident
    /// job (llb_generate_rgb). The tower always runs (native canvases aren't in the send-once cache). Greedy, `maxNew` ids,
    /// ending at the first stop id: what `-engineCheck look-golden` compares with the CLI golden.
    func lookNative(_ image: CGImage, prompt: String, maxNew: Int) throws -> (result: Result, canvas: NativeCanvas) {
        let t0 = Date()
        if let why = Self.unavailableReason { throw Failure("no-vision: \(why)") }
        lock.withLock { running += 1; lastUse = Date() }
        defer { lock.withLock { running -= 1 }; touch() }
        let tl = Date()
        let canvas = Self.nativeCanvas(image)
        let letterboxMs = Date().timeIntervalSince(tl) * 1000
        let tr = Date()
        try CameraEngine.shared.ensureReady(tower: true)
        let readyMs = Date().timeIntervalSince(tr) * 1000
        let tok = try tokenizerLocked()
        let text = try Self.r19Prompt(prompt)
        let pre = tok.encode(text.pre), post = tok.encode(text.post)
        let state = PieceState()
        let r = try LLBEngine.shared.run("chat") {
            try LLBEngine.shared.generateImageLocked(rgb: canvas.rgb, side: canvas.width, height: canvas.height, pre: pre, post: post,
                                                     imageTokens: canvas.tokens, nNew: max(maxNew, 1)) { _, id in
                if id == tok.imEnd || id == tok.endOfText { state.finish = "stop"; return false }
                if state.firstMs < 0 { state.firstMs = Date().timeIntervalSince(t0) * 1000 }
                state.ids.append(id)
                state.text += state.joiner.push(tok.piece(id))
                return true
            }
        }
        if let m = r.mismatch { EngineManager.log("look native stream \(m)") }
        state.text += state.joiner.flush()
        let st = r.stats
        let result = Result(text: state.text, ids: state.ids, promptIds: Int(st.n_ids), imageTokens: Int(st.n_img), finish: state.finish,
                            firstMs: state.firstMs, totalMs: Date().timeIntervalSince(t0) * 1000, towerMsSum: st.tower_ms,
                            llmMsSum: st.llm_ms, readyMs: readyMs, letterboxMs: letterboxMs, towerCalls: st.rows_kept != 0 ? 0 : 1,
                            prefillMs: st.prefill_ms, decodeMsPerId: st.steps > 0 ? st.step_ms_sum / Double(st.steps) : 0,
                            rowsKept: st.rows_kept != 0)
        return (result, canvas)
    }

    // MARK: send-once (IMGPERF)

    /// The photos the web handed over (POST /v1/bend/image on attach, or a look that carried its key), as canvases, by
    /// the web's content key: a look then names the key instead of sending the photo again. The last 4 (~0.6 MB each).
    private var canvases: [(key: String, canvas: Canvas)] = []
    static let keepCanvases = 4

    func store(_ canvas: Canvas, key: String) {
        lock.withLock {
            canvases.removeAll { $0.key == key }
            canvases.append((key, canvas))
            if canvases.count > Self.keepCanvases { canvases.removeFirst(canvases.count - Self.keepCanvases) }
        }
    }

    func canvas(key: String) -> Canvas? { lock.withLock { canvases.last { $0.key == key }?.canvas } }

    /// IMAGETRUE: the camera's shot, for the Look surface. Its look canvas (448, from the full-resolution shot) is held
    /// by a fresh key and the tower starts on it now, on an engine thread: by the time the page asks (its first look names
    /// the key), the rows are kept and the look is one prefill + decode. nil: this process can't look (the simulator,
    /// -kevChat, files missing); the page then hands its own copy over, as for any photo.
    static func hold(_ image: CGImage) -> String? {
        guard unavailableReason == nil else { return nil }
        let t0 = Date()
        let c = canvas(image)
        let key = "cam-" + UUID().uuidString.prefix(12).lowercased()
        shared.store(c, key: key)
        let letterboxMs = Date().timeIntervalSince(t0) * 1000
        EngineManager.onEngineThread("bend.engine.image") {
            do {
                let a = try shared.encodeAhead(c)
                EngineManager.log(String(format: "look shot key=%@ image=%dx%d letterbox_ms=%.1f ahead_ms=%.0f ready_ms=%.0f tower_ms=%.0f was_kept=%d footprint_mb=%d",
                                         key, image.width, image.height, letterboxMs, a.ms, a.readyMs, a.towerMs, a.wasKept ? 1 : 0, EngineManager.footprint()))
            } catch {
                EngineManager.log("look shot key=\(key) ahead failed: \(error.localizedDescription) (the look will encode it)")
            }
        }
        return key
    }

    struct Ahead: Sendable { var ms: Double; var towerMs: Double; var readyMs: Double; var wasKept: Bool }

    /// The photo encoded ahead of its look (it came with the chip, while the person types): chat warm, the tower open,
    /// its rows kept in the engine (llb_vision_keep). The look that follows runs no tower: only its one prefill + decode.
    func encodeAhead(_ canvas: Canvas) throws -> Ahead {
        if let why = Self.unavailableReason { throw Failure("no-vision: \(why)") }
        lock.withLock { running += 1; lastUse = Date() }
        defer { lock.withLock { running -= 1 }; touch() }
        let t0 = Date()
        if LLBEngine.shared.run("camera", { LLBEngine.shared.hasImageLocked(rgb: canvas.rgb, side: canvas.side) }) {
            return Ahead(ms: Date().timeIntervalSince(t0) * 1000, towerMs: 0, readyMs: 0, wasKept: true)
        }
        let tr = Date()
        try CameraEngine.shared.ensureReady(tower: true, ahead: true)
        let readyMs = Date().timeIntervalSince(tr) * 1000
        var st = llb_vstats()
        let rc = LLBEngine.shared.run("camera") {
            canvas.rgb.withUnsafeBufferPointer { llb_vision_keep($0.baseAddress, Int32(canvas.side), Int32(canvas.side), &st) }
        }
        if rc != 0 { throw Failure("vision: \(String(cString: llb_error()))") }
        return Ahead(ms: Date().timeIntervalSince(t0) * 1000, towerMs: st.tower_ms, readyMs: readyMs, wasKept: st.rows_kept != 0)
    }

    private final class PieceState: @unchecked Sendable {
        var ids: [UInt32] = []
        var text = ""
        var joiner = UTF8Joiner()
        var finish = "length"
        var firstMs = -1.0
    }

    private func tokenizerLocked() throws -> LLBTokenizer {
        try lock.withLock { () -> LLBTokenizer in
            if let t = tokenizer { return t }
            let t = try LLBTokenizer(dir: EngineManager.modelDir(.chat))
            tokenizer = t
            return t
        }
    }

    /// ATTACHVIS's look (the before path, `-lookLoop`): greedy decoding over llb_classify_rgb, one tower + prefill per id.
    func loopLook(_ image: CGImage, question: String, maxNew: Int = defaultMaxNew,
                  onPiece: (String) -> Bool = { _ in true }) throws -> Result {
        let t0 = Date()
        if let why = Self.unavailableReason { throw Failure("no-vision: \(why)") }
        lock.withLock { running += 1; lastUse = Date() }
        defer { lock.withLock { running -= 1 }; touch() }
        let tr = Date()
        try CameraEngine.shared.ensureReady()
        let readyMs = Date().timeIntervalSince(tr) * 1000
        var calls = 0
        let tok = try lock.withLock { () -> LLBTokenizer in
            if let t = tokenizer { return t }
            let t = try LLBTokenizer(dir: EngineManager.modelDir(.chat))
            tokenizer = t
            return t
        }
        let tl = Date()
        let canvas = Canvas.letterbox(image, side: Self.canvasSide)
        let letterboxMs = Date().timeIntervalSince(tl) * 1000
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        let pre = tok.encode(Self.prefixText())
        let post = tok.encode(Self.suffixText(q.isEmpty ? Self.defaultQuestion : q))
        let budget = min(max(maxNew, 1), Self.maxNew)
        var ids: [UInt32] = []
        var text = ""
        var joiner = UTF8Joiner()
        var finish = "length"
        var firstMs = -1.0, tower = 0.0, llm = 0.0
        var nImg = 0
        while ids.count < budget {
            // A no-op while the tower is open and chat resident; re-opens either if something took it between ids.
            if !ids.isEmpty { try CameraEngine.shared.ensureReady() }
            var st = llb_vstats()
            let tail = post + ids
            let side = Int32(canvas.side)
            let rc = try LLBEngine.shared.run("chat") { () throws -> Int32 in
                canvas.rgb.withUnsafeBufferPointer { rp in
                    pre.withUnsafeBufferPointer { pp in
                        tail.withUnsafeBufferPointer { sp in
                            llb_classify_rgb(rp.baseAddress, side, side, pp.baseAddress, Int32(pp.count), sp.baseAddress,
                                             Int32(sp.count), CameraConfig.yesID, CameraConfig.noID, &st)
                        }
                    }
                }
            }
            if rc != 0 { throw Failure("vision: \(String(cString: llb_error()))") }
            calls += 1
            tower += st.tower_ms; llm += st.llm_ms; nImg = Int(st.n_img)
            let id = st.first_id
            if id == tok.imEnd || id == tok.endOfText { finish = "stop"; break }
            if firstMs < 0 { firstMs = Date().timeIntervalSince(t0) * 1000 }
            ids.append(id)
            let piece = joiner.push(tok.piece(id))
            text += piece
            if !piece.isEmpty, !onPiece(piece) { finish = "aborted"; break }
        }
        let rest = joiner.flush()
        if !rest.isEmpty { text += rest; _ = onPiece(rest) }
        return Result(text: text, ids: ids, promptIds: pre.count + nImg + post.count, imageTokens: nImg, finish: finish,
                      firstMs: firstMs, totalMs: Date().timeIntervalSince(t0) * 1000, towerMsSum: tower, llmMsSum: llm,
                      readyMs: readyMs, letterboxMs: letterboxMs, towerCalls: calls)
    }

    /// Re-arms the idle close: the tower (641 MiB) goes back once looks stop, unless the camera surface holds it.
    private func touch() {
        lock.withLock {
            lastUse = Date()
            closer?.cancel()
            let item = DispatchWorkItem { [weak self] in
                guard let self else { return }
                let (idle, busy) = self.lock.withLock { (Date().timeIntervalSince(self.lastUse), self.running > 0) }
                if idle >= Self.idleClose - 1, !busy, !Self.cameraHeld { CameraEngine.shared.close() }
            }
            closer = item
            DispatchQueue.global().asyncAfter(deadline: .now() + Self.idleClose, execute: item)
        }
    }
}

/// CAMCAP: the camera card's caption on the engine: VisionLook with the caption question (the chat photo path, on the
/// phone's own tower). Blocking decode, so off the cooperative pool.
final class VisionCaptioner: Captioner {
    var unavailableReason: String? { VisionLook.unavailableReason }

    func caption(_ image: CGImage, onPiece: @escaping @Sendable (String) -> Bool) async throws -> CaptionResult {
        try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let r = try VisionLook.shared.look(image, question: CaptionConfig.question, maxNew: CaptionConfig.maxNew, onPiece: onPiece)
                    cont.resume(returning: CaptionResult(text: r.text, firstMs: r.firstMs, totalMs: r.totalMs, finish: r.finish))
                } catch { cont.resume(throwing: error) }
            }
        }
    }
}
