import Foundation

/// OUR engine for the camera: LLBEngine.xcframework's camera API (llb_engine.h, PHONECAM): the chat model's resident
/// session (KEVPHONE: llb_open, weights loaded once) + the 4B Metal vision tower kept loaded (llb_vision_open) +
/// one resident job per frame (llb_classify_rgb). No llama.cpp, no network: every call is in this process.
final class CameraEngine: VerdictEngine, @unchecked Sendable {
    static let shared = CameraEngine()

    struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ m: String) { errorDescription = m }
    }

    private let lock = NSLock()
    private var tokenizer: LLBTokenizer?
    private var prefix: [UInt32] = []
    private var suffixes: [String: [UInt32]] = [:]
    private var towerOpen = false
    /// BENDSDK: the tower was opened by a /v1/look, not by the camera surface (only those are closed when looks stop).
    private var openedByLook = false

    static var mmprojURL: URL {
        EngineManager.modelsDir.appendingPathComponent(CameraConfig.mmprojDir).appendingPathComponent(CameraConfig.mmprojFile)
    }

    /// Why this build/device can't judge frames, or nil. The simulator: the engine needs a device's Metal (the
    /// simulator caps one buffer at 256 MB and threadgroups below 1024: engine-phone2 §3), so the surface runs there
    /// with its gate and frames but no verdicts.
    static var unavailableReason: String? {
        // PHONEUX: the engine files first (the one fix the person can make), in the words the first screen uses.
        var missing: [String] = []
        if !EngineManager.hasModel(.chat) { missing.append("\(EngineID.chat.dir)/") }
        if !FileManager.default.fileExists(atPath: mmprojURL.path) { missing.append("\(CameraConfig.mmprojDir)/\(CameraConfig.mmprojFile)") }
        let setup = missing.isEmpty ? nil
            : "Engine files not set up — run the one-time setup (missing in Documents/models: \(missing.joined(separator: ", ")))"
        #if targetEnvironment(simulator)
        let sim = "Simulator: our engine needs the device's Metal (256 MB buffer cap here)"
        return setup.map { "\($0)\n\(sim)" } ?? sim
        #else
        if EngineID.kevChat { return "Camera vision needs Two models in Settings (Qwen3.5-4B is the tower's backbone)" }
        return setup
        #endif
    }

    /// At EngineManager.start, before any session opens: every session reserves the image area (llb_vision_area), so
    /// the camera never has to reopen the chat model. ~10 MB; nothing else changes for chat or decide.
    static func reserveArea() {
        if llb_vision_area(CameraConfig.imageArea) != 0 {
            EngineManager.log("camera: image area NOT reserved: \(String(cString: llb_error()))")
        }
    }

    func ensureReady() throws { try ensureReady(tower: true) }

    /// `tower: false` (IMGPERF: a look whose photo's rows the engine keeps): chat resident and the tokenizer, no tower.
    /// `ahead` (IMGPERF: a photo encoded while the person types): nobody waits on this warm yet, so chat's open runs its
    /// warm-up and the look that follows doesn't pay the session's first-job set-up.
    /// `forLook` (BENDSDK: a POST /v1/look): a tower it opens is closed again 30 s after the last look.
    func ensureReady(tower: Bool, ahead: Bool = false, forLook: Bool = false) throws {
        if let why = Self.unavailableReason { throw Failure(why) }
        // The chat model's weights resident (the lifecycle's warm: one load, kept; a no-op when already warm).
        if LLBEngine.shared.residentDir != EngineManager.modelDir(.chat).path {
            let done = DispatchSemaphore(value: 0)
            nonisolated(unsafe) var failure: Error?
            Task { @MainActor in
                do { try await EngineManager.shared.ensure(.chat, ahead: ahead) } catch { failure = error }
                done.signal()
            }
            done.wait()
            if let failure { throw failure }
        }
        try LLBEngine.shared.run("camera") {
            try lock.withLock {
                if tokenizer == nil {
                    let t = try LLBTokenizer(dir: EngineManager.modelDir(.chat))
                    let ids = t.encode(CameraConfig.prefixText)
                    guard ids == CameraConfig.vlive1Prefix else {
                        throw Failure("the prompt prefix tokenizes to \(ids.count) ids, not vlive1's 33")
                    }
                    tokenizer = t
                    prefix = ids
                }
                if towerOpen && !forLook { openedByLook = false }  // the surface holds it now
                if tower, !towerOpen {
                    let t0 = Date()
                    if llb_vision_open(Self.mmprojURL.path, CameraConfig.backboneWidth, CameraConfig.imageArea) != 0 {
                        throw Failure("vision tower: \(String(cString: llb_error()))")
                    }
                    towerOpen = true
                    openedByLook = forLook
                    EngineManager.log(String(format: "camera: tower resident (mmproj %@) open_ms=%.0f footprint_mb=%d",
                                             CameraConfig.mmprojFile, Date().timeIntervalSince(t0) * 1000, EngineManager.footprint()))
                }
            }
        }
    }

    /// BENDSDK: frees the tower if only looks were using it (EngineManager closes it 30 s after the last look).
    func closeIfOpenedByLook() {
        LLBEngine.shared.run("camera") {
            lock.withLock {
                if towerOpen && openedByLook {
                    llb_vision_close(); towerOpen = false; openedByLook = false
                    EngineManager.log("camera: tower closed (no look for 30 s)")
                }
            }
        }
    }

    /// Frees the tower (the surface closed). The chat session is the lifecycle's, untouched.
    func close() {
        LLBEngine.shared.run("camera") {
            lock.withLock {
                if towerOpen { llb_vision_close(); towerOpen = false }
            }
        }
    }

    func promptIDs(question: String) throws -> (prefix: [UInt32], suffix: [UInt32]) {
        try lock.withLock {
            guard let tokenizer else { throw Failure("camera engine not prepared") }
            if let s = suffixes[question] { return (prefix, s) }
            let s = tokenizer.encode(CameraConfig.suffixText(question))
            suffixes[question] = s
            return (prefix, s)
        }
    }

    func classify(_ canvas: Canvas, question: String) throws -> CameraVerdict {
        let (pre, post) = try promptIDs(question: question)
        return try LLBEngine.shared.run("camera") {
            var st = llb_vstats()
            let side = Int32(canvas.side)
            let rc = canvas.rgb.withUnsafeBufferPointer { rp in
                pre.withUnsafeBufferPointer { pp in
                    post.withUnsafeBufferPointer { sp in
                        llb_classify_rgb(rp.baseAddress, side, side, pp.baseAddress, Int32(pp.count), sp.baseAddress,
                                         Int32(sp.count), CameraConfig.yesID, CameraConfig.noID, &st)
                    }
                }
            }
            if rc != 0 { throw Failure(String(cString: llb_error())) }
            var v = CameraVerdict(score: st.logit_a - st.logit_b, yesBits: st.bits_a, noBits: st.bits_b,
                                  firstID: st.first_id, question: question)
            v.towerMs = st.tower_ms; v.llmMs = st.llm_ms; v.engineMs = st.total_ms
            return v
        }
    }
}
