import Foundation

/// Our engine, in process: LLBEngine.xcframework (engine-earlystop + KEVPHONE's resident session, the Bend-emitted fused
/// Qwen3.5 program, emitted C bccd7cea…; its GPU program is c13b9e28's byte for byte) through its C API
/// (include/llb_engine.h).
///
/// **Resident (KEVPHONE, the default):** a warm engine's weights stay loaded. `open` is llb_open: one load into the GPU
/// span, kept by an engine thread of the library's own; every llb_generate / llb_decide on that model is then a job on
/// it (prefill + decode, no load, no release), until `release` (llb_free) gives it all back. One model is resident at a
/// time (one engine per process): opening the other model closes this one first. `-residentOff` (or BEND_RESIDENT=0):
/// the one-shot path below, as before (the before receipt).
///
/// **One-shot:** every request is llb_load(model dir, cache dir), then one llb_generate (chat) or llb_decide
/// (decisions); the run releases what it set up when it returns.
///
/// The engine's limits shape everything above it (engine-repeat and engine-earlystop REPORT.md):
///   - **Any number of runs per process, one at a time.** A one-shot run maps the GPU span, reads its model's weights,
///     runs, then unmaps the span and joins its threads (~20–220 MB stays between runs instead of ~4.2 GB), so it pays
///     its load (~0.35 s Mac, ~1 s phone). A resident session pays that once. Runs are serialised here (`runLock`):
///     one engine per process, not thread-safe.
///   - Greedy decoding of up to `n_new` ids. There is no sampler (temperature is ignored). The run ends at the first
///     stop id (`<|im_end|>`, `<|endoftext|>`, read by the wrapper from tokenizer.bin), which is the last id returned;
///     the reply is cut at that same id here.
///   - Ids in, ids out. Chat text goes through the engine's own tokenizer (the sidecar's tokenizer.bin, linked in
///     the same library) and Qwen3.5's chat template, rendered below.
///   - A fatal engine error is the engine's _exit: it ends the app. A run whose release failed leaves the library
///     refusing every later run ("R1: … could not be released"); the lifecycle shows that as a failed warm + Retry.
///
/// Streaming: llb_generate only returns when the run is over, but the engine writes one `logits <pos> <argmax> …`
/// line per position to its stderr (unbuffered, the wrapper's <cache>/run/err.txt: fd 2 redirected for a one-shot run;
/// the session's own FILE, emptied after every job, while resident). Greedy picks the
/// argmax, so the lines at pos ≥ n_ids-1 are the new ids as they are produced; the poller below turns them into
/// tokens. The final ids from llb_generate are checked against what was streamed.
final class LLBEngine: @unchecked Sendable {
    static let shared = LLBEngine()

    /// The first 8 hex of the emitted C the library was built from (engine-earlystop build-ios.sh SRC_SHA). Names the
    /// Metal program cache dir: the engine reuses <cache>/sys1mac.gpu whenever it exists, so a stale archive from
    /// another program must never sit there.
    static let programTag = "b0ffd9af"

    struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    struct GenerateResult {
        var ids: [UInt32]
        var streamed: Int
        var stats: llb_stats
        var mismatch: String?
        /// The run found its model resident (no load), or nil on the one-shot path.
        var resident: Residency?
        /// G18 (hot state): the kept prefix's ids restored instead of prefilled (0: kept now, first time), or nil when
        /// the run was a plain llb_generate.
        var prefixIds: Int?
    }

    /// How a request met the resident session.
    enum Residency: String, Sendable {
        /// Its model's weights were resident: nothing loaded.
        case warm
        /// They were not (the session was closed under it, or another model held it): this request opened it (one load).
        case reloaded
    }

    /// KEVPHONE: keep a warm engine's weights loaded (llb_open). Off: `-residentOff` or BEND_RESIDENT=0.
    static let residentOn: Bool = !ProcessInfo.processInfo.arguments.contains("-residentOff")
        && ProcessInfo.processInfo.environment["BEND_RESIDENT"] != "0"

    /// The session's KV capacity in ids (prompt + new): 1024 new on a 3k-id thread fits. A longer request runs one-shot
    /// (the wrapper closes the session for it; the next request opens it again). G18: `-llbKvCap` (MemLevers), else 4096.
    static let residentContext: Int32 = MemLevers.kvCap

    /// The model dir whose weights are resident now, or nil.
    var residentDir: String? { llb_resident().map { String(cString: $0) } }

    /// One line: resident=yes|no model= cap= loads= jobs= load_ms= open_ms= compile_ms= warmup_ms= span_mb= footprint_mb=
    static var residentReport: String { String(cString: llb_resident_report()) }

    struct OpenResult: Sendable {
        var loaded: Bool
        var loadMs: Double
        var compileMs: Double
        var totalMs: Double
    }

    private let lock = NSLock()
    private var running: String?
    /// Held for a whole run (and by `release`): the library takes one call at a time.
    private let runLock = NSLock()

    /// The run in flight ("chat", "decision"), or nil.
    var runningBy: String? { lock.withLock { running } }

    /// Runs `body` as `what`'s engine run, after any run in flight (runs queue; none is refused).
    func run<T>(_ what: String, _ body: () throws -> T) rethrows -> T {
        runLock.lock()
        lock.withLock { running = what }
        defer {
            lock.withLock { running = nil }
            runLock.unlock()
        }
        return try body()
    }

    /// Has the library take `dir`'s model (llb_load: the sidecar and the GGUF checked), between runs.
    func arm(_ dir: URL) throws {
        try runLock.withLock { try Self.load(dir) }
    }

    /// Gives back what the library holds between runs (llb_free: the resident session too), once no run is in flight.
    func release() {
        runLock.withLock {
            if MemLevers.hotState, let d = residentDir { Self.saveHot(URL(fileURLWithPath: d)) }
            llb_free()
        }
    }

    /// G18: drops the session's kept prefix snapshots (memory pressure), once no run is in flight.
    func forgetPrefixes() {
        runLock.withLock { llb_prefix_forget() }
    }

    /// G18 (hot state): the session's text snapshots to `dir`'s hot file, then out of RAM (the snapshots outlive
    /// llb_close: ~60 MB each, kept while cold otherwise; loadHot brings them back after the next open). Under the run
    /// lock, session idle.
    private static func saveHot(_ dir: URL) {
        let t0 = Date()
        let n = llb_state_save(MemLevers.hotFile(dir).path)
        if n >= 0 { llb_prefix_forget() }
        EngineManager.log(n < 0 ? "hot state save \(dir.lastPathComponent) FAILED: \(String(cString: llb_error()))"
            : String(format: "hot state saved %@ slots=%d ms=%.1f", dir.lastPathComponent, n, Date().timeIntervalSince(t0) * 1000))
    }

    /// G18 (hot state): `dir`'s hot file into the session just opened, when there is one. A file from another engine
    /// build, model or cap is refused by the engine (nothing loaded): the first turn prefills as before.
    private static func loadHot(_ dir: URL) {
        let file = MemLevers.hotFile(dir)
        guard FileManager.default.fileExists(atPath: file.path) else { return EngineManager.log("hot state: no file for \(dir.lastPathComponent)") }
        let t0 = Date()
        let n = llb_state_load(file.path)
        EngineManager.log(n < 0 ? "hot state load \(dir.lastPathComponent) refused: \(String(cString: llb_error()))"
            : String(format: "hot state loaded %@ slots=%d ms=%.1f footprint_mb=%d", dir.lastPathComponent, n,
                     Date().timeIntervalSince(t0) * 1000, EngineManager.footprint()))
    }

    /// Loads `dir`'s weights and keeps them (llb_open). Returns once they are resident; at once if they already are.
    /// `warmUp`: the library then runs one throw-away 16-id chat, so the first request runs at the session's speed —
    /// for a warm nobody is waiting on (eager, watchdog). A request waiting on the warm skips it: its own first job
    /// pays that one-time set-up, which costs less than the warm-up.
    func open(_ dir: URL, warmUp: Bool = true) throws -> OpenResult {
        try run("open") { try Self.openLocked(dir, warmUp: warmUp) }
    }

    private static func openLocked(_ dir: URL, warmUp: Bool = false) throws -> OpenResult {
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let was = llb_resident().map { String(cString: $0) }
        // G18: llb_open on another model closes `was` first: its snapshots are written before that.
        if MemLevers.hotState, let was, was != dir.path { saveHot(URL(fileURLWithPath: was)) }
        if let note = MemLevers.beforeOpen(dir) { EngineManager.log(note) }
        var st = llb_stats()
        if llb_open(dir.path, cacheDir.path, residentContext, warmUp ? 0 : LLB_OPEN_NOWARMUP, &st) != 0 {
            throw Failure("open: \(String(cString: llb_error()))")
        }
        if MemLevers.packMap, was != dir.path { EngineManager.log("pack map \(dir.lastPathComponent) \(MemLevers.packReport)") }
        if MemLevers.hotState, was != dir.path { loadHot(dir) }
        return OpenResult(loaded: was != dir.path, loadMs: st.load_ms, compileMs: st.compile_ms, totalMs: st.total_ms)
    }

    /// Resident mode: the run's model is resident, opening it if it is not (a reload: logged by the caller).
    private static func residency(_ dir: URL) throws -> Residency? {
        guard residentOn else { return nil }
        if llb_resident().map({ String(cString: $0) }) == dir.path { return .warm }
        _ = try openLocked(dir)
        return .reloaded
    }

    /// The last run's release, one line (released=yes|NO release_ms= footprint_mb_before= footprint_mb_after=).
    static var releaseReport: String { String(cString: llb_release_report()) }

    static var cacheDir: URL {
        let base = ProcessInfo.processInfo.environment["BEND_CACHE_DIR"].map(URL.init(fileURLWithPath:))
            ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("llb-" + programTag)
    }

    /// The compiled Metal program the engine keeps in the cache (llb_gpu_cache: compiled and written by the first run
    /// into a cache dir without it, ~0.8–1 s; every later run, in any process, finds it and pays 0).
    static var programArchive: URL { cacheDir.appendingPathComponent("sys1mac.gpu") }

    static var programCached: Bool { FileManager.default.fileExists(atPath: programArchive.path) }

    static func load(_ dir: URL) throws {
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        if llb_load(dir.path, cacheDir.path) != 0 { throw Failure("load: \(String(cString: llb_error()))") }
    }

    // MARK: chat

    /// One greedy run. `onId(k, id)` gets new id k as the engine produces it (on a poller thread); return false to
    /// stop listening. The engine's run ends at its first stop id (or `nNew`). Blocks until that run is over.
    /// G18: `nPre` > 0 (MemLevers.prefix, hot state on) keeps / restores the first nPre ids (llb_generate_px) on the
    /// resident session; same ids out.
    func generate(dir: URL, ids: [UInt32], nPre: Int = 0, nNew: Int, onId: @escaping (Int, UInt32) -> Bool) throws -> GenerateResult {
        try run("chat") { try generateRun(dir: dir, ids: ids, nPre: nPre, nNew: nNew, onId: onId) }
    }

    private func generateRun(dir: URL, ids: [UInt32], nPre: Int, nNew: Int, onId: @escaping (Int, UInt32) -> Bool) throws -> GenerateResult {
        try Self.load(dir)
        let resident = try Self.residency(dir)
        let errPath = Self.cacheDir.appendingPathComponent("run/err.txt").path
        // One-shot: the wrapper recreates err.txt (O_TRUNC) for this run; a previous launch's file must not be read as
        // this one. Resident: the session holds err.txt open and empties it after every job; it stays.
        if resident == nil { unlink(errPath) }

        let poller = LogitsPoller(path: errPath, firstPos: ids.count - 1, onId: onId)
        poller.start()
        var out = [UInt32](repeating: 0, count: nNew)
        var st = llb_stats()
        let px = nPre > 0 && resident != nil
        var prefixIds: Int32 = -1
        let got = ids.withUnsafeBufferPointer { ip in
            out.withUnsafeMutableBufferPointer { op in
                px ? llb_generate_px(ip.baseAddress, Int32(ids.count), Int32(nPre), Int32(nNew), op.baseAddress, &st, &prefixIds)
                   : llb_generate(ip.baseAddress, Int32(ids.count), Int32(nNew), op.baseAddress, &st)
            }
        }
        let streamed = poller.finish()
        if got < 0 {
            let log = String(cString: llb_log()).split(separator: "\n").filter { !$0.hasPrefix("logits") }.suffix(6)
            throw Failure("generate: \(String(cString: llb_error())) exit=\(st.exit_code) \(log.joined(separator: " | "))")
        }
        let final = Array(out.prefix(Int(got)))
        var mismatch: String?
        if let k = zip(streamed, final).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset {
            mismatch = "streamed id \(k) = \(streamed[k]), engine's final id = \(final[k])"
        }
        // Ids the poller had not seen yet (it stopped early, or the last line landed after its last read).
        if streamed.count < final.count, mismatch == nil {
            for k in streamed.count..<final.count where !poller.stopped { if !onId(k, final[k]) { break } }
        }
        return GenerateResult(ids: final, streamed: streamed.count, stats: st, mismatch: mismatch, resident: resident,
                              prefixIds: px ? max(0, Int(prefixIds)) : nil)
    }

    // MARK: image (IMGPERF)

    struct ImageResult {
        var ids: [UInt32]
        var streamed: Int
        var stats: llb_vstats
        var mismatch: String?
    }

    /// One look (llb_generate_rgb): the canvas through the tower once (or not at all: the engine keeps the last 4
    /// photos' rows, by their exact bytes), then ONE resident job that prefills `pre` + the image rows + `post` once
    /// and decodes up to `nNew` ids, streamed through `onId` like chat's (the session's `logits` lines). The caller
    /// holds the run (LLBEngine.run) and has the chat session resident. `imageTokens` is the canvas's row count
    /// ((side / 32)², checked against the engine's after the run): the poller's first position depends on it.
    func generateImageLocked(rgb: [UInt8], side: Int, height: Int? = nil, pre: [UInt32], post: [UInt32], imageTokens: Int,
                             nNew: Int, onId: @escaping (Int, UInt32) -> Bool) throws -> ImageResult {
        let errPath = Self.cacheDir.appendingPathComponent("run/err.txt").path
        let poller = LogitsPoller(path: errPath, firstPos: pre.count + imageTokens + post.count - 1, onId: onId)
        poller.start()
        var out = [UInt32](repeating: 0, count: nNew)
        var st = llb_vstats()
        let got = rgb.withUnsafeBufferPointer { rp in
            pre.withUnsafeBufferPointer { pp in
                post.withUnsafeBufferPointer { sp in
                    out.withUnsafeMutableBufferPointer { op in
                        llb_generate_rgb(rp.baseAddress, Int32(side), Int32(height ?? side), pp.baseAddress, Int32(pp.count), sp.baseAddress,
                                         Int32(sp.count), Int32(nNew), op.baseAddress, &st)
                    }
                }
            }
        }
        let streamed = poller.finish()
        if got < 0 {
            let log = String(cString: llb_log()).split(separator: "\n").filter { !$0.hasPrefix("logits") }.suffix(6)
            throw Failure("look: \(String(cString: llb_error())) \(log.joined(separator: " | "))")
        }
        let final = Array(out.prefix(Int(got)))
        var mismatch: String?
        if Int(st.n_img) != imageTokens {
            mismatch = "image tokens \(st.n_img), expected \(imageTokens): nothing streamed was trusted"
        } else if let k = zip(streamed, final).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset {
            mismatch = "streamed id \(k) = \(streamed[k]), engine's final id = \(final[k])"
        }
        if streamed.count < final.count, mismatch == nil {
            for k in streamed.count..<final.count where !poller.stopped { if !onId(k, final[k]) { break } }
        }
        return ImageResult(ids: final, streamed: streamed.count, stats: st, mismatch: mismatch)
    }

    /// True when the engine keeps this canvas's tower rows (a look at it needs no tower). Under the run lock.
    func hasImageLocked(rgb: [UInt8], side: Int) -> Bool {
        rgb.withUnsafeBufferPointer { llb_vision_has($0.baseAddress, Int32(side), Int32(side)) != 0 }
    }

    // MARK: prime

    /// The smallest real path through the engine (R17): `ids` prefilled, one new id, the run released. Its point is the
    /// one-time Metal compile (the archive written into the cache), off any request's critical path. Queues like any
    /// run; reports as "prime" (no engine's run), and leaves nothing behind but the archive.
    /// PHONEUX: nil when, once its turn came, the archive was already there (the launch warm's llb_open wrote it), unless
    /// `always`: nothing then runs.
    func prime(dir: URL, ids: [UInt32], always: Bool = false) throws -> llb_stats? {
        try run("prime") {
            if Self.programCached && !always { return nil }
            try Self.load(dir)
            var out = [UInt32](repeating: 0, count: 1)
            var st = llb_stats()
            let got = ids.withUnsafeBufferPointer { ip in
                out.withUnsafeMutableBufferPointer { op in llb_generate(ip.baseAddress, Int32(ids.count), 1, op.baseAddress, &st) }
            }
            if got < 0 { throw Failure("prime: \(String(cString: llb_error())) exit=\(st.exit_code)") }
            return st
        }
    }

    // MARK: decide

    /// /v1/systemone bodies → reply lines (llama-server's layout), one engine run (a job on the session, resident).
    func decide(dir: URL, bodies: [String]) throws -> (replies: [String], stats: llb_dstats, resident: Residency?) {
        try run("decision") { try decideRun(dir: dir, bodies: bodies) }
    }

    private func decideRun(dir: URL, bodies: [String]) throws -> (replies: [String], stats: llb_dstats, resident: Residency?) {
        try Self.load(dir)
        let resident = try Self.residency(dir)
        var cBodies = bodies.map { strdup($0) }
        defer { cBodies.forEach { free($0) } }
        var replies = [UnsafeMutablePointer<CChar>?](repeating: nil, count: bodies.count)
        var st = llb_dstats()
        let got = cBodies.withUnsafeMutableBufferPointer { bp in
            bp.baseAddress!.withMemoryRebound(to: UnsafePointer<CChar>?.self, capacity: bodies.count) { b in
                replies.withUnsafeMutableBufferPointer { rp in llb_decide(b, Int32(bodies.count), rp.baseAddress, &st) }
            }
        }
        defer { replies.forEach { free($0) } }
        if got < 0 { throw Failure("decide: \(String(cString: llb_error())) exit=\(st.exit_code)") }
        return (replies.map { $0.map { String(cString: $0) } ?? #"{"error":"no reply"}"# }, st, resident)
    }
}

/// Tails the engine's stderr for `logits <pos> <argmax> …` lines while llb_generate runs.
private final class LogitsPoller: @unchecked Sendable {
    private let path: String
    private let firstPos: Int
    private let onId: (Int, UInt32) -> Bool
    private let done = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var finishing = false
    private var ids: [UInt32] = []
    private(set) var stopped = false
    private var offset: UInt64 = 0
    private var partial = Data()

    init(path: String, firstPos: Int, onId: @escaping (Int, UInt32) -> Bool) {
        self.path = path
        self.firstPos = firstPos
        self.onId = onId
    }

    func start() {
        let thread = Thread { [self] in
            while !lock.withLock({ finishing }) {
                read()
                usleep(15_000)
            }
            read()
            done.signal()
        }
        thread.name = "bend.engine.logits"
        thread.start()
    }

    /// Stops the poller after one last read; returns every new id it saw, in order.
    func finish() -> [UInt32] {
        lock.withLock { finishing = true }
        done.wait()
        return ids
    }

    private func read() {
        guard let handle = FileHandle(forReadingAtPath: path) else { return }
        defer { try? handle.close() }
        try? handle.seek(toOffset: offset)
        guard let data = try? handle.readToEnd(), !data.isEmpty else { return }
        offset += UInt64(data.count)
        partial.append(data)
        while let nl = partial.firstIndex(of: 0x0A) {
            let line = String(decoding: partial[partial.startIndex..<nl], as: UTF8.self)
            partial.removeSubrange(partial.startIndex...nl)
            let f = line.split(separator: " ", maxSplits: 3)
            guard f.count >= 3, f[0] == "logits", let pos = Int(f[1]), let id = UInt32(f[2]), pos >= firstPos else { continue }
            let k = pos - firstPos
            guard k == ids.count else { continue } // one line per position, in order
            ids.append(id)
            if !stopped, !onId(k, id) { stopped = true }
        }
    }
}

/// The Qwen3.5 tokenizer linked in LLBEngine (the sidecar's tokenizer.bin), and the GGUF's chat template rendered
/// for Bend's messages (text content, no tools).
final class LLBTokenizer: @unchecked Sendable {
    private var tok = Q35Tokenizer()
    let imEnd: UInt32
    let endOfText: UInt32

    init(dir: URL) throws {
        if q35_tok_load(&tok, dir.path) != 0 { throw LLBEngine.Failure("tokenizer: cannot load \(dir.path)/tokenizer.bin") }
        // By text, not the header's eos: Kev-4B's tokenizer.bin names <|endoftext|> as its eos (Qwen3.5's: <|im_end|>).
        var ime: Int32 = 0
        _ = q35_tok_encode(&tok, "<|im_end|>", 1, &ime, 1)
        imEnd = UInt32(ime)
        var eot: Int32 = 0
        _ = q35_tok_encode(&tok, "<|endoftext|>", 1, &eot, 1)
        endOfText = UInt32(eot)
    }

    deinit { q35_tok_free(&tok) }

    func encode(_ text: String) -> [UInt32] {
        var cap = max(64, text.utf8.count + 16)
        while true {
            var out = [Int32](repeating: 0, count: cap)
            let n = Int(q35_tok_encode(&tok, text, 1, &out, Int32(cap)))
            if n >= 0 { return out.prefix(n).map { UInt32($0) } }
            cap = -n
        }
    }

    func piece(_ id: UInt32) -> Data {
        var buf = [CChar](repeating: 0, count: 256)
        let n = Int(q35_tok_piece(&tok, Int32(id), &buf, 256))
        return Data(buf.prefix(max(0, min(n, 256))).map { UInt8(bitPattern: $0) })
    }

    struct Message {
        let role: String
        let content: String
    }

    private static func systemText(_ content: String) -> String { "<|im_start|>system\n\(content)<|im_end|>\n" }

    /// G18: the rendered leading system turn (render's first turn, the same text), or nil: the hot state's kept prefix.
    static func systemTurn(_ messages: [Message]) -> String? {
        guard let first = messages.first, first.role == "system" else { return nil }
        return systemText(first.content.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// tokenizer.chat_template of Qwen3.5-4B (the GGUF's own), for system/user/assistant text turns:
    /// a leading system turn, user turns as given, earlier assistant turns without their reasoning, then the
    /// generation prompt. enable_thinking false (what Bend sends, and the template's default) adds the empty think block.
    static func render(_ messages: [Message], enableThinking: Bool) throws -> String {
        guard let last = messages.lastIndex(where: { $0.role == "user" }) else {
            throw LLBEngine.Failure("No user query found in messages.")
        }
        let trim = { (s: String) in s.trimmingCharacters(in: .whitespacesAndNewlines) }
        var out = ""
        for (i, m) in messages.enumerated() {
            let content = trim(m.content)
            switch m.role {
            case "system":
                guard i == 0 else { throw LLBEngine.Failure("System message must be at the beginning.") }
                out += systemText(content)
            case "user":
                out += "<|im_start|>user\n\(content)<|im_end|>\n"
            case "assistant":
                var text = content
                var reasoning = ""
                if let r = text.range(of: "</think>") {
                    reasoning = String(text[..<r.lowerBound]).components(separatedBy: "<think>").last ?? ""
                    text = String(text[r.upperBound...]).replacingOccurrences(of: #"^\n+"#, with: "", options: .regularExpression)
                }
                out += i > last
                    ? "<|im_start|>assistant\n<think>\n\(trim(reasoning))\n</think>\n\n\(text)<|im_end|>\n"
                    : "<|im_start|>assistant\n\(text)<|im_end|>\n"
            default:
                throw LLBEngine.Failure("Unexpected message role: \(m.role)")
            }
        }
        return out + (enableThinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }
}

/// UTF-8 across token boundaries: a piece can end inside a multi-byte character; hold those bytes for the next one.
struct UTF8Joiner {
    private var pending = Data()

    mutating func push(_ bytes: Data) -> String {
        pending.append(bytes)
        var cut = pending.count
        // Back off over an incomplete trailing sequence (at most 3 bytes).
        for back in 1...min(3, pending.count) {
            let b = pending[pending.startIndex + pending.count - back]
            if b & 0xC0 == 0x80 { continue }            // continuation byte: keep looking for the lead
            let need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1
            if need > back { cut = pending.count - back }
            break
        }
        let text = String(decoding: pending.prefix(cut), as: UTF8.self)
        pending = Data(pending.suffix(from: pending.startIndex + cut))
        return text
    }

    mutating func flush() -> String {
        defer { pending = Data() }
        return String(decoding: pending, as: UTF8.self)
    }
}
