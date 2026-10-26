import CoreGraphics
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// IMAGETRUE: one earlier turn of a conversation about a photo: what was asked, what the look answered. The shell's
/// look route pairs them from the request (EngineManager.exchanges); VisionLook renders them after the image.
struct VisionLookExchange: Sendable, Equatable {
    var asked: String
    var answered: String
}

/// Which engine a request is for. The web app names them the same way.
enum EngineID: String, CaseIterable, Sendable {
    /// Kev-4B by default; Qwen3.5-4B in the optional two-model mode (llb_generate).
    case chat
    /// Kev-4B: /v1/systemone, the decision card and the Millionaire Jev (llb_decide, the Jev head).
    case decision

    /// PHONESINGLE: one resident Kev serves both routes by default. Settings persists the optional Qwen chat pairing
    /// for the next launch. Harness overrides do not change the saved choice.
    static let modeKey = "bend.engine.oneModel"
    static let kevChat: Bool = {
        let args = ProcessInfo.processInfo.arguments
        if args.contains("-kevChat") { return true }
        if args.contains("-twoModels") { return false }
        if let model = ProcessInfo.processInfo.environment["BEND_CHAT_MODEL"] {
            if model == "kev" { return true }
            if model == "qwen" { return false }
        }
        return (UserDefaults.standard.object(forKey: modeKey) as? Bool) ?? true
    }()

    /// The GGUF inside the model dir (status shows it; the web app shortens it to the model name).
    var model: String {
        switch self {
        case .chat: Self.kevChat ? "Kev-4B-Q4_K_M.gguf" : "Qwen3.5-4B-Q4_K_M.gguf"
        case .decision: "Kev-4B-Q4_K_M.gguf"
        }
    }

    /// Documents/models/<dir>/: the GGUF + its llb sidecar (manifest.json, table.bin, tokenizer.bin, chat.flags),
    /// the layout engine-phone2's device-run.sh copies.
    var dir: String {
        switch self {
        case .chat: Self.kevChat ? "kev-4b-q4km" : "qwen35-4b-q4km"
        case .decision: "kev-4b-q4km"
        }
    }
}

/// Owns Bend's engine: OUR fused engine (LLBEngine, see LLBEngine.swift) behind the same two loopback listeners the
/// llama.cpp shell had (docs/R7-NOTES.md, docs/ENGINE-NATIVE.md), so the web app runs unchanged:
///   127.0.0.1:8930  ControlServer: GET /engines/status, POST /engines/ensure
///   127.0.0.1:8931  EngineServer:  /v1/chat/completions, /v1/systemone, /health
///
/// Lifecycle (docs/ENGINES.md, EngineLifecycle.swift): each engine is cold → warming → warm → releasing → cold.
/// Every request ensures its engine is warm first; warm means the loopback surface answers, the model dir is verified,
/// chat's tokenizer is resident, the library has taken the model (llb_load) and — KEVPHONE, resident mode, the default —
/// its weights are loaded and kept (llb_open: one load, then every request is prefill + decode on them, nothing loaded
/// or released per request). Only one model fits resident at a time. The lifecycle is exclusive in two-model mode;
/// the default shares one Kev session across both routes, so no swap is needed. Release
/// (away from the screen, after the grace) is llb_free (the session closed, the GBs back) + the tokenizer dropped.
/// `-residentOff`: R17's path, each llb_generate / llb_decide loads and releases its weights (engine-repeat). Any
/// number of requests per launch; they run one at a time.
@MainActor
final class EngineManager: EngineRoutes, EngineBackend {
    static let shared = EngineManager()

    /// PHONEONE: BEND_PORT_BASE (the Mac harnesses only; unset in the app) moves the three loopback ports together
    /// (control, engine, voice = base, +1, +2), so a harness can measure while a simulator app holds 8930–8932.
    nonisolated static let portBase: UInt16 = ProcessInfo.processInfo.environment["BEND_PORT_BASE"].flatMap { UInt16($0) } ?? 8930
    nonisolated static let controlPort: UInt16 = portBase
    nonisolated static let enginePort = Int(portBase) + 1
    /// Kept in the status for the web types; release is the lifecycle's grace now (both engines).
    nonisolated static let decisionIdle: TimeInterval = 0
    /// Chat's new-id budget when the request names none, and the cap (the engine decodes all of them, greedy).
    nonisolated static let defaultMaxTokens = 512
    nonisolated static let maxNew = 1024

    enum State: String { case noModels = "no-models", idle, loading, live, error }

    struct EngineError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    /// Per-launch key for both listeners: other apps on the phone can reach loopback ports too.
    let apiKey: String = {
        #if os(macOS)
        // The Mac harness only (scripts/engine-mac.sh): a fixed key so a test client can call it.
        if let key = ProcessInfo.processInfo.environment["BEND_API_KEY"] { return key }
        #endif
        // BENDSDK: the simulator under -hostEngines hands the web the Mac harness's key (SIMCTL_CHILD_BEND_API_KEY).
        if EngineManager.hostEngines, let key = ProcessInfo.processInfo.environment["BEND_API_KEY"] { return key }
        #if targetEnvironment(simulator) && DEBUG
        // IOS-FOLD: a plain simulator run (no -hostEngines) honors BEND_API_KEY too, so a smoke can key its curls.
        // VoiceServer already does this on every platform; the engine port was a random key with no way to read it.
        if let key = ProcessInfo.processInfo.environment["BEND_API_KEY"] { return key }
        #endif
        return (0..<4).map { _ in String(format: "%08x", UInt32.random(in: .min ... .max)) }.joined()
    }()

    /// BENDSDK, simulator only: `-hostEngines` (with SIMCTL_CHILD_BEND_PORT_BASE and SIMCTL_CHILD_BEND_API_KEY) points the
    /// web layer at the Mac harness's listeners (scripts/engine-mac.sh run): the simulator shares the Mac's loopback, and
    /// it can't run the engine itself (below). Nothing starts in the simulator; the page talks to OUR engine on the Mac
    /// through the same transport, ports and key check it uses on the phone (scripts/sdk-sim.sh).
    /// FOLDIN-R5: one switch with PHONEUX's (the instance `hostEngines` below reads this): simulator Debug builds only.
    nonisolated static let hostEngines: Bool = {
        #if targetEnvironment(simulator) && DEBUG
        return ProcessInfo.processInfo.arguments.contains("-hostEngines")
        #else
        return false
        #endif
    }()

    /// Device and Mac harness: always. Simulator: only with `-simEngines` (and the engine will refuse there: the
    /// simulator's Metal caps one buffer at 256 MB and its threadgroups below 1024; engine-phone2 §3).
    let enabled: Bool = {
        #if targetEnvironment(simulator)
        return ProcessInfo.processInfo.arguments.contains("-simEngines")
        #else
        return true
        #endif
    }()

    /// The engine the last ensure asked for.
    private(set) var active: EngineID?
    /// The run in flight (LLBEngine keeps it, under its lock).
    private var running: EngineID? { LLBEngine.shared.runningBy.flatMap(EngineID.init(rawValue:)) }
    private var footprintMB: [EngineID: Int] = [:]
    /// Requests waiting on their engine's warm (admit → ensure): a warm with someone waiting skips the warm-ups.
    private var waiting: [EngineID: Int] = [:]
    private var loadSeconds: [EngineID: Double] = [:]
    /// PHONEUX: what each warm is doing now and what each of its stages took (status `stage` / `phases`).
    private var trace: [EngineID: WarmTrace] = [:]
    private var lastPhases: [EngineID: [WarmTrace.Phase]] = [:]
    private var control: ControlServer?
    private var engine: EngineServer?

    /// The lifecycle's policy: the grace is 90 s, or BEND_GRACE_SECONDS (the Mac lifecycle check uses a short one).
    nonisolated static var policy: EngineLifecycle.Policy {
        var policy = EngineLifecycle.Policy()
        if let s = ProcessInfo.processInfo.environment["BEND_GRACE_SECONDS"].flatMap(Double.init) { policy.grace = s }
        // G18: the same as a launch flag (a device launch passes args, not env): `-graceSeconds N`.
        if let i = Self.args.firstIndex(of: "-graceSeconds"), i + 1 < Self.args.count, let s = Double(Self.args[i + 1]) { policy.grace = s }
        // KEVPHONE: one model's weights resident at a time (one engine per process, ~4 GB each): warm one, release the
        // other. In one-model mode both routes use the same weights, so there is nothing to swap.
        policy.exclusive = LLBEngine.residentOn && EngineID.chat.dir != EngineID.decision.dir
        // PHONEUX: the first screen warms chat (after the page has painted, `pageReady`), so the person sees
        // cold → warming → warm with its stages instead of a first send that sits on a silent load. `-coldLaunch`:
        // R14's policy (a cold launch loads nothing unless chat was warm at the last exit).
        policy.eagerLaunch = !args.contains("-coldLaunch") && hasModel(.chat)
        return policy
    }

    private(set) lazy var lifecycle = EngineLifecycle(backend: self, clock: SystemClock(), store: DefaultsHotStore(),
                                                      policy: Self.policy, log: Self.log)

    /// What a warm engine keeps between runs: chat's tokenizer (the engine thread reads it, so it sits behind a lock).
    private final class Resident: @unchecked Sendable {
        private let lock = NSLock()
        private var tokenizers: [EngineID: LLBTokenizer] = [:]
        subscript(id: EngineID) -> LLBTokenizer? {
            get { lock.withLock { tokenizers[id] } }
            set { lock.withLock { tokenizers[id] = newValue } }
        }
    }
    nonisolated private let resident = Resident()

    #if canImport(UIKit)
    /// Background time asked for while away, so the grace can run; if iOS ends it first, release then.
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    #endif

    nonisolated static var modelsDir: URL {
        if let dir = ProcessInfo.processInfo.environment["BEND_MODELS_DIR"] { return URL(fileURLWithPath: dir) }
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("models")
    }

    nonisolated static func modelDir(_ id: EngineID) -> URL { modelsDir.appendingPathComponent(id.dir) }

    nonisolated static func log(_ line: String) {
        // One greppable line per event, in the Xcode console and `devicectl … --console`.
        console("BEND engine \(line)")
    }

    /// The console as it was at launch. During a run the engine wrapper points the process's fd 1 and 2 at its own
    /// files (<cache>/run/out.txt, err.txt) and parses out.txt as the new ids, so nothing else may print to stdout
    /// then: a line from another thread would land in the engine's output and cut its ids short. Bend's lines go to
    /// this duplicate of the launch-time stdout instead.
    /// IMGPERF: BEND_CONSOLE_FILE (the simulator receipts: a host path the UI test names) gets the console lines instead
    /// of stdout, which XCUITest's launch doesn't keep. Unset: stdout, as before.
    nonisolated static let consoleFD: Int32 = {
        // BENDFIX-1: `documents:<name>` = that file in the app's Documents (a device UI test can't name a host path;
        // the receipts pull it with devicectl).
        if let path = consoleFilePath() {
            let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
            if fd >= 0 { return fd }
        }
        return dup(STDOUT_FILENO)
    }()

    nonisolated static func consoleFilePath() -> String? {
        guard let path = ProcessInfo.processInfo.environment["BEND_CONSOLE_FILE"], !path.isEmpty else { return nil }
        guard path.hasPrefix("documents:") else { return path }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent(String(path.dropFirst("documents:".count))).path
    }

    nonisolated static func console(_ line: String) {
        let bytes = Array((line + "\n").utf8)
        _ = bytes.withUnsafeBytes { write(consoleFD, $0.baseAddress, $0.count) }
    }

    nonisolated private static var backend: String {
        #if targetEnvironment(simulator)
        return "SIMULATOR/LLB"
        #elseif os(iOS)
        return "DEVICE/LLB-Metal"
        #else
        return "MAC/LLB-Metal"
        #endif
    }

    nonisolated static func hasModel(_ id: EngineID) -> Bool {
        let dir = modelDir(id)
        return ["manifest.json", "tokenizer.bin", id.model].allSatisfy {
            FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
    }

    func hasModel(_ id: EngineID) -> Bool { Self.hasModel(id) }

    /// The web's per-engine state (unchanged contract) from the lifecycle phase: cold → idle (error once the warms
    /// failed: the next request retries), warming → loading, warm → live, releasing → idle.
    func state(_ id: EngineID) -> State {
        if !hasModel(id) { return .noModels }
        let slot = lifecycle.slot(id)
        switch slot.phase {
        case .cold: return slot.retry ? .error : .idle
        case .warming: return .loading
        case .warm: return .live
        case .releasing: return .idle
        }
    }

    // MARK: lifecycle

    func start() {
        guard enabled else {
            Self.log(hostEngines ? "disabled here; -hostEngines: the page uses the Mac harness on the host's loopback"
                                 : "disabled (simulator; launch with -simEngines to try them, or -hostEngines with the Mac harness)")
            return
        }
        _ = Self.consoleFD // taken now, before any engine run can redirect fd 1
        #if BEND_CAMERA
        CameraEngine.reserveArea()  // PHONECAM: before any session opens (llb_vision_area)
        #endif
        try? FileManager.default.createDirectory(at: Self.modelsDir, withIntermediateDirectories: true)
        if Self.args.contains("-dropProgramCache") {
            let gone = (try? FileManager.default.removeItem(at: LLBEngine.programArchive)) != nil
            Self.log("program cache \(gone ? "dropped" : "was not there") (-dropProgramCache): the next run compiles")
        }
        openListeners()
        Self.log("engine LLBEngine program=\(LLBEngine.programTag) cache=\(LLBEngine.cacheDir.path) backend=\(Self.backend)")
        Self.log(LLBEngine.residentOn
            ? "resident mode on: a warm engine keeps its weights loaded (llb_open, kv \(LLBEngine.residentContext)); one model at a time; \(EngineID.kevChat ? "Kev-4B serves chat and decisions" : "Qwen chat + Kev decisions")"
            : "resident mode off (-residentOff): every request loads and releases its weights")
        Self.log("device \(String(cString: llb_device_info()))")
        Self.log(MemLevers.line)  // G18: every lever off unless its flag is given
        if let note = MemLevers.resetIfAsked() { Self.log(note) }  // G18: before any llb_open
        for id in EngineID.allCases {
            Self.log("model \(id.rawValue) \(id.dir)/\(id.model) \(hasModel(id) ? "present" : "MISSING (no-models)")")
        }
        #if canImport(UIKit)
        let center = NotificationCenter.default
        center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { EngineManager.shared.memoryWarning() }
        }
        // Away from the screen: backgrounded, which a screen lock also does. Back: about to be on screen again.
        center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { EngineManager.shared.away() }
        }
        center.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { EngineManager.shared.back() }
        }
        #endif
        // PHONEUX: boot order. The page paints first (WebAppView's first load → pageReady), then the launch warm
        // starts, so the engine's GB of reads never race the first frame. A page that never reports: 3 s, then anyway.
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            self.launchLifecycle(why: "no page after 3 s")
        }
        Self.log(String(format: "boot start done at %.0f ms since process start", Boot.sinceStartMs()))
        EngineCheck.startIfAsked(key: apiKey)
        #if os(macOS)
        // The Mac harness has no page: its "UI is up" is the listeners being up.
        pageReady()
        #endif
    }

    // MARK: models (IOSFETCH)

    /// The setup card verified and moved these into Documents/models (Models/ModelStore.swift). Status picks them up by
    /// itself (`hasModel` reads the disk, so the web's no-models → idle on its next poll); chat is warmed now, as an
    /// eager warm (nobody waiting: its warm-ups run), so the first message after the download is a warm one.
    func modelsArrived(_ items: Set<ModelItem.ID>) {
        for id in EngineID.allCases {
            Self.log("model \(id.rawValue) \(id.dir)/\(id.model) \(hasModel(id) ? "present" : "MISSING (no-models)") (after download)")
        }
        guard enabled else { return Self.log("models arrived: engines are off here (simulator), nothing to warm") }
        if items.contains(.chat), hasModel(.chat), lifecycle.phase(.chat) == .cold {
            Task { @MainActor in try? await self.lifecycle.ensure(.chat, why: "eager (models downloaded)") }
        }
    }

    // MARK: prime (R17)

    nonisolated static let args = ProcessInfo.processInfo.arguments
    private var primed = false

    /// The web UI is on screen (WebAppView's first page load; on the Mac harness, the listeners). After a beat of idle,
    /// if the engine's compiled Metal program is not in the cache (a fresh install, a new program tag, or iOS purged
    /// Caches), one tiny real run pays the ~1 s compile now, so no person's first request does. With the archive
    /// there, nothing runs: the release policy is untouched and no weights are read for nothing.
    ///   -dropProgramCache  delete the archive first (the fresh-install receipt)
    ///   -noPrime           never prime (the before receipt)     -primeAlways  prime even with the archive there
    func pageReady() {
        guard enabled, !primed else { return }
        primed = true
        Self.log(String(format: "boot page ready at %.0f ms since process start", Boot.sinceStartMs()))
        launchLifecycle(why: "page ready")
        if Self.args.contains("-noPrime") { return Self.log("prime off (-noPrime)") }
        let always = Self.args.contains("-primeAlways")
        if LLBEngine.programCached && !always {
            return Self.log("prime not needed: the compiled program is in the cache (\(LLBEngine.programArchive.lastPathComponent))")
        }
        guard hasModel(.chat) else { return }
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(800)) // the page's first frames and status poll go first
            let dir = Self.modelDir(.chat)
            Self.onEngineThread("bend.engine.prime") {
                let t0 = Date()
                do {
                    let tok = try LLBTokenizer(dir: dir)
                    let ids = tok.encode(try LLBTokenizer.render([.init(role: "user", content: "Hi")], enableThinking: false))
                    Self.log("prime run ids=\(ids.count) n_new=1 (\(always ? "-primeAlways" : "the compiled program is not in the cache"))")
                    // The launch warm may compile it first (llb_open writes the archive): checked again once it's our turn.
                    guard let st = try LLBEngine.shared.prime(dir: dir, ids: ids, always: always) else {
                        return Self.log("prime not needed: the launch warm wrote the compiled program")
                    }
                    Self.log(String(format: "prime done compile_ms=%.0f load_ms=%.0f prefill_ms=%.0f total_ms=%.0f wall_ms=%.0f archive=%@ footprint_mb=%d",
                                    st.compile_ms, st.load_ms, st.prefill_ms, st.total_ms, Date().timeIntervalSince(t0) * 1000,
                                    LLBEngine.programCached ? "cached" : "MISSING", Self.footprint()))
                } catch {
                    Self.log("prime failed (the first request compiles instead): \(error.localizedDescription)")
                }
            }
        }
    }

    private var launchedLifecycle = false

    /// The lifecycle's launch rule, once: the page is up (or never said so).
    private func launchLifecycle(why: String) {
        guard !launchedLifecycle else { return }
        launchedLifecycle = true
        // G11: -engineCheck vision measures the vision stack alone; chat stays cold (no eager warm) unless -visionWithChat.
        if EngineCheck.visionOnly { Self.log("boot lifecycle launch skipped (\(why); -engineCheck vision: chat stays cold)"); return }
        Self.log("boot lifecycle launch (\(why))")
        lifecycle.launched()
    }

    /// Both loopback listeners, (re)opened.
    private func openListeners() {
        control?.stop()
        engine?.stop()
        do {
            let server = try ControlServer(port: Self.controlPort, manager: self)
            server.start()
            control = server
            let port = try EngineServer(port: UInt16(Self.enginePort), key: apiKey, routes: self)
            port.start()
            engine = port
        } catch {
            Self.log("listener failed: \(error)")
        }
    }

    /// The app (or the web page) left the screen: the lifecycle releases after its grace.
    func away() {
        lifecycle.away()
        #if canImport(UIKit)
        guard backgroundTask == .invalid else { return }
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "bend.engine.grace") {
            MainActor.assumeIsolated { EngineManager.shared.backgroundTimeOver() }
        }
        #endif
    }

    func back() {
        lifecycle.back()
        endBackgroundTask()
    }

    #if canImport(UIKit)
    /// iOS is about to suspend Bend before the grace ran out: release now, then let it.
    private func backgroundTimeOver() {
        lifecycle.suspending()
        Task { @MainActor in
            await self.lifecycle.settle()
            self.endBackgroundTask()
        }
    }
    #endif

    private func endBackgroundTask() {
        #if canImport(UIKit)
        guard backgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
        #endif
    }

    private func memoryWarning() {
        Self.log("memory warning: running=\(running?.rawValue ?? "none") footprint_mb=\(Self.footprint())")
        // G18 (hot state): the kept prefix snapshots (~60 MB each) are the first thing to give back.
        if MemLevers.hotState {
            Task.detached {
                LLBEngine.shared.forgetPrefixes()
                Self.log("memory warning: prefix snapshots dropped footprint_mb=\(Self.footprint())")
            }
        }
    }

    /// The web app's POST /engines/ensure: returns when `id` is warm (warming it if cold, with the bounded retries).
    /// `ahead` (PHONEVOICE, `&ahead=1`): warmed ahead of need, nobody waiting yet (voice mode opened): like the eager
    /// warm, it runs the warm-ups, so the first voice turn does not pay the first job's set-up.
    func ensure(_ id: EngineID, ahead: Bool = false) async throws {
        guard hasModel(id) else { throw EngineError("no-models: \(id.dir)/\(id.model) is not in Documents/models") }
        // The web ensures right before each request (transport.ts): a request is waiting on this warm.
        let waits = ahead ? 0 : 1
        waiting[id, default: 0] += waits
        defer { waiting[id, default: 0] -= waits }
        try await lifecycle.ensure(id)
        active = id
    }

    // MARK: lifecycle backend (EngineBackend)

    func warm(_ id: EngineID, progress: @escaping @MainActor (Double) -> Void) async throws {
        guard hasModel(id) else { throw EngineError("no-models: \(id.dir)/\(id.model) is not in Documents/models") }
        let t0 = Date()
        trace[id] = WarmTrace(startedAt: t0, expectWeightsMs: WarmTrace.expectedWeightsMs(id))
        defer { if let t = trace.removeValue(forKey: id), t.done { lastPhases[id] = t.phases } }
        stage(id, "listeners", progress, 0.1)
        // The surface first: iOS can tear a suspended app's listening sockets down.
        if !(await surfaceAnswers()) {
            Self.log("lifecycle: the loopback listeners do not answer, reopening them")
            openListeners()
            try? await Task.sleep(for: .milliseconds(300))
            guard await surfaceAnswers() else { throw EngineError("the loopback listeners did not come back") }
        }
        let dir = Self.modelDir(id)
        stage(id, "tokenizer", progress, 0.3)
        if id == .chat, resident[.chat] == nil {
            resident[.chat] = try await Task.detached { try LLBTokenizer(dir: dir) }.value
        }
        stage(id, "engine open", progress, 0.4)
        // The library takes the model (checks the sidecar and the GGUF). A library left spent by a failed release
        // refuses here, so it shows as a failed warm with Retry, never as a hang.
        try await Task.detached { try LLBEngine.shared.arm(dir) }.value
        stage(id, "weights", progress, 0.5)
        if LLBEngine.residentOn {
            // The weights, loaded and kept (the GBs; this is the warm the person sees as "warming… N%"). A request
            // waiting on it: no warm-ups (its own first run pays that set-up, for less); nobody waiting: warm them up.
            let warmUp = waiting[id, default: 0] == 0
            // llb_open says nothing until it returns: the bar moves on the last load's time on this device (an
            // estimate, and the status says so); the first load ever shows elapsed seconds and no estimate.
            let ticker = Task { @MainActor in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(250))
                    guard let t = self.trace[id], t.stage == "weights", let expect = t.expectWeightsMs, expect > 0 else { continue }
                    progress(0.5 + 0.4 * min(0.95, Date().timeIntervalSince(t.stageAt) * 1000 / expect))
                }
            }
            defer { ticker.cancel() }
            #if os(macOS)
            // Receipts only (the Mac harness; never in the app): BEND_SLOW_WEIGHTS_MS holds the weights stage that long
            // first, so the simulator's UI shows a phone-length warm (the Mac reads the weights in ~0.4 s).
            if let ms = ProcessInfo.processInfo.environment["BEND_SLOW_WEIGHTS_MS"].flatMap(Double.init) {
                try? await Task.sleep(for: .milliseconds(ms))
            }
            #endif
            let open = try await Self.engineThread("bend.engine.open") { try LLBEngine.shared.open(dir, warmUp: warmUp) }
            trace[id]?.note = String(format: "load_ms=%.0f compile_ms=%.0f", open.loadMs, open.compileMs)
            if open.loaded { WarmTrace.rememberWeightsMs(id, Date().timeIntervalSince(trace[id]?.stageAt ?? t0) * 1000) }
            Self.log(String(format: "resident open %@ loaded=%@ load_ms=%.0f compile_ms=%.0f open_ms=%.0f warm_up=%@ | %@", id.rawValue,
                            open.loaded ? "yes" : "no (already resident)", open.loadMs, open.compileMs, open.totalMs,
                            warmUp ? "yes" : "no (a request is waiting)", LLBEngine.residentReport))
            if id == .decision, warmUp {
                stage(id, "warm-up", progress, 0.9)
                // The decider's first call opens its head and tokenizer and sets up the hook's rows: a two-option
                // throw-away decide pays that here, so the first card doesn't.
                let t1 = Date()
                let ok = (try? await Self.engineThread("bend.engine.open") {
                    try LLBEngine.shared.decide(dir: dir, bodies: [Self.warmupDecide]).replies.first?.hasPrefix(#"{"model""#) ?? false
                }) ?? false
                Self.log(String(format: "resident warm-up decide %@ ms=%.0f", ok ? "ok" : "FAILED", Date().timeIntervalSince(t1) * 1000))
            }
        }
        stage(id, "ready", progress, 1)
        trace[id]?.done = true
        Self.log(String(format: "warm %@ ready_ms=%.0f footprint_mb=%d", id.rawValue, Date().timeIntervalSince(t0) * 1000, Self.footprint()))
        if let t = trace[id] { Self.log("warm \(id.rawValue) phases \(t.line)") }
    }

    /// The warm moved on to `name`: the stage before it is timed, progress set.
    private func stage(_ id: EngineID, _ name: String, _ progress: @MainActor (Double) -> Void, _ p: Double) {
        trace[id]?.enter(name)
        progress(p)
    }

    /// A two-option /v1/systemone body (transport.ts systemoneBody's shape) for the decider's warm-up.
    nonisolated static let warmupDecide =
        #"{"state":"Warm-up.","questions":{"pick":{"type":"choice","instructions":"Pick one.","criteria":{"Yes":null,"No":null}}}}"#

    func release(_ id: EngineID) async {
        resident[id] = nil
        // One-model mode: the other route is warm on the same weights; keep them until both release.
        if LLBEngine.residentOn, let other = EngineID.allCases.first(where: { $0 != id && $0.dir == id.dir }),
           lifecycle.slot(other).phase == .warm {
            return Self.log("release \(id.rawValue): \(other.rawValue) is warm on the same model, the session stays")
        }
        let held = LLBEngine.shared.residentDir == Self.modelDir(id).path
        await Task.detached { LLBEngine.shared.release() }.value
        Self.log("release \(id.rawValue) llb_free footprint_mb=\(Self.footprint())\(held ? " session closed: \(LLBEngine.releaseReport)" : "")")
    }

    func alive(_ id: EngineID) async -> Bool {
        if id == .chat, resident[.chat] == nil { return false }
        // Resident: warm means its weights are loaded. A session gone under a warm engine is relaunched (watchdog).
        if LLBEngine.residentOn, LLBEngine.shared.runningBy == nil, LLBEngine.shared.residentDir != Self.modelDir(id).path { return false }
        return await surfaceAnswers()
    }

    /// Both listeners answer on loopback within 2 s.
    private func surfaceAnswers() async -> Bool {
        for url in ["http://127.0.0.1:\(Self.enginePort)/health", "http://127.0.0.1:\(Self.controlPort)/engines/status"] {
            var request = URLRequest(url: URL(string: url)!)
            request.timeoutInterval = 2
            guard let (_, response) = try? await URLSession.shared.data(for: request),
                  (response as? HTTPURLResponse)?.statusCode == 200 else { return false }
        }
        return true
    }

    /// The Mac lifecycle check only: drops the engine listener the way iOS can while suspended.
    func dropEngineListener() {
        engine?.stop()
        engine = nil
        Self.log("check: engine listener dropped")
    }

    /// The jetsam-relevant number (what Xcode's memory gauge shows), in MB.
    nonisolated static func footprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint / 1_048_576) : -1
    }

    // MARK: status (GET /engines/status)

    func statusJSON() -> [String: Any] {
        func engine(_ id: EngineID) -> [String: Any] {
            var e: [String: Any] = ["state": state(id).rawValue, "model": id.model]
            if let mb = footprintMB[id] { e["footprintMB"] = mb }
            if let s = loadSeconds[id] { e["loadSeconds"] = (s * 100).rounded() / 100 }
            let slot = lifecycle.slot(id)
            e["lifecycle"] = slot.phase.rawValue
            e["progress"] = Int((slot.progress * 100).rounded())
            e["retry"] = slot.retry
            if slot.attempt > 0 { e["attempt"] = slot.attempt }
            if let error = slot.error { e["error"] = error }
            // PHONEUX (additive): the warm in flight, stage by stage, and the last warm's stage times.
            if let t = trace[id], slot.phase == .warming {
                let now = Date()
                e["stage"] = t.stage
                e["stageMs"] = Int(now.timeIntervalSince(t.stageAt) * 1000)
                e["warmMs"] = Int(now.timeIntervalSince(t.startedAt) * 1000)
                if let x = t.expectWeightsMs { e["expectMs"] = Int(x) }
                e["phases"] = t.phases.map { ["name": $0.name, "ms": Int($0.ms)] }
            } else if let phases = lastPhases[id] {
                e["phases"] = phases.map { ["name": $0.name, "ms": Int($0.ms)] }
            }
            return e
        }
        var status: [String: Any] = [
            "port": Self.enginePort,
            "backend": Self.backend,
            "engine": "llb",
            "program": LLBEngine.programTag,
            // Requests are taken from launch when chat's model is present: each one warms its engine first, so the
            // web never holds the composer for a cold engine.
            "warm": hasModel(.chat),
            "lifecycle": ["foreground": lifecycle.foreground, "graceSeconds": lifecycle.policy.grace, "recentlyHot": lifecycle.recentlyHot],
            "chat": engine(.chat),
            "decision": engine(.decision),
            "footprintMB": Self.footprint(),
            "decisionIdleSeconds": Int(Self.decisionIdle),
            // KEVPHONE (additive): whose weights are loaded now ("chat" | "decision" | null), and whether the mode is on.
            "resident": ["on": LLBEngine.residentOn,
                         "engine": EngineID.allCases.first { LLBEngine.shared.residentDir == Self.modelDir($0).path }?.rawValue as Any? ?? NSNull()],
        ]
        // ATTACHVIS (additive): whether a chat photo can be looked at here, and why not.
        #if BEND_CAMERA
        let why = VisionLook.unavailableReason
        status["vision"] = ["available": why == nil, "model": CameraConfig.mmprojFile, "canvas": VisionLook.canvasSide,
                            "reason": why as Any? ?? NSNull(),
                            // BENDSDK (additive): whether POST /v1/look (the camera's yes/no verdict, its own 320 canvas)
                            // can answer here, and why not (src/sdk/bend.ts reads it as the look capability).
                            "look": Self.visionStatus]
        #else
        status["vision"] = ["available": false, "reason": "this build has no vision engine", "look": Self.visionStatus]
        #endif
        // PHONEUX (additive): what's missing on this phone, and how long the process has been up.
        status["setup"] = Boot.setup()
        status["bootMs"] = Int(Boot.sinceStartMs())
        status["active"] = (running ?? active)?.rawValue ?? NSNull()
        status["running"] = running?.rawValue ?? NSNull()
        return status
    }

    /// `-autorun` launch flag: once live, the web app sends one chat turn and one decision by itself (useAutorun.ts).
    let autorun = ProcessInfo.processInfo.arguments.contains("-autorun")
    /// R17: `-noSpeculate` ranks decisions only on the tap (the before receipt); the web reads it from the bridge.
    let speculate = !ProcessInfo.processInfo.arguments.contains("-noSpeculate")
    /// IMGPERF: `-perfProbe` turns the web's frame probe on (src/app/perf.ts; the sim receipts).
    let perfProbe = ProcessInfo.processInfo.arguments.contains("-perfProbe")
    /// IMGPERF: `-imgperfOff` gives the web ATTACHVIS's photo wire (the before receipt; see src/engines/native.ts).
    let imgperf = !ProcessInfo.processInfo.arguments.contains("-imgperfOff")

    /// PHONEUX, simulator Debug builds only: `-hostEngines` points the page at the Mac harness's listeners
    /// (scripts/engine-mac.sh run, on the host's loopback, which a simulator shares) with the key in BEND_API_KEY
    /// (SIMCTL_CHILD_BEND_API_KEY). The simulator can't run the engine (256 MB Metal buffers), so this is how its UI
    /// meets a real one: real warming, stages and replies, from the same Swift layer on the Mac's GPU.
    /// FOLDIN-R5: the same switch as BENDSDK's `EngineManager.hostEngines` (BendApp reads that one before `shared` exists).
    var hostEngines: Bool { Self.hostEngines }

    /// What the web app reads at document start (WebAppView): where the engines are and the key for them.
    var bridgeScript: String? {
        #if targetEnvironment(simulator)
        // ATTACHVIS receipts: `-hostEngines <base> <key>` points the simulator's web UI at the Mac's engine harness
        // (scripts/engine-mac.sh with BEND_PORT_BASE=<base> BEND_API_KEY=<key>): the simulator shares the Mac's
        // loopback, and its own Metal can't run the engine. The same Swift engine layer answers, on the Mac's GPU.
        // WORKFIX's `-hostEngine PORT KEY` (its Work receipts, scripts/work-sim-e2e.sh) is the same switch: either spelling.
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(where: { $0 == "-hostEngines" || $0 == "-hostEngine" }), i + 2 < args.count, let base = Int(args[i + 1]) {
            return """
            window.__BEND_NATIVE__ = Object.freeze({ control: "http://127.0.0.1:\(base)", engine: "http://127.0.0.1:\(base + 1)", key: "\(args[i + 2])", autorun: false, speculate: \(speculate), perf: \(perfProbe), imgperf: \(imgperf), location: "Mac harness" });
            """
        }
        #endif
        // PHONEUX: `-hostEngines` alone (Debug) takes the key from BEND_API_KEY, on this build's own ports.
        var key = apiKey
        var location = "On device"
        if !enabled {
            guard hostEngines, let hostKey = ProcessInfo.processInfo.environment["BEND_API_KEY"] else { return nil }
            key = hostKey
            location = "Mac harness"
        }
        return """
        window.__BEND_NATIVE__ = Object.freeze({ control: "http://127.0.0.1:\(Self.controlPort)", engine: "http://127.0.0.1:\(Self.enginePort)", key: "\(key)", autorun: \(autorun), speculate: \(speculate), perf: \(perfProbe), imgperf: \(imgperf), location: "\(location)" });
        """
    }

    // MARK: engine port routes (EngineServer)

    func health() -> (Bool, String) {
        if !hasModel(.chat) { return (false, "no-models: \(EngineID.chat.dir) is not in Documents/models") }
        return (true, "")
    }

    /// A run is over. `died`: the engine failed under it; the lifecycle releases and restarts it.
    func ended(_ id: EngineID, loadMs: Double, died: String?) {
        footprintMB[id] = Self.footprint()
        if loadMs > 0 { loadSeconds[id] = loadMs / 1000 }
        lifecycle.end(id, died: died)
    }

    /// Every request ensures its engine is warm first, then holds it (a release waits for the run). Answers the
    /// request itself when it can't: 503 with why, after the lifecycle's bounded retries.
    func admit(_ id: EngineID, _ x: HTTPExchange) async -> Bool {
        waiting[id, default: 0] += 1
        defer { waiting[id, default: 0] -= 1 }
        do {
            try await lifecycle.ensure(id)
        } catch {
            x.json(503, EngineServer.error(503, error.localizedDescription, type: "unavailable_error"))
            return false
        }
        lifecycle.begin(id)
        return true
    }

    /// Answers the request when `id`'s model is not on the phone.
    nonisolated func present(_ id: EngineID, _ x: HTTPExchange) -> Bool {
        guard Self.hasModel(id) else {
            x.json(404, EngineServer.error(404, "no-models: \(id.dir)/\(id.model) is not in Documents/models", type: "not_found_error"))
            return false
        }
        return true
    }

    /// The engine's run on its own thread (the engine's main is deep; give it a main-thread-sized stack).
    nonisolated static func onEngineThread(_ name: String, _ body: @escaping @Sendable () -> Void) {
        let thread = Thread(block: body)
        thread.name = name
        thread.stackSize = 16 << 20
        thread.start()
    }

    /// `body` on an engine thread, awaited.
    nonisolated private static func engineThread<T: Sendable>(_ name: String, _ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<T, Error>) in
            onEngineThread(name) { c.resume(with: Result { try body() }) }
        }
    }

    /// The request's residency, for its stats line: resident=warm (nothing loaded) | reloaded (it loaded) | off.
    nonisolated private static func residency(_ r: LLBEngine.Residency?) -> String {
        "residency=\(r?.rawValue ?? "off") \(LLBEngine.residentOn ? LLBEngine.residentReport : "")"
    }

    nonisolated func chat(_ x: HTTPExchange) {
        guard let request = try? JSONSerialization.jsonObject(with: x.body) as? [String: Any],
              let raw = request["messages"] as? [[String: Any]] else {
            return x.json(400, EngineServer.error(400, "messages: expected an array of {role, content}", type: "invalid_request_error"))
        }
        let messages = raw.map { LLBTokenizer.Message(role: $0["role"] as? String ?? "", content: Self.text($0["content"])) }
        // IMGPERF: what crossed the wire for this turn (the whole body, and how much of it is image data URLs).
        Self.log("wire chat body_bytes=\(x.body.count) image_bytes=\(Self.imageBytes(raw)) messages=\(raw.count)")
        let stream = request["stream"] as? Bool ?? false
        let model = ProviderContract.model(request, EngineID.chat.model)
        let asked = (request["max_tokens"] as? Int) ?? (request["n_predict"] as? Int) ?? Self.defaultMaxTokens
        let nNew = min(max(asked, 1), Self.maxNew)
        // ATTACHVIS: a photo in a user turn (OpenAI `image_url` content) is a look: the vision path answers it.
        // IMAGETRUE: the photo's turn may be followed by a conversation about it (the Look surface): those turns ride
        // as `earlier` exchanges and the last user turn is the question (VisionLook.conversationText).
        if let at = raw.lastIndex(where: { $0["role"] as? String == "user" && Self.imageURL($0["content"]) != nil }) {
            let user = raw[at]
            let talk = Self.exchanges(raw[at...].map { (role: $0["role"] as? String ?? "", text: Self.text($0["content"])) })
            return look(x, image: Self.imageURL(user["content"])!, key: Self.imageKey(user["content"]), question: talk.question,
                        earlier: talk.earlier, nNew: asked, stream: stream, model: model)
        }
        let thinking = (request["chat_template_kwargs"] as? [String: Any])?["enable_thinking"] as? Bool ?? false
        let rendered: String
        do {
            rendered = try LLBTokenizer.render(messages, enableThinking: thinking)
        } catch {
            return x.json(400, EngineServer.error(400, error.localizedDescription, type: "invalid_request_error"))
        }
        guard present(.chat, x) else { return }
        Task { @MainActor in
            guard await self.admit(.chat, x) else { return }
            self.runChat(x, messages: messages, rendered: rendered, nNew: nNew, stream: stream, thinking: thinking, model: model)
        }
    }

    nonisolated private func runChat(_ x: HTTPExchange, messages: [LLBTokenizer.Message], rendered: String, nNew: Int, stream: Bool,
                                     thinking: Bool, model: String) {
        Self.onEngineThread("bend.engine.chat") { [resident] in
            let dir = Self.modelDir(.chat)
            let created = Int(Date().timeIntervalSince1970)
            let id = "chatcmpl-llb-\(created)"
            func frame(_ delta: [String: Any], finish: String?) -> [String: Any] {
                ["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [["index": 0, "delta": delta, "finish_reason": finish.map { $0 as Any } ?? NSNull()]]]
            }
            var promptIds: [UInt32] = []
            var newIds: [UInt32] = []
            var text = ""
            var answered = false
            let t0 = Date()
            var firstMs: Double?

            func answer(_ finish: String, _ extra: [String: Any]) {
                guard !answered else { return }
                answered = true
                var llb = extra
                llb["program"] = LLBEngine.programTag
                llb["prompt_ids"] = promptIds
                llb["ids"] = newIds
                let usage = ["prompt_tokens": promptIds.count, "completion_tokens": newIds.count,
                             "total_tokens": promptIds.count + newIds.count]
                if stream {
                    var last = frame([:], finish: finish)
                    last["usage"] = usage
                    last["llb"] = llb
                    x.event(last)
                    x.endStream()
                } else {
                    x.json(200, ["id": id, "object": "chat.completion", "created": created, "model": model,
                                 "choices": [["index": 0, "message": ["role": "assistant", "content": text], "finish_reason": finish]],
                                 "usage": usage, "llb": llb])
                }
                Self.log(String(format: "chat answered finish=%@ new=%d first_token_ms=%.0f reply_ms=%.0f",
                                finish, newIds.count, firstMs ?? -1, Date().timeIntervalSince(t0) * 1000))
            }

            do {
                let tok = try resident[.chat] ?? LLBTokenizer(dir: dir)
                promptIds = tok.encode(rendered)
                let nPre = MemLevers.prefix(messages: messages, promptIds: promptIds, tok: tok)
                Self.log("chat run prompt_ids=\(promptIds.count) n_new=\(nNew) stream=\(stream) thinking=\(thinking) footprint_mb=\(Self.footprint())\(nPre > 0 ? " n_pre=\(nPre)" : "")")
                if stream { x.beginStream() }
                var joiner = UTF8Joiner()
                let result = try LLBEngine.shared.generate(dir: dir, ids: promptIds, nPre: nPre, nNew: nNew) { _, tid in
                    if tid == tok.imEnd || tid == tok.endOfText {
                        text += joiner.flush()
                        answer("stop", [:])
                        return false
                    }
                    if firstMs == nil { firstMs = Date().timeIntervalSince(t0) * 1000 }
                    newIds.append(tid)
                    let piece = joiner.push(tok.piece(tid))
                    text += piece
                    if stream, !piece.isEmpty { x.event(frame(["content": piece], finish: nil)) }
                    return true
                }
                let st = result.stats
                let decode = st.steps > 0 ? st.step_ms_sum / Double(st.steps) : 0
                let stats: [String: Any] = ["load_ms": st.load_ms, "prefill_ms": st.prefill_ms, "decode_ms_per_token": decode,
                                            "steps": Int(st.steps), "compile_ms": st.compile_ms, "total_ms": st.total_ms,
                                            "span_mb": Int(st.span_bytes >> 20), "footprint_mb": Self.footprint()]
                    .merging(result.prefixIds.map { ["n_pre": nPre, "prefix_ids": $0] } ?? [:]) { a, _ in a }
                if !answered { text += joiner.flush() }
                answer("length", stats)
                Self.log("chat ids \(result.ids.map(String.init).joined(separator: " "))")
                Self.log("chat stream \(result.mismatch.map { "MISMATCH \($0)" } ?? "matches the engine's final ids (\(result.streamed)/\(result.ids.count) streamed)")")
                Self.log(String(format: "chat stats compile_ms=%.0f load_ms=%.0f prefill_ms=%.0f decode_ms_per_tok=%.2f tok_s=%.1f steps=%d total_ms=%.0f span_mb=%d footprint_mb=%d %@",
                                st.compile_ms, st.load_ms, st.prefill_ms, decode, decode > 0 ? 1000 / decode : 0, st.steps,
                                st.total_ms, Int(st.span_bytes >> 20), Self.footprint(), Self.residency(result.resident))
                         + (result.prefixIds.map { " n_pre=\(nPre) prefix_ids=\($0)" } ?? ""))
                Task { @MainActor in self.ended(.chat, loadMs: st.load_ms, died: nil) }
            } catch {
                Self.log("chat error: \(error.localizedDescription)")
                if stream, !answered {
                    x.event(["error": ["code": 500, "message": error.localizedDescription, "type": "server_error"]])
                    x.endStream()
                } else if !answered {
                    x.json(500, EngineServer.error(500, error.localizedDescription))
                }
                Task { @MainActor in self.ended(.chat, loadMs: 0, died: error.localizedDescription) }
            }
        }
    }

    /// ATTACHVIS: a look (src/engines/transport.ts `lookBody`): the photo + the person's words → VisionLook on the chat
    /// model's resident session with the 4B tower, streamed like chat. It is a chat run to the lifecycle (admit(.chat):
    /// warms chat if cold, holds it against release). A build that can't look answers at once, never a hang:
    /// 404 "no-models: …" (the tower or the model is not on the phone), 503 "no-vision: …" (the simulator's Metal,
    /// -kevChat, or a target without the vision code).
    nonisolated private func look(_ x: HTTPExchange, image url: String, key: String?, question: String, earlier: [VisionLookExchange] = [],
                                  nNew asked: Int, stream: Bool, model: String) {
        #if BEND_CAMERA
        if let why = VisionLook.unavailableReason {
            let missing = why.hasPrefix("no model") || why.hasPrefix("no vision tower")
            Self.log("look refused: \(why)")
            return x.json(missing ? 404 : 503, EngineServer.error(missing ? 404 : 503, "\(missing ? "no-models" : "no-vision"): \(why)",
                                                                  type: missing ? "not_found_error" : "unavailable_error"))
        }
        // IMGPERF send-once: `bend-image:<key>` names a photo the web handed over before (POST /v1/bend/image, or an
        // earlier look that carried its key). Not held here (a relaunch, or it fell out of the last 4): 409, and the
        // web sends the photo itself.
        let canvas: Canvas
        var letterboxMs = 0.0
        if url.hasPrefix(Self.imageRef) {
            let k = String(url.dropFirst(Self.imageRef.count))
            guard let held = VisionLook.shared.canvas(key: k) else {
                Self.log("look image-unknown key=\(k)")
                return x.json(409, EngineServer.error(409, "image-unknown: this photo isn't held here (send it)", type: "image_unknown"))
            }
            canvas = held
            Self.log("look by key=\(k): no upload, no decode, no letterbox")
        } else {
            let image: CGImage
            let td = Date()
            do { image = try VisionLook.decode(url) } catch {
                return x.json(400, EngineServer.error(400, error.localizedDescription, type: "invalid_request_error"))
            }
            let decodeMs = Date().timeIntervalSince(td) * 1000
            let tl = Date()
            canvas = VisionLook.canvas(image)
            letterboxMs = Date().timeIntervalSince(tl) * 1000
            if let key { VisionLook.shared.store(canvas, key: key) }
            Self.log(String(format: "look decode image=%dx%d url_bytes=%d decode_ms=%.1f letterbox_ms=%.1f key=%@", image.width, image.height,
                            url.utf8.count, decodeMs, letterboxMs, key ?? "-"))
        }
        let nNew = min(max(asked, 1), VisionLook.maxNew)
        Task { @MainActor in
            guard await self.admit(.chat, x) else { return }
            self.runLook(x, canvas: canvas, letterboxMs: letterboxMs, question: question, earlier: earlier, nNew: nNew, stream: stream, model: model)
        }
        #else
        x.json(503, EngineServer.error(503, "no-vision: this build has no vision engine", type: "unavailable_error"))
        #endif
    }

    /// IMGPERF send-once: the photo, handed over when it lands in the composer (POST /v1/bend/image {url, key}). It is
    /// decoded and letterboxed now and kept by its key; answered at once; then, on an engine thread, chat warms, the
    /// tower opens and the photo is encoded (its rows kept), all while the person types. The look names the key.
    nonisolated func image(_ x: HTTPExchange) {
        #if BEND_CAMERA
        guard let body = try? JSONSerialization.jsonObject(with: x.body) as? [String: Any], let url = body["url"] as? String,
              let key = body["key"] as? String, !key.isEmpty else {
            return x.json(400, EngineServer.error(400, "image: expected {url, key}", type: "invalid_request_error"))
        }
        Self.log("wire image body_bytes=\(x.body.count) key=\(key)")
        if let why = VisionLook.unavailableReason {
            let missing = why.hasPrefix("no model") || why.hasPrefix("no vision tower")
            return x.json(missing ? 404 : 503, EngineServer.error(missing ? 404 : 503, "\(missing ? "no-models" : "no-vision"): \(why)",
                                                                  type: missing ? "not_found_error" : "unavailable_error"))
        }
        let canvas: Canvas
        if let held = VisionLook.shared.canvas(key: key) {
            canvas = held
        } else {
            let td = Date()
            let image: CGImage
            do { image = try VisionLook.decode(url) } catch {
                return x.json(400, EngineServer.error(400, error.localizedDescription, type: "invalid_request_error"))
            }
            canvas = VisionLook.canvas(image)
            VisionLook.shared.store(canvas, key: key)
            Self.log(String(format: "image held key=%@ image=%dx%d url_bytes=%d decode_letterbox_ms=%.1f", key, image.width, image.height,
                            url.utf8.count, Date().timeIntervalSince(td) * 1000))
        }
        x.json(200, ["key": key, "canvas": VisionLook.canvasSide, "image_tokens": (VisionLook.canvasSide / 32) * (VisionLook.canvasSide / 32)])
        Self.onEngineThread("bend.engine.image") {
            do {
                let a = try VisionLook.shared.encodeAhead(canvas)
                Self.log(String(format: "image ahead key=%@ ms=%.0f ready_ms=%.0f tower_ms=%.0f was_kept=%d footprint_mb=%d", key, a.ms,
                                a.readyMs, a.towerMs, a.wasKept ? 1 : 0, Self.footprint()))
            } catch {
                Self.log("image ahead failed key=\(key): \(error.localizedDescription) (the look will encode it)")
            }
        }
        #else
        x.json(503, EngineServer.error(503, "no-vision: this build has no vision engine", type: "unavailable_error"))
        #endif
    }

    /// IMAGETRUE: the photo's turn and what followed it → the earlier exchanges and the question now. Turns pair up in
    /// order (a user turn, then its answer); a user turn with no answer yet folds into the next one (two questions in a
    /// row are one question), and the last user turn's words are the question.
    nonisolated static func exchanges(_ turns: [(role: String, text: String)]) -> (earlier: [VisionLookExchange], question: String) {
        var earlier: [VisionLookExchange] = []
        var asked: String?
        for t in turns {
            switch t.role {
            case "user": asked = [asked, t.text].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: "\n")
            case "assistant":
                if let a = asked { earlier.append(VisionLookExchange(asked: a, answered: t.text)) }
                asked = nil
            default: break
            }
        }
        return (earlier, asked ?? "")
    }

    /// IMGPERF: the URL scheme a look uses to name a photo held here.
    nonisolated static let imageRef = "bend-image:"

    #if BEND_CAMERA
    nonisolated private func runLook(_ x: HTTPExchange, canvas: Canvas, letterboxMs: Double, question: String, earlier: [VisionLook.Exchange],
                                     nNew: Int, stream: Bool, model: String) {
        Self.onEngineThread("bend.engine.look") {
            let created = Int(Date().timeIntervalSince1970)
            let id = "chatcmpl-llb-look-\(created)"
            func frame(_ delta: [String: Any], finish: String?) -> [String: Any] {
                ["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [["index": 0, "delta": delta, "finish_reason": finish.map { $0 as Any } ?? NSNull()]]]
            }
            Self.log("look run canvas=\(canvas.side) n_new=\(nNew) stream=\(stream) turns=\(earlier.count + 1) footprint_mb=\(Self.footprint())")
            if stream { x.beginStream() }
            do {
                // Stop (the web aborts its fetch) shows here as a closed exchange: decoding ends at the next id.
                let r = try VisionLook.shared.look(canvas: canvas, question: question, earlier: earlier, maxNew: nNew, letterboxMs: letterboxMs) { piece in
                    if stream { x.event(frame(["content": piece], finish: nil)) }
                    return !x.isClosed
                }
                let llb: [String: Any] = ["program": LLBEngine.programTag, "look": true, "ids": r.ids, "image_tokens": r.imageTokens,
                                          "canvas": VisionLook.canvasSide, "first_ms": r.firstMs, "total_ms": r.totalMs,
                                          "tower_ms_sum": r.towerMsSum, "llm_ms_sum": r.llmMsSum]
                let usage = ["prompt_tokens": r.promptIds, "completion_tokens": r.ids.count, "total_tokens": r.promptIds + r.ids.count]
                if stream {
                    var last = frame([:], finish: r.finish)
                    last["usage"] = usage
                    last["llb"] = llb
                    x.event(last)
                    x.endStream()
                } else {
                    x.json(200, ["id": id, "object": "chat.completion", "created": created, "model": model,
                                 "choices": [["index": 0, "message": ["role": "assistant", "content": r.text], "finish_reason": r.finish]],
                                 "usage": usage, "llb": llb])
                }
                Self.log(String(format: "look answered finish=%@ new=%d image_tokens=%d first_token_ms=%.0f reply_ms=%.0f ms_per_id=%.0f tower_ms_sum=%.0f llm_ms_sum=%.0f footprint_mb=%d ready_ms=%.0f letterbox_ms=%.1f tower_calls=%d prefill_ms=%.0f decode_ms_per_id=%.1f rows_kept=%d prefix_ids=%d prompt_ids=%d path=%@",
                                r.finish, r.ids.count, r.imageTokens, r.firstMs, r.totalMs, r.totalMs / Double(max(1, r.ids.count)),
                                r.towerMsSum, r.llmMsSum, Self.footprint(), r.readyMs, r.letterboxMs, r.towerCalls, r.prefillMs,
                                r.decodeMsPerId, r.rowsKept ? 1 : 0, r.prefixIds, r.promptIds, VisionLook.loop ? "loop" : "generate"))
                Self.log("look ids \(r.ids.map(String.init).joined(separator: " "))")
                Self.log("look text \(r.text.replacingOccurrences(of: "\n", with: " "))")
                Task { @MainActor in self.ended(.chat, loadMs: 0, died: nil) }
            } catch {
                Self.log("look error: \(error.localizedDescription)")
                if stream {
                    x.event(["error": ["code": 500, "message": error.localizedDescription, "type": "server_error"]])
                    x.endStream()
                } else {
                    x.json(500, EngineServer.error(500, error.localizedDescription))
                }
                // A vision failure (the tower, a bad canvas) is not the chat model dying: the lifecycle keeps chat warm.
                Task { @MainActor in self.ended(.chat, loadMs: 0, died: nil) }
            }
        }
    }
    #endif

    nonisolated func systemone(_ x: HTTPExchange) {
        let body = String(decoding: x.body, as: UTF8.self)
        guard (try? JSONSerialization.jsonObject(with: x.body)) is [String: Any] else {
            return x.json(400, EngineServer.error(400, "systemone: the body is not a JSON object", type: "invalid_request_error"))
        }
        guard present(.decision, x) else { return }
        Task { @MainActor in
            guard await self.admit(.decision, x) else { return }
            self.runDecide(x, body: body, model: nil)
        }
    }

    nonisolated func runDecide(_ x: HTTPExchange, body: String, model: String?) {
        Self.onEngineThread("bend.engine.decide") {
            let t0 = Date()
            do {
                Self.log("decide run body_bytes=\(x.body.count) footprint_mb=\(Self.footprint())")
                let (replies, st, residency) = try LLBEngine.shared.decide(dir: Self.modelDir(.decision), bodies: [body])
                let reply = replies[0]
                if reply.hasPrefix(#"{"error""#) {
                    let message = ((try? JSONSerialization.jsonObject(with: Data(reply.utf8))) as? [String: Any])?["error"] as? String ?? reply
                    x.json(400, EngineServer.error(400, message, type: "invalid_request_error"))
                } else {
                    x.reply(200, Data((model.map { ProviderContract.fillModel(reply, model: $0) } ?? reply).utf8))
                }
                Self.log("decide reply \(reply)")
                Self.log(String(format: "decide stats fastpf=%d compile_ms=%.0f load_ms=%.0f head_open_ms=%.1f forward_ms=%.1f head_ms=%.2f chunks=%d chunk_tokens=%d questions=%d input_tokens=%d total_ms=%.0f wall_ms=%.0f span_mb=%d footprint_mb=%d %@",
                                st.fastpf, st.compile_ms, st.load_ms, st.head_open_ms, st.forward_ms, st.head_ms, st.chunks,
                                st.chunk_tokens, st.questions, st.input_tokens, st.total_ms, Date().timeIntervalSince(t0) * 1000,
                                Int(st.span_bytes >> 20), Self.footprint(), Self.residency(residency)))
                Task { @MainActor in self.ended(.decision, loadMs: st.load_ms, died: nil) }
            } catch {
                Self.log("decide error: \(error.localizedDescription)")
                x.json(500, EngineServer.error(500, error.localizedDescription))
                Task { @MainActor in self.ended(.decision, loadMs: 0, died: error.localizedDescription) }
            }
        }
    }

    /// IMGPERF: the bytes of every `image_url` part in the request (the measurement's upload column).
    nonisolated static func imageBytes(_ messages: [[String: Any]]) -> Int {
        messages.reduce(0) { sum, m in
            guard let parts = m["content"] as? [[String: Any]] else { return sum }
            return sum + parts.reduce(0) { s, part in
                let u = (part["image_url"] as? [String: Any])?["url"] as? String ?? part["image_url"] as? String ?? ""
                return s + u.utf8.count
            }
        }
    }

    /// IMGPERF: the first `image_url` part's `key` (the web's content key for send-once), or nil.
    nonisolated private static func imageKey(_ content: Any?) -> String? {
        guard let parts = content as? [[String: Any]] else { return nil }
        for part in parts where part["type"] as? String == "image_url" {
            if let key = (part["image_url"] as? [String: Any])?["key"] as? String, !key.isEmpty { return key }
        }
        return nil
    }

    /// ATTACHVIS: the first `image_url` part's URL (a data URL from the web app), or nil.
    nonisolated private static func imageURL(_ content: Any?) -> String? {
        guard let parts = content as? [[String: Any]] else { return nil }
        for part in parts where part["type"] as? String == "image_url" {
            if let url = (part["image_url"] as? [String: Any])?["url"] as? String { return url }
            if let url = part["image_url"] as? String { return url }
        }
        return nil
    }

    // MARK: look (BENDSDK)

    /// {available, reason?, canvas, tokens}: the camera engine's vision path in this build, on this device.
    nonisolated static var visionStatus: [String: Any] {
        #if BEND_CAMERA
        let why = CameraEngine.unavailableReason
        var v: [String: Any] = ["available": why == nil, "canvas": CameraConfig.canvasSide, "tokens": CameraConfig.imageTokens,
                                "tower": CameraConfig.mmprojFile]
        if let why { v["reason"] = why }
        return v
        #else
        return ["available": false, "reason": "this build has no vision path (the camera engine is compiled into the main (camera) target only)"]
        #endif
    }

    /// G27: POST /v1/bend/pocket {url, mask?}: the photo (a data URL, as the composer holds it) → VisionPocket on its own
    /// thread: the detector's boxes, the top box's mask (a PNG when `mask` is not false) and the sentence. CoreML only:
    /// no LLM session, so no admit(); the packages load on the first request and stay. 503 "no-vision: …" when they are
    /// not staged (Documents/ml, scripts/vision-stage.sh).
    nonisolated func pocket(_ x: HTTPExchange) {
        guard let request = try? JSONSerialization.jsonObject(with: x.body) as? [String: Any], let url = request["url"] as? String else {
            return x.json(400, EngineServer.error(400, "pocket: expected {url: a data URL of the photo, mask?: bool}", type: "invalid_request_error"))
        }
        let withMask = request["mask"] as? Bool ?? true
        let b64 = url.hasPrefix("data:") ? String(url[(url.firstIndex(of: ",").map { url.index(after: $0) } ?? url.endIndex)...]) : url
        guard let data = Data(base64Encoded: b64, options: .ignoreUnknownCharacters), let image = VisionPocket.decode(data) else {
            return x.json(400, EngineServer.error(400, "pocket: couldn't decode the photo", type: "invalid_request_error"))
        }
        Task.detached(priority: .userInitiated) {
            guard let p = await PocketHost.shared.get(log: { Self.log($0) }) else {
                return x.json(503, EngineServer.error(503, "no-vision: the pocket packages are not staged (Documents/ml)", type: "unavailable_error"))
            }
            do {
                let a = try p.run(image)
                Self.log(String(format: "pocket image=%dx%d boxes=%d top=%@ mask_area=%.3f prompt=%@ ms=%.1f %@ footprint_mb=%d", image.width, image.height,
                                a.boxes.count, a.boxes.first.map { "\($0.label):\(String(format: "%.2f", $0.score))" } ?? "-", a.mask?.area ?? -1, a.prompt,
                                a.total, a.ms.map { String(format: "%@_ms=%.1f", $0.0, $0.1) }.joined(separator: " "), Self.footprint()))
                x.json(200, VisionPocket.json(a, segName: p.segName, detName: p.detName, withMask: withMask))
            } catch {
                Self.log("pocket error: \(error.localizedDescription)")
                x.json(500, EngineServer.error(500, "pocket: \(error.localizedDescription)"))
            }
        }
    }

    /// One image + one yes/no question → the camera's verdict, on the chat model's resident session (the image area
    /// every session reserves) and the vision tower. A request like any other: it ensures chat warm and holds it.
    nonisolated func look(_ x: HTTPExchange) {
        #if BEND_CAMERA
        guard let request = try? JSONSerialization.jsonObject(with: x.body) as? [String: Any],
              let question = (request["question"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !question.isEmpty,
              let w = request["width"] as? Int, let h = request["height"] as? Int, w > 0, h > 0, w <= 2048, h <= 2048,
              let rgb = (request["rgb"] as? String).flatMap({ Data(base64Encoded: $0) }), rgb.count == w * h * 3 else {
            return x.json(400, EngineServer.error(400, "look: expected {question, width, height, rgb: base64 of width*height*3 bytes}", type: "invalid_request_error"))
        }
        if let why = CameraEngine.unavailableReason {
            return x.json(503, EngineServer.error(503, "vision unavailable: \(why)", type: "unavailable_error"))
        }
        guard present(.chat, x) else { return }
        Task { @MainActor in
            guard await self.admit(.chat, x) else { return }
            Self.onEngineThread("bend.engine.look") {
                let t0 = Date()
                do {
                    try CameraEngine.shared.ensureReady(tower: true, forLook: true)
                    let canvas = Canvas.letterbox(rgb: [UInt8](rgb), width: w, height: h)
                    let prepMs = Date().timeIntervalSince(t0) * 1000
                    var v = try CameraEngine.shared.classify(canvas, question: question)
                    v.frameMs = Date().timeIntervalSince(t0) * 1000
                    Self.log(v.logLine("look"))
                    x.json(200, ["verdict": v.yes ? "yes" : "no", "yes": v.yes, "score": Double(v.score), "confidence": v.confidence,
                                 "bits": ["yes": String(format: "%08x", v.yesBits), "no": String(format: "%08x", v.noBits)],
                                 "first_id": Int(v.firstID), "question": question, "canvas": canvas.side, "source": ["width": w, "height": h],
                                 "ms": ["total": v.frameMs, "prepare": prepMs, "engine": v.engineMs, "tower": v.towerMs, "llm": v.llmMs]])
                    Task { @MainActor in self.ended(.chat, loadMs: 0, died: nil); self.closeTowerWhenIdle() }
                } catch {
                    Self.log("look error: \(error.localizedDescription)")
                    x.json(500, EngineServer.error(500, "look: \(error.localizedDescription)"))
                    Task { @MainActor in self.ended(.chat, loadMs: 0, died: nil) }
                }
            }
        }
        #else
        x.json(503, EngineServer.error(503, "vision unavailable: \(Self.visionStatus["reason"] as? String ?? "")", type: "unavailable_error"))
        #endif
    }

    #if BEND_CAMERA
    private var lastLook = Date.distantPast
    /// The tower (641 MiB) a look opened is closed again after 30 s without one; the camera surface's own tower is its own.
    private func closeTowerWhenIdle() {
        lastLook = Date()
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(30))
            guard Date().timeIntervalSince(self.lastLook) >= 29.5 else { return }
            Self.onEngineThread("bend.engine.look.close") { CameraEngine.shared.closeIfOpenedByLook() }
        }
    }
    #endif

    /// OpenAI content: a string, or an array of {type:"text", text} parts.
    nonisolated private static func text(_ content: Any?) -> String {
        if let s = content as? String { return s }
        if let parts = content as? [[String: Any]] { return parts.compactMap { $0["text"] as? String }.joined() }
        return ""
    }
}
