import Foundation
import UIKit

/// Which model serves the engine port right now. The web app names them the same way.
enum EngineID: String, CaseIterable, Sendable {
    /// Qwen3.5-4B: chat, propose, the Millionaire Emcee.
    case chat
    /// Kev-4B: /v1/systemone, the decision card and the Millionaire Jev.
    case decision

    var model: String {
        switch self {
        case .chat: "Qwen3.5-4B-Q4_K_M.gguf"
        case .decision: "Kev-4B-Q4_K_M.gguf"
        }
    }
}

/// Owns the in-process llama-server (llama.cpp tools/server, see LlamaServerHost) and the policy for it.
///
/// llama-server keeps process-global state, so exactly one model serves at a time, on `enginePort`.
/// Two 4B models would not both fit in the phone's memory budget anyway (2.7 + 3.0 GB of weights).
/// Policy: the engine lifecycle (EngineLifecycle.swift, docs/ENGINES.md), exclusive — warming one model stops the
/// server and starts it on the other:
///   - warm on first need (POST /engines/ensure before each request); chat eagerly at launch only if it was warm
///     when the app last left; a warm model whose server died is relaunched before the request (`ensureAlive`);
///   - kept warm while foreground; away from the screen → the server is stopped after the grace (memory back).
/// The web layer serialises its own requests, so a swap never cuts a request in flight.
@MainActor
final class EngineManager: EngineBackend {
    static let shared = EngineManager()

    nonisolated static let controlPort: UInt16 = 8930
    nonisolated static let enginePort = 8931
    /// Kept in the status for the web types; release is the lifecycle's grace now (both engines).
    nonisolated static let decisionIdle: TimeInterval = 0

    enum State: String { case noModels = "no-models", idle, loading, live, error }

    struct EngineError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    /// Per-launch key for both listeners: other apps on the phone can reach loopback ports too.
    let apiKey: String = (0..<4).map { _ in String(format: "%08x", UInt32.random(in: .min ... .max)) }.joined()

    /// Device: always. Simulator: only with the `-simEngines` launch flag (CPU only, see `offloadArgs`),
    /// so a plain simulator build runs the demo engines as before.
    let enabled: Bool = {
        #if targetEnvironment(simulator)
        return ProcessInfo.processInfo.arguments.contains("-simEngines")
        #else
        return true
        #endif
    }()

    /// The model on the engine port, once /health says 200.
    private(set) var loaded: EngineID?
    private var footprintMB: [EngineID: Int] = [:]
    private var loadSeconds: [EngineID: Double] = [:]
    private var control: ControlServer?
    /// Background time asked for while away, so the grace can run; if iOS ends it first, release then.
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid

    private(set) lazy var lifecycle: EngineLifecycle = {
        var policy = EngineLifecycle.Policy()
        policy.exclusive = true
        return EngineLifecycle(backend: self, clock: SystemClock(), store: DefaultsHotStore(), policy: policy, log: Self.log)
    }()

    nonisolated static var modelsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("models")
    }

    nonisolated static func log(_ line: String) {
        // One greppable line per event, in the Xcode console and `devicectl … --console`.
        print("BEND engine \(line)")
        fflush(stdout)
    }

    /// Device: full Metal offload. Simulator: CPU, because ggml-metal gives garbage on the simulator's Metal
    /// device (bend-phone-prep R2), and 8 threads, because all host cores spin-wait on a busy Mac.
    nonisolated private static var offloadArgs: [String] {
        #if targetEnvironment(simulator)
        return ["-ngl", "0", "--device", "none", "-t", "8"]
        #else
        return ["-ngl", "99"]
        #endif
    }

    nonisolated private static var backend: String {
        #if targetEnvironment(simulator)
        return "SIMULATOR/CPU"
        #else
        return "DEVICE/Metal"
        #endif
    }

    func hasModel(_ id: EngineID) -> Bool {
        FileManager.default.fileExists(atPath: Self.modelsDir.appendingPathComponent(id.model).path)
    }

    /// The web's per-engine state (unchanged contract) from the lifecycle phase, as in the LLB target.
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

    /// The web UI is on screen (WebAppView). R17's compile prime is an LLB thing; llama-server has nothing to prime.
    /// PHONEUX: this was missing, so the BendLlama target stopped building at R17.
    func pageReady() {}

    func start() {
        guard enabled else {
            Self.log("disabled (simulator; launch with -simEngines to run them on the CPU)")
            return
        }
        try? FileManager.default.createDirectory(at: Self.modelsDir, withIntermediateDirectories: true)
        do {
            let server = try ControlServer(port: Self.controlPort, manager: self)
            server.start()
            control = server
        } catch {
            Self.log("control listener failed: \(error)")
        }
        for id in EngineID.allCases {
            Self.log("model \(id.rawValue) \(id.model) \(hasModel(id) ? "present" : "MISSING (no-models)")")
        }
        let center = NotificationCenter.default
        center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { EngineManager.shared.memoryWarning() }
        }
        center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { EngineManager.shared.away() }
        }
        center.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { EngineManager.shared.back() }
        }
        lifecycle.launched()
    }

    /// The app (or the web page) left the screen: the lifecycle stops the server after its grace.
    func away() {
        lifecycle.away()
        guard backgroundTask == .invalid else { return }
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "bend.engine.grace") {
            MainActor.assumeIsolated {
                let manager = EngineManager.shared
                manager.lifecycle.suspending()
                Task { @MainActor in
                    await manager.lifecycle.settle()
                    manager.endBackgroundTask()
                }
            }
        }
    }

    func back() {
        lifecycle.back()
        endBackgroundTask()
    }

    private func endBackgroundTask() {
        guard backgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
    }

    private func memoryWarning() {
        Self.log("memory warning: loaded=\(loaded?.rawValue ?? "none") footprint_mb=\(Self.footprint())")
    }

    /// Returns once `id` is live on the engine port: warming (or swapping) it as needed, and relaunching a server
    /// that died under a warm model.
    /// PHONEVOICE: ControlServer passes `ahead`; llama-server has no warm-ups, so it is the plain ensure.
    func ensure(_ id: EngineID, ahead: Bool) async throws { try await ensure(id) }

    func ensure(_ id: EngineID) async throws {
        guard hasModel(id) else { throw EngineError("no-models: \(id.model) is not in Documents/models") }
        if loaded != id { Self.log("ensure \(id.rawValue): swapping in (loaded=\(loaded?.rawValue ?? "none"))") }
        try await lifecycle.ensureAlive(id)
    }

    // MARK: lifecycle backend (EngineBackend)

    func warm(_ id: EngineID, progress: @escaping @MainActor (Double) -> Void) async throws {
        try await load(id, progress: progress)
    }

    func release(_ id: EngineID) async {
        guard loaded == id || loaded == nil else { return } // the server holds the other model
        await stopServer()
    }

    func alive(_ id: EngineID) async -> Bool {
        guard loaded == id, LlamaServerHost.isRunning() else { return false }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(Self.enginePort)/health")!)
        request.timeoutInterval = 2
        guard let (_, response) = try? await URLSession.shared.data(for: request) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    private func stopServer() async {
        guard LlamaServerHost.isRunning() else { loaded = nil; return }
        let was = loaded
        loaded = nil
        let rc = await Task.detached { LlamaServerHost.stop() }.value
        Self.log("server stopped model=\(was?.rawValue ?? "?") rc=\(rc) footprint_mb=\(Self.footprint())")
    }

    private func load(_ id: EngineID, progress: @escaping @MainActor (Double) -> Void) async throws {
        let path = Self.modelsDir.appendingPathComponent(id.model).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw EngineError("no-models: \(id.model) is not in Documents/models")
        }
        await stopServer()
        // llama-server reports no load progress: estimate it from the last load of this model (else ~8 s).
        let expected = loadSeconds[id] ?? 8

        let t0 = Date()
        let args = ["-m", path, "--host", "127.0.0.1", "--port", "\(Self.enginePort)",
                    "-c", "4096", "-np", "1", "--no-webui", "--api-key", apiKey] + Self.offloadArgs
        Self.log("server start \(id.rawValue) \(args.filter { $0 != apiKey }.joined(separator: " "))")
        guard LlamaServerHost.start(withArguments: args) else {
            throw fail(id, "a server is already running")
        }
        // /health is public (no key): 503 while loading, 200 when ready.
        let health = URL(string: "http://127.0.0.1:\(Self.enginePort)/health")!
        while true {
            if !LlamaServerHost.isRunning() { throw fail(id, "llama-server exited while loading \(id.model)") }
            if let (_, response) = try? await URLSession.shared.data(from: health),
               (response as? HTTPURLResponse)?.statusCode == 200 { break }
            progress(min(0.95, Date().timeIntervalSince(t0) / expected))
            if Date().timeIntervalSince(t0) > 600 {
                await stopServer()
                throw fail(id, "\(id.model) did not load in 10 minutes")
            }
            try await Task.sleep(for: .milliseconds(200))
        }
        let seconds = Date().timeIntervalSince(t0)
        loaded = id
        loadSeconds[id] = seconds
        footprintMB[id] = Self.footprint()
        Self.log(String(format: "live %@ model=%@ backend=%@ port=%d load_s=%.2f footprint_mb=%d",
                        id.rawValue, id.model, Self.backend, Self.enginePort, seconds, footprintMB[id] ?? -1))
    }

    private func fail(_ id: EngineID, _ message: String) -> EngineError {
        Self.log("error \(id.rawValue): \(message)")
        return EngineError(message)
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
            return e
        }
        var status: [String: Any] = [
            "port": Self.enginePort,
            "backend": Self.backend,
            // Requests are taken from launch when chat's model is present: each one warms its engine first.
            "warm": hasModel(.chat),
            "lifecycle": ["foreground": lifecycle.foreground, "graceSeconds": lifecycle.policy.grace, "recentlyHot": lifecycle.recentlyHot],
            "chat": engine(.chat),
            "decision": engine(.decision),
            "footprintMB": Self.footprint(),
            "decisionIdleSeconds": Int(Self.decisionIdle),
        ]
        status["active"] = loaded?.rawValue ?? NSNull()
        return status
    }

    /// `-autorun` launch flag: once live, the web app sends one chat turn and one decision by itself (useAutorun.ts),
    /// so a launch from the command line exercises web view → control → swap → engine with no taps.
    let autorun = ProcessInfo.processInfo.arguments.contains("-autorun")

    /// What the web app reads at document start (WebAppView): where the engines are and the key for them.
    var bridgeScript: String? {
        guard enabled else { return nil }
        return """
        window.__BEND_NATIVE__ = Object.freeze({ control: "http://127.0.0.1:\(Self.controlPort)", engine: "http://127.0.0.1:\(Self.enginePort)", key: "\(apiKey)", autorun: \(autorun) });
        """
    }
}
