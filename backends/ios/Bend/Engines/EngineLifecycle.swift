import Foundation

/// The engine lifecycle (R13 policy, R14; docs/ENGINES.md), per engine:
///
///     cold ──need──▶ warming ──ok──▶ warm ──away + grace──▶ releasing ──▶ cold
///                     │ fail: retry ×2 (backoff), then cold + `retry` (the next need tries again)
///
///   - **Warm on first real need**: every request `ensure`s its engine first (cold → warming → warm, with progress).
///     On launch / return, chat is warmed eagerly only if it was warm when the app last left (`recentlyHot`).
///   - **Kept warm while foreground.** Away (background, screen locked, page hidden) starts a grace timer; when it
///     runs out every engine is released (memory given back on purpose, before jetsam decides). Back within the
///     grace cancels it.
///   - **Never stuck cold.** A failed warm is retried twice with a backoff, then the engine is `cold` with
///     `retry` set and the error: the request fails with a clear message (the web shows Retry), and the next request
///     starts a fresh round. A run that dies mid-use releases the engine and restarts it (same bounded retries). On
///     return to foreground the watchdog asks every warm engine whether it still answers and relaunches the ones
///     that don't. No state is terminal.
///
/// Pure policy: the engine work is behind `EngineBackend`, time behind `LifecycleClock`, the hot flag behind
/// `HotStore`, so the tests run it with fakes (ios/Tests/LifecycleTests.swift, scripts/lifecycle-test.sh).
enum LifePhase: String, Sendable { case cold, warming, warm, releasing }

/// What the lifecycle drives. Calls for one engine never overlap.
@MainActor
protocol EngineBackend: AnyObject {
    /// Makes `id` ready to take a request, reporting progress in 0…1. Throws when it can't.
    func warm(_ id: EngineID, progress: @escaping @MainActor (Double) -> Void) async throws
    /// Gives `id`'s memory back. Leaves it cold whatever happens.
    func release(_ id: EngineID) async
    /// Whether a warm `id` still answers (the watchdog).
    func alive(_ id: EngineID) async -> Bool
}

@MainActor
protocol LifecycleTimer: AnyObject { func cancel() }

@MainActor
protocol LifecycleClock: AnyObject {
    var now: Date { get }
    func sleep(seconds: Double) async
    /// Runs `action` after `seconds` unless the timer is cancelled first.
    func schedule(after seconds: Double, _ action: @escaping @MainActor () -> Void) -> LifecycleTimer
}

/// When chat was last warm as the app left; kept across launches.
@MainActor
protocol HotStore: AnyObject { var hotAt: Date? { get set } }

@MainActor
final class EngineLifecycle {
    struct Policy {
        /// Away this long → release.
        var grace: Double = 90
        /// Retries after a failed warm (so 1 + maxRetries attempts), with these waits before each.
        var maxRetries = 2
        var backoff: [Double] = [0.5, 1.5]
        /// "Recently hot": chat was warm when the app left, at most this long ago.
        var recentWindow: Double = 24 * 3600
        /// Only one engine warm at a time (llama-server serves one model); warming one releases the other.
        var exclusive = false
        /// PHONEUX: launch warms chat whatever the hot flag says (the app's first screen shows it warming). Off:
        /// R14's rule, a cold launch loads nothing unless chat was warm when the app last left.
        var eagerLaunch = false
    }

    struct Slot {
        var phase = LifePhase.cold
        /// Of the warm in flight, 0…1.
        var progress = 0.0
        /// The warm attempt in flight or last made (1…1+maxRetries).
        var attempt = 0
        /// Set when the last round of warms failed: the next request retries.
        var retry = false
        var error: String?
        /// Requests running on the engine now: a release waits for them.
        var inUse = 0
        var warmSeconds: Double?
    }

    struct WarmFailed: LocalizedError {
        let id: EngineID
        let attempts: Int
        let why: String
        var errorDescription: String? {
            "The \(id.rawValue) engine didn't load (\(attempts) tries: \(why)). Retry to try again."
        }
    }

    let policy: Policy
    private let backend: EngineBackend
    private let clock: LifecycleClock
    private let store: HotStore
    private let log: (String) -> Void

    private(set) var slots: [EngineID: Slot] = Dictionary(uniqueKeysWithValues: EngineID.allCases.map { ($0, Slot()) })
    private(set) var foreground = true
    private var awayAt: Date?
    private var grace: LifecycleTimer?
    private var warming: [EngineID: Task<Void, Error>] = [:]
    private var releasing: [EngineID: Task<Void, Never>] = [:]
    /// Released once their last request ends (grace ran out mid-request).
    private var releaseAfterUse: Set<EngineID> = []
    /// Fire-and-forget work (eager warms, restarts, the grace release), so tests can wait for it: `settle()`.
    private var chores: [Task<Void, Never>] = []

    init(backend: EngineBackend, clock: LifecycleClock, store: HotStore, policy: Policy = Policy(), log: @escaping (String) -> Void = { _ in }) {
        self.backend = backend
        self.clock = clock
        self.store = store
        self.policy = policy
        self.log = log
    }

    func slot(_ id: EngineID) -> Slot { slots[id]! }
    func phase(_ id: EngineID) -> LifePhase { slots[id]!.phase }

    var recentlyHot: Bool {
        guard let at = store.hotAt else { return false }
        return clock.now.timeIntervalSince(at) <= policy.recentWindow
    }

    // MARK: needs

    /// Returns once `id` is warm, warming it first if needed. Throws `WarmFailed` after the bounded retries; the
    /// engine is then cold with `retry` set, and the next call starts over.
    func ensure(_ id: EngineID, why: String = "request") async throws {
        while true {
            switch slots[id]!.phase {
            case .warm:
                return
            case .releasing:
                await releasing[id]?.value
            case .warming:
                if let task = warming[id] { try await task.value; return }
                slots[id]!.phase = .cold // a warm task always exists while warming; never wait on nothing
            case .cold:
                try await startWarm(id, why: why).value
                return
            }
        }
    }

    /// `ensure`, but a warm engine is first asked whether it still answers (llama-server can die under the app):
    /// a dead one is released and warmed again, so the request never goes to an engine that is gone.
    func ensureAlive(_ id: EngineID, why: String = "request") async throws {
        if slots[id]!.phase == .warm, slots[id]!.inUse == 0, !(await backend.alive(id)) {
            log("lifecycle \(id.rawValue) says warm but does not answer: relaunching")
            await forceCold(id)
        }
        try await ensure(id, why: why)
    }

    /// A request is about to run on `id` (after `ensure`). Pair with `end`.
    func begin(_ id: EngineID) { slots[id]!.inUse += 1 }

    /// The request on `id` is over. `died`: the engine failed under it, so it is released and restarted.
    func end(_ id: EngineID, died: String? = nil) {
        slots[id]!.inUse = max(0, slots[id]!.inUse - 1)
        if let died {
            self.died(id, died)
        } else if slots[id]!.inUse == 0, releaseAfterUse.remove(id) != nil, !foreground {
            chore { await self.release(id, why: "grace over (after its last request)") }
        }
    }

    /// The engine is gone (killed, crashed, the surface stopped answering): release what's left, then (in
    /// foreground) warm it again with the bounded retries. If that fails too, it is cold with `retry`.
    func died(_ id: EngineID, _ why: String) {
        log("lifecycle \(id.rawValue) died: \(why)")
        slots[id]!.error = why
        // PHONEUX: the release starts now, not in the chore. A chore runs later; if a request (a Retry) warmed the
        // engine again before it ran, a late release would pull that fresh engine out from under the request.
        let dropping = slots[id]!.phase == .warm ? startDrop(id, why: "dead") : releasing[id]
        chore {
            await dropping?.value
            if self.foreground { try? await self.ensure(id, why: "restart after: \(why)") }
        }
    }

    /// Launch: chat warms eagerly if it was warm when the app last left, or always with `eagerLaunch`.
    func launched() {
        if recentlyHot || policy.eagerLaunch {
            let why = recentlyHot ? "eager (recently hot)" : "eager (launch)"
            log("lifecycle launch: \(recentlyHot ? "chat was warm at last exit" : "the first screen"), warming it")
            chore { try? await self.ensure(.chat, why: why) }
        } else {
            log("lifecycle launch: cold (chat warms on first need)")
        }
    }

    // MARK: away / back

    /// The app left the screen (background, lock, page hidden): release everything after the grace.
    func away() {
        guard foreground else { return }
        foreground = false
        awayAt = clock.now
        let hot = [.warm, .warming].contains(slots[.chat]!.phase)
        store.hotAt = hot ? clock.now : nil
        log(String(format: "lifecycle away: release in %.0f s unless back (chat %@)", policy.grace, slots[.chat]!.phase.rawValue))
        grace?.cancel()
        grace = clock.schedule(after: policy.grace) { [weak self] in self?.graceOver() }
    }

    /// The app is back on screen: cancel the grace, check every warm engine still answers (the watchdog), and
    /// warm chat if it was hot.
    func back() {
        guard !foreground else { return }
        foreground = true
        grace?.cancel()
        grace = nil
        let gone = awayAt.map { clock.now.timeIntervalSince($0) } ?? 0
        awayAt = nil
        releaseAfterUse.removeAll()
        log(String(format: "lifecycle back after %.0f s (%@)", gone, gone < policy.grace ? "within grace: kept warm" : "after the release"))
        chore {
            await self.watchdog()
            if self.slots[.chat]!.phase == .cold, self.recentlyHot { try? await self.ensure(.chat, why: "eager (recently hot)") }
        }
    }

    /// Still away and the OS is about to suspend the app (its background time ran out before the grace did):
    /// release now, so the memory goes back by our hand rather than jetsam's.
    func suspending() {
        guard !foreground else { return }
        grace?.cancel()
        grace = nil
        log("lifecycle suspending before the grace ran out: releasing now")
        chore {
            for id in EngineID.allCases { await self.release(id, why: "suspending") }
        }
    }

    /// Every warm, idle engine must still answer; the ones that don't are relaunched before the next send needs them.
    func watchdog() async {
        for id in EngineID.allCases where slots[id]!.phase == .warm && slots[id]!.inUse == 0 {
            if await backend.alive(id) { continue }
            log("lifecycle watchdog: \(id.rawValue) says warm but does not answer, relaunching")
            await forceCold(id)
            try? await ensure(id, why: "watchdog")
        }
    }

    private func graceOver() {
        grace = nil
        guard !foreground else { return }
        log(String(format: "lifecycle grace over (%.0f s away): releasing", policy.grace))
        chore {
            for id in EngineID.allCases { await self.release(id, why: "grace over") }
        }
    }

    // MARK: release

    /// warm → releasing → cold. Waits for a warm in flight; a request in flight defers it to that request's end.
    func release(_ id: EngineID, why: String) async {
        if let task = warming[id] { _ = try? await task.value }
        if let task = releasing[id] { await task.value; return }
        guard slots[id]!.phase == .warm else { return }
        if slots[id]!.inUse > 0 {
            releaseAfterUse.insert(id)
            log("lifecycle \(id.rawValue) release (\(why)) waits for its request")
            return
        }
        await drop(id, why: why)
    }

    /// Release now, whatever the phase (a dead engine): leaves it cold.
    private func forceCold(_ id: EngineID) async {
        if let task = warming[id] { _ = try? await task.value }
        if let task = releasing[id] { await task.value }
        guard slots[id]!.phase != .cold else { return }
        await drop(id, why: "dead")
    }

    private func drop(_ id: EngineID, why: String) async {
        await startDrop(id, why: why).value
    }

    /// warm → releasing now; cold once the backend has released.
    private func startDrop(_ id: EngineID, why: String) -> Task<Void, Never> {
        slots[id]!.phase = .releasing
        slots[id]!.progress = 0
        let task = Task { @MainActor in
            await self.backend.release(id)
            self.slots[id]!.phase = .cold
            self.releasing[id] = nil
            self.log("lifecycle \(id.rawValue) released (\(why)) → cold")
        }
        releasing[id] = task
        return task
    }

    // MARK: warm

    private func startWarm(_ id: EngineID, why: String) -> Task<Void, Error> {
        slots[id]!.phase = .warming
        slots[id]!.progress = 0
        let task = Task { @MainActor in
            defer { self.warming[id] = nil }
            if self.policy.exclusive {
                for other in EngineID.allCases where other != id {
                    await self.release(other, why: "\(id.rawValue) needs the memory")
                }
            }
            var last = ""
            let attempts = 1 + self.policy.maxRetries
            for attempt in 1...attempts {
                self.slots[id]!.phase = .warming
                self.slots[id]!.attempt = attempt
                self.slots[id]!.progress = 0
                self.log("lifecycle \(id.rawValue) cold → warming (\(why), attempt \(attempt)/\(attempts))")
                let t0 = self.clock.now
                do {
                    try await self.backend.warm(id) { p in
                        guard self.slots[id]!.phase == .warming else { return }
                        self.slots[id]!.progress = min(max(p, 0), 1)
                    }
                    let seconds = self.clock.now.timeIntervalSince(t0)
                    self.slots[id]!.phase = .warm
                    self.slots[id]!.progress = 1
                    self.slots[id]!.retry = false
                    self.slots[id]!.error = nil
                    self.slots[id]!.warmSeconds = seconds
                    if id == .chat, self.foreground { self.store.hotAt = self.clock.now }
                    self.log(String(format: "lifecycle %@ warming → warm in %.2f s", id.rawValue, seconds))
                    return
                } catch {
                    last = error.localizedDescription
                    self.log("lifecycle \(id.rawValue) warm attempt \(attempt)/\(attempts) failed: \(last)")
                    await self.backend.release(id) // whatever it got half way
                    if attempt < attempts {
                        await self.clock.sleep(seconds: self.policy.backoff[min(attempt - 1, self.policy.backoff.count - 1)])
                    }
                }
            }
            self.slots[id]!.phase = .cold
            self.slots[id]!.progress = 0
            self.slots[id]!.retry = true
            self.slots[id]!.error = last
            self.log("lifecycle \(id.rawValue) → cold, retry offered (the next request tries again)")
            throw WarmFailed(id: id, attempts: attempts, why: last)
        }
        warming[id] = task
        return task
    }

    // MARK: chores

    private func chore(_ body: @escaping @MainActor () async -> Void) {
        chores.append(Task { @MainActor in await body() })
    }

    /// Waits until no background work is left (tests; the Mac lifecycle check).
    func settle() async {
        while !chores.isEmpty {
            let now = chores
            chores.removeAll()
            for task in now { await task.value }
        }
    }
}

/// The real clock.
@MainActor
final class SystemClock: LifecycleClock {
    private final class Timer: LifecycleTimer {
        var task: Task<Void, Never>?
        func cancel() { task?.cancel() }
    }

    var now: Date { Date() }

    func sleep(seconds: Double) async { try? await Task.sleep(for: .seconds(seconds)) }

    func schedule(after seconds: Double, _ action: @escaping @MainActor () -> Void) -> LifecycleTimer {
        let timer = Timer()
        timer.task = Task { @MainActor in
            try? await Task.sleep(for: .seconds(seconds))
            if !Task.isCancelled { action() }
        }
        return timer
    }
}

/// The hot flag in UserDefaults (the app), or a file (the Mac harness: BEND_HOT_FILE, so it stays out of ~/Library).
@MainActor
final class DefaultsHotStore: HotStore {
    private let key = "bend.engine.chatHotAt"
    private let file = ProcessInfo.processInfo.environment["BEND_HOT_FILE"].map(URL.init(fileURLWithPath:))

    var hotAt: Date? {
        get {
            if let file {
                return (try? String(contentsOf: file, encoding: .utf8)).flatMap(Double.init).map(Date.init(timeIntervalSince1970:))
            }
            return UserDefaults.standard.object(forKey: key) as? Date
        }
        set {
            if let file {
                if let newValue { try? String(newValue.timeIntervalSince1970).write(to: file, atomically: true, encoding: .utf8) }
                else { try? FileManager.default.removeItem(at: file) }
                return
            }
            UserDefaults.standard.set(newValue, forKey: key)
        }
    }
}
