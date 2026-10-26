import Foundation

// The lifecycle state machine (ios/Bend/Engines/EngineLifecycle.swift) under a fake clock and a fake engine.
// Run: scripts/lifecycle-test.sh (compiles this file with EngineLifecycle.swift into a Mac executable).

/// The app's two engines (each target's EngineManager.swift defines the same cases).
enum EngineID: String, CaseIterable, Sendable { case chat, decision }

// MARK: fakes

@MainActor
final class FakeClock: LifecycleClock {
    final class Timer: LifecycleTimer {
        let at: Date
        let action: @MainActor () -> Void
        var cancelled = false
        init(at: Date, action: @escaping @MainActor () -> Void) { self.at = at; self.action = action }
        func cancel() { cancelled = true }
    }

    var now = Date(timeIntervalSince1970: 1_000_000)
    var slept: [Double] = []
    private var timers: [Timer] = []

    func sleep(seconds: Double) async {
        slept.append(seconds)
        now += seconds
    }

    func schedule(after seconds: Double, _ action: @escaping @MainActor () -> Void) -> LifecycleTimer {
        let timer = Timer(at: now + seconds, action: action)
        timers.append(timer)
        return timer
    }

    /// Moves time on, firing the timers that come due.
    func advance(_ seconds: Double) {
        now += seconds
        let due = timers.filter { !$0.cancelled && $0.at <= now }
        timers.removeAll { $0.cancelled || $0.at <= now }
        due.forEach { $0.action() }
    }
}

@MainActor
final class MemoryHotStore: HotStore { var hotAt: Date? }

struct FakeError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

/// Models the library as R13 met it and as engine-repeat fixed it: a run "spends" the engine; a spent engine can
/// only be warmed again after a release. `reopens == false` is R13's library (release never re-arms it).
@MainActor
final class FakeEngine: EngineBackend {
    var reopens = true
    /// The next N warms fail.
    var failWarms = 0
    var warms: [EngineID] = []
    var releases: [EngineID] = []
    var progressSeen: [Double] = []
    private(set) var resident: Set<EngineID> = []
    private(set) var spent: Set<EngineID> = []
    /// Killed under the app: still "warm" to the lifecycle, but it no longer answers.
    var killed: Set<EngineID> = []

    func warm(_ id: EngineID, progress: @escaping @MainActor (Double) -> Void) async throws {
        warms.append(id)
        progress(0.25)
        progressSeen.append(0.25)
        if failWarms > 0 {
            failWarms -= 1
            throw FakeError("fake: warm refused")
        }
        if spent.contains(id) { throw FakeError("R1: one run per process") }
        progress(0.62)
        progressSeen.append(0.62)
        resident.insert(id)
        killed.remove(id)
    }

    func release(_ id: EngineID) async {
        releases.append(id)
        resident.remove(id)
        if reopens { spent.remove(id) }
    }

    func alive(_ id: EngineID) async -> Bool { resident.contains(id) && !killed.contains(id) }

    /// One request: needs the engine resident and not spent; leaves it spent (the R13 library).
    func run(_ id: EngineID) throws -> String {
        guard resident.contains(id), !killed.contains(id) else { throw FakeError("fake: \(id) not loaded") }
        guard !spent.contains(id) else { throw FakeError("R1: one run per process") }
        spent.insert(id)
        return "answer from \(id.rawValue)"
    }
}

// MARK: harness

@MainActor
var failures = 0
@MainActor
var checks = 0

@MainActor
func expect(_ ok: Bool, _ what: String, file: String = #file, line: Int = #line) {
    checks += 1
    if !ok {
        failures += 1
        print("  FAIL \(what) (line \(line))")
    }
}

@MainActor
func make(_ policy: EngineLifecycle.Policy = .init()) -> (EngineLifecycle, FakeEngine, FakeClock, MemoryHotStore, LogBox) {
    let engine = FakeEngine()
    let clock = FakeClock()
    let store = MemoryHotStore()
    let logs = LogBox()
    let life = EngineLifecycle(backend: engine, clock: clock, store: store, policy: policy) { logs.lines.append($0) }
    return (life, engine, clock, store, logs)
}

final class LogBox: @unchecked Sendable { var lines: [String] = [] }

@MainActor
func test(_ name: String, _ body: () async throws -> Void) async {
    let before = failures
    do { try await body() } catch {
        failures += 1
        print("  FAIL threw \(error)")
    }
    print("\(failures == before ? "PASS" : "FAIL") \(name)")
}

/// Ensure + run + end, the way a route does it.
@MainActor
func request(_ life: EngineLifecycle, _ engine: FakeEngine, _ id: EngineID) async throws -> String {
    try await life.ensure(id)
    life.begin(id)
    do {
        let answer = try engine.run(id)
        life.end(id)
        return answer
    } catch {
        life.end(id, died: error.localizedDescription)
        throw error
    }
}

// MARK: tests

@MainActor
func run() async {
    await test("cold → warming → warm → answer (warm on first need, with progress)") {
        let (life, engine, _, _, _) = make()
        expect(life.phase(.chat) == .cold, "starts cold")
        let answer = try await request(life, engine, .chat)
        expect(answer == "answer from chat", "answered")
        expect(life.phase(.chat) == .warm, "warm after")
        expect(engine.warms == [.chat], "one warm")
        expect(engine.progressSeen == [0.25, 0.62], "progress reported")
        expect(life.slot(.chat).progress == 1, "progress 100% once warm")
        expect(life.phase(.decision) == .cold, "decision untouched")
    }

    await test("warm → away → grace → released (both engines), not before") {
        let (life, engine, clock, store, logs) = make()
        try await life.ensure(.chat)
        try await life.ensure(.decision)
        life.away()
        expect(store.hotAt != nil, "chat warm at exit → hot")
        clock.advance(89)
        await life.settle()
        expect(life.phase(.chat) == .warm && life.phase(.decision) == .warm, "still warm at 89 s")
        clock.advance(2)
        await life.settle()
        expect(life.phase(.chat) == .cold && life.phase(.decision) == .cold, "cold after the grace")
        expect(Set(engine.releases) == [.chat, .decision], "both released")
        expect(logs.lines.contains { $0.contains("grace over") }, "logged the release")
    }

    await test("back within the grace → stays warm, no release, no re-warm") {
        let (life, engine, clock, _, _) = make()
        try await life.ensure(.chat)
        life.away()
        clock.advance(60)
        life.back()
        await life.settle()
        clock.advance(120)
        await life.settle()
        expect(life.phase(.chat) == .warm, "warm")
        expect(engine.releases.isEmpty, "nothing released")
        expect(engine.warms == [.chat], "no second warm")
    }

    await test("eager warm only if recently hot (launch and return)") {
        let (life, engine, _, store, _) = make()
        life.launched()
        await life.settle()
        expect(engine.warms.isEmpty && life.phase(.chat) == .cold, "cold launch: nothing loaded")
        store.hotAt = Date(timeIntervalSince1970: 1_000_000 - 3600)
        life.launched()
        await life.settle()
        expect(life.phase(.chat) == .warm, "hot an hour ago: chat warmed at launch")
        expect(life.phase(.decision) == .cold, "decision never eager")

        let (stale, staleEngine, _, staleStore, _) = make()
        staleStore.hotAt = Date(timeIntervalSince1970: 1_000_000 - 3 * 24 * 3600)
        stale.launched()
        await stale.settle()
        expect(staleEngine.warms.isEmpty, "hot three days ago: not recent, stays cold")
    }

    await test("PHONEUX eagerLaunch: launch warms chat even when not recently hot; a failed launch warm ends in retry, not a hang") {
        var policy = EngineLifecycle.Policy()
        policy.eagerLaunch = true
        let (life, engine, _, _, _) = make(policy)
        life.launched()
        await life.settle()
        expect(engine.warms == [.chat] && life.phase(.chat) == .warm, "launch: chat warmed")
        expect(life.phase(.decision) == .cold, "decision never eager")

        let (bad, badEngine, _, _, _) = make(policy)
        badEngine.failWarms = 3
        bad.launched()
        await bad.settle()
        expect(badEngine.warms.count == 3, "bounded: 3 attempts")
        expect(bad.phase(.chat) == .cold && bad.slot(.chat).retry, "cold + retry offered, not stuck warming")
        _ = try await request(bad, badEngine, .chat)
        expect(bad.phase(.chat) == .warm, "the next request warms it")
    }

    await test("away → released → back → chat warms again (it was hot); cold at exit → stays cold") {
        let (life, engine, clock, _, _) = make()
        try await life.ensure(.chat)
        life.away()
        clock.advance(91)
        await life.settle()
        expect(life.phase(.chat) == .cold, "released")
        life.back()
        await life.settle()
        expect(life.phase(.chat) == .warm, "re-warmed on return")
        expect(engine.warms == [.chat, .chat], "two warms")

        let (cold, coldEngine, coldClock, coldStore, _) = make()
        cold.away()
        expect(coldStore.hotAt == nil, "cold at exit → not hot")
        coldClock.advance(91)
        cold.back()
        await cold.settle()
        expect(coldEngine.warms.isEmpty, "nothing eager")
    }

    await test("spent → released → the next request re-opens (succeeds); without a reopening library it can't") {
        let (life, engine, clock, _, _) = make()
        _ = try await request(life, engine, .chat)
        expect(engine.spent.contains(.chat), "the run spent it")
        life.away()
        clock.advance(91)
        await life.settle()
        expect(life.phase(.chat) == .cold && !engine.spent.contains(.chat), "released, re-armed")
        life.back()
        await life.settle()
        let again = try await request(life, engine, .chat)
        expect(again == "answer from chat", "second request answered after the release")

        // R13's library: a release does not re-arm, so re-warming fails ×3 and the lifecycle offers Retry
        // instead of hanging (the wall that blocked R13, now only in this fake).
        let (old, oldEngine, oldClock, _, _) = make()
        oldEngine.reopens = false
        _ = try await request(old, oldEngine, .chat)
        old.away()
        oldClock.advance(91)
        await old.settle()
        do {
            _ = try await request(old, oldEngine, .chat)
            expect(false, "R13 library should refuse")
        } catch {
            expect(old.slot(.chat).retry && old.phase(.chat) == .cold, "cold + retry, not stuck warming")
        }
    }

    await test("warm fails → retried ×2 with backoff → then cold + Retry; the next request tries again") {
        let (life, engine, clock, _, _) = make()
        engine.failWarms = 3
        do {
            try await life.ensure(.chat)
            expect(false, "should fail after 3 attempts")
        } catch let error as EngineLifecycle.WarmFailed {
            expect(error.attempts == 3, "3 attempts")
            expect(error.localizedDescription.contains("Retry"), "says Retry")
        }
        expect(engine.warms.count == 3, "1 + 2 retries")
        expect(clock.slept == [0.5, 1.5], "backoff 0.5 s, 1.5 s")
        expect(life.phase(.chat) == .cold && life.slot(.chat).retry, "cold with retry offered")
        expect(life.slot(.chat).error == "fake: warm refused", "the reason kept")
        // The next send: a fresh round, which works now.
        let answer = try await request(life, engine, .chat)
        expect(answer == "answer from chat", "Retry works")
        expect(!life.slot(.chat).retry && life.slot(.chat).error == nil, "retry flag cleared")
    }

    await test("one failed attempt → auto-retried, the request never sees it") {
        let (life, engine, _, _, _) = make()
        engine.failWarms = 1
        let answer = try await request(life, engine, .chat)
        expect(answer == "answer from chat", "answered")
        expect(engine.warms.count == 2, "warmed on the 2nd attempt")
    }

    await test("killed mid-use → released + restarted (bounded ×2 retries) → Retry → next request works") {
        let (life, engine, clock, _, _) = make()
        try await life.ensure(.chat)
        life.begin(.chat)
        engine.killed.insert(.chat)
        engine.failWarms = 3 // the restart fails too
        life.end(.chat, died: "fake: killed")
        await life.settle()
        expect(engine.releases.contains(.chat), "released what was left")
        expect(engine.warms.count == 1 + 3, "restart: 3 attempts")
        expect(clock.slept == [0.5, 1.5], "with backoff")
        expect(life.phase(.chat) == .cold && life.slot(.chat).retry, "cold + Retry, never offline")
        let answer = try await request(life, engine, .chat)
        expect(answer == "answer from chat", "the next send works")
    }

    await test("killed mid-use, restart works → warm again without the user noticing") {
        let (life, engine, _, _, _) = make()
        try await life.ensure(.chat)
        life.begin(.chat)
        engine.killed.insert(.chat)
        life.end(.chat, died: "fake: killed")
        await life.settle()
        expect(life.phase(.chat) == .warm && !life.slot(.chat).retry, "restarted")
    }

    await test("watchdog on return: warm but not answering → relaunched before the next send") {
        let (life, engine, clock, _, logs) = make()
        try await life.ensure(.chat)
        life.away()
        clock.advance(30)
        engine.killed.insert(.chat) // e.g. iOS tore the listener down while suspended
        life.back()
        await life.settle()
        expect(logs.lines.contains { $0.contains("watchdog: chat") }, "watchdog noticed")
        expect(engine.warms == [.chat, .chat], "relaunched")
        expect(life.phase(.chat) == .warm, "warm")
        expect(try engine.run(.chat) == "answer from chat", "answers")
    }

    await test("suspended before the grace ends (iOS background time ran out) → released now; in foreground ignored") {
        let (life, engine, clock, _, _) = make()
        try await life.ensure(.chat)
        life.suspending()
        await life.settle()
        expect(life.phase(.chat) == .warm, "foreground: nothing to do")
        life.away()
        clock.advance(25)
        life.suspending()
        await life.settle()
        expect(life.phase(.chat) == .cold && engine.releases == [.chat], "released at 25 s")
        clock.advance(100)
        await life.settle()
        expect(engine.releases == [.chat], "the cancelled grace does not fire again")
    }

    await test("ensureAlive: a warm engine that died (llama-server gone) is relaunched before the request") {
        let (life, engine, _, _, _) = make()
        try await life.ensure(.chat)
        engine.killed.insert(.chat)
        try await life.ensureAlive(.chat)
        expect(engine.releases == [.chat] && engine.warms == [.chat, .chat], "released and warmed again")
        expect(try engine.run(.chat) == "answer from chat", "answers")
        try await life.ensureAlive(.chat)
        expect(engine.warms.count == 2, "a live warm engine is left alone")
    }

    await test("grace runs out mid-request → released when the request ends") {
        let (life, engine, clock, _, _) = make()
        try await life.ensure(.chat)
        life.begin(.chat)
        life.away()
        clock.advance(91)
        await life.settle()
        expect(life.phase(.chat) == .warm && engine.releases.isEmpty, "not released under a running request")
        life.end(.chat)
        await life.settle()
        expect(life.phase(.chat) == .cold, "released after it")
    }

    await test("requests during warming share the one warm; during releasing wait and re-warm") {
        let (life, engine, _, _, _) = make()
        async let a: Void = life.ensure(.chat)
        async let b: Void = life.ensure(.chat)
        _ = try await (a, b)
        expect(engine.warms == [.chat], "one warm for two waiters")
        // PHONEUX: the release must be under way before the ensure asks (two async lets start in either order: an ensure
        // that ran first found chat warm and returned, then the release left it cold, a flake of ~1 in 200 under load).
        let r = Task { await life.release(.chat, why: "test") }
        for _ in 0..<1000 where life.phase(.chat) == .warm { await Task.yield() }
        try await life.ensure(.chat)
        await r.value
        expect(life.phase(.chat) == .warm, "warm after release + ensure")
    }

    await test("exclusive (llama-server, one model): warming one releases the other") {
        var policy = EngineLifecycle.Policy()
        policy.exclusive = true
        let (life, engine, _, _, _) = make(policy)
        try await life.ensure(.chat)
        try await life.ensure(.decision)
        expect(life.phase(.chat) == .cold && life.phase(.decision) == .warm, "swapped")
        expect(engine.releases == [.chat], "chat released for decision")
        _ = try await request(life, engine, .chat)
        expect(life.phase(.decision) == .cold && life.phase(.chat) == .warm, "and back")
    }

    await test("PHONEUX: an engine that died is released at once; a request warmed after it is never pulled by the late restart") {
        let (life, engine, _, _, _) = make()
        try await life.ensure(.chat)
        life.begin(.chat)
        life.end(.chat, died: "fake: died")
        expect(life.phase(.chat) == .releasing, "released at once, not in a later chore")
        _ = try await request(life, engine, .chat)   // the Retry: waits for the release, warms, answers
        await life.settle()                           // the restart chore: nothing to undo
        expect(life.phase(.chat) == .warm, "still warm after the restart chore")
        expect(engine.releases == [.chat], "one release (the death's), none from the late chore")
    }

    await test("no terminal state: from any state a random walk reaches, a healthy engine answers the next request") {
        var seed: UInt64 = 0x5EED_B3ED
        func next(_ n: Int) -> Int {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Int((seed >> 33) % UInt64(n))
        }
        let (life, engine, clock, _, _) = make()
        var stuck = 0
        for step in 0..<400 {
            let id: EngineID = next(2) == 0 ? .chat : .decision
            switch next(8) {
            case 0: _ = try? await request(life, engine, id)
            case 1: life.away()
            case 2: life.back()
            case 3: clock.advance(Double(next(120)))
            case 4: engine.failWarms = next(4)
            case 5: engine.killed.insert(id)
            case 6: life.begin(id); life.end(id, died: next(2) == 0 ? "fake: died" : nil)
            default: await life.release(id, why: "walk")
            }
            await life.settle()
            expect(LifePhase(rawValue: life.phase(id).rawValue) != nil, "a known phase")
            // Heal the engine, then the very next request must answer (a Retry is just that next request).
            if step % 10 == 9 {
                engine.failWarms = 0
                engine.killed.removeAll()
                life.back()
                await life.settle()
                do { _ = try await request(life, engine, id) } catch {
                    // A spent engine (answered, still warm) is the library's to re-arm: in Bend each run re-loads;
                    // here a release does it. One more request after a release must answer.
                    await life.release(id, why: "re-arm")
                    do { _ = try await request(life, engine, id) } catch { stuck += 1 }
                }
            }
        }
        expect(stuck == 0, "never stuck (\(stuck) stuck)")
    }

    print(failures == 0 ? "RESULT lifecycle PASS (\(checks) checks)" : "RESULT lifecycle FAIL (\(failures)/\(checks) checks failed)")
}

@main
struct LifecycleTests {
    static func main() async {
        await run()
        exit(await MainActor.run { failures } == 0 ? 0 : 1)
    }
}
