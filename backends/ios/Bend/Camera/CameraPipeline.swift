import Foundation

/// The camera's loop, UI-free (the surface, the Mac harness and the tests drive the same object):
///   frame(canvas, t) → the pixel gate (live mode) → fire → the engine (one request in flight; a fire that lands while
///   it is busy waits, the latest one wins) → verdict. check(canvas) is the tap: judged now, and its view marked
///   judged in the gate. Keep-warm (camprod): while live (or touched in the last minute), a frame the gate skips runs
///   one discarded request when the engine has idled longer than CameraConfig.keepWarmAfter.
/// Callbacks arrive on the pipeline's queues; the surface hops to the main actor.
final class CameraPipeline: @unchecked Sendable {
    enum Origin: String, Sendable { case tap, live, warm }

    struct Stats: Sendable, Equatable {
        var frames = 0, fires = 0, skipped = 0, warms = 0, verdicts = 0, errors = 0
        var gateState: PixelGate.State = .moving
        var engineBusy = false
        /// Engine time over the last 10 s / 10 s (warm-ups included).
        var duty: Double = 0
        var lastError: String?
    }

    let engine: VerdictEngine
    var onVerdict: (@Sendable (CameraVerdict, Origin, Canvas) -> Void)?
    var onStats: (@Sendable (Stats) -> Void)?
    var onReady: (@Sendable (Result<Void, Error>) -> Void)?

    private let lock = NSLock()
    private let engineQ = DispatchQueue(label: "bend.camera.engine", qos: .userInitiated)
    private var gate = PixelGate()
    private var stats = Stats()
    private var live = false
    private var question = CameraConfig.defaultQuestion
    private var ready = false
    private var busy = false
    /// CAMCAP: a caption is running on the engine. Frames still feed the gate's clock, but nothing new fires or warms
    /// (a fire would queue between the look's ids); the tap's `check` still runs, between the look's ids.
    private var held = false
    private var pending: (Canvas, Origin, Double)?
    private var lastEngineEnd = CFAbsoluteTimeGetCurrent()
    private var lastTouch = CFAbsoluteTimeGetCurrent()
    private var busyLog: [(end: Double, ms: Double)] = []
    private let clock: @Sendable () -> Double

    init(engine: VerdictEngine, clock: @escaping @Sendable () -> Double = { CFAbsoluteTimeGetCurrent() }) {
        self.engine = engine
        self.clock = clock
    }

    var isLive: Bool { lock.withLock { live } }
    func currentStats() -> Stats { lock.withLock { stats } }

    /// Brings the engine up (model resident, tower loaded) off the main thread; onReady says how it went. Then one
    /// discarded request on a gray canvas: a session's first job pays a one-time set-up (Mac: ~356 vs ~150 ms,
    /// receipts/mac-pipeline r1), so the first real fire doesn't.
    func prepare() {
        engineQ.async { [self] in
            let r = Result {
                try engine.ensureReady()
                let side = CameraConfig.canvasSide
                _ = try engine.classify(Canvas(side: side, rgb: [UInt8](repeating: 128, count: side * side * 3)),
                                        question: lock.withLock { question })
                lock.withLock { lastEngineEnd = clock(); stats.warms += 1 }
            }
            lock.withLock { ready = (try? r.get()) != nil; if case .failure(let e) = r { stats.lastError = e.localizedDescription } }
            onReady?(r)
            publish()
        }
    }

    func setLive(_ on: Bool) {
        lock.withLock { live = on; lastTouch = clock() }
    }

    func setQuestion(_ q: String) {
        lock.withLock { question = q; gate.rearm(); lastTouch = clock() }
    }

    func touched() { lock.withLock { lastTouch = clock() } }

    /// A view the person froze without asking for a verdict (Describe): the live gate won't fire on it either.
    func markJudged(_ canvas: Canvas) {
        lock.withLock {
            lastTouch = clock()
            canvas.rgb.withUnsafeBufferPointer { gate.markJudged($0, width: canvas.side, height: canvas.side) }
        }
    }

    func hold(_ on: Bool) { lock.withLock { held = on; lastTouch = clock() } }

    /// One frame (capture timestamp tMs). Gate work is on the caller's queue (the capture queue); ~1 ms.
    func frame(_ canvas: Canvas, tMs: Double) {
        let t0 = clock()
        var run: (Canvas, Origin)?
        lock.withLock {
            stats.frames += 1
            if held { return }
            guard live else {
                if ready && !busy && t0 - lastTouch < CameraConfig.keepWarmTouchWindow
                    && t0 - lastEngineEnd > CameraConfig.keepWarmAfter {
                    run = (canvas, .warm)
                    busy = true
                }
                return
            }
            let step = canvas.rgb.withUnsafeBufferPointer { gate.step($0, width: canvas.side, height: canvas.side, tMs: tMs) }
            stats.gateState = step.state
            if step.fire {
                stats.fires += 1
                if busy { pending = (canvas, .live, t0) } else { run = (canvas, .live) }
            } else {
                stats.skipped += 1
                if ready && !busy && pending == nil && t0 - lastEngineEnd > CameraConfig.keepWarmAfter {
                    run = (canvas, .warm)
                }
            }
            if run != nil { busy = true }
        }
        if let (c, o) = run { submit(c, o, t0) }
        publish()
    }

    /// The tap: this frame, judged now (after a request in flight, if any); the gate won't re-fire on this view.
    func check(_ canvas: Canvas) {
        let t0 = clock()
        var now = false
        lock.withLock {
            lastTouch = t0
            canvas.rgb.withUnsafeBufferPointer { gate.markJudged($0, width: canvas.side, height: canvas.side) }
            if busy { pending = (canvas, .tap, t0) } else { busy = true; now = true }
        }
        if now { submit(canvas, .tap, t0) }
        publish()
    }

    private func submit(_ canvas: Canvas, _ origin: Origin, _ t0: Double) {
        lock.withLock { stats.engineBusy = true }
        engineQ.async { [self] in
            let q = lock.withLock { question }
            let e0 = clock()
            let result = Result { () throws -> CameraVerdict in
                if !lock.withLock({ ready }) {
                    try engine.ensureReady()
                    lock.withLock { ready = true; stats.lastError = nil }
                }
                return try engine.classify(canvas, question: q)
            }
            let end = clock()
            var next: (Canvas, Origin, Double)?
            lock.withLock {
                lastEngineEnd = end
                busyLog.append((end, (end - e0) * 1000))
                busyLog.removeAll { end - $0.end > 10 }
                stats.duty = busyLog.reduce(0) { $0 + $1.ms } / 10_000
                switch result {
                case .success: if origin == .warm { stats.warms += 1 } else { stats.verdicts += 1 }
                case .failure(let e):
                    stats.errors += 1
                    stats.lastError = e.localizedDescription
                    ready = false  // the next request re-ensures (e.g. the lifecycle released the model)
                }
                next = pending
                pending = nil
                busy = next != nil
                stats.engineBusy = busy
            }
            if case .success(var v) = result, origin != .warm {
                v.frameMs = (end - t0) * 1000
                onVerdict?(v, origin, canvas)
            }
            publish()
            if let (c, o, t) = next { submit(c, o, t) }
        }
    }

    private func publish() {
        let s = lock.withLock { stats }
        onStats?(s)
    }

    /// Waits until no request is in flight or pending (tests and the Mac harness).
    func drain() {
        while true {
            engineQ.sync {}
            if !lock.withLock({ busy || pending != nil }) { return }
            usleep(1000)
        }
    }
}
