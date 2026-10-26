import CoreGraphics
import Foundation

// PHONECAM: the camera's logic without the engine or a camera: PixelGate, Canvas, CameraVerdict, CameraPipeline (fake
// engine, fake clock) and the synthetic source. Run: scripts/camera-test.sh (a Mac executable, like lifecycle-test.sh).
// The engine-backed checks (bits vs the references, timing, the clip replay) are the Mac harness's (scripts/cam-mac.sh).

nonisolated(unsafe) var checks = 0
nonisolated(unsafe) var failures: [String] = []
func check(_ ok: Bool, _ what: String, file: String = #file, line: Int = #line) {
    checks += 1
    if !ok { failures.append("\(what) (line \(line))") }
}

final class ManualClock: @unchecked Sendable {
    private let lock = NSLock()
    private var t = 1000.0
    func now() -> Double { lock.withLock { t } }
    func advance(_ s: Double) { lock.withLock { t += s } }
}

final class FakeEngine: VerdictEngine, @unchecked Sendable {
    let lock = NSLock()
    var inFlight = 0, maxInFlight = 0, calls = 0, ensures = 0
    var failNext = false, failEnsure = false
    var delay: useconds_t = 2000
    var questions: [String] = []
    func ensureReady() throws {
        lock.withLock { ensures += 1 }
        if lock.withLock({ failEnsure }) { throw CameraEngineError.down }
    }
    func classify(_ canvas: Canvas, question: String) throws -> CameraVerdict {
        lock.withLock { inFlight += 1; maxInFlight = max(maxInFlight, inFlight); calls += 1; questions.append(question) }
        usleep(delay)
        defer { lock.withLock { inFlight -= 1 } }
        if lock.withLock({ let f = failNext; failNext = false; return f }) { throw CameraEngineError.down }
        // a stand-in score: the canvas's mean brightness above or below mid gray
        let mean = Double(canvas.rgb.reduce(0) { $0 + Int($1) }) / Double(canvas.rgb.count)
        return CameraVerdict(score: Float(mean - 128) / 16, yesBits: 0, noBits: 0, firstID: 0, question: question)
    }
}
enum CameraEngineError: Error { case down }

final class Collector: @unchecked Sendable {
    let lock = NSLock()
    var verdicts: [(CameraVerdict, CameraPipeline.Origin)] = []
    var ready: [Bool] = []
}

func flat(_ v: UInt8, side: Int = CameraConfig.canvasSide) -> Canvas {
    Canvas(side: side, rgb: [UInt8](repeating: v, count: side * side * 3))
}

@main
struct CameraTests {
    @MainActor static func main() async {
        verdictMath()
        canvas()
        gateUnit()
        gateSynthetic()
        pipelineLive()
        pipelineTapAndQuestion()
        pipelineKeepWarm()
        pipelineFailure()
        pipelineHold()
        await captionCard()
        if failures.isEmpty {
            print("RESULT camera PASS (\(checks) checks)")
            exit(0)
        }
        failures.forEach { print("FAIL \($0)") }
        print("RESULT camera FAIL (\(failures.count) of \(checks) checks)")
        exit(1)
    }

    static func verdictMath() {
        let a = CameraVerdict(score: 0, yesBits: 0, noBits: 0, firstID: 0, question: "")
        check(!a.yes && abs(a.confidence - 0.5) < 1e-12, "score 0: NO at 50%")
        // vlive1/camprod's confidence: max(p, 1-p), p = sigmoid(score); b01's score on the 4B (receipts/battery-r1)
        let b = CameraVerdict(score: 3.0975189208984375, yesBits: 0, noBits: 0, firstID: 0, question: "")
        check(b.yes && abs(b.confidence - 0.95679028688594314) < 1e-12, "b01: YES at 95.68%")
        let c = CameraVerdict(score: -4.909227, yesBits: 0, noBits: 0, firstID: 0, question: "")
        check(!c.yes && c.confidence > 0.99, "b03: NO at >99%")
    }

    static func canvas() {
        let s = CameraConfig.canvasSide
        check(CameraConfig.imageTokens == 100, "canvas 320 → 100 image tokens")
        var rgb = [UInt8](repeating: 0, count: s * s * 3)
        for i in rgb.indices { rgb[i] = UInt8(truncatingIfNeeded: i &* 31) }
        let id = Canvas.letterbox(rgb: rgb, width: s, height: s)
        check(id.rgb == rgb, "an S×S frame goes in byte for byte (the fixture path)")
        let back = Canvas.ppm(id.ppmData())
        check(back?.rgb == rgb, "PPM round trip")
        check(Canvas.ppm(Data("P6\n2 2\n255\n".utf8) + Data([UInt8](repeating: 1, count: 12))) == nil, "a PPM of another size is refused")
        // a 640×480 white frame: letterboxed to 320×240 centered → black bands of 40 rows top and bottom
        let wide = Canvas.letterbox(rgb: [UInt8](repeating: 255, count: 640 * 480 * 3), width: 640, height: 480)
        func px(_ x: Int, _ y: Int) -> UInt8 { wide.rgb[(y * s + x) * 3] }
        check(px(160, 10) == 0 && px(160, 309) == 0, "letterbox: black bands top and bottom")
        check(px(160, 160) == 255 && px(5, 160) == 255 && px(314, 160) == 255, "letterbox: the frame fills the width")
        // a portrait 720×1280 frame: pillarboxed
        let tall = Canvas.letterbox(rgb: [UInt8](repeating: 200, count: 720 * 1280 * 3), width: 720, height: 1280)
        check(tall.rgb[(160 * s + 10) * 3] == 0 && tall.rgb[(160 * s + 160) * 3] == 200, "letterbox: portrait frames are pillarboxed")
    }

    static func gateUnit() {
        var g = PixelGate()
        let a = flat(50), b = flat(200)
        func step(_ c: Canvas, _ t: Double) -> PixelGate.Step { c.rgb.withUnsafeBufferPointer { g.step($0, width: c.side, height: c.side, tMs: t) } }
        check(step(a, 0).state == .moving, "the first frame opens a window (moving)")
        check(!step(a, 100).fire, "still 100 ms: not settled")
        check(step(a, 299).state == .static, "still 299 ms: not settled")
        check(step(a, 300).fire, "still 300 ms and new: fire")
        check(!step(a, 400).fire && !step(a, 2000).fire, "the same view never fires again")
        check(step(b, 2100).state == .moving, "a new view: moving")
        check(!step(b, 2300).fire && step(b, 2400).fire, "a new view, settled 300 ms: fire")
        // small: 0.299/0.587/0.114 gray, block means over the cropped grid
        var rgb = [UInt8](repeating: 0, count: 320 * 320 * 3)
        for p in 0..<(320 * 320) { rgb[p * 3] = 100; rgb[p * 3 + 1] = 50; rgb[p * 3 + 2] = 10 }
        let s = rgb.withUnsafeBufferPointer { g.small($0, width: 320, height: 320) }
        let want = Float(100) * 0.299 + Float(50) * 0.587 + Float(10) * 0.114
        check(s.count == 36 * 64 && s.allSatisfy { abs($0 - want) < 1e-4 }, "small(): 36×64 gray block means")
        // a drift whose every step is under the motion bar still counts as motion (the anchor rule)
        var d = PixelGate()
        var fired = false
        for i in 0..<40 {
            let c = flat(UInt8(60 + i * 3))
            if c.rgb.withUnsafeBufferPointer({ d.step($0, width: 320, height: 320, tMs: Double(i) * 66) }).fire && i > 5 { fired = true }
        }
        check(!fired, "a slow drift (3 levels a frame) never settles")
    }

    static func gateSynthetic() {
        let seq = SyntheticSequence(poses: SyntheticSequence.patterns(6))
        var g = PixelGate()
        var perPose: [Int: Int] = [:], outside = 0
        for i in 0..<seq.count {
            let f = seq.frame(i)
            let st = f.canvas.rgb.withUnsafeBufferPointer { g.step($0, width: 320, height: 320, tMs: f.tMs) }
            if st.fire { if let p = f.pose { perPose[p, default: 0] += 1 } else { outside += 1 } }
        }
        check((0..<6).allSatisfy { perPose[$0] == 1 }, "synthetic 6 poses: the gate fires once per pose (\(perPose))")
        check(outside == 0, "synthetic: no fire inside a transition")
        check(seq.frame(17).canvas.rgb == seq.frame(17).canvas.rgb, "the synthetic source is deterministic")
    }

    static func makePipe(_ e: FakeEngine, _ clock: ManualClock) -> (CameraPipeline, Collector) {
        let p = CameraPipeline(engine: e, clock: { clock.now() })
        let c = Collector()
        p.onVerdict = { v, o, _ in c.lock.withLock { c.verdicts.append((v, o)) } }
        p.onReady = { r in c.lock.withLock { c.ready.append((try? r.get()) != nil) } }
        return (p, c)
    }

    static func pipelineLive() {
        let e = FakeEngine(), clock = ManualClock()
        e.delay = 30_000  // 30 ms a request: fires land while one is in flight
        let (p, c) = makePipe(e, clock)
        p.prepare(); p.drain()
        check(c.ready == [true] && e.calls == 1, "prepare: ready, one discarded warm-up request")
        p.setLive(true)
        let seq = SyntheticSequence(poses: SyntheticSequence.patterns(6))
        for i in 0..<seq.count {
            let f = seq.frame(i)
            p.frame(f.canvas, tMs: f.tMs)  // as fast as it goes: the clock does not move, so no keep-warm
        }
        p.drain()
        let live = c.verdicts.filter { $0.1 == .live }
        check(live.count == 6, "live: 6 poses → 6 verdicts (got \(live.count))")
        check(e.maxInFlight == 1, "one request in flight at a time")
        let s = p.currentStats()
        check(s.fires == 6 && s.skipped == seq.count - 6 && s.warms == 1, "stats: fires 6, the rest skipped, the one prepare warm-up")
    }

    static func pipelineTapAndQuestion() {
        let e = FakeEngine(), clock = ManualClock()
        let (p, c) = makePipe(e, clock)
        p.prepare(); p.drain()
        let view = flat(90)
        p.check(view); p.drain()
        check(c.verdicts.last?.1 == .tap, "a tap is judged at once")
        // live after the tap: the same still view is already judged → no fire
        p.setLive(true)
        for i in 0..<20 { p.frame(view, tMs: Double(i) * 66) }
        p.drain()
        check(c.verdicts.filter { $0.1 == .live }.isEmpty, "a tapped view is marked judged: live does not re-fire on it")
        // a new question re-arms: the same still view fires again
        p.setQuestion("Is there a cat in this image?")
        for i in 20..<40 { p.frame(view, tMs: Double(i) * 66) }
        p.drain()
        check(c.verdicts.filter { $0.1 == .live }.count == 1, "a new question re-arms the gate (one fire)")
        check(e.questions.last == "Is there a cat in this image?", "the engine gets the new question")
    }

    static func pipelineKeepWarm() {
        let e = FakeEngine(), clock = ManualClock()
        let (p, c) = makePipe(e, clock)
        p.prepare(); p.drain()
        p.setLive(true)
        let view = flat(70)
        for i in 0..<10 { p.frame(view, tMs: Double(i) * 66) }  // fires once
        p.drain()
        let calls0 = e.calls
        clock.advance(0.5)
        p.frame(view, tMs: 700); p.drain()
        check(e.calls == calls0, "keep-warm: nothing before \(CameraConfig.keepWarmAfter) s idle")
        clock.advance(0.5)
        p.frame(view, tMs: 800); p.drain()
        check(e.calls == calls0 + 1 && p.currentStats().warms == 2, "keep-warm: a skipped frame after >0.8 s idle runs one discarded request")
        check(c.verdicts.filter { $0.1 == .warm }.isEmpty, "keep-warm verdicts are never shown")
        // tap mode: warms only while touched in the last minute
        p.setLive(false)
        clock.advance(1)
        p.frame(view, tMs: 900); p.drain()
        check(e.calls == calls0 + 2, "tap mode, touched just now: keep-warm runs")
        clock.advance(61)
        p.frame(view, tMs: 1000); p.drain()
        check(e.calls == calls0 + 2, "tap mode, untouched for a minute: no keep-warm")
    }

    static func pipelineFailure() {
        let e = FakeEngine(), clock = ManualClock()
        let (p, c) = makePipe(e, clock)
        p.prepare(); p.drain()
        e.failNext = true
        p.check(flat(10)); p.drain()
        check(p.currentStats().errors == 1 && p.currentStats().lastError != nil, "a failed request is counted, with why")
        let ensures = e.ensures
        p.check(flat(240)); p.drain()
        check(e.ensures == ensures + 1 && c.verdicts.last?.0.yes == true, "after a failure the next request re-ensures the engine, then judges")
        let (q, qc) = makePipe(FakeEngine(), clock)
        (q.engine as! FakeEngine).failEnsure = true
        q.prepare(); q.drain()
        check(qc.ready == [false], "an engine that can't come up reports not ready")
    }

    /// CAMCAP: while a caption runs the pipeline fires and warms nothing new; a tap still judges.
    static func pipelineHold() {
        let e = FakeEngine(), clock = ManualClock()
        let (p, c) = makePipe(e, clock)
        p.prepare(); p.drain()
        p.setLive(true)
        p.hold(true)
        let calls0 = e.calls
        clock.advance(2)
        for i in 0..<20 { p.frame(flat(UInt8(i % 2 == 0 ? 30 : 220)), tMs: Double(i) * 66) }
        p.drain()
        check(e.calls == calls0, "held (a caption is running): no fire, no warm-up reaches the engine")
        p.check(flat(240)); p.drain()
        check(c.verdicts.last?.1 == .tap, "held: the tap still judges")
        p.hold(false)
        clock.advance(2)
        p.frame(flat(100), tMs: 5000); p.drain()
        check(e.calls > calls0 + 1, "released: keep-warm runs again")
    }

    @MainActor static func until(_ what: String, _ cond: @MainActor () -> Bool) async {
        for _ in 0..<400 where !cond() { try? await Task.sleep(for: .milliseconds(5)) }
        check(cond(), "waited for: \(what)")
    }

    /// CAMCAP: the caption card's controller: loading → streaming → done, stop, replace, every failure with words.
    @MainActor static func captionCard() async {
        let img = flat(120).cgImage()!
        let cap = FakeCaptioner(pieces: ["A yellow ", "dump truck ", "on a road."])
        let c = CaptionController(captioner: cap)
        var busy: [Bool] = [], done: [String] = []
        c.onBusy = { busy.append($0) }
        c.onDone = { done.append($0) }
        check(c.state == .idle && c.text == nil && !c.isBusy, "idle at first")
        c.ask(img)
        check(c.state == .loading && c.isBusy, "asked: loading at once, before the engine says anything")
        await until("done") { c.text != nil }
        check(c.state == .done("A yellow dump truck on a road.", ms: 12), "done: the pieces joined, with the engine's ms")
        check(busy == [true, false] && done == ["A yellow dump truck on a road."], "busy true then false; onDone once with the text")
        check(cap.seen == 1 && !c.isBusy, "one engine call")

        // streaming: the card grows, never shortens
        let slow = FakeCaptioner(pieces: ["One ", "two ", "three"], pieceDelayMs: 30)
        let s = CaptionController(captioner: slow)
        s.ask(img)
        var grew: [Int] = []
        for _ in 0..<200 where s.text == nil {
            if case .streaming(let t) = s.state { grew.append(t.count) }
            try? await Task.sleep(for: .milliseconds(5))
        }
        check(!grew.isEmpty && grew == grew.sorted(), "streaming: the text only grows (\(grew.count) samples)")
        check(s.text == "One two three", "streamed caption completes")

        // stop mid-run: back to idle, the engine is told to stop, the late result is dropped
        let stopper = FakeCaptioner(pieces: Array(repeating: "x ", count: 40), pieceDelayMs: 20)
        let t = CaptionController(captioner: stopper)
        var tBusy: [Bool] = []
        t.onBusy = { tBusy.append($0) }
        t.ask(img)
        await until("streaming") { if case .streaming = t.state { return true } else { return false } }
        t.cancel()
        check(t.state == .idle && tBusy == [true, false], "cancel: idle, busy released")
        await until("engine stopped") { stopper.stoppedEarly }
        try? await Task.sleep(for: .milliseconds(80))
        check(t.state == .idle && t.text == nil, "a cancelled run's late result never lands on the card")

        // a new ask replaces the running one
        let r = FakeCaptioner(pieces: ["first ", "run ", "text ", "here ", "now"], pieceDelayMs: 25)
        let u = CaptionController(captioner: r)
        u.ask(img)
        await until("first streaming") { if case .streaming = u.state { return true } else { return false } }
        r.pieces = ["second"]; r.pieceDelayMs = 0
        u.ask(img)
        await until("second done") { u.text != nil }
        try? await Task.sleep(for: .milliseconds(150))
        check(u.text == "second", "the newer ask wins; the older run's words never mix in (got \(u.text ?? "nil"))")

        // failures: words, and the next ask works
        let f = FakeCaptioner(pieces: ["ok"])
        let v = CaptionController(captioner: f)
        f.failure = "vision: out of memory"
        v.ask(img)
        await until("failed") { if case .failed = v.state { return true } else { return false } }
        check(v.state == .failed("vision: out of memory") && !v.isBusy, "an engine error is the card's words, with Retry")
        f.failure = nil
        v.ask(img)
        await until("retry done") { v.text != nil }
        check(v.text == "ok", "Retry after a failure captions")
        let e = FakeCaptioner(pieces: ["  \n"])
        let w = CaptionController(captioner: e)
        w.ask(img)
        await until("empty") { if case .failed = w.state { return true } else { return false } }
        check(w.text == nil, "an empty caption is a failure with words, never a blank card")
        let n = FakeCaptioner(pieces: ["x"])
        n.unavailable = "Simulator: our engine needs the device's Metal"
        let x = CaptionController(captioner: n)
        x.ask(img)
        check(x.state == .unavailable("Simulator: our engine needs the device's Metal") && n.seen == 0, "an unavailable build says why at once, no engine call")
        let m = FakeCaptioner(pieces: ["x"])
        m.failure = "no-models: the vision tower isn't on this phone"
        let y = CaptionController(captioner: m)
        y.ask(img)
        await until("no-models") { if case .unavailable = y.state { return true } else { return false } }
        check(y.isBusy == false, "a models-missing error reads as unavailable")
        let none = CaptionController(captioner: nil)
        none.ask(img)
        check({ if case .unavailable = none.state { return true } else { return false } }(), "no captioner: unavailable")
        let z = CaptionController(captioner: cap)
        z.expect()
        check(z.state == .loading, "expect(): loading while the frame is fetched")
        z.fail("No frame to describe.")
        check(z.state == .failed("No frame to describe."), "fail(): words")
    }
}

final class FakeCaptioner: Captioner, @unchecked Sendable {
    private let lock = NSLock()
    var pieces: [String]
    var pieceDelayMs: UInt32
    var failure: String?
    var unavailable: String?
    private(set) var seen = 0
    private(set) var stoppedEarly = false
    init(pieces: [String], pieceDelayMs: UInt32 = 0) { self.pieces = pieces; self.pieceDelayMs = pieceDelayMs }
    var unavailableReason: String? { lock.withLock { unavailable } }
    func caption(_ image: CGImage, onPiece: @escaping @Sendable (String) -> Bool) async throws -> CaptionResult {
        let (list, delay, fail) = lock.withLock { seen += 1; return (pieces, pieceDelayMs, failure) }
        if let fail { throw CaptionFailure(fail) }
        var text = ""
        for p in list {
            if delay > 0 { try? await Task.sleep(for: .milliseconds(Int(delay))) }
            text += p
            if !onPiece(p) { lock.withLock { stoppedEarly = true }; return CaptionResult(text: text, firstMs: 1, totalMs: 12, finish: "aborted") }
        }
        return CaptionResult(text: text, firstMs: 1, totalMs: 12, finish: "stop")
    }
}
struct CaptionFailure: LocalizedError {
    let errorDescription: String?
    init(_ m: String) { errorDescription = m }
}
