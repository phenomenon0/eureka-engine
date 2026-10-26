import AVFoundation
import Foundation

/// VOICE-R2: the hands-free line. One microphone (or the synthetic one) stays open for the whole conversation; the
/// `Endpointer` watches it. In order:
///
///   armed     waiting for the person: speech starts a turn (the last ~600 ms of audio goes to the recognizer first, so the
///             first word is whole), and `silenceMs` after the last voiced frame the turn is decided (`onEnd`)
///   decided   the turn is over; audio no longer goes to the recognizer, the web takes it from here (`release`)
///   monitor   the reply is on its way (thinking, speaking): the mic is still listening, with the barge-in thresholds
///             (longer, louder, and over what the speaker is playing); speech here is a barge-in (`onStart(barge: true)`)
///
/// Everything is on the audio thread's callback; the callbacks only record and hand work to Tasks.
final class Line: @unchecked Sendable {
    enum State { case armed, decided, monitor }

    /// What the line tells Voice. Times are on `Voice.now()`'s clock.
    struct Handlers: @unchecked Sendable {
        /// Speech began. `barge`: the line was monitoring a reply. `at`: the onset (first voiced frame). Returns the turn.
        var onStart: @Sendable (_ barge: Bool, _ at: Double) -> Int
        /// The turn ended: `lastVoiced` is the end of the last voiced frame, `decided` when the line knew.
        var onEnd: @Sendable (_ turn: Int, _ lastVoiced: Double, _ decided: Double) -> Void
        /// A sound that began but was never an utterance (a cough, a bump): the turn goes back to waiting.
        var onDrop: @Sendable (_ turn: Int) -> Void
        /// The level of what the speaker is playing (dBFS), nil when nothing is.
        var reference: @Sendable () -> Double?
    }

    private let lock = NSLock()
    private var ep: Endpointer
    private var cutter: FrameCutter?
    private var ring: [AVAudioPCMBuffer] = []
    private var ringSeconds = 0.0
    private var backlog: [AVAudioPCMBuffer] = []
    private var forwarding = false
    private var hearing: Hearing?
    private(set) var state = State.monitor
    private(set) var turn = 0
    private let handlers: Handlers
    private var _level = -120.0

    init(config: Endpointer.Config, handlers: Handlers) {
        ep = Endpointer(config)
        self.handlers = handlers
    }

    var level: Double { lock.withLock { _level } }
    var inSpeech: Bool { lock.withLock { ep.inSpeech } }
    var floorDb: Double { lock.withLock { ep.floorDb } }

    /// Wait for the next utterance as `turn`. Returns false if a turn is already open (armed or in speech).
    func arm(turn: Int) -> Bool {
        lock.withLock {
            if state == .armed { return false }
            self.turn = turn
            state = .armed
            ep.barge = false
            if !ep.inSpeech { ep.rearm() }
            return true
        }
    }

    /// The turn was released (the web has the words): watch for a barge-in while the reply runs.
    func monitor() {
        lock.withLock {
            guard state != .armed else { return }
            state = .monitor
            ep.barge = true
            ep.rearm()
            hearing = nil
            backlog = []
            forwarding = false
        }
    }

    /// The recognizer for the open turn is ready: it gets everything held back, in order, then the live audio.
    func attach(_ h: Hearing) {
        lock.withLock {
            hearing = h
            for b in backlog { h.feed(b) }
            backlog = []
        }
    }

    /// The person (or the web) ended the turn now, without waiting for the silence window.
    func endNow() {
        let fired = lock.withLock { () -> (Int, Double)? in
            guard state == .armed, forwarding else { return nil }
            state = .decided
            forwarding = false
            ep.rearm()
            return (turn, Voice.now())
        }
        if let (t, now) = fired { handlers.onEnd(t, now, now) }
    }

    /// The tap may deliver several channels (a loopback device, a voice-processed mic on the Mac): the line and the recognizer
    /// get one, the average, in the same sample rate: the shape the phone's mic and the fixtures have.
    static func mono(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard buffer.format.channelCount > 1, let data = buffer.floatChannelData,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: buffer.format.sampleRate, channels: 1, interleaved: false),
              let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: buffer.frameLength) else { return buffer }
        out.frameLength = buffer.frameLength
        let n = Int(buffer.frameLength), channels = Int(buffer.format.channelCount)
        let target = out.floatChannelData![0]
        for i in 0..<n {
            var sum: Float = 0
            for c in 0..<channels { sum += data[c][i] }
            target[i] = sum / Float(channels)
        }
        return out
    }

    func ingest(_ input: AVAudioPCMBuffer) {
        guard let buffer = Line.mono(input), let channel = buffer.floatChannelData?[0], buffer.frameLength > 0 else { return }
        let rate = buffer.format.sampleRate
        let now = Voice.now()
        var starts: [(barge: Bool, at: Double)] = []
        var ends: [(Double, Double)] = []
        var drops = false
        var turnNow = 0
        lock.withLock {
            if cutter == nil { cutter = FrameCutter(sampleRate: rate) }
            ring.append(buffer)
            ringSeconds += Double(buffer.frameLength) / rate
            while ringSeconds > 0.6, ring.count > 1 {
                ringSeconds -= Double(ring[0].frameLength) / rate
                ring.removeFirst()
            }
            if forwarding {
                if let hearing { hearing.feed(buffer) } else { backlog.append(buffer) }
            }
            ep.barge = state == .monitor
            let ref = state == .monitor ? handlers.reference() : nil
            cutter?.feed(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength))) { db, zcr in
                _level = db
                guard let event = ep.push(db: db, zcr: zcr, ref: ref) else { return }
                switch event {
                case .speechStart(let at):
                    // `at` is on the endpointer's clock; the wall time of the onset is `now` minus what has run since.
                    starts.append((state == .monitor, now - (ep.clock - at)))
                    if state == .monitor { state = .armed; ep.barge = false } // a barge-in opens a turn
                    forwarding = true
                    backlog = ring
                    hearing = nil
                case .speechEnd(let last, let decidedAt):
                    state = .decided
                    forwarding = false
                    ends.append((now - (decidedAt - last), now))
                case .dropped:
                    forwarding = false
                    backlog = []
                    hearing = nil
                    drops = true
                }
            }
            turnNow = turn
        }
        for s in starts {
            let t = handlers.onStart(s.barge, s.at)
            lock.withLock { turn = t }
            turnNow = t
        }
        for (last, decided) in ends { handlers.onEnd(turnNow, last, decided) }
        if drops { handlers.onDrop(turnNow) }
    }
}
