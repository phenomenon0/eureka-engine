import Foundation

/// VOICE-R2: the endpointer. Pure Foundation (no audio framework), so the Mac check (scripts/vad-check.sh) and the app run
/// the same code. Energy + zero-crossing VAD over 20 ms frames, in the PixelGate shape (camera area): a noise floor that is
/// learned only from frames judged quiet, an onset window (speech must hold for `onsetFrames` of the last `onsetWindow`
/// before it counts) and a settle window (silence must hold for `silenceMs` before the turn ends). Every threshold is a
/// named field, and scripts/vad-check.sh measures them on the fixtures (false cuts, overhears, endpoint lag).
struct Endpointer {
    struct Config {
        /// Silence after speech that ends a turn (the brief: ~600–800 ms).
        var silenceMs = 700.0
        /// Speech onset: `onsetFrames` voiced of the last `onsetWindow` frames (20 ms each): 100 ms of 160.
        var onsetFrames = 5
        var onsetWindow = 8
        /// Barge-in (the reply is playing): longer and louder, so the speaker's own leakage into the mic is not a voice.
        var bargeOnsetFrames = 12
        var bargeOnsetWindow = 16
        var bargeExtraDb = 6.0
        /// A frame is voiced when its level is this far over the learned floor, and over the absolute floor.
        var overFloorDb = 11.0
        var absFloorDb = -52.0
        /// Hiss (a high zero-crossing rate at a level near the floor) is not voice.
        var hissZcr = 0.42
        var hissMarginDb = 18.0
        /// The floor never learns above this (loud rooms still gate on the absolute floor + margin).
        var floorCeilDb = -38.0
        /// The floor is the `floorPercentile` of the last `floorFrames` frames (5 s), seeded by the first `calibrateFrames`
        /// (300 ms); frozen while a turn is open, so speech never raises it.
        var floorFrames = 250
        var floorPercentile = 0.15
        var calibrateFrames = 15
        /// A turn this long is ended whatever the level says (a fan that came on mid-turn must not hold it open forever).
        var maxTurnMs = 30_000.0
        /// Barge-in only: the reply's own leakage into the mic is taken to be the playback level minus this much.
        var couplingDb = 26.0
        /// A turn this short in total voiced time is a click or a cough, not an utterance: dropped, not endpointed.
        var minVoicedMs = 240.0
    }

    enum Event: Equatable {
        /// Speech began; `at` is the (stream) time of the first voiced frame of the onset window, in ms.
        case speechStart(at: Double)
        /// The turn ended: `lastVoicedAt` is the end of the last voiced frame, `at` when it was decided (≥ silenceMs later).
        case speechEnd(lastVoicedAt: Double, at: Double)
        /// A sound that was voiced for a moment but never became an utterance (below `minVoicedMs`).
        case dropped(at: Double)
    }

    var config: Config
    /// True while the reply is playing: the barge-in thresholds apply.
    var barge = false
    private(set) var floorDb = -70.0
    private var levels: [Double] = []
    private var sinceFloor = 0
    private(set) var inSpeech = false
    private(set) var voicedMs = 0.0
    private var recent: [Bool] = []
    private var silentMs = 0.0
    private var lastVoicedEnd = 0.0
    private var onsetAt = 0.0
    private(set) var clock = 0.0
    private(set) var lastDb = -120.0

    init(_ config: Config = Config()) { self.config = config }

    static let frameMs = 20.0

    /// Level (dBFS RMS) and zero-crossing rate of one frame of mono samples.
    static func measure(_ x: UnsafeBufferPointer<Float>) -> (db: Double, zcr: Double) {
        guard x.count > 1 else { return (-120, 0) }
        var sum = 0.0
        var crossings = 0
        var prev = x[0]
        for v in x { sum += Double(v) * Double(v); if (v >= 0) != (prev >= 0) { crossings += 1 }; prev = v }
        let rms = (sum / Double(x.count)).squareRoot()
        return (20 * log10(max(rms, 1e-6)), Double(crossings) / Double(x.count))
    }

    /// One frame (20 ms). Returns the event it caused, if any.
    /// `ref`: the level of what the speaker is playing right now (dBFS), when the reply is playing.
    mutating func push(db: Double, zcr: Double, ref: Double? = nil) -> Event? {
        let t = clock
        clock += Self.frameMs
        lastDb = db
        learn(db)
        var thr = max(config.absFloorDb, floorDb + config.overFloorDb)
        if barge {
            thr += config.bargeExtraDb
            if let ref { thr = max(thr, ref - config.couplingDb) }
        }
        let hiss = zcr > config.hissZcr && db < floorDb + config.hissMarginDb
        let voiced = db > thr && !hiss
        recent.append(voiced)
        let window = barge ? config.bargeOnsetWindow : config.onsetWindow
        if recent.count > max(config.onsetWindow, config.bargeOnsetWindow) { recent.removeFirst() }
        if !inSpeech {
            let need = barge ? config.bargeOnsetFrames : config.onsetFrames
            let have = recent.suffix(window).filter { $0 }.count
            if have >= need {
                inSpeech = true
                silentMs = 0
                voicedMs = Double(have) * Self.frameMs
                lastVoicedEnd = clock
                // The onset is the first voiced frame in the window, not the frame that completed it.
                let w = Array(recent.suffix(window))
                onsetAt = t - Double(w.count - 1 - (w.firstIndex(of: true) ?? 0)) * Self.frameMs
                return .speechStart(at: max(0, onsetAt))
            }
            return nil
        }
        if voiced {
            silentMs = 0
            voicedMs += Self.frameMs
            lastVoicedEnd = clock
            return nil
        }
        silentMs += Self.frameMs
        if silentMs >= config.silenceMs || clock - onsetAt >= config.maxTurnMs {
            inSpeech = false
            recent.removeAll()
            let ended = voicedMs >= config.minVoicedMs
            let last = lastVoicedEnd
            voicedMs = 0
            return ended ? .speechEnd(lastVoicedAt: last, at: clock) : .dropped(at: clock)
        }
        return nil
    }

    /// The noise floor: the first frames set it, then a low percentile of the last few seconds; frozen inside a turn.
    private mutating func learn(_ db: Double) {
        let db = max(db, -100)
        levels.append(db)
        if levels.count > config.floorFrames { levels.removeFirst() }
        if levels.count <= config.calibrateFrames {
            floorDb = min(levels.reduce(0, +) / Double(levels.count), config.floorCeilDb)
            return
        }
        sinceFloor += 1
        guard !inSpeech, sinceFloor >= 5 else { return }
        sinceFloor = 0
        let sorted = levels.sorted()
        floorDb = min(sorted[Int(Double(sorted.count - 1) * config.floorPercentile)], config.floorCeilDb)
    }

    /// Back to waiting for speech (a new turn), keeping the learned floor.
    mutating func rearm() {
        inSpeech = false
        recent.removeAll()
        silentMs = 0
        voicedMs = 0
    }
}

/// Cuts a stream of mono buffers (any size) into 20 ms frames for an `Endpointer`.
struct FrameCutter {
    private var carry: [Float] = []
    let frame: Int
    init(sampleRate: Double) { frame = Int(sampleRate * Endpointer.frameMs / 1000) }

    mutating func feed(_ samples: UnsafeBufferPointer<Float>, _ body: (Double, Double) -> Void) {
        carry.append(contentsOf: samples)
        var at = 0
        while carry.count - at >= frame {
            carry.withUnsafeBufferPointer { p in
                let m = Endpointer.measure(UnsafeBufferPointer(rebasing: p[at..<(at + frame)]))
                body(m.db, m.zcr)
            }
            at += frame
        }
        if at > 0 { carry.removeFirst(at) }
    }
}
