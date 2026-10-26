import AVFoundation
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// SOUND-DESIGN: the voice surface's quiet room: a faint bed (brown-noise room tone + a soft open-fifth pad, both
/// synthesized here, nothing licensed) under a voice session, and three micro-cues (listen, heard, answer).
/// Its own AVAudioEngine on the session Voice already set: it never sets a category or activates the session, and
/// nothing on the voice path waits for it (every call hops to `queue`).
///
/// Levels are dBFS of the bed's own RMS (measured in the render block, logged):
///   thinking (heard → first audio): −46. The "presence" you hear while the engine thinks.
///   listening (the mic is open):    −52. Through the ~26 dB speaker→mic coupling (Endpointer.couplingDb) that is ≈ −78
///                                   at the mic, far under the VAD's −52 floor. The recognizer never hears it.
///   voice (the reply is playing):   the reply's level − 34 dB (the −30…−38 band), never above −52.
///   idle: lingers 4 s after a turn, fades over 1.5 s, then the engine STOPS (no render callbacks, no battery).
/// The cues are ≤ 120 ms, peak −34…−32 dBFS. The setting (Speech › Sound) turns all of it off. Default on.
final class Ambience: @unchecked Sendable {
    static let shared = Ambience()
    static let key = "BendSoundAmbience"
    static let changed = Notification.Name("BendSoundAmbienceChanged")

    /// The person's choice (UserDefaults; `-BendSoundAmbience NO` at launch overrides it for that run). Default on.
    static var enabled: Bool {
        get { flag(UserDefaults.standard.object(forKey: key)) ?? true }
        set {
            UserDefaults.standard.set(newValue, forKey: key)
            NotificationCenter.default.post(name: changed, object: nil)
            shared.queue.async { shared.apply(reason: "setting \(newValue ? "on" : "off")") }
        }
    }

    /// VOICEFIX: the setting stores a Bool, but a launch arg lands in the argument domain as a String ("NO"), which
    /// `as? Bool` read as nil (so: on). Both forms count; anything else is "not set".
    static func flag(_ value: Any?) -> Bool? {
        if let s = value as? String {
            switch s.lowercased() {
            case "yes", "1", "true": return true
            case "no", "0", "false": return false
            default: return nil
            }
        }
        return value as? Bool
    }

    /// VOICEFIX: `-voiceSilent` / BEND_VOICE_SILENT (the Mac harness, the simulator) plays nothing, the room included:
    /// no engine, no cue. Where output never runs an IO cycle, the cue's `play()` throws an uncaught NSException.
    private static var on: Bool { enabled && !Voice.silent }

    static let thinkingDb: Float = -46
    static let listeningDb: Float = -52
    static let underVoiceDb: Float = 34
    static let lingerS = 4.0
    static let fadeS = 1.5

    enum Cue: String { case listen, heard, answer }
    private enum Mode: String { case off, listening, thinking, voice, linger }

    private let queue = DispatchQueue(label: "bend.voice.ambience", qos: .userInitiated)
    private let engine = AVAudioEngine()
    private let cuePlayer = AVAudioPlayerNode()
    private let format = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!
    private let dsp = Bed(rate: 48_000)
    private var source: AVAudioSourceNode!
    private var cues: [Cue: AVAudioPCMBuffer] = [:]
    private var mode = Mode.off
    private var voicePhase = Voice.Phase.idle
    private var gen = 0
    private var started = 0.0
    private var wired = false

    private init() {
        for cue in [Cue.listen, .heard, .answer] { cues[cue] = Self.render(cue, format) }
        NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { [weak self] _ in
            guard let self else { return }
            self.queue.async { if self.mode != .off && !self.engine.isRunning { self.startEngine(why: "output changed") } }
        }
        #if os(iOS)
        NotificationCenter.default.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: nil) { [weak self] _ in
            guard let self else { return }
            self.queue.async { if self.mode != .off { self.halt(why: "background") } }
        }
        #endif
    }

    // MARK: what Voice tells it (any thread; returns at once)

    func phase(_ p: Voice.Phase) { queue.async { self.voicePhase = p; self.apply(reason: "phase \(p.rawValue)") } }

    /// The reply's first audio is playing at `speechDb` (the Speaker's level): the bed tucks under it.
    func voiceStarted(speechDb: Double?) {
        queue.async {
            guard self.voicePhase == .speaking, Self.on, self.mode != .off else { return }
            let target = min(Self.listeningDb, Float(speechDb ?? -20) - Self.underVoiceDb)
            self.to(.voice, db: target, tau: 0.12, reason: String(format: "voice speech_db=%.1f", speechDb ?? .nan))
        }
    }

    func cue(_ cue: Cue) {
        queue.async {
            guard Self.on, let buffer = self.cues[cue] else { return }
            if !self.engine.isRunning { self.startEngine(why: "cue") }
            guard self.engine.isRunning else { return }
            self.cuePlayer.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
            if !self.cuePlayer.isPlaying { self.cuePlayer.play() }
            Voice.log(String(format: "ambience cue %@ ms=%.0f", cue.rawValue, Double(buffer.frameLength) / 48))
            if self.mode == .off { self.scheduleStop(after: 0.4) }  // a cue alone does not keep the engine up
        }
    }

    // MARK: the state machine (queue only)

    private func apply(reason: String) {
        guard Self.on else { if mode != .off { halt(why: reason) }; return }
        switch voicePhase {
        case .listening: to(.listening, db: Self.listeningDb, tau: 0.25, reason: reason)
        case .transcribing, .speaking: if mode != .voice || voicePhase == .transcribing { to(.thinking, db: Self.thinkingDb, tau: 0.6, reason: reason) }
        case .idle, .preparing:
            guard mode != .off, mode != .linger else { return }
            to(.linger, db: Self.thinkingDb, tau: 0.6, reason: reason)
            let g = gen
            queue.asyncAfter(deadline: .now() + Self.lingerS) {
                guard self.gen == g else { return }
                self.dsp.set(0, tau: Self.fadeS / 3)
                self.scheduleStop(after: Self.fadeS)
            }
        }
    }

    private func to(_ m: Mode, db: Float, tau: Double, reason: String) {
        gen += 1
        if !engine.isRunning { startEngine(why: reason) }
        guard engine.isRunning else { return }
        if mode != m { Voice.log(String(format: "ambience %@ → %@ target_db=%.1f bed_db=%.1f (%@)", mode.rawValue, m.rawValue, db, dsp.meterDb(), reason)) }
        mode = m
        dsp.set(powf(10, db / 20), tau: tau)
    }

    private func scheduleStop(after s: Double) {
        let g = gen
        queue.asyncAfter(deadline: .now() + s) { if self.gen == g { self.halt(why: "idle") } }
    }

    /// The graph is wired at the first start, not in `init`: `mainMixerNode` makes the output node, and where output
    /// never runs an IO cycle (the simulator) that RPC times out and aborts. `Ambience.shared` alone must stay inert.
    private func wire() {
        guard !wired else { return }
        wired = true
        let dsp = self.dsp
        source = AVAudioSourceNode(format: format) { silent, _, frames, abl -> OSStatus in
            let out = UnsafeMutableAudioBufferListPointer(abl)
            guard let p = out[0].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            dsp.render(p, Int(frames))
            if dsp.quiet { silent.pointee = true }
            return noErr
        }
        engine.attach(source)
        engine.attach(cuePlayer)
        engine.connect(source, to: engine.mainMixerNode, format: format)
        engine.connect(cuePlayer, to: engine.mainMixerNode, format: format)
        // A receipt run is muted like the Speaker (-voiceAudible plays it); the level meter still reads the signal.
        if Voice.checking && !Voice.args.contains("-voiceAudible") { engine.mainMixerNode.outputVolume = 0 }
    }

    private func startEngine(why: String) {
        let t0 = Voice.now()
        wire()
        engine.prepare()
        do { try engine.start() } catch { Voice.log("ambience could not start: \(error.localizedDescription)"); return }
        started = Voice.now()
        Voice.log(String(format: "ambience engine start ms=%.1f (%@)", started - t0, why))
    }

    /// Off: the bed silenced and the engine stopped (no render thread left running).
    private func halt(why: String) {
        gen += 1
        dsp.set(0, tau: 0, now: true)
        if wired { cuePlayer.stop() }
        let on = engine.isRunning
        if on { engine.stop() }
        if mode != .off || on {
            Voice.log(String(format: "ambience off engine_stopped=%d ran_s=%.1f peak_bed_db=%.1f (%@)", on ? 1 : 0, (Voice.now() - started) / 1000, dsp.takePeakDb(), why))
        }
        mode = .off
    }

    // MARK: sound

    /// One cue: a short sine (or two) with a 4 ms rise and an exponential fall, rendered once at launch.
    private static func render(_ cue: Cue, _ format: AVAudioFormat) -> AVAudioPCMBuffer {
        // (partials Hz, length s, peak dBFS, decay s): listen rises a breath, heard settles, answer is an open fifth.
        let (hz, len, peakDb, decay): ([Double], Double, Double, Double) = switch cue {
        case .listen: ([660, 990], 0.09, -32, 0.03)
        case .heard: ([523.25], 0.07, -34, 0.022)
        case .answer: ([784, 1174.66], 0.12, -34, 0.045)
        }
        let rate = format.sampleRate
        let n = AVAudioFrameCount(rate * len)
        let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: n)!
        buf.frameLength = n
        let p = buf.floatChannelData![0]
        let amp = pow(10, peakDb / 20) / Double(hz.count)
        for i in 0..<Int(n) {
            let t = Double(i) / rate
            let env = min(1, t / 0.004) * exp(-t / decay) * min(1, (len - t) / 0.01)
            var s = 0.0
            for (k, f) in hz.enumerated() { s += sin(2 * .pi * f * t * (cue == .listen ? 1 + 0.25 * t / len : 1)) / Double(k + 1) }
            p[i] = Float(amp * env * s)
        }
        return buf
    }

    /// The bed itself, on the render thread: brown noise band-limited to 110 Hz–1.6 kHz (room tone) + a G/D/G pad
    /// 16 dB under it, breathing at 0.06 Hz. Normalized once so a gain of 1.0 is 0 dBFS RMS; then the smoothed gain.
    final class Bed: @unchecked Sendable {
        private let rate: Float
        private var rng: UInt32 = 0x9E37_79B9
        private var brown: Float = 0, hpX: Float = 0, hpY: Float = 0, lp: Float = 0
        private var phases: [Float] = [0, 0, 0]
        private let steps: [Float]
        private var lfo: Float = 0
        private var norm: Float = 1
        // Written by the queue, read by the render thread (word-sized; a stale read is one buffer late).
        private var gain: Float = 0
        private var target: Float = 0
        private var coef: Float = 1
        private var sumSq: Double = 0, count = 0, peakDb: Double = -120

        init(rate: Float) {
            self.rate = rate
            steps = [196, 293.66, 392].map { 2 * .pi * $0 / rate }
            var probe = [Float](repeating: 0, count: Int(rate))
            target = 1; gain = 1
            probe.withUnsafeMutableBufferPointer { raw($0.baseAddress!, $0.count) }
            let rms = sqrt(probe.reduce(0) { $0 + $1 * $1 } / Float(probe.count))
            norm = rms > 0 ? 1 / rms : 1
            target = 0; gain = 0; sumSq = 0; count = 0
        }

        var quiet: Bool { gain < 1e-5 && target == 0 }

        func set(_ g: Float, tau: Double, now: Bool = false) {
            coef = tau <= 0 ? 1 : 1 - expf(-1 / (Float(tau) * rate))
            target = g
            if now { gain = g }
        }

        /// The bed's RMS since the last read (dBFS of what went out), and the loudest such window.
        func meterDb() -> Double {
            let db = count > 0 ? 10 * log10(max(sumSq / Double(count), 1e-12)) : -120
            sumSq = 0; count = 0
            peakDb = max(peakDb, db)
            return db
        }
        func takePeakDb() -> Double { let p = max(peakDb, meterDb()); peakDb = -120; return p }

        func render(_ p: UnsafeMutablePointer<Float>, _ n: Int) {
            if quiet { for i in 0..<n { p[i] = 0 }; return }
            raw(p, n)
            var ss: Double = 0
            for i in 0..<n {
                gain += (target - gain) * coef
                p[i] *= norm * gain
                ss += Double(p[i] * p[i])
            }
            sumSq += ss; count += n
        }

        private func raw(_ p: UnsafeMutablePointer<Float>, _ n: Int) {
            let hpA: Float = 1 / (1 + 2 * .pi * 110 / rate)
            let lpA: Float = 1 - expf(-2 * .pi * 1600 / rate)
            let lfoStep: Float = 2 * .pi * 0.06 / rate
            for i in 0..<n {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5
                let white = Float(rng) / Float(UInt32.max) * 2 - 1
                brown = (brown + 0.02 * white) / 1.02
                let hp = hpA * (hpY + brown - hpX); hpX = brown; hpY = hp
                lp += (hp - lp) * lpA
                var pad: Float = 0
                for k in 0..<3 { phases[k] += steps[k]; if phases[k] > 2 * .pi { phases[k] -= 2 * .pi }; pad += sinf(phases[k]) }
                lfo += lfoStep; if lfo > 2 * .pi { lfo -= 2 * .pi }
                // The pad sits 16 dB under the noise (offline check: noise RMS 0.040, pad RMS 0.0064; crest 12 dB).
                p[i] = lp + pad * 0.006 * (0.6 + 0.4 * sinf(lfo))
            }
        }
    }
}
