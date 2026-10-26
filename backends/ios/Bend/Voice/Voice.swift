import AVFoundation
import Foundation
import Speech
#if canImport(UIKit)
import UIKit
import os
#endif

/// PHONEVOICE: the on-device half of voice mode (docs/VOICE.md). Hearing is Apple's SpeechAnalyzer + SpeechTranscriber
/// (on device, no Siri/Dictation switch needed, no network once the OS has the en-US model); speaking is the phone's own
/// neural voice (VOICEREAL: NeuralVoice, Supertonic 2 F1 on onnxruntime), with AVSpeechSynthesizer (`write`) as the
/// fallback, played through an AVAudioEngine and kept as a .caf receipt per turn.
/// The reply itself comes from OUR engine: the web app sends the transcript through its transport to the warm resident
/// chat engine (127.0.0.1:8931), exactly as a typed message goes. This file never talks to the engine.
///
/// One turn: listen (mic, or a fixture clip fed at real-time pace) → release (= speech end: the transcript is
/// finalized) → speak(text)… as the reply streams → finish (waits for the audio to play out) → receipt.
/// Clock: one monotonic ms clock (`Voice.now`); every receipt time is relative to the turn's speech end.
final class Voice: @unchecked Sendable {
    static let shared = Voice()
    static let port: UInt16 = (ProcessInfo.processInfo.environment["BEND_PORT_BASE"].flatMap { UInt16($0) } ?? 8930) + 2  // PHONEONE: BEND_PORT_BASE
    static let locale = Locale(identifier: "en-US")
    static let args = ProcessInfo.processInfo.arguments
    /// `-voiceCheck` (the PTT fixture turns) or `-voiceLine` (VOICE-R2: the hands-free fixture run): a check run, not a person.
    static let checking = args.contains("-voiceCheck") || args.contains("-voiceLine")

    /// `-voiceSilent` / BEND_VOICE_SILENT=1 (the Mac harness): render and keep the audio, play nothing. First audio is
    /// then the first rendered buffer, not the first one handed to the output.
    static let silent: Bool = args.contains("-voiceSilent") || ProcessInfo.processInfo.environment["BEND_VOICE_SILENT"] == "1"

    static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

    static func log(_ line: String) {
        let bytes = Array("BEND voice \(line)\n".utf8)
        _ = bytes.withUnsafeBytes { write(Voice.consoleFD, $0.baseAddress, $0.count) }
    }
    /// A dup of the launch-time stdout, taken in VoiceServer.start before any engine run: an engine run points fd 1 at
    /// its own file and parses it as the new ids, so nothing else may print there (ENGINE-NATIVE.md §2).
    /// BENDFIX-1: BEND_CONSOLE_FILE (as EngineManager's: a path, or `documents:<name>` on a device UI test run) gets the
    /// voice lines too, so one file holds the whole console.
    static let consoleFD: Int32 = {
        if let raw = ProcessInfo.processInfo.environment["BEND_CONSOLE_FILE"], !raw.isEmpty {
            let path = raw.hasPrefix("documents:")
                ? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(String(raw.dropFirst("documents:".count))).path
                : raw
            let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
            if fd >= 0 { return fd }
        }
        return dup(STDOUT_FILENO)
    }()

    enum Phase: String { case idle, preparing, listening, transcribing, speaking }

    struct Receipt {
        var turn: Int
        var source: String
        var startedAt: Double
        var speechEndAt: Double?
        var finalAt: Double?
        var text = ""
        var firstPartialAt: Double?
        var firstTextAt: Double?
        var firstAudioAt: Double?
        var reply = ""
        var chunks = 0
        var frames = 0
        var sampleRate = 0.0
        var file: URL?
        var playedAt: Double?
        var cancelled = false
        /// Who produced `text`: the recognizer's name, or "fixture-text" (see KnownText).
        var heardBy = ""
        /// VOICE-R2: hands-free turns. `speechEndAt` is then the last voiced frame; `endpointAt` is when the line decided.
        var auto = false
        var endpointAt: Double?
        /// Barge-in: this turn began by talking over the reply; the speaker stopped at `bargeStoppedAt`.
        var barged = false
        var bargeStoppedAt: Double?
        /// VOICEREAL: who spoke the reply, in order of first use ("Supertonic F1"; "+Samantha" if a chunk fell back).
        var voices: [String] = []
        /// VOICESHOP2: the kitchen timer of the turn's first audible chunk (Speaker.Timing).
        var timing: Speaker.Timing?

        /// One line per turn, every leg relative to the one before, summing to first_audio_ms:
        /// prep (speech end → the first speakable text: hearing + the engine's first clause) → queue (waiting for the
        /// render queue: a voice re-opening, an earlier chunk) → first_chunk_synth (the voice renders the first chunk) →
        /// play_start (into the .caf, the player scheduled) = first_audio; heard_ms adds the output's own latency.
        func timer() -> String {
            func ms(_ a: Double?, _ b: Double?) -> String { guard let a, let b else { return "-" }; return String(format: "%.0f", a - b) }
            let t = timing
            return "prep_ms=\(ms(firstTextAt, speechEndAt)) queue_ms=\(ms(t?.synthStart, t?.queued)) first_chunk_synth_ms=\(ms(t?.synthEnd, t?.synthStart)) "
                + "play_start_ms=\(ms(t?.scheduled, t?.synthEnd)) first_audio_ms=\(ms(firstAudioAt, speechEndAt)) "
                + "heard_ms=\(ms(t.map { $0.scheduled + $0.outputLatencyMs }, speechEndAt)) out_latency_ms=\(t.map { String(format: "%.0f", $0.outputLatencyMs) } ?? "-") "
                + "first_chunk_words=\(t?.words ?? 0) first_chunk_audio_ms=\(t.map { String(format: "%.0f", $0.audioMs) } ?? "-") runtime=\(t?.runtime ?? "-") voice=\(voices.joined(separator: "+"))"
        }

        func json() -> [String: Any] {
            func rel(_ t: Double?) -> Any { guard let t, let end = speechEndAt else { return NSNull() }; return (t - end).rounded() }
            var out: [String: Any] = [
                "turn": turn, "source": source, "text": text, "heardBy": heardBy, "voice": voices.joined(separator: "+"), "reply": reply, "chunks": chunks, "cancelled": cancelled,
                "audioMs": sampleRate > 0 ? (Double(frames) / sampleRate * 1000).rounded() : 0,
                "file": file?.path ?? NSNull(), "fileBytes": file.flatMap { try? FileManager.default.attributesOfItem(atPath: $0.path)[.size] as? Int } ?? 0,
                // Relative to speech end (push-to-talk release, or the fixture's last sample fed).
                "finalMs": rel(finalAt), "firstTextMs": rel(firstTextAt), "firstAudioMs": rel(firstAudioAt), "playedMs": rel(playedAt),
                "listenMs": speechEndAt.map { ($0 - startedAt).rounded() } ?? NSNull(),
            ]
            if let p = firstPartialAt { out["firstPartialMs"] = (p - startedAt).rounded() } // from listen start
            if auto {
                out["auto"] = true
                out["endpointMs"] = rel(endpointAt)
                // Endpoint decided → first audio: what the loop adds after it knows the person stopped.
                out["endpointToAudioMs"] = (firstAudioAt != nil && endpointAt != nil) ? (firstAudioAt! - endpointAt!).rounded() : NSNull()
            }
            if barged {
                out["barged"] = true
                // Voice onset (the VAD's estimate) → the speaker stopped.
                out["bargeStopMs"] = bargeStoppedAt.map { ($0 - startedAt).rounded() } ?? NSNull()
            }
            if let t = timing, let end = speechEndAt {
                out["timer"] = ["prepMs": firstTextAt.map { ($0 - end).rounded() } ?? NSNull(), "queueMs": (t.synthStart - t.queued).rounded(),
                                "firstChunkSynthMs": (t.synthEnd - t.synthStart).rounded(), "playStartMs": (t.scheduled - t.synthEnd).rounded(),
                                "heardMs": (t.scheduled + t.outputLatencyMs - end).rounded(), "firstChunkWords": t.words, "runtime": t.runtime]
            }
            return out
        }
    }

    private let lock = NSLock()
    private var _phase = Phase.idle
    private var _partial = ""
    private var _turn = 0
    private var receipts: [Int: Receipt] = [:]
    private var listener: Hearing?
    private var source: Source?
    // VOICE-R2: the hands-free line (one mic for the conversation) and what it has decided.
    private var line: Line?
    private var lineSource: Source?
    private var synthetic: SyntheticSource?
    private var hearings: [Int: Task<Hearing, Error>] = [:]
    private var finals: [Int: Task<String, Error>] = [:]
    private var _endpointed = 0
    private var _barged = 0
    let speaker = Speaker()
    private(set) var prepared = false

    var phase: Phase { lock.withLock { _phase } }
    var turn: Int { lock.withLock { _turn } }
    private func set(_ phase: Phase) {
        lock.withLock { _phase = phase }
        Ambience.shared.phase(phase)  // SOUND-DESIGN: the bed follows the phase (its own queue; never waits)
    }

    func receipt(_ turn: Int) -> Receipt? { lock.withLock { receipts[turn] } }
    private func update(_ turn: Int, _ change: (inout Receipt) -> Void) {
        lock.withLock { if var r = receipts[turn] { change(&r); receipts[turn] = r } }
    }

    // MARK: status

    func status() async -> [String: Any] {
        let (phase, partial, turn) = lock.withLock { (_phase, _partial, _turn) }
        let line = lock.withLock { self.line }
        return [
            "line": line != nil, "speech": line?.inSpeech ?? false, "level": line?.level ?? -120,
            // VOICESHADE: what the speaker is playing at this moment (dBFS, 20 ms frames of the reply's PCM on the
            // playback clock; −120 when silent). The voice surface draws the reply's waveform from it.
            "out": speaker.outLevel,
            "endpointed": lock.withLock { _endpointed }, "barged": lock.withLock { _barged }, "mic": Self.micState,
            "available": SpeechTranscriber.isAvailable, "assets": await Self.assets(), "locale": Self.locale.identifier,
            "recognizer": Self.recognizer,
            "phase": phase.rawValue, "partial": partial, "turn": turn, "prepared": prepared, "silent": Self.silent,
            "voice": speaker.name, "voiceEngine": speaker.engineKind, "voiceFallback": speaker.fallback ?? NSNull(),
            "voiceID": speaker.packID ?? NSNull(), "voiceChoice": VoiceLibrary.shared.choice,
            "appleVoice": speaker.voice.name, "appleVoiceQuality": speaker.voice.quality.rawValue,
        ]
    }

    /// "installed" when the en-US transcriber model is on the device; "supported" means the OS can download it
    /// (Settings-style system download, done only on the person's tap: `install`); "unsupported" otherwise.
    static func assets() async -> String {
        let installed = await SpeechTranscriber.installedLocales.contains { $0.identifier(.bcp47) == locale.identifier(.bcp47) }
        if installed { return "installed" }
        let supported = await SpeechTranscriber.supportedLocales.contains { $0.identifier(.bcp47) == locale.identifier(.bcp47) }
        return supported ? "supported" : "unsupported"
    }

    /// Which recognizer hears: SpeechAnalyzer (iOS 26+ / macOS 26+ with the model), else the older SFSpeechRecognizer
    /// held to on-device recognition (the iOS simulator has no SpeechTranscriber; older phones may lack it), else none.
    static var recognizer: String {
        #if BEND_EARS
        if EarsASR.enabled { return EarsASR.label }  // G38: -voiceASR ears (EarsASR.swift)
        #endif
        if SpeechTranscriber.isAvailable { return "SpeechAnalyzer" }
        if SFSpeechRecognizer(locale: locale)?.supportsOnDeviceRecognition == true { return "SFSpeechRecognizer(on-device)" }
        return "none"
    }

    /// The recognizer failed to start at prepare (e.g. the simulator's missing ASR asset).
    nonisolated(unsafe) static var noRecognizerHere = false

    /// `ask`: may show the system's speech-recognition prompt (the fallback only; SpeechAnalyzer has none). Opening the
    /// surface never asks; the first real press does; a -voiceCheck run never does.
    static func hearing(ask: Bool = true, onPartial: @escaping @Sendable (String) -> Void) async throws -> Hearing {
        #if BEND_EARS
        if EarsASR.enabled { return EarsListener(onPartial: onPartial) }  // G38: the ASR tower, behind -voiceASR ears
        #endif
        if SpeechTranscriber.isAvailable { return try await Listener(onPartial: onPartial) }
        return try await LegacyListener(ask: ask && !args.contains("-voiceCheck"), onPartial: onPartial)
    }

    /// The person asked for the speech model: the OS downloads it once (Apple's asset server, not ours). Never automatic.
    func install() async throws -> String {
        let module = Listener.transcriber()
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [module]) {
            Self.log("assets: installing \(Self.locale.identifier) (system download, asked for by the person)")
            try await request.downloadAndInstall()
        }
        return await Self.assets()
    }

    // MARK: prepare (the surface opened)

    /// Pays the cold costs once, before the first turn: the transcriber's model (a 0.3 s silent pass), the synthesizer's
    /// voice (a throw-away render; ~0.4 s cold on the Mac, 4–11 ms warm) and the output engine. Idempotent.
    func prepare() async -> [String: Any] {
        let t0 = Self.now()
        if phase == .idle { set(.preparing) }
        var out: [String: Any] = [:]
        do {
            #if os(iOS)
            try Self.session()
            #endif
            let tl = Self.now()
            let warm = try await Self.hearing(ask: false, onPartial: { _ in })
            out["recognizer"] = Self.recognizer
            let silence = AVAudioPCMBuffer(pcmFormat: warm.format, frameCapacity: AVAudioFrameCount(warm.format.sampleRate * 0.3))!
            silence.frameLength = silence.frameCapacity
            warm.feed(silence)
            _ = try await warm.finish()
            out["listenerMs"] = (Self.now() - tl).rounded()
        } catch {
            out["listenerError"] = "\(error)"
            Self.noRecognizerHere = true
        }
        let ts = Self.now()
        out.merge(await speaker.warm()) { a, _ in a }
        out["speakerMs"] = (Self.now() - ts).rounded()
        prepared = true
        if phase == .preparing { set(.idle) }
        out["ms"] = (Self.now() - t0).rounded()
        #if os(iOS)
        out["output"] = Self.output()
        if Self.args.contains("-voiceCheck") { Self.awake(true) }
        #endif
        Self.log("prepare \(out.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))")
        return out
    }

    #if os(iOS)
    /// One category for the whole surface: .playAndRecord (the mic hears, the voice speaks, no category switch between
    /// them). It is NOT silenced by the Ring/Silent switch (only .ambient and .soloAmbient are) nor by the screen lock;
    /// .defaultToSpeaker sends it to the loudspeaker, not the earpiece. Headphones / Bluetooth take over when connected.
    static func session(voiceChat: Bool = false) throws {
        let s = AVAudioSession.sharedInstance()
        // Hands-free: .voiceChat turns on the system's echo cancellation, so the reply does not hear itself.
        let mode: AVAudioSession.Mode = voiceChat && !args.contains("-voiceNoAEC") ? .voiceChat : .default
        if s.category != .playAndRecord || !s.categoryOptions.contains(.defaultToSpeaker) || s.mode != mode {
            try s.setCategory(.playAndRecord, mode: mode, options: [.defaultToSpeaker, .allowBluetoothHFP, .allowBluetoothA2DP])
        }
        try s.setActive(true)
        speakerRoute()
    }

    /// No headphones assumption: with nothing plugged in, the route is the built-in speaker. If it ever lands on the
    /// earpiece (the receiver: what .playAndRecord does without .defaultToSpeaker), force the speaker.
    static func speakerRoute() {
        let s = AVAudioSession.sharedInstance()
        if s.currentRoute.outputs.contains(where: { $0.portType == .builtInReceiver }) { try? s.overrideOutputAudioPort(.speaker) }
    }

    /// The output, in one line: what the voice plays through and whether anything would mute it.
    static func output() -> String {
        let s = AVAudioSession.sharedInstance()
        let mutedBySwitch = s.category == .ambient || s.category == .soloAmbient
        return "category=\(s.category.rawValue.replacingOccurrences(of: "AVAudioSessionCategory", with: "")) mode=\(s.mode.rawValue.replacingOccurrences(of: "AVAudioSessionMode", with: "")) "
            + "options=\(s.categoryOptions.rawValue) silent_switch_mutes=\(mutedBySwitch) route=\(s.currentRoute.outputs.map { $0.portType.rawValue }.joined(separator: "+")) "
            + String(format: "volume=%.2f out_latency_ms=%.0f io_buffer_ms=%.0f", s.outputVolume, s.outputLatency * 1000, s.ioBufferDuration * 1000)
    }
    #endif

    /// The screen stays on while the voice is in a turn (and for 30 s after), and for the whole -voiceCheck run: an
    /// auto-lock mid-reply backgrounds the app and the voice stops (the device run's "away" lines).
    static func awake(_ on: Bool, after delay: Double = 0) {
        #if os(iOS)
        if args.contains("-voiceCheck") && !on { return }
        DispatchQueue.main.async {
            awakeGen += 1
            let gen = awakeGen
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                guard awakeGen == gen else { return }
                if on, !UIApplication.shared.isIdleTimerDisabled { heldAwake = true; UIApplication.shared.isIdleTimerDisabled = true }
                if !on, heldAwake { heldAwake = false; UIApplication.shared.isIdleTimerDisabled = false }
            }
        }
        #endif
    }
    // Main queue only. Only what the voice turned on is turned off (a model download keeps the screen on too).
    nonisolated(unsafe) private static var awakeGen = 0
    nonisolated(unsafe) private static var heldAwake = false

    // MARK: listen

    /// Starts a turn. `fixture`: a clip name in the bundle's voice-check/ (or an absolute path), fed at real-time pace
    /// in 100 ms buffers as a mic would deliver it; nil = the microphone.
    func listen(fixture: String?, auto: Bool = false, silenceMs: Double? = nil) async throws -> Int {
        if auto { return try await listenAuto(fixture: fixture, silenceMs: silenceMs) }
        await cancel(reason: "new turn")
        #if os(iOS)
        if fixture == nil {
            // The first real press asks for the microphone (the system prompt, once); a refusal says where to fix it.
            if AVAudioApplication.shared.recordPermission == .undetermined { _ = await AVAudioApplication.requestRecordPermission() }
            guard AVAudioApplication.shared.recordPermission == .granted else {
                throw VoiceError("Eureka can’t hear you: allow the microphone in Settings › Eureka.")
            }
            try Self.session()
        }
        #else
        if fixture == nil { try await Self.askMic() }
        #endif
        // VOICESHOP2: a voice released under critical memory comes back now, while the person speaks (the render queue
        // runs the load before any chunk of this turn), never Apple's voice in its place.
        speaker.reopenIfReleased(why: "next turn")
        Self.awake(true)
        let turn = lock.withLock { _turn += 1; _partial = ""; return _turn }
        let started = Self.now()
        lock.withLock { receipts[turn] = Receipt(turn: turn, source: fixture ?? "mic", startedAt: started) }
        set(.listening)
        let listener: Hearing, source: Source
        do {
            let onPartial: @Sendable (String) -> Void = { [weak self] text in
                guard let self else { return }
                self.lock.withLock { self._partial = text }
                self.update(turn) { if $0.firstPartialAt == nil && !text.isEmpty { $0.firstPartialAt = Voice.now() } }
            }
            if let fixture, Self.recognizer == "none" || Self.noRecognizerHere, Self.args.contains("-voiceCheck") {
                // The iOS simulator has no on-device recognizer (no SpeechTranscriber, no ASR asset): -voiceCheck
                // there runs the rest of the loop on the clip's known words, and every receipt says so.
                listener = try KnownText(Self.fixtureURL(fixture))
                update(turn) { $0.heardBy = KnownText.label }
            } else {
                listener = try await Self.hearing(onPartial: onPartial)
                update(turn) { $0.heardBy = Self.recognizer }
            }
            source = fixture != nil ? try FixtureSource(Self.fixtureURL(fixture!), into: listener) : try MicSource(into: listener)
        } catch {
            set(.idle)
            Self.log("turn \(turn) could not listen: \(error.localizedDescription)")
            throw error
        }
        lock.withLock { self.listener = listener; self.source = source }
        source.start()
        Ambience.shared.cue(.listen)
        Self.log("turn \(turn) listening source=\(fixture ?? "mic")")
        return turn
    }

    static func fixtureURL(_ name: String) throws -> URL {
        if name.hasPrefix("/") { return URL(fileURLWithPath: name) }
        // IMAGETRUE: `look/ask.wav` = a clip under Documents (the device receipts copy the Look asks there).
        if name.contains("/"), let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
           FileManager.default.fileExists(atPath: docs.appendingPathComponent(name).path) {
            return docs.appendingPathComponent(name)
        }
        if let dir = ProcessInfo.processInfo.environment["BEND_VOICE_FIXTURES"] {
            return URL(fileURLWithPath: dir).appendingPathComponent(name)
        }
        guard let url = Bundle.main.url(forResource: (name as NSString).deletingPathExtension, withExtension: "wav", subdirectory: "voice-check") else {
            throw VoiceError("no fixture \(name)")
        }
        return url
    }

    /// Push-to-talk released: the speech has ended. Stops the source (a fixture first plays to its end, as a person
    /// lets go after speaking), finalizes, and returns the transcript.
    func release(_ turn: Int) async throws -> Receipt {
        if receipt(turn)?.auto == true { return try await releaseAuto(turn) }
        let (listener, source) = lock.withLock { (self.listener, self.source) }
        guard turn == self.turn, let listener, let source else { throw VoiceError("turn \(turn) is not listening") }
        await source.stop()
        let end = Self.now()
        Ambience.shared.cue(.heard)
        update(turn) { $0.speechEndAt = end }
        speaker.prewarm(turn: turn)
        set(.transcribing)
        let text = try await listener.finish()
        let fin = Self.now()
        lock.withLock { self.listener = nil; self.source = nil; _partial = text }
        update(turn) { $0.finalAt = fin; $0.text = text }
        set(.speaking) // the web goes to the engine now; the state line there says "thinking" until text lands
        Self.log(String(format: "turn %d heard final_ms=%.0f by=%@ text=%@", turn, fin - end, receipt(turn)?.heardBy ?? "?", text.debugDescription))
        return receipt(turn)!
    }

    // MARK: hands-free (VOICE-R2)

    /// The microphone's state for the surface: whether hands-free can start without a prompt.
    static var micState: String {
        #if os(iOS)
        switch AVAudioApplication.shared.recordPermission {
        case .granted: return "granted"
        case .denied: return "denied"
        default: return "undetermined"
        }
        #else
        switch AVCaptureDevice.authorizationStatus(for: .audio) {  // MACVOICE: the Mac's microphone grant (TCC)
        case .authorized: return "granted"
        case .denied, .restricted: return "denied"
        default: return "undetermined"
        }
        #endif
    }

    /// Opens the line (once) and waits for the next utterance. `fixture` "line" = the synthetic mic (the Mac harness and
    /// the simulator: silence + room noise at real-time pace, clips injected by `inject`). Idempotent while a turn is open.
    func listenAuto(fixture: String?, silenceMs: Double?) async throws -> Int {
        if lock.withLock({ line == nil }) {
            #if os(iOS)
            if fixture == nil {
                if AVAudioApplication.shared.recordPermission == .undetermined { _ = await AVAudioApplication.requestRecordPermission() }
                guard AVAudioApplication.shared.recordPermission == .granted else {
                    throw VoiceError("Eureka can’t hear you: allow the microphone in Settings › Eureka.")
                }
                try Self.session(voiceChat: true)
            }
            #else
            if fixture == nil { try await Self.askMic() }
            #endif
            var config = Endpointer.Config()
            if UserDefaults.standard.bool(forKey: "BendFastEndpoint") { config.silenceMs = 500 }  // G28: -BendFastEndpoint YES (default 700)
            if let ms = silenceMs ?? ProcessInfo.processInfo.environment["BEND_VAD_SILENCE_MS"].flatMap(Double.init) { config.silenceMs = ms }
            let line = Line(config: config, handlers: Line.Handlers(
                onStart: { [unowned self] barge, at in self.lineStarted(barge: barge, at: at) },
                onEnd: { [unowned self] turn, last, decided in self.lineEnded(turn, lastVoiced: last, decided: decided) },
                onDrop: { [unowned self] turn in self.lineDropped(turn) },
                reference: { [unowned self] in self.speaker.referenceDb }
            ))
            let source: Source
            if fixture != nil {
                let s = SyntheticSource { line.ingest($0) }
                lock.withLock { synthetic = s }
                source = s
            } else {
                source = try MicSource(voiceProcessing: !Self.args.contains("-voiceNoAEC")) { line.ingest($0) }
            }
            lock.withLock { self.line = line; lineSource = source }
            source.start()
            Ambience.shared.cue(.listen)
            Self.log("line open source=\(fixture == nil ? "mic" : "synthetic") silence_ms=\(config.silenceMs)")
        }
        let (line, turn) = lock.withLock { (self.line!, _turn + 1) }
        lock.withLock { _partial = "" }
        if line.arm(turn: turn) {
            lock.withLock {
                _turn = turn
                receipts[turn] = Receipt(turn: turn, source: synthetic != nil ? "line(synthetic)" : "mic(line)", startedAt: Self.now(), auto: true)
            }
            set(.listening)
            Self.log("turn \(turn) armed (hands-free)")
            return turn
        }
        return self.turn
    }

    func endpointNow() { lock.withLock { line }?.endNow() }

    /// A clip played into the synthetic mic now (the test driver's voice; absent on a device with a real mic).
    func inject(_ fixture: String) throws {
        guard let synthetic = lock.withLock({ synthetic }) else { throw VoiceError("no synthetic line is open") }
        try synthetic.inject(Self.fixtureURL(fixture))
    }

    /// Line callback: speech began. Returns the turn it belongs to. A barge-in stops the reply right here, on the audio
    /// callback's thread: the speaker stops before the web has even been told.
    private func lineStarted(barge: Bool, at: Double) -> Int {
        let turn: Int
        if barge {
            // The old turn is marked and the new one opened BEFORE the speaker stops: stopping lets the web's `finish` of the
            // old turn return, and it must find the barge already recorded.
            turn = lock.withLock { () -> Int in
                let old = _turn
                receipts[old]?.cancelled = true
                _turn += 1
                _partial = ""
                _barged = _turn
                receipts[_turn] = Receipt(turn: _turn, source: synthetic != nil ? "line(synthetic)" : "mic(line)", startedAt: at, auto: true, barged: true)
                return _turn
            }
            speaker.stop()
            let stoppedAt = Self.now()
            update(turn) { $0.bargeStoppedAt = stoppedAt }
            Self.log(String(format: "turn %d barge-in: speaker stopped %.0f ms after the voice began (cancelled turn %d)", turn, stoppedAt - at, turn - 1))
        } else {
            turn = lock.withLock { _turn }
            update(turn) { $0.startedAt = at }
        }
        set(.listening)
        let onPartial: @Sendable (String) -> Void = { [weak self] text in
            guard let self else { return }
            self.lock.withLock { self._partial = text }
            self.update(turn) { if $0.firstPartialAt == nil && !text.isEmpty { $0.firstPartialAt = Voice.now() } }
        }
        let known = lock.withLock { synthetic?.lastClip }
        let line = lock.withLock { self.line }
        let task = Task<Hearing, Error> {
            let h: Hearing
            if let known, Self.recognizer == "none" || Self.noRecognizerHere, Self.checking {
                h = try KnownText(known)
                self.update(turn) { $0.heardBy = KnownText.label }
            } else {
                h = try await Self.hearing(onPartial: onPartial)
                self.update(turn) { $0.heardBy = Self.recognizer }
            }
            line?.attach(h)
            return h
        }
        lock.withLock { hearings[turn] = task }
        return turn
    }

    /// Line callback: the person stopped. The recognizer is finalized right now (the web's `release` only collects it).
    private func lineEnded(_ turn: Int, lastVoiced: Double, decided: Double) {
        update(turn) { $0.speechEndAt = lastVoiced; $0.endpointAt = decided }
        speaker.prewarm(turn: turn)
        Ambience.shared.cue(.heard)
        set(.transcribing)
        lock.withLock { _endpointed = turn }
        let hearing = lock.withLock { hearings[turn] }
        let task = Task<String, Error> {
            guard let hearing else { return "" }
            let text = try await hearing.value.finish()
            self.update(turn) { $0.finalAt = Voice.now(); $0.text = text }
            return text
        }
        lock.withLock { finals[turn] = task }
        Self.log(String(format: "turn %d endpoint: silence held %.0f ms after the last voiced frame", turn, decided - lastVoiced))
    }

    private func lineDropped(_ turn: Int) {
        let task = lock.withLock { hearings.removeValue(forKey: turn) }
        Task { task?.cancel(); if let h = try? await task?.value { await h.cancel() } }
        lock.withLock { _partial = "" }
        Self.log("turn \(turn) dropped a short sound (not an utterance)")
    }

    /// The web has the endpoint: the transcript, and the line goes back to watching for a barge-in.
    func releaseAuto(_ turn: Int) async throws -> Receipt {
        guard let line = lock.withLock({ line }), turn == self.turn else { throw VoiceError("turn \(turn) is not listening") }
        if lock.withLock({ finals[turn] }) == nil { line.endNow() } // released before the silence window ran out
        guard let final = lock.withLock({ finals[turn] }) else { throw VoiceError("turn \(turn) heard nothing yet") }
        let text = try await final.value
        lock.withLock { _partial = text; finals[turn] = nil; hearings[turn] = nil }
        line.monitor()
        set(.speaking)
        Self.log(String(format: "turn %d heard final_ms=%.0f by=%@ text=%@", turn, (receipt(turn)?.finalAt ?? 0) - (receipt(turn)?.speechEndAt ?? 0), receipt(turn)?.heardBy ?? "?", text.debugDescription))
        return receipt(turn)!
    }

    // MARK: say (UX-VOICE-DRAW)

    /// A speak-only turn: no mic, no hearing. "Read aloud" on a reply (G-6), a reply to the composer's held mic, and
    /// speak(ref) playing a note all speak through it: the web then sends its chunks and finishes it like any turn
    /// (speak / finish), and cancel (or the next listen) stops it. The receipt's times are from this call.
    func say() async throws -> Int {
        await cancel(reason: "read aloud")
        #if os(iOS)
        try Self.session()
        #endif
        speaker.reopenIfReleased(why: "read aloud")
        Self.awake(true)
        let turn = lock.withLock { _turn += 1; _partial = ""; return _turn }
        let started = Self.now()
        lock.withLock { receipts[turn] = Receipt(turn: turn, source: "say", startedAt: started) }
        update(turn) { $0.speechEndAt = started; $0.finalAt = started; $0.heardBy = "none (say)" }
        set(.speaking)
        Self.log("turn \(turn) say: a speak-only turn (no mic)")
        return turn
    }

    // MARK: speak

    func speak(_ turn: Int, _ text: String) throws {
        guard turn == self.turn, receipt(turn)?.cancelled == false else { throw VoiceError("turn \(turn) is over") }
        let at = Self.now()
        if receipt(turn)?.firstTextAt == nil { Ambience.shared.cue(.answer) }  // SOUND-DESIGN: thinking → answer
        update(turn) { r in
            if r.firstTextAt == nil { r.firstTextAt = at }
            r.reply += (r.reply.isEmpty ? "" : " ") + text
            r.chunks += 1
        }
        speaker.say(text, turn: turn) { [weak self] event in
            switch event {
            case .firstAudio(let t):
                guard let self else { return }
                if self.receipt(turn)?.firstAudioAt == nil { Ambience.shared.voiceStarted(speechDb: self.speaker.referenceDb) }
                self.update(turn) { if $0.firstAudioAt == nil { $0.firstAudioAt = t } }
            case .file(let url, let frames, let rate): self?.update(turn) { $0.file = url; $0.frames = frames; $0.sampleRate = rate }
            case .voice(let name): self?.update(turn) { if !$0.voices.contains(name) { $0.voices.append(name) } }
            case .timing(let t): self?.update(turn) { if $0.timing == nil { $0.timing = t } }
            }
        }
    }

    /// The reply is complete: waits until every chunk has been rendered and played out, then closes the turn.
    func finish(_ turn: Int) async -> Receipt? {
        await speaker.drain(turn: turn)
        let played = Self.now()
        update(turn) { $0.playedAt = played }
        if self.turn == turn { set(lock.withLock { line != nil } ? .listening : .idle) }
        guard let r = receipt(turn) else { return nil }
        let j = r.json()
        Self.log("turn \(turn) receipt source=\(r.source) heard_by=\(r.heardBy) voice=\(j["voice"]!) final_ms=\(j["finalMs"]!) first_text_ms=\(j["firstTextMs"]!) first_audio_ms=\(j["firstAudioMs"]!) audio_ms=\(j["audioMs"]!) chunks=\(r.chunks) file=\(r.file?.lastPathComponent ?? "none") bytes=\(j["fileBytes"]!) cancelled=\(r.cancelled)")
        Self.log("turn \(turn) timer \(r.timer())")
        Self.awake(false, after: 30)
        // VOICESHOP2: `-voiceMemoryWarningAfter N`: the OS's memory warning, posted once turn N has played (the harness's
        // memory-policy proof: turn N+1 must still speak in the house voice). With `-voiceMemoryCritical` it takes the
        // critical branch (release after playback, re-open before the next turn) instead of keeping the voice resident.
        if let i = Self.args.firstIndex(of: "-voiceMemoryWarningAfter"), i + 1 < Self.args.count, Int(Self.args[i + 1]) == turn {
            #if os(iOS)
            await MainActor.run { NotificationCenter.default.post(name: UIApplication.didReceiveMemoryWarningNotification, object: nil) }
            #else
            speaker.memoryWarning()
            #endif
        }
        // VOICESHOP: `-voicesSwapAfter N <id>`: the picker's select, by itself, once turn N has played (the device run's
        // hot-swap receipt: the phone's loopback isn't reachable from the Mac). The app keeps running; turn N+1 speaks in <id>.
        if let i = Self.args.firstIndex(of: "-voicesSwapAfter"), i + 2 < Self.args.count, Int(Self.args[i + 1]) == turn {
            _ = try? await speaker.select(Self.args[i + 2])
        }
        return r
    }

    /// Cancel: stops hearing and speaking at once. The turn's receipt keeps what happened.
    func cancel(reason: String = "cancel") async {
        let (listener, source, turn, line, lineSource) = lock.withLock { () -> (Hearing?, Source?, Int, Line?, Source?) in
            let out = (self.listener, self.source, _turn, self.line, self.lineSource)
            self.listener = nil; self.source = nil; self.line = nil; self.lineSource = nil; self.synthetic = nil
            return out
        }
        // The hands-free line goes with everything else: mic off, recognizer dropped.
        if line != nil {
            await lineSource?.stop()
            for task in lock.withLock({ Array(hearings.values) }) { task.cancel(); if let h = try? await task.value { await h.cancel() } }
            lock.withLock { hearings = [:]; finals = [:] }
        }
        let wasBusy = (phase != .idle && phase != .preparing) || line != nil
        await source?.stop()
        await listener?.cancel()
        speaker.stop()
        if wasBusy {
            update(turn) { $0.cancelled = true }
            Self.log("turn \(turn) cancelled (\(reason))")
        }
        set(.idle)
    }
}

struct VoiceError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

// MARK: - hearing

/// One utterance's recognizer: audio in (any PCM format), partials out, the final text at the end.
protocol Hearing: AnyObject, Sendable {
    var format: AVAudioFormat { get }
    func feed(_ buffer: AVAudioPCMBuffer)
    func finish() async throws -> String
    func cancel() async
}

/// One SpeechAnalyzer session over one utterance. Volatile (partial) results go to `onPartial` as they come; `finish`
/// finalizes through the end of the input and returns the final text.
final class Listener: Hearing, @unchecked Sendable {
    let format: AVAudioFormat
    private let analyzer: SpeechAnalyzer
    private let input: AsyncStream<AnalyzerInput>.Continuation
    private var reader: Task<Void, Error>?
    private let lock = NSLock()
    private var finals = ""
    private var converter: AVAudioConverter?

    static func transcriber() -> SpeechTranscriber {
        SpeechTranscriber(locale: Voice.locale, transcriptionOptions: [], reportingOptions: [.volatileResults, .fastResults], attributeOptions: [])
    }

    init(onPartial: @escaping @Sendable (String) -> Void) async throws {
        let transcriber = Self.transcriber()
        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
            throw VoiceError("no audio format for the transcriber (is the en-US speech model installed?)")
        }
        self.format = format
        // The model stays loaded for the life of the process: a turn's analyzer is ~3 ms once the first one ran.
        analyzer = SpeechAnalyzer(modules: [transcriber], options: SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .processLifetime))
        let (stream, input) = AsyncStream<AnalyzerInput>.makeStream()
        self.input = input
        try await analyzer.prepareToAnalyze(in: format)
        reader = Task { [lock] in
            var finals = ""
            for try await result in transcriber.results {
                let text = String(result.text.characters)
                if result.isFinal {
                    finals += text
                    lock.withLock { self.finals = finals }
                    onPartial(finals.trimmingCharacters(in: .whitespaces))
                } else {
                    onPartial((finals + text).trimmingCharacters(in: .whitespaces))
                }
            }
        }
        try await analyzer.start(inputSequence: stream)
    }

    /// Any PCM buffer: converted to the analyzer's format here (the mic is 48 kHz float; the analyzer wants 16 kHz Int16).
    func feed(_ buffer: AVAudioPCMBuffer) {
        if buffer.format == format { input.yield(AnalyzerInput(buffer: buffer)); return }
        let converter = lock.withLock { () -> AVAudioConverter? in
            if self.converter?.inputFormat != buffer.format { self.converter = AVAudioConverter(from: buffer.format, to: format) }
            return self.converter
        }
        guard let converter else { return }
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * format.sampleRate / buffer.format.sampleRate + 64)
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return }
        var given = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if given { status.pointee = .noDataNow; return nil }
            given = true
            status.pointee = .haveData
            return buffer
        }
        if error == nil, out.frameLength > 0 { input.yield(AnalyzerInput(buffer: out)) }
    }

    func finish() async throws -> String {
        input.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        _ = try? await reader?.value
        return lock.withLock { finals }.trimmingCharacters(in: .whitespaces)
    }

    func cancel() async {
        input.finish()
        await analyzer.cancelAndFinishNow()
        reader?.cancel()
    }
}

/// The fallback: SFSpeechRecognizer with `requiresOnDeviceRecognition` (it never sends audio to a server: with no
/// on-device model it fails instead). Needs the speech-recognition permission (asked once) and, on macOS, Siri &
/// Dictation switched on; SpeechAnalyzer needs neither, which is why it is the first choice.
final class LegacyListener: Hearing, @unchecked Sendable {
    let format: AVAudioFormat
    private let request = SFSpeechAudioBufferRecognitionRequest()
    private var task: SFSpeechRecognitionTask?
    private let lock = NSLock()
    private var text = ""
    private var done: CheckedContinuation<String, Error>?
    private var result: Result<String, Error>?

    init(ask: Bool, onPartial: @escaping @Sendable (String) -> Void) async throws {
        guard let recognizer = SFSpeechRecognizer(locale: Voice.locale), recognizer.supportsOnDeviceRecognition else {
            throw VoiceError("no on-device speech recognizer for \(Voice.locale.identifier) on this device")
        }
        if SFSpeechRecognizer.authorizationStatus() == .notDetermined, ask {
            _ = await withCheckedContinuation { (c: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
                SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0) }
            }
        }
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            throw VoiceError("speech recognition is not allowed: allow it in Settings › Eureka")
        }
        format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        task = recognizer.recognitionTask(with: request) { [weak self] res, err in
            guard let self else { return }
            if let res {
                let t = res.bestTranscription.formattedString
                self.lock.withLock { self.text = t }
                onPartial(t)
                if res.isFinal { self.settle(.success(t)) }
            } else if let err {
                self.settle(.failure(err))
            }
        }
    }

    private func settle(_ r: Result<String, Error>) {
        let c = lock.withLock { () -> CheckedContinuation<String, Error>? in
            if result != nil { return nil }
            result = r
            defer { done = nil }
            return done
        }
        c?.resume(with: r)
    }

    func feed(_ buffer: AVAudioPCMBuffer) { request.append(buffer) }

    func finish() async throws -> String {
        request.endAudio()
        return try await withCheckedThrowingContinuation { c in
            let ready = lock.withLock { () -> Result<String, Error>? in
                if let result { return result }
                done = c
                return nil
            }
            if let ready { c.resume(with: ready) }
        }
    }

    func cancel() async {
        task?.cancel()
        settle(.success(lock.withLock { text }))
    }
}

/// -voiceCheck on a device with no recognizer at all (the iOS simulator): the clip's known words (`<clip>.txt` next
/// to the clip) stand in for the transcript. The audio is still fed at real-time pace, so speech end is where it would
/// be; the receipt's `heardBy` is `fixture-text`, never a recognizer's name. Never used outside -voiceCheck.
final class KnownText: Hearing, @unchecked Sendable {
    static let label = "fixture-text (no STT in this simulator)"
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
    private let text: String
    init(_ clip: URL) throws {
        text = try String(contentsOf: clip.deletingPathExtension().appendingPathExtension("txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    func feed(_ buffer: AVAudioPCMBuffer) {}
    func finish() async throws -> String { text }
    func cancel() async {}
}

/// Where a turn's audio comes from.
protocol Source: Sendable {
    func start()
    /// Returns once no more audio will be fed.
    func stop() async
}

/// A clip fed at real-time pace in 100 ms buffers: what the microphone tap delivers, without a microphone. The self-test
/// (`-voiceCheck`, scripts/voice-measure.ts) uses it; release waits for the clip's last buffer, then speech has ended.
final class FixtureSource: Source, @unchecked Sendable {
    private let buffers: [AVAudioPCMBuffer]
    private let listener: Hearing
    private var task: Task<Void, Never>?

    init(_ url: URL, into listener: Hearing) throws {
        let file = try AVAudioFile(forReading: url)
        let step = AVAudioFrameCount(file.processingFormat.sampleRate / 10)
        var buffers: [AVAudioPCMBuffer] = []
        // G38: 100 ms reads until the end. One read(into:) of the whole length comes back short on macOS 27 (the EARS
        // clips lost their last 263–791 frames), so a clip's tail never reached the recognizer.
        while file.framePosition < file.length {
            let b = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: step)!
            try file.read(into: b, frameCount: step)
            if b.frameLength == 0 { break }
            buffers.append(b)
        }
        self.buffers = buffers
        self.listener = listener
    }

    func start() {
        let buffers = buffers, listener = listener
        task = Task.detached(priority: .userInitiated) {
            let t0 = Voice.now()
            var fed = 0.0
            for b in buffers {
                if Task.isCancelled { return }
                listener.feed(b)
                fed += Double(b.frameLength) / b.format.sampleRate * 1000
                let wait = t0 + fed - Voice.now()
                if wait > 0 { try? await Task.sleep(nanoseconds: UInt64(wait * 1e6)) }
            }
        }
    }

    func stop() async { await task?.value }
}

/// The microphone: an input-node tap on its own AVAudioEngine. Hands-free: `voiceProcessing` turns on the system's echo
/// cancellation for the input (iOS), so the reply playing out of the speaker is taken out of what the tap hears.
final class MicSource: Source, @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let sink: @Sendable (AVAudioPCMBuffer) -> Void
    private let voiceProcessing: Bool

    init(into listener: Hearing) throws {
        sink = { listener.feed($0) }
        voiceProcessing = false
        try Self.check()
    }

    init(voiceProcessing: Bool, sink: @escaping @Sendable (AVAudioPCMBuffer) -> Void) throws {
        self.sink = sink
        self.voiceProcessing = voiceProcessing
        try Self.check()
    }

    private static func check() throws {
        #if os(iOS)
        guard AVAudioApplication.shared.recordPermission == .granted else { throw VoiceError("microphone permission: \(AVAudioApplication.shared.recordPermission)") }
        #else
        guard Voice.micState == "granted" else { throw VoiceError("microphone permission: \(Voice.micState)") }
        #endif
    }

    func start() {
        #if os(iOS)
        if voiceProcessing {
            do { try engine.inputNode.setVoiceProcessingEnabled(true) } catch { Voice.log("echo cancellation unavailable: \(error)") }
        }
        #endif
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        #if os(iOS)
        let aec = voiceProcessing
        #else
        let aec = false
        #endif
        Voice.log("mic format \(format.sampleRate) Hz \(format.channelCount) ch echo_cancellation=\(aec)")
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [sink] buffer, _ in sink(buffer) }
        do { try engine.start() } catch { Voice.log("mic failed to start: \(error)") }
    }

    func stop() async {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}

/// A microphone that hears a quiet room (pink noise at -62 dBFS) at real-time pace in 20 ms buffers, 48 kHz mono float:
/// the shape the iPhone tap delivers. `inject` plays a clip into it at the next buffer, as if the person began talking
/// then. Used by the Mac harness and the simulator (no microphone there); never on a device with a real one.
final class SyntheticSource: Source, @unchecked Sendable {
    static let rate = 48_000.0
    private let sink: @Sendable (AVAudioPCMBuffer) -> Void
    private let lock = NSLock()
    private var queue: [Float] = []
    private var task: Task<Void, Never>?
    private(set) var lastClip: URL?
    private var rng: UInt64 = 0x9E3779B97F4A7C15

    init(sink: @escaping @Sendable (AVAudioPCMBuffer) -> Void) { self.sink = sink }

    func inject(_ url: URL) throws {
        let file = try AVAudioFile(forReading: url)
        let all = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: all)
        guard file.processingFormat.sampleRate == Self.rate else { throw VoiceError("fixture \(url.lastPathComponent) is not 48 kHz") }
        let samples = Array(UnsafeBufferPointer(start: all.floatChannelData![0], count: Int(all.frameLength)))
        lock.withLock { queue.append(contentsOf: samples); lastClip = url }
    }

    func start() {
        // BEND_VOICE_SYNTH_CHANNELS=2: the same audio on two channels, as a loopback device or a voice-processed mic delivers it.
        let channels = AVAudioChannelCount(ProcessInfo.processInfo.environment["BEND_VOICE_SYNTH_CHANNELS"].flatMap { UInt32($0) } ?? 1)
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Self.rate, channels: channels, interleaved: false)!
        task = Task.detached(priority: .userInitiated) { [self] in
            let frames = Int(Self.rate / 50)
            let t0 = Voice.now()
            var n = 0.0
            var a = 0.0, b = 0.0
            let gain = pow(10, -62.0 / 20) * 3.2
            while !Task.isCancelled {
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))!
                buffer.frameLength = AVAudioFrameCount(frames)
                let out = buffer.floatChannelData![0]
                let clip = lock.withLock { () -> [Float] in
                    let k = min(frames, queue.count)
                    let head = Array(queue.prefix(k))
                    queue.removeFirst(k)
                    return head
                }
                for i in 0..<frames {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    let w = Double(rng >> 11) / Double(1 << 53) * 2 - 1
                    a = 0.97 * a + w; b = 0.6 * b + w
                    out[i] = Float((a * 0.25 + b * 0.5) * gain) + (i < clip.count ? clip[i] : 0)
                }
                for c in 1..<Int(channels) { memcpy(buffer.floatChannelData![c], out, frames * 4) }
                sink(buffer)
                n += 20
                let wait = t0 + n - Voice.now()
                if wait > 0 { try? await Task.sleep(nanoseconds: UInt64(wait * 1e6)) }
            }
        }
    }

    func stop() async { task?.cancel(); await task?.value }
}

// MARK: - speaking

/// The reply's text → PCM → an AVAudioPlayerNode, chunk by chunk as the reply streams in, and the same PCM into
/// `<Caches>/voice/turn-N.caf` (the receipt that audio was produced). Chunks render one at a time, in order.
///
/// VOICEREAL: two voices. The phone's own neural voice (NeuralVoice: Supertonic 2 F1 on onnxruntime, ~0.2 s a clause,
/// rendered on its own queue) speaks when its files are present and it loads; Apple's AVSpeechSynthesizer (`write`, the
/// best en-US voice the device has, e.g. Samantha) is the fallback: no voice files, a load failure (never a memory warning: VOICESHOP2),
/// a chunk the neural voice fails on, or the setting `BendVoiceEngine=apple` (UserDefaults: `-BendVoiceEngine apple` at
/// launch, or BEND_VOICE_ENGINE=apple on the Mac harness). A fallback chunk inside a neural turn is converted to the
/// turn's format, so one turn is one stream and one .caf.
final class Speaker: @unchecked Sendable {
    enum Event { case firstAudio(Double), file(URL, Int, Double), voice(String), timing(Timing) }
    /// VOICESHOP2: one chunk's kitchen timer (Voice.now ms): queued by say(), render start/end, handed to the player.
    struct Timing: Sendable {
        var queued: Double, synthStart: Double, synthEnd: Double, scheduled: Double
        var outputLatencyMs: Double, words: Int, audioMs: Double, runtime: String
    }

    private let synth = AVSpeechSynthesizer()
    let voice: AVSpeechSynthesisVoice
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var connected: AVAudioFormat?
    private let lock = NSLock()
    private var pending = 0            // utterances not fully rendered
    private var scheduled = 0          // buffers handed to the player, not yet played out
    private var file: AVAudioFile?
    private var fileTurn = 0
    private var frames = 0
    private var drained: [CheckedContinuation<Void, Never>] = []
    private var generation = 0
    private var recentDb: [Double] = []   // the level of the last buffers handed to the player (hands-free barge-in gate)
    // VOICESHADE: the reply's loudness, 20 ms frames, laid on the playback clock as buffers are scheduled (the player
    // plays them back to back), so `outLevel` reads the frame playing now. Works the same when -voiceSilent paces instead.
    private var envelope: [(at: Double, db: [Double])] = []
    private var envelopeEnd = 0.0
    private static let envelopeFrameMs = 20.0

    /// What the speaker is playing right now (dBFS), or −120 when nothing is.
    var outLevel: Double {
        lock.withLock {
            let now = Voice.now()
            guard scheduled > 0, let span = envelope.last(where: { $0.at <= now }) else { return -120 }
            let i = Int((now - span.at) / Self.envelopeFrameMs)
            return i < span.db.count ? span.db[i] : -120
        }
    }

    /// Lays one buffer's 20 ms loudness frames after what is already scheduled (starting no earlier than `start`).
    private func layEnvelope(_ pcm: AVAudioPCMBuffer, start: Double) {
        guard let ch = pcm.floatChannelData?[0], pcm.frameLength > 0 else { return }
        let step = max(1, Int(pcm.format.sampleRate * Self.envelopeFrameMs / 1000))
        let n = Int(pcm.frameLength)
        var db: [Double] = []
        var at = 0
        while at < n {
            db.append(Endpointer.measure(UnsafeBufferPointer(start: ch + at, count: min(step, n - at))).db)
            at += step
        }
        lock.withLock {
            let begin = max(envelopeEnd, start)
            envelope.append((begin, db))
            if envelope.count > 64 { envelope.removeFirst() }
            envelopeEnd = begin + Double(n) / pcm.format.sampleRate * 1000
        }
    }

    /// What the speaker is playing right now, in dBFS (the loudest of the last buffers), or nil when it is not playing.
    /// The line's barge-in gate: the mic hears the reply at about this level minus the phone's coupling loss.
    var referenceDb: Double? {
        lock.withLock { scheduled > 0 ? recentDb.max() : nil }
    }

    /// VOICESHOP: the voice that speaks: any engine's (VoiceLibrary: Kokoro's voices, Supertonic F1, packs from the shop
    /// or dropped into Documents/voices), and the pack it came from.
    private var neural: (any VoiceEngine)?
    private var neuralPack: VoicePack?
    var packID: String? { lock.withLock { neuralPack?.id } }
    /// Why the neural voice is not speaking (nil while it is).
    private var fallbackReason: String?
    var fallback: String? { lock.withLock { fallbackReason } }
    private let renderQueue = DispatchQueue(label: "bend.voice.render", qos: .userInitiated)

    /// "neural" (default) or "apple". VOICESHOP: picking "System voice (Apple)" in the picker is the same setting.
    static var setting: String {
        if VoiceLibrary.shared.choice == appleID { return "apple" }
        return ProcessInfo.processInfo.environment["BEND_VOICE_ENGINE"] ?? UserDefaults.standard.string(forKey: "BendVoiceEngine") ?? "neural"
    }
    static let appleID = "apple:system"
    /// VOICESHOP: posted after every select (picker tap, shop "Use this voice", POST /voice/select).
    static let voiceChanged = Notification.Name("BendVoiceChanged")

    init() {
        let voices = AVSpeechSynthesisVoice.speechVoices().filter { $0.language == Voice.locale.identifier }
        // The best en-US voice the device has (premium > enhanced > the default compact one).
        voice = voices.max { $0.quality.rawValue < $1.quality.rawValue } ?? AVSpeechSynthesisVoice(language: Voice.locale.identifier)!
        engine.attach(player)
        KokoroRuntime.log = { Voice.log($0) }
        if Voice.checking && !Voice.args.contains("-voiceAudible") { player.volume = 0 }
        #if os(iOS)
        // VOICESHOP2: the house voice is never swapped for Apple's mid-room (memoryWarning), and leaves only with the app.
        let center = NotificationCenter.default
        center.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: nil) { [weak self] _ in self?.memoryWarning() }
        center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { [weak self] _ in self?.away() }
        center.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { [weak self] _ in self?.back() }
        #endif
        // A route or format change (headphones in/out, the mic's engine starting on the same session) stops this engine
        // and its player: start them again, so a chunk scheduled next is heard and a turn's drain can't hang.
        NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { [weak self] _ in
            guard let self else { return }
            let restarted = self.lock.withLock { () -> String in
                guard self.connected != nil, !self.engine.isRunning else { return "running" }
                do { try self.engine.start(); self.player.play(); return "restarted" } catch { return "failed: \(error.localizedDescription)" }
            }
            #if os(iOS)
            Voice.speakerRoute()
            Voice.log("speaker: output changed (\(restarted)) \(Voice.output())")
            #else
            Voice.log("speaker: output changed (\(restarted))")
            #endif
        }
    }

    // MARK: VOICESHOP2: memory policy: keep the house voice resident

    /// Below this much headroom (os_proc_available_memory) a memory warning is critical. The neural voice is ~100–300 MB
    /// resident; above the line, releasing it would buy nothing jetsam needs, and cost the person their voice.
    static let criticalMB = 300
    /// Released under critical memory or by the away grace: the pack that comes back (re-open), and why it left.
    private var released: (pack: VoicePack, why: String, name: String)?
    private var reopening = false
    private var awayTimer: DispatchWorkItem?
    /// The engine's own away grace (EngineLifecycle.Policy.grace): the voice and the chat engine leave together.
    static let awayGrace: Double = 90

    static func availableMB() -> Int {
        #if os(iOS)
        return Int(os_proc_available_memory() / 1_048_576)
        #else
        return -1
        #endif
    }

    /// The OS's memory warning. Not critical (the usual case: the 4B chat model resident beside the voice): keep the
    /// voice, say so. Critical: let the current playback finish, release, and re-open before the next turn, so the
    /// next words are still the house voice; Apple's voice is never the stand-in.
    func memoryWarning() {
        let avail = Self.availableMB()
        // 0 = no limit known (the simulator, the Mac): not critical by itself.
        let critical = avail > 0 && avail < Self.criticalMB || Voice.args.contains("-voiceMemoryCritical")
        guard let pack = lock.withLock({ neuralPack }) else {
            Voice.log("speaker: memory warning available_mb=\(avail): no neural voice loaded, nothing to do")
            return
        }
        guard critical else {
            Voice.log("speaker: memory warning available_mb=\(avail) policy=keep-resident: \(pack.name) stays (critical below \(Self.criticalMB) MB)")
            return
        }
        Voice.log("speaker: memory warning available_mb=\(avail) policy=critical: \(pack.name) finishes this playback, then releases and re-opens before the next turn")
        Task {
            await drainAll()
            releaseNeural(why: "memory critical")
            // Eager: the memory went back while nothing was speaking; take it again now, before anyone needs it.
            try? await Task.sleep(nanoseconds: 500_000_000)
            reopenIfReleased(why: "after critical memory")
        }
    }

    /// Waits until nothing is rendering or playing (any turn).
    private func drainAll() async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            let now = lock.withLock { () -> Bool in
                if pending == 0 && scheduled == 0 { return true }
                drained.append(c)
                return false
            }
            if now { c.resume() }
        }
    }

    /// Drops the neural voice and its engine's shared model; it comes back with `reopenIfReleased`, not as Apple's voice.
    private func releaseNeural(why: String) {
        renderQueue.async {
            let pack = self.lock.withLock { () -> VoicePack? in
                guard let p = self.neuralPack else { return nil }
                self.released = (p, why, self.neural?.name ?? p.name)
                self.neural = nil; self.neuralPack = nil
                return p
            }
            guard let pack else { return }
            VoiceLibrary.shared.adapter(pack.engine)?.release()
            Voice.log("speaker: \(pack.name) released (\(why)) available_mb=\(Self.availableMB()); it re-opens before the next words")
        }
    }

    /// The released voice loads again. Runs on the render queue, so any chunk queued after it waits for it.
    func reopenIfReleased(why: String) {
        // One re-open at a time (the next turn and the eager re-open may both ask).
        guard let (pack, was, _) = lock.withLock({ () -> (VoicePack, String, String)? in
            guard let r = released, !reopening else { return nil }
            reopening = true
            return r
        }) else { return }
        Task {
            let t0 = Voice.now()
            defer { lock.withLock { reopening = false } }
            do {
                try await load(pack)
                lock.withLock { released = nil }
                Voice.log(String(format: "speaker: %@ re-opened (%@; it left for: %@) ms=%.0f available_mb=%d", pack.name, why, was, Voice.now() - t0, Self.availableMB()))
            } catch {
                Voice.log("speaker: \(pack.name) failed to re-open (\(error.localizedDescription)); the next prepare tries again")
            }
        }
    }

    /// Main queue. The app left the screen: the voice goes after the same grace as the chat engine, unless back.
    private func away() {
        awayTimer?.cancel()
        let item = DispatchWorkItem { [weak self] in self?.releaseNeural(why: "away \(Int(Self.awayGrace)) s") }
        awayTimer = item
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.awayGrace, execute: item)
        Voice.log("speaker: away: release in \(Int(Self.awayGrace)) s unless back")
    }

    /// Main queue. Back on screen: keep it, or (released while away) open it again now, before the first press.
    private func back() {
        awayTimer?.cancel()
        awayTimer = nil
        reopenIfReleased(why: "back on screen")
    }

    /// The voice that speaks now: "Kokoro Heart", "Supertonic F1" or Apple's (e.g. "Samantha").
    var name: String { lock.withLock { neural?.name ?? released?.name } ?? voice.name }
    var engineKind: String { lock.withLock { neural != nil || released != nil } ? "neural" : "apple" }

    static var dir: URL {
        let base = ProcessInfo.processInfo.environment["BEND_CACHE_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("voice")
    }

    /// The surface opened: load the neural voice (~0.2 s) and render one throw-away clause so onnxruntime's first run is
    /// paid now, not in a turn. Without it (no files, a load error, the setting), Apple's synthesizer is warmed instead.
    func warm() async -> [String: Any] {
        var out: [String: Any] = [:]
        let t0 = Voice.now()
        if Self.setting == "apple" {
            lock.withLock { neural = nil; neuralPack = nil; fallbackReason = VoiceLibrary.shared.choice == Self.appleID ? "picked: System voice" : "setting BendVoiceEngine=apple" }
        } else if lock.withLock({ neural == nil }) {
            // VOICESHOP: the picked voice, else the default, else any pack that loads (VoiceLibrary.candidates).
            let candidates = VoiceLibrary.shared.candidates()
            lock.withLock { released = nil }
            var why = candidates.isEmpty ? "no neural voice files (scripts/fetch-voice.sh)" : ""
            for pack in candidates {
                do {
                    try await load(pack)
                    why = ""
                    break
                } catch {
                    why = "\(pack.name) failed to load: \(error.localizedDescription)"
                    Voice.log("speaker: \(why)")
                }
            }
            if !why.isEmpty { lock.withLock { fallbackReason = why } }
        }
        out["voiceID"] = packID ?? (Self.setting == "apple" ? Self.appleID : NSNull())
        out["voice"] = name
        out["voiceEngine"] = engineKind
        out["neuralMs"] = (Voice.now() - t0).rounded()
        if let why = lock.withLock({ fallbackReason }) {
            out["voiceFallback"] = why
            Voice.log("speaker: Apple's voice (\(voice.name)) speaks: \(why)")
            await warmApple()
        }
        return out
    }

    /// One throw-away render: the synthesizer loads its voice (the cold ~0.4 s) now rather than in a turn.
    private func warmApple() async {
        let u = AVSpeechUtterance(string: "Ready.")
        u.voice = voice
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            var done = false
            synth.write(u) { buffer in
                guard (buffer as? AVAudioPCMBuffer)?.frameLength == 0, !done else { return }
                done = true
                c.resume()
            }
        }
    }

    // MARK: VOICESHOP: picking a voice

    /// Loads `pack` on the render queue (so it never races a chunk) and renders a throw-away word, then makes it the voice.
    /// Chunks already queued keep their order; the next one renders in the new voice: a swap needs no restart.
    @discardableResult
    private func load(_ pack: VoicePack) async throws -> Double {
        let t0 = Voice.now()
        let engine: any VoiceEngine = try await withCheckedThrowingContinuation { c in
            renderQueue.async {
                c.resume(with: Result {
                    guard let adapter = VoiceLibrary.shared.adapter(pack.engine) else { throw NeuralVoiceError("no \(pack.engine) engine in this build") }
                    let v = try adapter.load(pack)
                    _ = try v.synthesize("Ready.")
                    return v
                })
            }
        }
        let previous = lock.withLock { () -> String? in
            defer { neural = engine; neuralPack = pack; fallbackReason = nil; released = nil }
            return neuralPack?.engine
        }
        // Another engine's shared model is no longer needed (Kokoro's 92 MB when Supertonic takes over, and back).
        if let previous, previous != pack.engine { VoiceLibrary.shared.adapter(previous)?.release() }
        return Voice.now() - t0
    }

    /// The picker's tap: speak with `id` from the next chunk on, and remember it. `apple:system` = Apple's voice.
    func select(_ id: String) async throws -> [String: Any] {
        let t0 = Voice.now()
        if id == Self.appleID {
            VoiceLibrary.shared.choice = id
            lock.withLock { neural = nil; neuralPack = nil; fallbackReason = "picked: System voice" }
            await warmApple()
        } else {
            guard let pack = VoiceLibrary.shared.packs(refresh: true).first(where: { $0.id == id }) else { throw NeuralVoiceError("no voice \(id)") }
            try await load(pack)
            VoiceLibrary.shared.choice = id
        }
        let ms = (Voice.now() - t0).rounded()
        Voice.log("speaker: select \(id) → \(name) ms=\(ms)")
        NotificationCenter.default.post(name: Speaker.voiceChanged, object: nil)  // the picker and shop show it, whoever asked
        return ["voice": name, "voiceID": packID ?? id, "voiceEngine": engineKind, "ms": ms]
    }

    /// The demo line in `id`'s voice, without changing the choice: rendered once per voice (Caches/voice/previews), then
    /// played. Returns the file (the receipts keep it) and the render time (0 when it came from the cache).
    func preview(_ id: String, play: Bool = true) async throws -> [String: Any] {
        let dir = Self.dir.appendingPathComponent("previews")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("\(id.replacingOccurrences(of: ":", with: "_")).caf")
        var renderMs = 0.0, audioMs = 0.0
        if id == Self.appleID {
            throw NeuralVoiceError("the system voice has no preview here")
        }
        if !FileManager.default.fileExists(atPath: url.path) {
            guard let pack = VoiceLibrary.shared.packs(refresh: true).first(where: { $0.id == id }), let adapter = VoiceLibrary.shared.adapter(pack.engine) else {
                throw NeuralVoiceError("no voice \(id)")
            }
            let t0 = Voice.now()
            let (samples, rate): ([Float], Int) = try await withCheckedThrowingContinuation { c in
                renderQueue.async {
                    c.resume(with: Result {
                        let v = try adapter.load(pack)
                        return (try v.synthesize(VoiceLibrary.demoLine), v.sampleRate)
                    })
                }
            }
            renderMs = Voice.now() - t0
            audioMs = Double(samples.count) / Double(rate) * 1000
            let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(rate), channels: 1, interleaved: false)!
            guard let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { throw NeuralVoiceError("empty preview") }
            pcm.frameLength = pcm.frameCapacity
            samples.withUnsafeBufferPointer { pcm.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
            let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
            try file.write(from: pcm)
        }
        if play, !Voice.silent { playPreview(url) }
        Voice.log(String(format: "speaker: preview %@ render_ms=%.0f audio_ms=%.0f file=%@", id, renderMs, audioMs, url.lastPathComponent))
        return ["voiceID": id, "file": url.path, "renderMs": renderMs.rounded(), "audioMs": audioMs.rounded()]
    }

    private var previewPlayer: AVAudioPlayer?
    func playPreview(_ url: URL) {
        #if os(iOS)
        // VOICESHOP2: Settings may play a preview before the voice surface ever set the session: iOS's default
        // (.soloAmbient) is muted by the Silent switch. .playback isn't; the surface's .playAndRecord isn't either.
        let s = AVAudioSession.sharedInstance()
        if s.category != .playAndRecord && s.category != .playback {
            try? s.setCategory(.playback, mode: .spokenAudio)
            Voice.log("speaker: preview output \(Voice.output())")
        }
        try? s.setActive(true)
        #endif
        DispatchQueue.main.async {
            self.previewPlayer?.stop()
            self.previewPlayer = try? AVAudioPlayer(contentsOf: url)
            self.previewPlayer?.play()
        }
    }

    func stopPreview() { DispatchQueue.main.async { self.previewPlayer?.stop() } }

    /// G28: speech-end prewarm (`-voicePrewarm`, UserDefaults BendVoicePrewarm, BEND_VOICE_PREWARM=1; off by default).
    /// The person stopped: while the transcript is finalized and the engine prefills (prep_ms, 0.9–1.5 s on the phone)
    /// the render queue is idle, so one throw-away render of a first-chunk-sized clause runs there now: the voice's caches,
    /// the cores' clocks and (with the ANE bucket) the Neural Engine are hot when the reply's first words land. On the
    /// render queue, so it can only delay a first chunk that arrives while it runs (queue_ms in the turn's timer).
    static var prewarmOn: Bool {
        Voice.args.contains("-voicePrewarm") || ProcessInfo.processInfo.environment["BEND_VOICE_PREWARM"] == "1" || UserDefaults.standard.bool(forKey: "BendVoicePrewarm")
    }
    static let prewarmText = "Sure, here is one."
    func prewarm(turn: Int) {
        guard Self.prewarmOn else { return }
        renderQueue.async {
            guard let v = self.lock.withLock({ self.neural }) else { return }
            let t0 = Voice.now()
            _ = try? v.synthesize(Self.prewarmText)
            Voice.log(String(format: "speaker: turn %d prewarm render_ms=%.0f runtime=%@", turn, Voice.now() - t0, (v as? KokoroVoice)?.runtime.label ?? "-"))
        }
    }

    private struct Chunk { let text: String; let turn: Int; let event: @Sendable (Event) -> Void; var queued = Voice.now() }
    private var queue: [Chunk] = []
    private var rendering = false

    /// Queues one chunk. Chunks render one at a time, in order, on the main queue: two `write`s in flight at once lose
    /// the second one's audio (seen on the Mac harness). Rendering runs well ahead of playback (warm: tens of ms per
    /// sentence against seconds of audio), so the player never waits on the next chunk.
    ///
    /// VOICESHOP2: early speak. A turn's first chunk is cut again here (`earlySplit`: ≤6 words, then ≤14, then the
    /// rest), the way the engine streams tokens: the first few words render (~0.3–0.6 s) and play while the rest of the
    /// sentence renders behind them, instead of the whole sentence rendering before a sound.
    func say(_ text: String, turn: Int, on event: @escaping @Sendable (Event) -> Void) {
        let pieces = lock.withLock { () -> [String] in
            let first = fileTurn != turn
            if fileTurn != turn { file = nil; fileTurn = turn; frames = 0 }
            let pieces = first && !Voice.args.contains("-voiceNoEarlySplit") ? Self.earlySplit(text) : [text]
            pending += pieces.count
            queue += pieces.map { Chunk(text: $0, turn: turn, event: event) }
            return pieces
        }
        if pieces.count > 1 { Voice.log("speaker: turn \(turn) early speak: \(pieces.map { "\(Self.words($0))w" }.joined(separator: " + "))") }
        DispatchQueue.main.async { self.next() }
    }

    static func words(_ s: String) -> Int { s.split(whereSeparator: { $0 == " " || $0 == "\n" }).count }

    /// The first chunk of a turn in growing pieces: at most `caps` words each (the last takes the rest), cut at the
    /// last clause mark (, ; : —) inside the cap if there is one with ≥3 words before it, else at the cap's word end.
    /// A chunk already short stays whole.
    static func earlySplit(_ text: String, caps: [Int] = [4, 12]) -> [String] {
        var rest = text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        var out: [String] = []
        for cap in caps where rest.count > cap + 2 {  // a 1–2 word tail is not worth its own render
            var cut = cap
            if let mark = (3 ... cap).last(where: { i in rest[i - 1].last.map { ",;:—–".contains($0) } ?? false }) { cut = mark }
            out.append(rest[..<cut].joined(separator: " "))
            rest.removeFirst(cut)
        }
        if !rest.isEmpty { out.append(rest.joined(separator: " ")) }
        return out
    }

    /// Main queue only.
    private func next() {
        let (chunk, gen, neural) = lock.withLock { () -> (Chunk?, Int, Bool) in
            guard !rendering, !queue.isEmpty else { return (nil, generation, false) }
            rendering = true
            return (queue.removeFirst(), generation, self.neural != nil || released != nil)
        }
        guard let chunk else { return }
        if neural {
            // The voice is read on the render queue, at render time: a re-open queued there before this chunk is done
            // by then (VOICESHOP2: a voice that is re-opening is waited for, not replaced by Apple's).
            renderQueue.async {
                if let v = self.lock.withLock({ self.neural }) {
                    self.render(chunk, gen: gen, with: v)
                } else {
                    Voice.log("speaker: the neural voice did not come back (\(self.lock.withLock { self.released?.why ?? "?" })); Apple's voice says this chunk")
                    DispatchQueue.main.async { self.renderApple(chunk, gen: gen) }
                }
            }
        } else {
            renderApple(chunk, gen: gen)
        }
    }

    /// The neural voice, off the main queue: one chunk → one buffer (a clause renders whole: Supertonic ~0.2 s, Kokoro ~RTF 0.4).
    private func render(_ chunk: Chunk, gen: Int, with neural: any VoiceEngine) {
        let samples: [Float]
        let synthStart = Voice.now()
        do {
            samples = try neural.synthesize(chunk.text)
        } catch {
            Voice.log("speaker: neural voice failed on a chunk (\(error.localizedDescription)); Apple's voice says it")
            DispatchQueue.main.async { self.renderApple(chunk, gen: gen) }
            return
        }
        if lock.withLock({ generation != gen }) { return } // stopped since
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(neural.sampleRate), channels: 1, interleaved: false)!
        if !samples.isEmpty, let raw = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) {
            raw.frameLength = raw.frameCapacity
            samples.withUnsafeBufferPointer { raw.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
            let target = lock.withLock { fileTurn == chunk.turn ? file?.processingFormat : nil }
            let pcm = target.flatMap { Self.convert(raw, to: $0) } ?? raw
            let synthEnd = Voice.now()
            keep(pcm, turn: chunk.turn, on: chunk.event)
            if !Voice.silent { play(pcm) }
            let scheduled = Voice.now()
            chunk.event(.firstAudio(scheduled))
            chunk.event(.voice(neural.name))
            chunk.event(.timing(Timing(queued: chunk.queued, synthStart: synthStart, synthEnd: synthEnd, scheduled: scheduled, outputLatencyMs: Self.outputLatencyMs,
                                       words: Self.words(chunk.text), audioMs: Double(samples.count) / Double(neural.sampleRate) * 1000,
                                       runtime: (neural as? KokoroVoice)?.runtime.label ?? "onnx-cpu")))
        }
        lock.withLock { pending -= 1; rendering = false }
        settle()
        DispatchQueue.main.async { self.next() }
    }

    /// Apple's synthesizer (main queue): the fallback, or the voice when the neural one is off.
    private func renderApple(_ chunk: Chunk, gen: Int) {
        let u = AVSpeechUtterance(string: chunk.text)
        u.voice = voice
        u.prefersAssistiveTechnologySettings = false
        var first = true
        synth.write(u) { [weak self] buffer in
            guard let self, let pcm = buffer as? AVAudioPCMBuffer else { return }
            if self.lock.withLock({ self.generation != gen }) { return } // stopped since
            if pcm.frameLength == 0 {
                self.lock.withLock { self.pending -= 1; self.rendering = false }
                self.settle()
                DispatchQueue.main.async { self.next() }
                return
            }
            // Inside a turn the neural voice started, Apple's PCM joins in the turn's format (one stream, one .caf).
            let target = self.lock.withLock { self.fileTurn == chunk.turn ? self.file?.processingFormat : nil }
            guard let pcm = target.map({ Self.convert(pcm, to: $0) }) ?? pcm else { return }
            self.keep(pcm, turn: chunk.turn, on: chunk.event)
            if !Voice.silent { self.play(pcm) } else if Voice.args.contains("-voiceLine") { self.pace(pcm) }
            if first {
                let now = Voice.now()
                chunk.event(.firstAudio(now)); chunk.event(.voice(self.voice.name))
                chunk.event(.timing(Timing(queued: chunk.queued, synthStart: chunk.queued, synthEnd: now, scheduled: now, outputLatencyMs: Self.outputLatencyMs,
                                           words: Self.words(chunk.text), audioMs: Double(pcm.frameLength) / pcm.format.sampleRate * 1000, runtime: "apple")))
            }
            first = false
        }
    }

    /// `pcm` in `format` (Apple's 22.05 kHz Int16/Float → the neural voice's 44.1 kHz Float32); nil if it cannot be.
    static func convert(_ pcm: AVAudioPCMBuffer, to format: AVAudioFormat) -> AVAudioPCMBuffer? {
        if pcm.format == format { return pcm }
        guard let converter = AVAudioConverter(from: pcm.format, to: format),
              let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(Double(pcm.frameLength) * format.sampleRate / pcm.format.sampleRate + 64))
        else { return nil }
        var given = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if given { status.pointee = .endOfStream; return nil }
            given = true
            status.pointee = .haveData
            return pcm
        }
        return error == nil && out.frameLength > 0 ? out : nil
    }

    private func keep(_ pcm: AVAudioPCMBuffer, turn: Int, on event: @escaping @Sendable (Event) -> Void) {
        lock.withLock {
            if file == nil {
                try? FileManager.default.createDirectory(at: Self.dir, withIntermediateDirectories: true)
                let url = Self.dir.appendingPathComponent("turn-\(turn).caf")
                try? FileManager.default.removeItem(at: url)
                file = try? AVAudioFile(forWriting: url, settings: pcm.format.settings, commonFormat: pcm.format.commonFormat, interleaved: pcm.format.isInterleaved)
            }
            try? file?.write(from: pcm)
            frames += Int(pcm.frameLength)
            if let url = file?.url { event(.file(url, frames, pcm.format.sampleRate)) }
        }
    }

    private func play(_ pcm: AVAudioPCMBuffer) {
        layEnvelope(pcm, start: Voice.now() + Self.outputLatencyMs)
        if let ch = pcm.floatChannelData?[0], pcm.frameLength > 0 {
            let m = Endpointer.measure(UnsafeBufferPointer(start: ch, count: Int(pcm.frameLength)))
            lock.withLock { recentDb.append(m.db); if recentDb.count > 24 { recentDb.removeFirst() } }
        }
        lock.withLock {
            if connected != pcm.format {
                if engine.isRunning { engine.stop() }
                engine.connect(player, to: engine.mainMixerNode, format: pcm.format)
                connected = pcm.format
            }
            if !engine.isRunning {
                do { try engine.start() } catch { Voice.log("speaker: the output engine did not start (\(error.localizedDescription)): this chunk is in the .caf but not heard") }
            }
            if !player.isPlaying { player.play() }
            scheduled += 1
        }
        let gen = lock.withLock { generation }
        player.scheduleBuffer(pcm, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self else { return }
            self.lock.withLock { if self.generation == gen { self.scheduled -= 1 } }
            self.settle()
        }
    }

    /// `-voiceSilent -voiceLine` (the simulator, whose audio output cannot start): nothing is played, but the reply
    /// still takes as long to "play" as it would, so the loop sees a real speaking phase (and can be talked over).
    private var paceEnd = 0.0
    private func pace(_ pcm: AVAudioPCMBuffer) {
        layEnvelope(pcm, start: Voice.now())
        if let ch = pcm.floatChannelData?[0], pcm.frameLength > 0 {
            let m = Endpointer.measure(UnsafeBufferPointer(start: ch, count: Int(pcm.frameLength)))
            lock.withLock { recentDb.append(m.db); if recentDb.count > 24 { recentDb.removeFirst() } }
        }
        let (gen, delay) = lock.withLock { () -> (Int, Double) in
            scheduled += 1
            let now = Voice.now()
            paceEnd = max(paceEnd, now) + Double(pcm.frameLength) / pcm.format.sampleRate * 1000
            return (generation, paceEnd - now)
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + delay / 1000) { [weak self] in
            guard let self else { return }
            self.lock.withLock { if self.generation == gen { self.scheduled -= 1 } }
            self.settle()
        }
    }

    /// What the hardware adds after a buffer is handed over (output latency + one IO buffer); 0 off iOS.
    static var outputLatencyMs: Double {
        #if os(iOS)
        let s = AVAudioSession.sharedInstance()
        return (s.outputLatency + s.ioBufferDuration) * 1000
        #else
        return 0
        #endif
    }

    private func settle() {
        let waiting = lock.withLock { () -> [CheckedContinuation<Void, Never>] in
            guard pending == 0, scheduled == 0 else { return [] }
            defer { drained = [] }
            return drained
        }
        waiting.forEach { $0.resume() }
    }

    func drain(turn: Int) async {
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            let now = lock.withLock { () -> Bool in
                if pending == 0 && scheduled == 0 { return true }
                drained.append(c)
                return false
            }
            if now { c.resume() }
        }
    }

    /// Cancel: the synthesizer stops, the player drops what it holds, and anyone draining is let go.
    func stop() {
        lock.withLock { generation += 1; pending = 0; scheduled = 0; queue = []; rendering = false; recentDb = []; paceEnd = 0; envelope = []; envelopeEnd = 0 }
        synth.stopSpeaking(at: .immediate)
        player.stop()
        settle()
    }
}
