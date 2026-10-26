import Foundation
import Network

/// PHONEVOICE: voice mode's loopback listener, 127.0.0.1:8932 (docs/VOICE.md). Its own port so it serves wherever the
/// web app runs in the shell: device, simulator (where the engines are off and the demo engines answer) and the Mac
/// harness (scripts/voice-measure.sh). The web reads where it is from `window.__BEND_VOICE__` (WebAppView).
///
///   GET  /voice/status                  {available, assets, phase, partial, turn, prepared, voice, …} (public)
///   POST /voice/prepare                 warm the transcriber + synthesizer (+ output) → {ms, listenerMs, speakerMs}
///   POST /voice/install                 the person asked: the OS downloads the en-US speech model → {assets}
///   POST /voice/listen   {fixture?}     a turn starts: the mic, or a clip fed at real-time pace → {turn}
///   POST /voice/listen   {fixture?, auto?, silenceMs?}   VOICE-R2 hands-free: `auto` opens the line (one mic for the whole
///                                      conversation, auto endpointing, barge-in) and waits for the next utterance;
///                                      `fixture` with auto = the synthetic mic (no hardware: Mac harness, simulator)
///   POST /voice/inject   {fixture}      synthetic mic only: a clip is played into the line now (the test driver's voice)
///   POST /voice/endpoint                the person tapped to end the turn now (hands-free), without waiting for the silence
///   POST /voice/release  {turn}         speech ended (push-to-talk up): finalize → the receipt so far (text, finalMs)
///   POST /voice/speak    {turn, text}   one speakable chunk of the reply, queued in order → {ok}
///   POST /voice/finish   {turn}         the reply is complete: returns once its audio has played out → the receipt
///   POST /voice/cancel                  stop hearing and speaking now → {ok}
///   POST /voice/synthesize {text, file?}  MACVOICE: text → a 16-bit WAV, nothing played → {file, audioMs, voice, …}
///   POST /voice/transcribe {file}       MACVOICE: an audio file → {text, heardBy, ms}
///   POST /voice/permission              MACVOICE: ask for the microphone (system prompt, once) → {mic}
///   POST /voice/log      {line}         one web line on the console (`BEND voice web …`), for receipts
///   GET  /voice/voices                  VOICESHOP: {voices: [{id, name, engine, summary, origin, bytes, license}], choice, speaking}
///   POST /voice/select   {id}           VOICESHOP: speak with that voice from the next chunk (no restart); remembered
///   POST /voice/preview  {id, play?}    VOICESHOP: the demo line in that voice (rendered once, cached) → {file, renderMs}
///
/// POSTs need the per-launch key as `X-Api-Key` (other apps on the phone can reach loopback too). CORS open; the
/// page's origin is `null` (file://). One request per connection.
final class VoiceServer: @unchecked Sendable {
    static let shared = VoiceServer()

    let key: String = {
        if let key = ProcessInfo.processInfo.environment["BEND_API_KEY"] { return key } // Mac harness / tests
        return (0..<4).map { _ in String(format: "%08x", UInt32.random(in: .min ... .max)) }.joined()
    }()
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "bend.voice.http")
    private let voice = Voice.shared

    /// `-voiceCheck`: the web runs the three fixture turns by itself once it is up (src/voice/check.ts).
    let check = Voice.args.contains("-voiceCheck")
    /// `-voiceLine`: the web runs the hands-free fixture run by itself once voice is up (src/voice/linecheck.ts).
    let lineCheck = Voice.args.contains("-voiceLine")

    var bridgeScript: String {
        "window.__BEND_VOICE__ = Object.freeze({ url: \"http://127.0.0.1:\(Voice.port)\", key: \"\(key)\", check: \(check), line: \(lineCheck) });"
    }

    func start() {
        _ = Voice.consoleFD
        // VOICESHOP2: `-voiceBench` (the device script's step 4): the first-chunk render per Kokoro runtime, then done.
        if Voice.args.contains("-voiceBench") { Task.detached(priority: .userInitiated) { await VoiceBench.run() } }
        do {
            let parameters = NWParameters.tcp
            parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: Voice.port)!)
            parameters.allowLocalEndpointReuse = true
            let listener = try NWListener(using: parameters)
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready: Voice.log("listening on 127.0.0.1:\(Voice.port) silent=\(Voice.silent) ambience=\(Ambience.enabled) check=\(Voice.args.contains("-voiceCheck"))")
                case .failed(let error): Voice.log("listener failed: \(error)")
                default: break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                guard let self else { return }
                connection.start(queue: self.queue)
                self.receive(connection, Data())
            }
            listener.start(queue: queue)
            self.listener = listener
        } catch {
            Voice.log("listener failed: \(error)")
        }
    }

    private func receive(_ connection: NWConnection, _ buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, done, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let end = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buffer[..<end.lowerBound], as: UTF8.self)
                let lines = head.components(separatedBy: "\r\n")
                var headers: [String: String] = [:]
                for line in lines.dropFirst() {
                    guard let colon = line.firstIndex(of: ":") else { continue }
                    headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
                }
                let length = Int(headers["content-length"] ?? "0") ?? 0
                let body = buffer[end.upperBound...]
                if body.count >= length {
                    let parts = lines.first?.split(separator: " ") ?? []
                    let method = parts.count > 0 ? String(parts[0]) : ""
                    let path = parts.count > 1 ? (URLComponents(string: String(parts[1]))?.path ?? "") : ""
                    let json = (try? JSONSerialization.jsonObject(with: Data(body.prefix(length)))) as? [String: Any] ?? [:]
                    return self.route(method, path, headers, json, connection)
                }
            }
            if done || error != nil || buffer.count > 1 << 20 { connection.cancel(); return }
            self.receive(connection, buffer)
        }
    }

    private func route(_ method: String, _ path: String, _ headers: [String: String], _ body: [String: Any], _ c: NWConnection) {
        if method == "OPTIONS" { return reply(c, 204, nil) }
        if method == "GET", path == "/voice/status" {
            Task { reply(c, 200, await voice.status()) }
            return
        }
        if method == "GET", path == "/voice/voices" {  // VOICESHOP: the picker's list (any engine), the choice, the one speaking
            let packs = VoiceLibrary.shared.packs(refresh: true).map { p -> [String: Any] in
                ["id": p.id, "name": p.name, "engine": p.engine, "summary": p.summary, "origin": p.origin.rawValue, "bytes": p.bytes, "license": p.license]
            }
            return reply(c, 200, ["voices": packs, "choice": VoiceLibrary.shared.choice, "speaking": voice.speaker.packID ?? NSNull(), "demoLine": VoiceLibrary.demoLine])
        }
        guard method == "POST" else { return reply(c, 404, ["error": "no route \(method) \(path)"]) }
        guard headers["x-api-key"] == key else { return reply(c, 401, ["error": "bad or missing X-Api-Key"]) }
        let turn = body["turn"] as? Int ?? -1
        let auto = body["auto"] as? Bool ?? false, silenceMs = body["silenceMs"] as? Double
        let fixture = body["fixture"] as? String, text = body["text"] as? String ?? "", line = body["line"] as? String ?? "?"
        let file = body["file"] as? String
        let id = body["id"] as? String ?? "", play = body["play"] as? Bool ?? true
        Task { [voice] in
            do {
                switch path {
                case "/voice/prepare":
                    reply(c, 200, await voice.prepare())
                case "/voice/install":
                    reply(c, 200, ["assets": try await voice.install()])
                case "/voice/listen":
                    reply(c, 200, ["turn": try await voice.listen(fixture: fixture, auto: auto, silenceMs: silenceMs)])
                case "/voice/inject":
                    try voice.inject(fixture ?? "")
                    reply(c, 200, ["ok": true])
                case "/voice/endpoint":
                    voice.endpointNow()
                    reply(c, 200, ["ok": true])
                case "/voice/release":
                    reply(c, 200, try await voice.release(turn).json())
                case "/voice/say":  // UX-VOICE-DRAW: a speak-only turn (Read aloud, speak(ref)): {turn}
                    reply(c, 200, ["turn": try await voice.say()])
                case "/voice/speak":
                    try voice.speak(turn, text)
                    reply(c, 200, ["ok": true])
                case "/voice/finish":
                    reply(c, 200, await voice.finish(turn)?.json() ?? ["error": "no turn \(turn)"])
                case "/voice/cancel":
                    await voice.cancel()
                    reply(c, 200, ["ok": true])
                case "/voice/select":  // VOICESHOP: hot swap: the next chunk speaks in the new voice; remembered
                    reply(c, 200, try await voice.speaker.select(id))
                case "/voice/preview":  // VOICESHOP: the demo line in that voice (rendered once, cached), played
                    reply(c, 200, try await voice.speaker.preview(id, play: play))
                case "/voice/synthesize":  // MACVOICE: text → a WAV, nothing played (the headless TTS path)
                    reply(c, 200, try await Voice.synthesize(text, file: file, voice: voice.speaker.voice))
                case "/voice/transcribe":  // MACVOICE: an audio file → its words (the headless STT path)
                    reply(c, 200, try await Voice.transcribe(file: file ?? ""))
                case "/voice/permission":  // MACVOICE: ask for the microphone now (the system prompt, once) → {mic}
                    reply(c, 200, await Voice.askMicState())
                case "/voice/log":
                    Voice.log("web \(line)")
                    reply(c, 200, ["ok": true])
                default:
                    reply(c, 404, ["error": "no route POST \(path)"])
                }
            } catch {
                reply(c, 409, ["error": error.localizedDescription])
            }
        }
    }

    private func reply(_ c: NWConnection, _ code: Int, _ body: [String: Any]?) {
        let payload = body.flatMap { try? JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys, .withoutEscapingSlashes]) } ?? Data()
        let reason = [200: "OK", 204: "No Content", 401: "Unauthorized", 404: "Not Found", 409: "Conflict"][code] ?? "Error"
        var head = "HTTP/1.1 \(code) \(reason)\r\nContent-Type: application/json\r\nContent-Length: \(payload.count)\r\n"
        head += "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n"
        head += "Cache-Control: no-store\r\nConnection: close\r\n\r\n"
        c.send(content: Data(head.utf8) + payload, completion: .contentProcessed { _ in c.cancel() })
    }
}
