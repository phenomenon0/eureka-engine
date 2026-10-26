import Foundation
import Network

/// The engine port (127.0.0.1:8931) served by our engine: the slice of llama-server's HTTP surface that the web app
/// uses (src/engines/transport.ts), so the web layer runs unchanged on LLBEngine.
///
///   GET  /health                 public. 200 {"status":"ok"} when chat's model is present, else 503 (no-models).
///   POST /v1/chat/completions    OpenAI-style. stream=true → SSE `data:` frames with choices[0].delta.content, then
///                                `data: [DONE]`; stream=false → choices[0].message.content. Greedy (temperature is
///                                ignored), cut at the first <|im_end|>. Extra field `llb` on the last frame / body:
///                                the engine's ids and timings.
///   POST /v1/systemone           the body goes to llb_decide as is; the reply line comes back as is.
///   POST /v1/decide              IOS-FOLD F2: the five-verb alias for systemone; `model` filled, `int` refused.
///   POST /v1/see                 IOS-FOLD F3: {image: data URL, prompt} → {text, usage, llb}; task:detect → boxes or 501.
///   POST /v1/hear                IOS-FOLD F4: {audio: WAV data URL} → transcript through the speech pipeline.
///   POST /v1/speak                IOS-FOLD F5: {text, voice, style} → 24 kHz PCM16 WAV bytes.
///   POST /v1/look                BENDSDK: {question, width, height, rgb: base64 of width×height×3 bytes} → the camera's
///                                verdict on that image (letterboxed to the canvas, the tower, the Yes/No readout), as
///                                JSON. 503 with why when this build or device has no vision path (src/sdk/bend.ts).
///   POST /v1/bend/pocket         G27, only with -visionPocket (else 404, as before): {url: data URL of a photo, mask?} →
///                                the detector's boxes, the top box's mask (PNG) and one sentence to speak (VisionPocket.swift).
///
/// Both POST routes need the per-launch key as `X-Api-Key` (or `Authorization: Bearer`), as llama-server's --api-key
/// did. CORS: preflights are answered before the key check and the request's Origin is echoed (the web view's
/// origin is `null`, file://), with `Allow-Headers: *` — the same answers llama-server gave (docs/R7-NOTES.md).
/// Errors use the five-verb contract's canonical shape, {"error":{"message","type","param","code"}}; the 8 MiB body cap
/// is `serve.c`'s MAX_BODY. Both are `ProviderContract`'s (IOS-FOLD F1).
final class EngineServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "bend.engines.engine")
    private let key: String
    private let handler: EngineRoutes

    init(port: UInt16, key: String, routes: EngineRoutes) throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
        parameters.allowLocalEndpointReuse = true
        listener = try NWListener(using: parameters)
        self.key = key
        handler = routes
    }

    func start() {
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready: EngineManager.log("engine listening on 127.0.0.1:\(EngineManager.enginePort) (LLBEngine \(LLBEngine.programTag))")
            case .failed(let error): EngineManager.log("engine listener failed: \(error)")
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            connection.start(queue: self.queue)
            self.receive(HTTPExchange(connection), buffer: Data())
        }
        listener.start(queue: queue)
    }

    /// Closes the listener (the lifecycle's watchdog replaces a listener that stopped answering).
    func stop() { listener.cancel() }

    /// Reads the head, then Content-Length bytes of body.
    private func receive(_ exchange: HTTPExchange, buffer: Data) {
        exchange.connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, done, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let end = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buffer[..<end.lowerBound], as: UTF8.self)
                exchange.parse(head)
                let length = Int(exchange.headers["content-length"] ?? "0") ?? 0
                let body = buffer[end.upperBound...]
                if body.count >= length {
                    exchange.body = Data(body.prefix(length))
                    return self.route(exchange)
                }
                if length > ProviderContract.maxBody {
                    return exchange.fail(.tooLarge())
                }
            }
            if done || error != nil || buffer.count > 8 << 20 { exchange.connection.cancel(); return }
            self.receive(exchange, buffer: buffer)
        }
    }

    static func error(_ code: Int, _ message: String, type: String = "server_error") -> [String: Any] {
        ["error": ["code": code, "message": message, "type": type]]
    }

    private func route(_ x: HTTPExchange) {
        if x.method == "OPTIONS" { return x.reply(204, nil) }
        if x.path == ProviderContract.healthPath {
            guard x.method == "GET" else { return x.fail(.methodNotAllowed(x.method, allow: "GET")) }
            Task { @MainActor [handler] in
                let (ok, why) = handler.health()
                x.json(ok ? 200 : 503, ok ? ["status": "ok"] : Self.error(503, why, type: "unavailable_error"))
            }
            return
        }
        // The pocket route keeps its historical 404 when the flag is off (G27: the 404 every build gave).
        if x.path == ProviderContract.pocketPath, !VisionPocket.enabled {
            return x.fail(.notFound())
        }
        guard ProviderContract.postPaths.contains(x.path) || x.path == ProviderContract.pocketPath else {
            return x.fail(.notFound())
        }
        guard x.method == "POST" else {
            return x.fail(.methodNotAllowed(x.method, allow: "POST"))
        }
        let bearer = x.headers["authorization"].map { $0.replacingOccurrences(of: "Bearer ", with: "") }
        guard x.headers["x-api-key"] == key || bearer == key else {
            return x.fail(.unauthorized())
        }
        switch x.path {
        case ProviderContract.decidePath: handler.decide(x)
        case ProviderContract.seePath: handler.see(x)
        case ProviderContract.hearPath: handler.hear(x)
        case ProviderContract.speakPath: handler.speak(x)
        case ProviderContract.systemonePath: handler.systemone(x)
        case ProviderContract.imagePath: handler.image(x)
        case ProviderContract.lookPath: handler.look(x)
        case ProviderContract.pocketPath: handler.pocket(x)
        default: handler.chat(x)
        }
    }
}

/// What the engine port does with a request (EngineManager implements it).
protocol EngineRoutes: Sendable {
    @MainActor func health() -> (Bool, String)
    func chat(_ exchange: HTTPExchange)
    func systemone(_ exchange: HTTPExchange)
    /// IOS-FOLD F2: /v1/decide, the five-verb alias over the systemone/decide path.
    func decide(_ exchange: HTTPExchange)
    /// IOS-FOLD F3: /v1/see, one image in, text out (or detect boxes when the pocket is staged).
    func see(_ exchange: HTTPExchange)
    /// IOS-FOLD F4: /v1/hear, one WAV clip in, transcript out.
    func hear(_ exchange: HTTPExchange)
    /// IOS-FOLD F5: /v1/speak, text in, 24 kHz PCM16 WAV out.
    func speak(_ exchange: HTTPExchange)
    /// BENDSDK: one image + one yes/no question → the camera engine's verdict.
    func look(_ exchange: HTTPExchange)
    /// IMGPERF: a photo handed over ahead of its look (send-once): POST /v1/bend/image {url, key}.
    func image(_ exchange: HTTPExchange)
    /// G27: one photo → boxes, the top box's mask, one sentence (VisionPocket.swift); only with -visionPocket.
    func pocket(_ exchange: HTTPExchange)
}

extension EngineRoutes {
    /// A shell without the pocket (the CoreML stack): as if the route were off.
    func pocket(_ exchange: HTTPExchange) {
        exchange.json(404, EngineServer.error(404, "File Not Found", type: "not_found_error"))
    }

    /// A shell without the in-app vision path (the BendLlama target): the web keeps sending the photo with each look.
    func image(_ exchange: HTTPExchange) {
        exchange.json(503, EngineServer.error(503, "no-vision: this build keeps no photos", type: "unavailable_error"))
    }
}

/// One request and its response on one connection (`Connection: close`).
final class HTTPExchange: @unchecked Sendable {
    let connection: NWConnection
    private(set) var method = ""
    private(set) var path = ""
    private(set) var headers: [String: String] = [:]
    var body = Data()
    private let lock = NSLock()
    private var closed = false

    init(_ connection: NWConnection) { self.connection = connection }

    func parse(_ head: String) {
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines.first?.split(separator: " ") ?? []
        method = parts.count > 0 ? String(parts[0]) : ""
        path = parts.count > 1 ? (URLComponents(string: String(parts[1]))?.path ?? "") : ""
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
    }

    private func cors() -> String {
        // llama-server (cors_origins "*", credentials on) echoes the request's Origin; `null` for the file:// web view.
        "Access-Control-Allow-Origin: \(headers["origin"] ?? "*")\r\n"
            + "Access-Control-Allow-Credentials: true\r\n"
            + "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            + "Access-Control-Allow-Headers: *\r\n"
    }

    private static let reasons = [200: "OK", 204: "No Content", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
                                  405: "Method Not Allowed", 409: "Conflict", 413: "Payload Too Large",
                                  500: "Internal Server Error", 501: "Not Implemented", 503: "Service Unavailable"]

    func reply(_ code: Int, _ payload: Data?, type: String = "application/json", headers: [String: String] = [:]) {
        var head = "HTTP/1.1 \(code) \(Self.reasons[code] ?? "Error")\r\n"
        head += "Content-Type: \(type)\r\nContent-Length: \(payload?.count ?? 0)\r\n"
        for (name, value) in headers.sorted(by: { $0.key < $1.key }) { head += "\(name): \(value)\r\n" }
        head += cors()
        head += "Cache-Control: no-store\r\nConnection: close\r\n\r\n"
        send(Data(head.utf8) + (payload ?? Data()), close: true)
    }

    func json(_ code: Int, _ object: [String: Any]) {
        // IOS-FOLD F1: every error goes through the canonical encoder, whatever route built it. Key order is the
        // contract (message, type, param, code); JSONSerialization would lose it.
        if let e = object["error"] as? [String: Any], let c = e["code"] as? Int, let message = e["message"] as? String {
            return reply(code, ProviderContract.errorJSON(ProviderContract.APIError(
                code: c, message: message, type: e["type"] as? String ?? "server_error", param: e["param"] as? String)))
        }
        reply(code, (try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes])) ?? Data())
    }

    // MARK: SSE (chunked)

    func beginStream() {
        var head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n" + cors()
        head += "Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        send(Data(head.utf8), close: false)
    }

    func event(_ object: [String: Any]) {
        guard let json = try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes]) else { return }
        chunk(Data("data: ".utf8) + json + Data("\n\n".utf8))
    }

    func endStream() {
        chunk(Data("data: [DONE]\n\n".utf8))
        send(Data("0\r\n\r\n".utf8), close: true)
    }

    private func chunk(_ data: Data) {
        send(Data(String(data.count, radix: 16).utf8) + Data("\r\n".utf8) + data + Data("\r\n".utf8), close: false)
    }

    /// True once the response is complete (or the client went away).
    var isClosed: Bool { lock.withLock { closed } }

    private func send(_ data: Data, close: Bool) {
        let skip = lock.withLock { () -> Bool in
            if closed { return true }
            if close { closed = true }
            return false
        }
        if skip { return }
        connection.send(content: data, completion: .contentProcessed { [weak self, connection] error in
            // ATTACHVIS: a failed send is the client gone (Stop): isClosed says so, and a look stops decoding on it.
            if error != nil, let self { self.lock.withLock { self.closed = true } }
            if close || error != nil { connection.cancel() }
        })
    }
}
