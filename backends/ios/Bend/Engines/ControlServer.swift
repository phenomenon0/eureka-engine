import Foundation
import Network

/// The engines' control listener on 127.0.0.1:8930: a few lines of HTTP/1.1, one request per connection.
///
///   GET  /engines/status                  → EngineManager.statusJSON (public: no secrets in it)
///   POST /engines/ensure?engine=chat|decision[&ahead=1]
///                                         → blocks until that model serves the engine port, then the status.
///                                           `ahead=1` (PHONEVOICE): no request waits on it yet (voice opened), so the
///                                           warm runs its warm-ups as an eager one does.
///                                           Needs `X-Api-Key` (the per-launch key), since it can swap models.
///                                           Warms the engine first if it is cold (the lifecycle, docs/ENGINES.md);
///                                           503 with `error` after the bounded retries (the web offers Retry).
///   POST /engines/visibility?state=hidden|visible
///                                           the web page's visibilitychange: away starts the release grace,
///                                           visible cancels it and runs the watchdog. Needs `X-Api-Key`.
///
/// CORS is open (`*`): the web app's origin is `null` (file://). `X-Api-Key` rather than `Authorization`,
/// because a wildcard Access-Control-Allow-Headers does not cover Authorization, and llama-server takes either.
final class ControlServer: @unchecked Sendable {
    private let listener: NWListener
    private let queue = DispatchQueue(label: "bend.engines.control")
    private let manager: EngineManager

    init(port: UInt16, manager: EngineManager) throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
        parameters.allowLocalEndpointReuse = true
        listener = try NWListener(using: parameters)
        self.manager = manager
    }

    func start() {
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready: EngineManager.log("control listening on 127.0.0.1:\(EngineManager.controlPort)")
            case .failed(let error): EngineManager.log("control listener failed: \(error)")
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        listener.start(queue: queue)
    }

    /// Closes the listener (the lifecycle's watchdog replaces a listener that stopped answering).
    func stop() { listener.cancel() }

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        receive(connection, buffer: Data())
    }

    /// Reads until the end of the headers. The routes take no body, so anything after it is ignored.
    private func receive(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16_384) { [weak self] data, _, done, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let end = buffer.range(of: Data("\r\n\r\n".utf8)) {
                self.handle(String(decoding: buffer[..<end.lowerBound], as: UTF8.self), on: connection)
            } else if done || error != nil || buffer.count > 65_536 {
                connection.cancel()
            } else {
                self.receive(connection, buffer: buffer)
            }
        }
    }

    private func handle(_ head: String, on connection: NWConnection) {
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines.first?.split(separator: " ") ?? []
        guard parts.count >= 2 else { return reply(connection, 400, ["error": "bad request"]) }
        let method = String(parts[0])
        let target = URLComponents(string: String(parts[1]))
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let path = target?.path ?? ""
        let engine = target?.queryItems?.first { $0.name == "engine" }?.value
        let visibility = target?.queryItems?.first { $0.name == "state" }?.value
        let ahead = target?.queryItems?.first { $0.name == "ahead" }?.value == "1"
        let key = headers["x-api-key"]

        if method == "OPTIONS" { return reply(connection, 204, nil) }
        Task { @MainActor [manager] in
            switch (method, path) {
            case ("GET", "/engines/status"):
                self.reply(connection, 200, manager.statusJSON())
            case ("POST", "/engines/ensure"):
                guard key == manager.apiKey else { return self.reply(connection, 401, ["error": "bad or missing X-Api-Key"]) }
                guard let id = engine.flatMap(EngineID.init(rawValue:)) else {
                    return self.reply(connection, 400, ["error": "engine must be chat or decision"])
                }
                do {
                    try await manager.ensure(id, ahead: ahead)
                    self.reply(connection, 200, manager.statusJSON())
                } catch {
                    var body = manager.statusJSON()
                    body["error"] = error.localizedDescription
                    self.reply(connection, manager.hasModel(id) ? 503 : 404, body)
                }
            case ("POST", "/engines/visibility"):
                guard key == manager.apiKey else { return self.reply(connection, 401, ["error": "bad or missing X-Api-Key"]) }
                switch visibility {
                case "hidden": manager.away()
                case "visible": manager.back()
                default: return self.reply(connection, 400, ["error": "state must be hidden or visible"])
                }
                self.reply(connection, 200, manager.statusJSON())
            default:
                self.reply(connection, 404, ["error": "no route \(method) \(path)"])
            }
        }
    }

    private nonisolated func reply(_ connection: NWConnection, _ code: Int, _ body: [String: Any]?) {
        let payload = body.flatMap { try? JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys]) } ?? Data()
        let reason = [200: "OK", 204: "No Content", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 503: "Service Unavailable"][code] ?? "Error"
        var head = "HTTP/1.1 \(code) \(reason)\r\n"
        head += "Content-Type: application/json\r\n"
        head += "Content-Length: \(payload.count)\r\n"
        head += "Access-Control-Allow-Origin: *\r\n"
        head += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        head += "Access-Control-Allow-Headers: *\r\n"
        head += "Cache-Control: no-store\r\n"
        head += "Connection: close\r\n\r\n"
        connection.send(content: Data(head.utf8) + payload, completion: .contentProcessed { _ in connection.cancel() })
    }
}
