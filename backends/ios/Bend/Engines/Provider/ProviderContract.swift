import Foundation

/// IOS-FOLD F0/F1: the five-verb wire contract (chat · decide · see · hear · speak) in Swift, for the in-app provider on
/// 127.0.0.1:8931.
///
/// Folded from `runs/backend-parity/metal-5of5/SCHEMAS.md` and `serve.c`'s route *semantics* — not from `serve.c`'s code.
/// iOS is one process with a static engine: there is no engine child, no `fork`/`exec`, and no second HTTP server. The
/// unit of fold is therefore the route contract: paths, methods, status codes, the canonical error envelope, the 8 MiB
/// body cap, and the model echo/fill rule.
enum ProviderContract {
    /// `serve.c` MAX_BODY. Enforced before the body is read.
    static let maxBody = 8 << 20

    static let chatPath = "/v1/chat/completions"
    static let decidePath = "/v1/decide"
    static let systemonePath = "/v1/systemone"
    static let seePath = "/v1/see"
    static let hearPath = "/v1/hear"
    static let speakPath = "/v1/speak"
    static let lookPath = "/v1/look"
    static let imagePath = "/v1/bend/image"
    static let pocketPath = "/v1/bend/pocket"
    static let healthPath = "/health"

    /// Every POST route the engine port serves (the pocket only answers when its flag is on; EngineServer keeps its 404).
    static let postPaths: Set<String> = [chatPath, decidePath, systemonePath, seePath, hearPath, speakPath, lookPath, imagePath]

    /// The canonical error, field order and `param` included: every error on the provider is this shape.
    /// `{"error":{"message":"…","type":"invalid_request_error","param":null,"code":400}}`
    struct APIError: Error, Sendable {
        var code: Int
        var message: String
        var type: String
        var param: String?
    }

    // MARK: errors (one encoder, every route)

    static func invalid(_ message: String, param: String? = nil) -> APIError {
        APIError(code: 400, message: message, type: "invalid_request_error", param: param)
    }

    static func unauthorized() -> APIError {
        APIError(code: 401, message: "Invalid API Key", type: "authentication_error", param: nil)
    }

    static func notFound() -> APIError {
        APIError(code: 404, message: "File Not Found", type: "not_found_error", param: nil)
    }

    static func methodNotAllowed(_ method: String, allow: String) -> APIError {
        APIError(code: 405, message: "\(method) is not allowed here; use \(allow)", type: "invalid_request_error", param: nil)
    }

    static func tooLarge() -> APIError {
        APIError(code: 413, message: "request body too large", type: "invalid_request_error", param: nil)
    }

    static func unsupported(_ message: String, param: String? = nil) -> APIError {
        APIError(code: 501, message: message, type: "unsupported_error", param: param)
    }

    static func unavailable(_ message: String) -> APIError {
        APIError(code: 503, message: message, type: "unavailable_error", param: nil)
    }

    static func internalError(_ message: String) -> APIError {
        APIError(code: 500, message: message, type: "server_error", param: nil)
    }

    /// JSON string literal, escaped (the error envelope is built by hand so its key order is stable).
    static func quote(_ s: String) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }

    /// The canonical error bytes. Key order is `message, type, param, code` — `serve.c`'s encoder.
    /// `extra` adds sibling keys after `error` (the honest-501 payload); absent for every ordinary error.
    static func errorJSON(_ e: APIError, extra: [String: Any] = [:]) -> Data {
        var out = "{\"error\":{\"message\":\(quote(e.message)),\"type\":\(quote(e.type)),"
        out += "\"param\":\(e.param.map(quote) ?? "null"),\"code\":\(e.code)}"
        for key in extra.keys.sorted() {
            guard let value = extra[key], JSONSerialization.isValidJSONObject(value),
                  let data = try? JSONSerialization.data(withJSONObject: value, options: [.withoutEscapingSlashes]),
                  let text = String(data: data, encoding: .utf8) else { continue }
            out += ",\(quote(key)):\(text)"
        }
        return Data((out + "}").utf8)
    }

    // MARK: body helpers

    static func object(_ data: Data) throws -> [String: Any] {
        guard let o = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw invalid("the body is not a JSON object")
        }
        return o
    }

    /// The request's `model` when it names one, else the provider fixture's alias. Every verb echoes it.
    static func model(_ request: [String: Any], _ fallback: String) -> String {
        if let m = request["model"] as? String, !m.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return m }
        return fallback
    }

    static func string(_ request: [String: Any], _ key: String, param: String? = nil) throws -> String {
        guard let s = request[key] as? String, !s.isEmpty else {
            throw invalid("\(key): expected a non-empty string", param: param ?? key)
        }
        return s
    }

    static func optionalString(_ request: [String: Any], _ key: String) -> String? {
        guard let s = request[key] as? String, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return s
    }

    /// A `data:` URL or bare base64 (the parent FOLD brief's `image_b64` / `audio_b64` spelling). `http(s)` is refused
    /// on iOS: the app is offline-first and does not fetch remote media.
    static func base64(_ value: String, param: String) throws -> Data {
        if value.hasPrefix("http://") || value.hasPrefix("https://") {
            throw invalid("\(param): http(s) URLs are not fetched on iOS; send a data: URL (base64)", param: param)
        }
        var b64 = Substring(value)
        if value.hasPrefix("data:") {
            guard let comma = value.firstIndex(of: ",") else {
                throw invalid("\(param): a data URL without data", param: param)
            }
            b64 = value[value.index(after: comma)...]
        }
        guard let data = Data(base64Encoded: String(b64), options: .ignoreUnknownCharacters), !data.isEmpty else {
            throw invalid("\(param): not base64", param: param)
        }
        return data
    }

    /// A 16-bit mono PCM WAV at `rate` (RIFF/WAVE). Built here so no route can send an empty or truncated body.
    static func wav(_ samples: [Float], rate: Int) -> Data {
        var pcm = Data(capacity: samples.count * 2)
        for sample in samples {
            let clamped = max(-1, min(1, sample))
            var v = Int16((clamped * 32767).rounded()).littleEndian
            withUnsafeBytes(of: &v) { pcm.append(contentsOf: $0) }
        }
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let bytes = UInt32(pcm.count)
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + bytes); d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(rate)); u32(UInt32(rate * 2)); u16(2); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(bytes)
        d.append(pcm)
        return d
    }

    /// RIFF/WAVE with a PCM `fmt ` chunk and a non-empty `data` chunk: validated before a 200 is written.
    static func isWAV(_ d: Data) -> Bool {
        guard d.count >= 44 else { return false }
        let bytes = [UInt8](d.prefix(44))
        func str(_ r: Range<Int>) -> String { String(bytes: bytes[r], encoding: .ascii) ?? "" }
        func u16(_ o: Int) -> UInt16 { UInt16(bytes[o]) | (UInt16(bytes[o + 1]) << 8) }
        func u32(_ o: Int) -> UInt32 {
            UInt32(bytes[o]) | (UInt32(bytes[o + 1]) << 8) | (UInt32(bytes[o + 2]) << 16) | (UInt32(bytes[o + 3]) << 24)
        }
        guard str(0..<4) == "RIFF", str(8..<12) == "WAVE", str(12..<16) == "fmt " else { return false }
        guard u16(20) == 1 else { return false }  // PCM (the format field is 2 bytes at offset 20)
        let dataSize = Int(u32(40))
        return dataSize > 0 && d.count >= 44 + dataSize
    }

    /// Adds `model` to a decide reply without re-encoding its numbers: the reply line is a flat JSON object and the
    /// answer floats must survive byte for byte. Already named → unchanged.
    static func fillModel(_ reply: String, model: String) -> String {
        guard let obj = (try? JSONSerialization.jsonObject(with: Data(reply.utf8))) as? [String: Any] else { return reply }
        if obj["model"] != nil { return reply }
        guard let end = reply.lastIndex(of: "}") else { return reply }
        var head = String(reply[..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
        if head.hasSuffix(",") { head.removeLast() }
        if head.hasSuffix("{") { return head + "\"model\":\(quote(model))}" }
        return head + ",\"model\":\(quote(model))}"
    }
}

extension HTTPExchange {
    /// The one error path: canonical bytes, correct status.
    func fail(_ e: ProviderContract.APIError, extra: [String: Any] = [:]) {
        reply(e.code, ProviderContract.errorJSON(e, extra: extra), type: "application/json")
    }
}

/// Shorthand so a route can write `x.fail(.invalid(…))`; the constructors live on `ProviderContract`.
extension ProviderContract.APIError {
    static func invalid(_ message: String, param: String? = nil) -> Self {
        ProviderContract.invalid(message, param: param)
    }

    static func unauthorized() -> Self { ProviderContract.unauthorized() }
    static func notFound() -> Self { ProviderContract.notFound() }
    static func methodNotAllowed(_ method: String, allow: String) -> Self {
        ProviderContract.methodNotAllowed(method, allow: allow)
    }
    static func tooLarge() -> Self { ProviderContract.tooLarge() }
    static func unsupported(_ message: String, param: String? = nil) -> Self {
        ProviderContract.unsupported(message, param: param)
    }
    static func unavailable(_ message: String) -> Self { ProviderContract.unavailable(message) }
    static func internalError(_ message: String) -> Self { ProviderContract.internalError(message) }
}
