#if os(iOS)
import CoreGraphics
import Foundation

/// CAMCAP receipts: the simulator has no Metal for our engine, so `-camCaptionHost <base> <key>` captions through the
/// Mac's engine harness (scripts/engine-mac.sh with BEND_PORT_BASE=<base> BEND_API_KEY=<key>) on the wire the web's
/// vision.look uses (docs/ATTACH.md § The wire): POST /engines/ensure?engine=chat, then a streamed
/// /v1/chat/completions with the frame as an image_url part. The harness runs the same Swift VisionLook on the Mac's GPU.
final class HostCaptioner: Captioner {
    let base: Int
    let key: String
    init(base: Int, key: String) { self.base = base; self.key = key }

    var unavailableReason: String? { nil }

    struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ m: String) { errorDescription = m }
    }

    func caption(_ image: CGImage, onPiece: @escaping @Sendable (String) -> Bool) async throws -> CaptionResult {
        let t0 = Date()
        guard let jpeg = OSBridge.jpeg(image, edge: 1280, quality: 0.85) else { throw Failure("couldn't encode the frame") }
        var ensure = URLRequest(url: URL(string: "http://127.0.0.1:\(base)/engines/ensure?engine=chat")!)
        ensure.httpMethod = "POST"
        ensure.setValue(key, forHTTPHeaderField: "X-Api-Key")
        ensure.timeoutInterval = 300
        _ = try? await URLSession.shared.data(for: ensure)
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(base + 1)/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue(key, forHTTPHeaderField: "X-Api-Key")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 300
        let body: [String: Any] = [
            "messages": [["role": "user", "content": [
                ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(jpeg.base64EncodedString())"]],
                ["type": "text", "text": CaptionConfig.question]]]],
            "stream": true, "max_tokens": CaptionConfig.maxNew, "chat_template_kwargs": ["enable_thinking": false],
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (bytes, response) = try await URLSession.shared.bytes(for: req)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            var raw = ""
            for try await line in bytes.lines { raw += line }
            throw Failure(raw.isEmpty ? "the engine said \(http.statusCode)" : String(raw.prefix(200)))
        }
        var text = "", first = -1.0, finish = "stop"
        for try await line in bytes.lines {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" { break }
            guard let obj = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any],
                  let choice = (obj["choices"] as? [[String: Any]])?.first else { continue }
            if let piece = (choice["delta"] as? [String: Any])?["content"] as? String, !piece.isEmpty {
                if first < 0 { first = Date().timeIntervalSince(t0) * 1000 }
                text += piece
                if !onPiece(piece) { finish = "aborted"; break }
            }
        }
        return CaptionResult(text: text, firstMs: first, totalMs: Date().timeIntervalSince(t0) * 1000, finish: finish)
    }
}
#endif
