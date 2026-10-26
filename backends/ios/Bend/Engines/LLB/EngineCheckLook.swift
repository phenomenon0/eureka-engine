import Foundation

/// IMGPERF: `-engineCheck look`: the photo conversation the owner found slow, over the app's own HTTP surface (the
/// requests the web sends), on the phone, one launch. Every turn logs `BEND check look <turn> first_ms= done_ms= ids=…`.
///   control   two text turns (p2: the first warms chat; both must equal the Mac's 32 ids)
///   attachvis ATTACHVIS's web: the 1280 px photo rides every look (L1 opens the tower, L2 the same photo again) + a
///             text follow-up (the digest's "[Photo] …")
///   imgperf   IMGPERF's web: the 448 px wire copy handed over first (POST /v1/bend/image + key, as the photo lands in
///             the composer), a 3 s typing pause, then L1 / L2 by key + the text follow-up; L3 by key after the tower's
///             idle close (BEND_LOOK_IDLE_S, the device script sets 5)
/// Bundled photos: engine-check/look-news-1280.jpg (the web's 1280 px prepare of a 3000 px photo) and look-news-wire.jpg
/// (its 448 px wire copy). PASS: every turn answered, p2 parity, and the looks' answers name the headline.
/// `-lookLoop` runs the shell's ATTACHVIS engine path (one tower + prefill per id): the before on the same build.
extension EngineCheck {
    private static func lookPost(_ path: String, _ body: Data, key: String) -> URLRequest {
        var r = URLRequest(url: URL(string: "http://127.0.0.1:\(EngineManager.enginePort)\(path)")!)
        r.httpMethod = "POST"
        r.httpBody = body
        r.timeoutInterval = 600
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.setValue(key, forHTTPHeaderField: "X-Api-Key")
        return r
    }

    struct LookTurn { var ok: Bool; var firstMs: Double; var doneMs: Double; var text: String; var ids: [Int]; var bytes: Int }

    /// One streamed /v1/chat/completions; first / done from the send, the final frame's llb ids.
    private static func turn(_ body: [String: Any], key: String) async -> LookTurn {
        let data = try! JSONSerialization.data(withJSONObject: body)
        let t0 = Date()
        do {
            let (bytes, response) = try await URLSession.shared.bytes(for: lookPost("/v1/chat/completions", data, key: key))
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            var text = "", first: Double?, ids: [Int] = []
            for try await line in bytes.lines {
                guard code == 200 else { text += line; continue }
                guard line.hasPrefix("data:") else { continue }
                let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                if payload == "[DONE]" { break }
                guard let frame = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any] else { continue }
                if let piece = ((frame["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["content"] as? String, !piece.isEmpty {
                    if first == nil { first = Date().timeIntervalSince(t0) * 1000 }
                    text += piece
                }
                if let llb = frame["llb"] as? [String: Any], let i = llb["ids"] as? [Int] { ids = i }
            }
            if code != 200 { say("look HTTP \(code): \(text.prefix(200))") }
            return LookTurn(ok: code == 200 && !text.isEmpty, firstMs: first ?? -1, doneMs: Date().timeIntervalSince(t0) * 1000,
                            text: text, ids: ids, bytes: data.count)
        } catch {
            say("look error: \(error.localizedDescription)")
            return LookTurn(ok: false, firstMs: -1, doneMs: Date().timeIntervalSince(t0) * 1000, text: "", ids: [], bytes: data.count)
        }
    }

    private static func lookBody(_ url: String, key: String?, _ question: String) -> [String: Any] {
        var image: [String: Any] = ["url": url]
        if let key { image["key"] = key }
        return ["messages": [["role": "user", "content": [["type": "image_url", "image_url": image], ["type": "text", "text": question]]]],
                "stream": true, "max_tokens": 96, "chat_template_kwargs": ["enable_thinking": false]]
    }

    private static func textBody(_ messages: [[String: String]], maxTokens: Int = 64) -> [String: Any] {
        ["messages": messages, "stream": true, "max_tokens": maxTokens, "temperature": 0.7, "chat_template_kwargs": ["enable_thinking": false]]
    }

    private static func log(_ name: String, _ t: LookTurn) {
        say(String(format: "look %@ first_ms=%.0f done_ms=%.0f bytes=%d ids=%d ok=%@ text=\"%@\"", name, t.firstMs, t.doneMs, t.bytes, t.ids.count,
                   t.ok ? "yes" : "NO", t.text.replacingOccurrences(of: "\n", with: " ").prefix(120).description))
        say("look \(name) ids \(t.ids.map(String.init).joined(separator: " "))")
    }

    static func look(key: String) async -> Bool {
        let dir = fixtures
        guard let full = FileManager.default.contents(atPath: dir.appendingPathComponent("look-news-1280.jpg").path),
              let wire = FileManager.default.contents(atPath: dir.appendingPathComponent("look-news-wire.jpg").path) else {
            say("look: the bundled photos are missing (engine-check/look-news-*.jpg)")
            return false
        }
        let fullURL = "data:image/jpeg;base64,\(full.base64EncodedString())"
        let wireURL = "data:image/jpeg;base64,\(wire.base64EncodedString())"
        let wireKey = "check-\(wire.count)"
        let path = ProcessInfo.processInfo.arguments.contains("-lookLoop") ? "loop (ATTACHVIS engine path)" : "one job (IMGPERF engine path)"
        say("look path=\(path) photo=look-news 1280 px \(fullURL.utf8.count) B, wire 448 px \(wireURL.utf8.count) B")
        var ok = true
        // control: text turns (the first warms chat)
        for name in ["C1-warms", "C2"] {
            let t = await turn(textBody([["role": "user", "content": p2Text]], maxTokens: 32), key: key)  // p2: the Mac's 32 ids
            log("control-\(name)", t)
            if t.ids.map(UInt32.init) != macP2 { say("look control \(name) p2 PARITY FAIL"); ok = false }
            ok = ok && t.ok
        }
        let q1 = "What is the headline?", q2 = "What newspaper is this, and what is the date?", f = "Why was that such big news at the time?"
        // ATTACHVIS's web: the photo rides every look
        let a1 = await turn(lookBody(fullURL, key: nil, q1), key: key); log("attachvis-L1-tower-cold", a1)
        let a2 = await turn(lookBody(fullURL, key: nil, q2), key: key); log("attachvis-L2-same", a2)
        let af = await turn(textBody([["role": "user", "content": "[Photo] \(q1)"], ["role": "assistant", "content": a1.text],
                                      ["role": "user", "content": "[Photo] \(q2)"], ["role": "assistant", "content": a2.text],
                                      ["role": "user", "content": f]]), key: key)
        log("attachvis-F-text", af)
        // IMGPERF's web: handed over as it lands in the composer, then named by key. Both conversations start with chat
        // warm and the tower closed (past its idle close, when BEND_LOOK_IDLE_S is short).
        let idle = ProcessInfo.processInfo.environment["BEND_LOOK_IDLE_S"].flatMap(Double.init) ?? 90
        if idle <= 30 { try? await Task.sleep(for: .seconds(idle + 2)) }
        let th = Date()
        let (hd, hr) = (try? await URLSession.shared.data(for: lookPost("/v1/bend/image",
            try! JSONSerialization.data(withJSONObject: ["url": wireURL, "key": wireKey]), key: key))) ?? (Data(), URLResponse())
        say(String(format: "look imgperf-attach handover HTTP %d in %.0f ms (%d B up) %@", (hr as? HTTPURLResponse)?.statusCode ?? -1,
                   Date().timeIntervalSince(th) * 1000, wireURL.utf8.count, String(decoding: hd, as: UTF8.self)))
        try? await Task.sleep(for: .seconds(3))  // the person types the question
        let ref = "\(EngineManager.imageRef)\(wireKey)"
        let b1 = await turn(lookBody(ref, key: nil, q1), key: key); log("imgperf-L1-after-attach", b1)
        let b2 = await turn(lookBody(ref, key: nil, q2), key: key); log("imgperf-L2-same", b2)
        let bf = await turn(textBody([["role": "user", "content": "[Photo] \(q1)"], ["role": "assistant", "content": b1.text],
                                      ["role": "user", "content": "[Photo] \(q2)"], ["role": "assistant", "content": b2.text],
                                      ["role": "user", "content": f]]), key: key)
        log("imgperf-F-text", bf)
        if idle <= 30 {
            try? await Task.sleep(for: .seconds(idle + 2))  // past the tower's idle close
            let b3 = await turn(lookBody(ref, key: nil, "What does the photo on the front page show?"), key: key)
            log("imgperf-L3-after-tower-close", b3)
            ok = ok && b3.ok
        }
        let named = [a1, b1].allSatisfy { $0.text.uppercased().contains("MOON") }
        if !named { say("look: an L1 answer doesn't name the headline (MEN WALK ON MOON)") }
        ok = ok && [a1, a2, af, b1, b2, bf].allSatisfy(\.ok) && named
        say(String(format: "look summary path=%@ attachvis L1 %.0f / L2 %.0f / F %.0f ms; imgperf L1 %.0f / L2 %.0f / F %.0f ms (done)",
                   path, a1.doneMs, a2.doneMs, af.doneMs, b1.doneMs, b2.doneMs, bf.doneMs))
        return ok
    }

    // MARK: look-golden (BENDHARVEST H2)

    static let goldenPrompt = "Describe this image."

    /// `-engineCheck look-golden`: the crew's R19 vision gate on this tree's engine. engine-check/invoice.png (1024×640)
    /// at its native aspect (640 image tokens, no letterbox) with R19's prompt ("Describe this image.", no system turn),
    /// 32 greedy ids through VisionLook.lookNative (llb_generate_rgb), three times in one process with a chat turn (p2,
    /// the Mac's 32 ids) between the second and third. PASS: every run's 32 ids equal the CLI golden
    /// (engine-check/invoice.golden.ids, ENGINE-IPHONEVIS Q3, R19's device PASS) and each other, and p2 holds. A DIFF is
    /// reported with its first divergence and the run's ids; the golden is never loosened here (BENDHARVEST REPORT).
    static func lookGolden(key: String) async -> Bool {
        #if BEND_CAMERA
        guard let png = FileManager.default.contents(atPath: fixtures.appendingPathComponent("invoice.png").path),
              let golden = (try? String(contentsOf: fixtures.appendingPathComponent("invoice.golden.ids"), encoding: .utf8))?
                .split(whereSeparator: \.isWhitespace).compactMap({ UInt32($0) }), golden.count == 32 else {
            say("look-golden: engine-check/invoice.png or invoice.golden.ids (32 ids) missing")
            return false
        }
        guard let image = try? VisionLook.decode(png.base64EncodedString()) else { say("look-golden: invoice.png doesn't decode"); return false }
        say("look-golden image \(image.width)x\(image.height) prompt \"\(goldenPrompt)\" golden \(golden.prefix(3).map(String.init).joined(separator: " ")) … (32 ids) footprint_mb=\(EngineManager.footprint())")
        var runs: [[UInt32]] = []
        var ok = true
        for n in 1...3 {
            if n == 3 {
                let t = await turn(textBody([["role": "user", "content": p2Text]], maxTokens: 32), key: key)
                log("golden-chat-between", t)
                let parity = t.ids.map(UInt32.init) == macP2
                say("look-golden chat between (p2 parity) \(parity ? "OK" : "FAILED")")
                ok = ok && t.ok && parity
            }
            do {
                let (r, canvas) = try await Task.detached { try VisionLook.shared.lookNative(image, prompt: goldenPrompt, maxNew: 32) }.value
                runs.append(r.ids)
                let same = r.ids == golden
                let at = (0..<min(r.ids.count, golden.count)).first { r.ids[$0] != golden[$0] } ?? min(r.ids.count, golden.count)
                say(String(format: "look-golden %d canvas=%dx%d image_tokens=%d prompt_ids=%d new_ids=%d first_ms=%.0f total_ms=%.0f tower_ms=%.0f prefill_ms=%.0f decode_ms_per_id=%.1f",
                           n, canvas.width, canvas.height, r.imageTokens, r.promptIds, r.ids.count, r.firstMs, r.totalMs, r.towerMsSum, r.prefillMs, r.decodeMsPerId))
                say("look-golden \(n) ids \(r.ids.map(String.init).joined(separator: " "))")
                say("look-golden \(n) reply \(String(reflecting: r.text))")
                say("look-golden \(n) GOLDEN \(same ? "OK 32/32 ids identical to the CLI golden" : "DIFF: first divergence at id \(at)")")
                ok = ok && same
            } catch {
                say("look-golden \(n) failed: \(error.localizedDescription)")
                ok = false
            }
        }
        let identical = runs.count == 3 && runs.allSatisfy { $0 == runs[0] }
        say("look-golden runs 1, 2, 3 \(identical ? "BYTE-IDENTICAL ids" : "DIFFER") footprint_mb=\(EngineManager.footprint())")
        return ok && identical
        #else
        say("look-golden: this build has no vision engine (BEND_CAMERA off)")
        return false
        #endif
    }
}
