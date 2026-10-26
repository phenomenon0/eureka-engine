import Foundation

/// `-engineCheck chat|decide|reopen|lifecycle` launch flag: once the listeners are up, Bend sends requests to its own engine
/// port (127.0.0.1:8931, with the per-launch key) — the same HTTP surface the web app uses — and judges the answers
/// against the Mac's. `-engineCheckExit` exits after the verdict.
///
///   chat    POST /v1/chat/completions, stream, the one user turn of engine-phone2's p2 prompt, max_tokens 32.
///           Rendered by Bend's template and tokenized by the engine's tokenizer it must be p2's 31 ids, and the 32
///           greedy ids must equal the Mac's (engine-4b receipts, = engine-phone2 EngineProbe.macQ4["p2"]).
///   decide  POST /v1/systemone with engine-phone2's travel request (Resources/decide/travel.json); the reply must
///           equal the Mac's fast llb_decide reply byte for byte (travel.mac.json: Amsterdam 0.554900763070548).
///
///   reopen  many runs per launch, all over HTTP: chat (p2) → chat (a distinct prompt) → decide (travel) → chat (p2
///           again), in one process. Each run loads, answers and releases (engine-repeat, engine-earlystop); the footprint is logged
///           after each. PASS only if all four answer and the p2 chats and the decide match the Mac.
///
///   lifecycle  (R14) cold launch → warm on a send → kept warm within the grace → released past it → eager
///           re-warm on return → a decision from cold → the watchdog reopening a dropped listener. See lifecycle().
///
///   resident  (KEVPHONE) the weights kept loaded: a send from cold opens chat (one load); then chats and decides,
///           all over HTTP, each judged against the Mac AND against the session's load count (a warm request must
///           not load); the swap to the decider and back is one load each; away past the grace closes the session
///           and gives the GBs back. See resident().
///
///   mem     (G18) the memory levers (pack map, hot state, KV cap: MemLevers.swift) on the chat model: footprint and
///           prefill per step, ids judged against the Mac's flags-off ids. See EngineCheckMem.swift.
///
///   vision  (G11) the CoreML detector (+ segmenter) from Documents/ml, N frames, ms + phys_footprint +
///           os_proc_available_memory per frame; no LLM unless -visionWithChat. See EngineCheckVision.swift.
///
///   pocket  (G27) photo → boxes → mask → sentence over POST /v1/bend/pocket (it turns -visionPocket on); ms, footprint,
///           the sentence, the mask PNG. See EngineCheckPocket.swift and VisionPocket.swift.
///
///   ears    (G38) EARS (the Metal Qwen3-ASR tower, -voiceASR ears) hearing golden clips through Voice.listen/release;
///           text (and ids) judged against the EARS-G0 goldens, ms + footprint per turn. See EngineCheckEars.swift.
///
/// Lines start with `BEND check`; scripts/engine-device-run.sh greps them.
enum EngineCheck {
    static let p2Text = "Explain in two sentences why the sky is blue, then list three colors of a sunset."
    static let p2Ids: [UInt32] = [248045, 846, 198, 814, 20139, 303, 1330, 22157, 3069, 279, 12515, 369, 6105, 11, 1179, 1103, 2250, 7736, 314, 264, 41564, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
    static let macP2: [UInt32] = [760, 12515, 7701, 6105, 1521, 33470, 303, 279, 8964, 579, 16078, 43000, 2716, 89661, 314, 37728, 11, 1680, 430, 6105, 321, 76990, 11, 777, 13057, 1056, 975, 7736, 13, 10022, 76990, 3000]

    static func say(_ line: String) {
        EngineManager.console("BEND check \(line)")
    }

    static var fixtures: URL {
        if let dir = ProcessInfo.processInfo.environment["BEND_CHECK_DIR"] { return URL(fileURLWithPath: dir) }
        return Bundle.main.bundleURL.appendingPathComponent("engine-check")
    }

    static func startIfAsked(key: String) {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-engineCheck"), i + 1 < args.count else { return }
        let which = args[i + 1]
        let exitAfter = args.contains("-engineCheckExit")
        Task.detached {
            try? await Task.sleep(for: .seconds(1)) // the listeners come up on their queues
            let ok: Bool
            switch which {
            case "chat": ok = await chat(key: key)
            case "decide": ok = await decide(key: key)
            case "reopen": ok = await reopen(key: key)
            case "lifecycle": ok = await lifecycle(key: key)
            case "resident": ok = await resident(key: key)
            case "look": ok = await look(key: key)  // IMGPERF (EngineCheckLook.swift)
            case "look-golden": ok = await lookGolden(key: key)  // BENDHARVEST H2: the crew's R19 invoice golden (EngineCheckLook.swift)
            case "vision": ok = await vision(key: key)  // G11: CoreML detector + segmenter (EngineCheckVision.swift)
            case "mem": ok = await mem(key: key)  // G18: the memory levers' footprint (EngineCheckMem.swift)
            case "pocket": ok = await pocket(key: key)  // G27: photo → boxes → mask → sentence (EngineCheckPocket.swift)
            #if BEND_EARS
            case "ears": ok = await ears(key: key)  // G38: the ASR tower through the voice path (EngineCheckEars.swift)
            #endif
            default: say("unknown check \(which) (chat | decide | reopen | lifecycle | resident | look | look-golden | vision | mem | pocket)"); ok = false
            }
            say("RESULT \(which) \(ok ? "PASS" : "FAIL")")
            if exitAfter { exit(ok ? 0 : 1) }
        }
    }

    static func post(_ path: String, _ body: Data, key: String) -> URLRequest {
        var r = URLRequest(url: URL(string: "http://127.0.0.1:\(EngineManager.enginePort)\(path)")!)
        r.httpMethod = "POST"
        r.httpBody = body
        r.timeoutInterval = 600
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.setValue(key, forHTTPHeaderField: "X-Api-Key")
        return r
    }

    static func chat(key: String) async -> Bool {
        let body = try! JSONSerialization.data(withJSONObject: [
            "messages": [["role": "user", "content": p2Text]], "stream": true, "max_tokens": 32, "temperature": 0.7,
            "chat_template_kwargs": ["enable_thinking": false],
        ])
        let t0 = Date()
        do {
            let (bytes, response) = try await URLSession.shared.bytes(for: post("/v1/chat/completions", body, key: key))
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                var text = ""
                for try await line in bytes.lines { text += line }
                say("chat HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1): \(text)")
                return false
            }
            var text = ""
            var deltas = 0
            var firstMs: Double?
            var llb: [String: Any] = [:]
            for try await line in bytes.lines where line.hasPrefix("data:") {
                let data = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                if data == "[DONE]" { break }
                guard let frame = try? JSONSerialization.jsonObject(with: Data(data.utf8)) as? [String: Any] else { continue }
                if let error = frame["error"] { say("chat stream error: \(error)"); return false }
                if let piece = ((frame["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any])?["content"] as? String {
                    if firstMs == nil { firstMs = Date().timeIntervalSince(t0) * 1000 }
                    text += piece
                    deltas += 1
                }
                if let x = frame["llb"] as? [String: Any] { llb = x }
            }
            let wall = Date().timeIntervalSince(t0) * 1000
            let prompt = (llb["prompt_ids"] as? [NSNumber])?.map { $0.uint32Value } ?? []
            let ids = (llb["ids"] as? [NSNumber])?.map { $0.uint32Value } ?? []
            say("chat reply \(String(reflecting: text))")
            say("chat ids \(ids.map(String.init).joined(separator: " "))")
            let promptOK = prompt == p2Ids
            say("chat PROMPT \(promptOK ? "OK" : "DIFF"): Eureka's template + the engine's tokenizer gave \(prompt.count) ids (p2 has \(p2Ids.count))")
            let n = min(ids.count, macP2.count)
            let parityOK: Bool
            if let k = (0..<n).first(where: { ids[$0] != macP2[$0] }) {
                say("chat PARITY DIFF vs Mac: first divergence at new id \(k): bend=\(ids[k]) mac=\(macP2[k])")
                parityOK = false
            } else {
                parityOK = n == macP2.count
                say("chat PARITY \(parityOK ? "OK" : "SHORT") vs Mac (\(n)/\(macP2.count) ids identical)")
            }
            say(String(format: "chat timing deltas=%d first_delta_ms=%.0f wall_ms=%.0f load_ms=%@ prefill_ms=%@ decode_ms_per_token=%@",
                       deltas, firstMs ?? -1, wall, "\(llb["load_ms"] ?? "-")", "\(llb["prefill_ms"] ?? "-")", "\(llb["decode_ms_per_token"] ?? "-")"))
            return promptOK && parityOK && deltas > 0
        } catch {
            say("chat request failed: \(error)")
            return false
        }
    }

    static func decide(key: String) async -> Bool {
        guard let body = try? Data(contentsOf: fixtures.appendingPathComponent("travel.json")),
              let mac = try? String(contentsOf: fixtures.appendingPathComponent("travel.mac.json"), encoding: .utf8) else {
            say("decide: fixtures missing in \(fixtures.path)")
            return false
        }
        let t0 = Date()
        do {
            let (data, response) = try await URLSession.shared.data(for: post("/v1/systemone", body, key: key))
            let wall = Date().timeIntervalSince(t0) * 1000
            let reply = String(decoding: data, as: UTF8.self)
            say("decide HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1) wall_ms=\(Int(wall))")
            say("decide reply \(reply)")
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let dest = (obj["answers"] as? [String: Any])?["destination"] as? [String: Any],
                  let choice = dest["choice"] as? String,
                  let p = (dest["probabilities"] as? [String: Any])?[choice] as? Double else {
                say("decide: no destination answer")
                return false
            }
            say(String(format: "decide DECIDE travel destination=%@ p=%.6f", choice, p))
            let same = reply.trimmingCharacters(in: .whitespacesAndNewlines) == mac.trimmingCharacters(in: .whitespacesAndNewlines)
            say("decide MACREF travel vs Mac fast llb_decide: \(same ? "BYTE-IDENTICAL" : "DIFFERS")")
            return choice == "Amsterdam" && abs(p - 0.5549) < 0.001 && same
        } catch {
            say("decide request failed: \(error)")
            return false
        }
    }

    /// Chat with a prompt other than p2 (a different prompt length and reply), over HTTP: answered or not.
    static func chatOther(key: String) async -> Bool {
        let body = try! JSONSerialization.data(withJSONObject: [
            "messages": [["role": "user", "content": "Name one fruit."]], "stream": false, "max_tokens": 8,
            "chat_template_kwargs": ["enable_thinking": false],
        ])
        do {
            let (data, response) = try await URLSession.shared.data(for: post("/v1/chat/completions", body, key: key))
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            let text = ((obj?["choices"] as? [[String: Any]])?.first?["message"] as? [String: Any])?["content"] as? String
            let ids = ((obj?["llb"] as? [String: Any])?["ids"] as? [NSNumber])?.map { $0.stringValue }.joined(separator: " ") ?? "-"
            say("chat-other HTTP \(code) reply \(String(reflecting: text ?? String(decoding: data, as: UTF8.self))) ids \(ids)")
            return code == 200 && !(text ?? "").isEmpty
        } catch {
            say("chat-other request failed: \(error)")
            return false
        }
    }

    /// chat, chat, decide, chat in one process, every run through Bend's HTTP layer.
    static func reopen(key: String) async -> Bool {
        say("reopen footprint_mb=\(EngineManager.footprint()) before any run")
        let steps: [(String, () async -> Bool)] = [
            ("chat (p2, parity vs Mac)", { await chat(key: key) }),
            ("chat (distinct prompt)", { await chatOther(key: key) }),
            ("decide (travel, byte-identical vs Mac)", { await decide(key: key) }),
            ("chat again (p2, parity vs Mac)", { await chat(key: key) }),
        ]
        var answered = 0
        for (n, (what, run)) in steps.enumerated() {
            let ok = await run()
            if ok { answered += 1 }
            say("reopen step \(n + 1) \(what) \(ok ? "answered" : "FAILED")")
            // Sample after the run has released (it ends ~20 ms after the reply's <|im_end|>).
            while LLBEngine.shared.runningBy != nil { try? await Task.sleep(for: .milliseconds(20)) }
            say("reopen step \(n + 1) footprint_mb=\(EngineManager.footprint()) \(LLBEngine.releaseReport)")
        }
        say("reopen runs answered \(answered)/\(steps.count) in one process")
        return answered == steps.count
    }

    // MARK: resident (KEVPHONE)

    /// The session's load count (llb_resident_report's loads=).
    private static var loads: Int {
        let r = LLBEngine.residentReport
        guard let at = r.range(of: "loads=") else { return -1 }
        return Int(r[at.upperBound...].prefix { $0.isNumber }) ?? -1
    }

    static func resident(key: String) async -> Bool {
        var ok = true
        func expect(_ cond: Bool, _ what: String) {
            say("resident \(cond ? "OK" : "FAIL") \(what)")
            ok = ok && cond
        }
        guard LLBEngine.residentOn else {
            say("resident: the mode is off (-residentOff)")
            return false
        }
        let reps = Int(ProcessInfo.processInfo.environment["BEND_CHECK_REPS"] ?? "") ?? 5
        let grace = await EngineManager.shared.lifecycle.policy.grace
        say("resident reps=\(reps) grace_s=\(grace) footprint_mb=\(EngineManager.footprint()) | \(LLBEngine.residentReport)")
        func idle() async { while LLBEngine.shared.runningBy != nil { try? await Task.sleep(for: .milliseconds(20)) } }

        // A request judged twice: its answer (parity / byte-identical) and the loads it caused.
        func step(_ what: String, wantLoads: Int, _ run: () async -> Bool) async {
            let l0 = loads
            let t0 = Date()
            let answered = await run()
            await idle()
            let l1 = loads
            expect(answered && l1 - l0 == wantLoads, String(format: "%@: answered=%@ loads %d → %d (%@) wall_ms=%.0f footprint_mb=%d", what,
                answered ? "yes" : "NO", l0, l1, l1 == l0 ? "no reload" : "loaded", Date().timeIntervalSince(t0) * 1000, EngineManager.footprint()))
        }
        // A hot launch (chat was warm at the last exit) has already opened chat: then the first send loads nothing.
        await EngineManager.shared.lifecycle.settle()
        let eager = LLBEngine.shared.residentDir == EngineManager.modelDir(.chat).path
        await step(eager ? "chat, eager-warmed at launch (p2 parity): no load on the send" : "chat from cold (p2 parity): the warm opens chat, one load",
                   wantLoads: eager ? 0 : 1) { await chat(key: key) }
        for r in 1...reps {
            await step("chat warm #\(r) (p2 parity)", wantLoads: 0) { await chat(key: key) }
        }
        await step("chat warm (another prompt)", wantLoads: 0) { await chatOther(key: key) }
        await step("decide: the swap to the decider (travel byte-identical), one load", wantLoads: 1) { await decide(key: key) }
        for r in 1...reps {
            await step("decide warm #\(r) (travel byte-identical)", wantLoads: 0) { await decide(key: key) }
        }
        await step("chat: the swap back (p2 parity), one load", wantLoads: 1) { await chat(key: key) }
        await step("chat warm after the swap (p2 parity)", wantLoads: 0) { await chat(key: key) }

        let held = EngineManager.footprint()
        _ = await control("POST", "/engines/visibility?state=hidden", key: key)
        try? await Task.sleep(for: .seconds(grace + 1.5))
        await EngineManager.shared.lifecycle.settle()
        let s = await control("GET", "/engines/status", key: key)
        expect(LLBEngine.shared.residentDir == nil && phase(s, .chat) == "cold" && phase(s, .decision) == "cold",
               "away past the grace: session closed, footprint_mb \(held) → \(EngineManager.footprint()): \(phases(s)) | \(LLBEngine.releaseReport)")
        let lb = loads
        _ = await control("POST", "/engines/visibility?state=visible", key: key)
        await EngineManager.shared.lifecycle.settle()
        expect(loads == lb + 1 && LLBEngine.shared.residentDir == EngineManager.modelDir(.chat).path,
               "back: chat was warm at exit, so the return re-opened it before any send (loads \(lb) → \(loads))")
        await step("back: the first send after the return (p2 parity)", wantLoads: 0) { await chat(key: key) }
        say("resident \(LLBEngine.residentReport)")
        return ok
    }

    // MARK: lifecycle (R14)

    static func control(_ method: String, _ path: String, key: String) async -> [String: Any] {
        var r = URLRequest(url: URL(string: "http://127.0.0.1:\(EngineManager.controlPort)\(path)")!)
        r.httpMethod = method
        r.timeoutInterval = 600
        r.setValue(key, forHTTPHeaderField: "X-Api-Key")
        guard let (data, _) = try? await URLSession.shared.data(for: r) else { return [:] }
        return (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    private static func phases(_ status: [String: Any]) -> String {
        EngineID.allCases.map { id in
            let e = status[id.rawValue] as? [String: Any] ?? [:]
            return "\(id.rawValue)=\(e["lifecycle"] ?? "?")/\(e["state"] ?? "?")"
        }.joined(separator: " ")
    }

    private static func phase(_ status: [String: Any], _ id: EngineID) -> String {
        (status[id.rawValue] as? [String: Any])?["lifecycle"] as? String ?? "?"
    }

    /// The lifecycle through Bend's own layer, driven as the web drives it (control port: status, visibility; the
    /// engine port's requests ensure their engine), with a short grace (BEND_GRACE_SECONDS):
    ///   launch (cold; or chat eager-warmed if it was warm at the last exit) → a send warms chat and answers (p2 parity) → away + back within the grace: kept warm →
    ///   away past the grace: both released → back: chat eager-warmed (it was hot) → decide from cold (byte-identical)
    ///   → the engine listener dropped while away → back: the watchdog reopens it → chat answers again (p2 parity).
    static func lifecycle(key: String) async -> Bool {
        var ok = true
        func expect(_ cond: Bool, _ what: String) {
            say("lifecycle \(cond ? "OK" : "FAIL") \(what)")
            ok = ok && cond
        }
        let grace = await EngineManager.shared.lifecycle.policy.grace
        var s = await control("GET", "/engines/status", key: key)
        let hot = (s["lifecycle"] as? [String: Any])?["recentlyHot"] as? Bool ?? false
        say("lifecycle grace_s=\(grace) recently_hot=\(hot) footprint_mb=\(EngineManager.footprint()) \(phases(s))")
        if hot {
            await EngineManager.shared.lifecycle.settle()
            s = await control("GET", "/engines/status", key: key)
            expect(phase(s, .chat) == "warm" && phase(s, .decision) == "cold", "launch: chat was warm at the last exit → eager-warmed, decision not: \(phases(s))")
        } else {
            expect(phase(s, .chat) == "cold" && phase(s, .decision) == "cold", "launch: both cold (not recently hot, nothing loaded)")
        }

        expect(await chat(key: key), "send from cold: chat warmed and answered (p2 parity)")
        s = await control("GET", "/engines/status", key: key)
        expect(phase(s, .chat) == "warm", "chat warm after the send: \(phases(s))")

        _ = await control("POST", "/engines/visibility?state=hidden", key: key)
        try? await Task.sleep(for: .seconds(grace / 2))
        _ = await control("POST", "/engines/visibility?state=visible", key: key)
        await EngineManager.shared.lifecycle.settle()
        try? await Task.sleep(for: .seconds(grace))
        s = await control("GET", "/engines/status", key: key)
        expect(phase(s, .chat) == "warm", "away \(grace / 2) s, back within the grace: still warm, nothing released: \(phases(s))")

        try? await Task.sleep(for: .seconds(1))
        let before = EngineManager.footprint()
        _ = await control("POST", "/engines/visibility?state=hidden", key: key)
        try? await Task.sleep(for: .seconds(grace + 1.5))
        await EngineManager.shared.lifecycle.settle()
        s = await control("GET", "/engines/status", key: key)
        expect(phase(s, .chat) == "cold" && phase(s, .decision) == "cold",
               "away past the grace: released, footprint_mb \(before) → \(EngineManager.footprint()): \(phases(s))")

        _ = await control("POST", "/engines/visibility?state=visible", key: key)
        await EngineManager.shared.lifecycle.settle()
        s = await control("GET", "/engines/status", key: key)
        expect(phase(s, .chat) == "warm" && phase(s, .decision) == "cold", "back: chat eager-warmed (it was warm at exit), decision not: \(phases(s))")

        expect(await decide(key: key), "decision from cold: loaded, then the verdict (travel byte-identical)")
        while LLBEngine.shared.runningBy != nil { try? await Task.sleep(for: .milliseconds(20)) }
        s = await control("GET", "/engines/status", key: key)
        expect(phase(s, .decision) == "warm", "decision warm after: \(phases(s))")

        _ = await control("POST", "/engines/visibility?state=hidden", key: key)
        await EngineManager.shared.dropEngineListener()
        _ = await control("POST", "/engines/visibility?state=visible", key: key)
        await EngineManager.shared.lifecycle.settle()
        s = await control("GET", "/engines/status", key: key)
        // KEVPHONE: with one model resident (exclusive), the decider is the warm engine here (it swapped chat out), so
        // the watchdog relaunches the decider; chat was not warm at the exit, so nothing eager-warms it.
        let watched: EngineID = await EngineManager.shared.lifecycle.policy.exclusive ? .decision : .chat
        expect(phase(s, watched) == "warm", "watchdog: the dropped listener was noticed and reopened, \(watched.rawValue) warm: \(phases(s))")
        expect(await chat(key: key), "the next send just works (p2 parity)")
        while LLBEngine.shared.runningBy != nil { try? await Task.sleep(for: .milliseconds(20)) }

        s = await control("GET", "/engines/status", key: key)
        let states = EngineID.allCases.map { (s[$0.rawValue] as? [String: Any])?["state"] as? String ?? "?" }
        expect(!states.contains("error"), "no engine left in error: \(phases(s)) footprint_mb=\(EngineManager.footprint())")
        return ok
    }
}
