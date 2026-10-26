import Foundation

/// G18 `-engineCheck mem`: the memory levers (MemLevers.swift) measured in one launch, on the chat model, over the
/// same HTTP surface the web app uses. Run it once per lever set (launch flags) to fill G18's EXPECTED vs MEASURED table:
///
///   (none)            the baseline: the vendored behaviour (every lever off)
///   -llbKvCap 2048    the KV halved (expected −128 MB)
///   -llbPackMap       first launch writes the pack file, the next maps it (expected: Mac −3.1 GB; iOS is the question)
///   -llbHotState      the system turn kept, saved at release and restored after the next open (prefill 143 ids → ~0)
///
/// Steps: settle → turn A (system + U1: opens chat if cold; with hot state the system turn is kept) → turn B (system +
/// U2: restored in-session with hot state) → away past the grace (the session closes: hot state saved) → back (chat was
/// warm, so the lifecycle re-opens it: hot state loaded) → turn B again (restored from the file with hot state).
/// Every answer's ids must equal the Mac's flags-off ids for the same prompt (the G3 harness's system turn and U2: 164
/// ids, n_pre 143), so no lever may change an id. Pass `-graceSeconds 3` (or BEND_GRACE_SECONDS) for a short away.
///
/// Lines: `BEND check mem …` per step (footprint_mb / available_mb: PHONE-CEILING's two columns, fp + avail = the
/// limit), then one `mem summary …` line that scripts/mem-levers.sh tabulates.
extension EngineCheck {
    /// g3_host.c's SYS (the system turn's content), U1 and U2.
    static let memSys = "You are Bend, a calm personal assistant that lives on the person's own phone. You help them decide, plan and remember. Keep answers short and concrete: one or two sentences unless they ask for more. When they are choosing between options, name the option you would pick and the one reason that matters most. Never invent facts about their life; if you do not know, say so and ask one short question. You run fully on the device: nothing they say leaves the phone. Today is a weekday; the person is at home in the evening. Prefer plain words to jargon, avoid lists unless asked, and do not add follow-up offers at the end of a reply."
    static let memU1 = "What is a good way to wind down before sleep?"
    static let memU2 = "Should I cook pasta or order takeaway tonight?"
    /// Mac, flags off (G18 library = the vendored imgfast program; receipts g18/ref-{Q,K}.jsonl step B): the first 16
    /// new ids for system + U2. The check asks for 16, so the reply ends on length and carries its stats.
    static let memGoldU2: [String: [UInt32]] = [
        "qwen35-4b-q4km": [40, 4035, 3608, 16725, 35243, 1521, 424, 579, 3841, 321, 488, 3172, 599, 264, 7478, 14704],
        "kev-4b-q4km": [40, 4035, 4052, 16725, 35243, 13, 1049, 579, 3841, 11, 488, 628, 30069, 424, 11, 321],
    ]

    private struct MemTurn {
        var ids: [UInt32] = []
        var llb: [String: Any] = [:]
        var error: String?
        func num(_ k: String) -> Double { (llb[k] as? NSNumber)?.doubleValue ?? -1 }
    }

    private static func memTurn(_ user: String, key: String) async -> MemTurn {
        let body = try! JSONSerialization.data(withJSONObject: [
            "messages": [["role": "system", "content": memSys], ["role": "user", "content": user]],
            "stream": false, "max_tokens": 16, "chat_template_kwargs": ["enable_thinking": false],
        ])
        do {
            let (data, response) = try await URLSession.shared.data(for: post("/v1/chat/completions", body, key: key))
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let llb = json["llb"] as? [String: Any] else {
                return MemTurn(error: "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1): \(String(decoding: data.prefix(300), as: UTF8.self))")
            }
            return MemTurn(ids: (llb["ids"] as? [NSNumber])?.map { $0.uint32Value } ?? [], llb: llb)
        } catch {
            return MemTurn(error: "\(error)")
        }
    }

    static func mem(key: String) async -> Bool {
        var ok = true
        func expect(_ cond: Bool, _ what: String) {
            say("mem \(cond ? "OK" : "FAIL") \(what)")
            ok = ok && cond
        }
        let model = EngineManager.modelDir(.chat)
        let gold = memGoldU2[model.lastPathComponent]
        var peak = VisionCheck.footprint(), availMin = VisionCheck.available()
        func sample() -> String {
            let fp = VisionCheck.footprint(), av = VisionCheck.available()
            peak = max(peak, fp)
            if av >= 0 { availMin = availMin < 0 ? av : min(availMin, av) }
            return "footprint_mb=\(fp) available_mb=\(av)"
        }
        let base = VisionCheck.footprint()  // at the check's start (a hot launch has already opened chat)
        say("mem \(MemLevers.line) model=\(model.lastPathComponent) golden=\(gold == nil ? "none" : "mac-flags-off") \(sample())")
        await EngineManager.shared.lifecycle.settle()
        let eager = LLBEngine.shared.residentDir == model.path
        say("mem launch chat \(eager ? "eager-warmed (hot at the last exit)" : "cold") \(sample()) | \(LLBEngine.residentReport)")

        // One turn judged: the ids (vs the Mac's flags-off), and with hot state, the prefix it restored.
        func turn(_ name: String, _ user: String, wantRestored: Bool?) async -> MemTurn {
            let t = await memTurn(user, key: key)
            if let e = t.error {
                expect(false, "\(name): \(e)")
                return t
            }
            let fp = sample()
            var what = String(format: "%@: new=%d prefill_ms=%.0f load_ms=%.0f decode_ms_per_token=%.1f n_pre=%d prefix_ids=%d %@",
                              name, t.ids.count, t.num("prefill_ms"), t.num("load_ms"), t.num("decode_ms_per_token"),
                              Int(t.num("n_pre")), Int(t.num("prefix_ids")), fp)
            if user == memU2, let gold {
                what += t.ids == gold ? " ids=golden" : " ids DIFF vs golden: \(t.ids.map(String.init).joined(separator: " "))"
                expect(t.ids == gold, what)
            } else {
                say("mem \(what) ids=\(t.ids.map(String.init).joined(separator: " "))")
            }
            if let want = wantRestored, MemLevers.hotState {
                expect((t.num("prefix_ids") > 0) == want, "\(name): hot state \(want ? "restored" : "kept, nothing restored yet") (prefix_ids=\(Int(t.num("prefix_ids"))) of n_pre=\(Int(t.num("n_pre"))))")
            }
            return t
        }

        // A: kept the first time; restored already when this launch's open read a hot file from the last one (reported only).
        let a = await turn("turn A (system + U1)", memU1, wantRestored: nil)
        let open = VisionCheck.footprint()
        func lastLoadMs() -> String {
            LLBEngine.residentReport.components(separatedBy: " ").first { $0.hasPrefix("load_ms=") }.map { String($0.dropFirst(8)) } ?? "?"
        }
        let loadOpen = lastLoadMs()
        say("mem open \(LLBEngine.residentReport) | packmap \(MemLevers.packReport)")
        let b = await turn("turn B (system + U2, same session)", memU2, wantRestored: true)

        let grace = await EngineManager.shared.lifecycle.policy.grace
        _ = await control("POST", "/engines/visibility?state=hidden", key: key)
        try? await Task.sleep(for: .seconds(grace + 1.5))
        await EngineManager.shared.lifecycle.settle()
        let released = VisionCheck.footprint()
        let hotFile = MemLevers.hotFile(model)
        let hotMB = ((try? FileManager.default.attributesOfItem(atPath: hotFile.path)[.size] as? NSNumber)?.intValue ?? 0) >> 20
        expect(LLBEngine.shared.residentDir == nil, "away \(grace) s + 1.5: session closed, footprint_mb \(open) → \(released)\(MemLevers.hotState ? ", hot file \(hotMB) MB" : "") | \(LLBEngine.releaseReport)")

        _ = await control("POST", "/engines/visibility?state=visible", key: key)
        await EngineManager.shared.lifecycle.settle()
        let reopened = VisionCheck.footprint()
        let loadReopen = lastLoadMs()
        say("mem back: chat re-opened \(sample()) | \(LLBEngine.residentReport) | packmap \(MemLevers.packReport)")
        let c = await turn("turn B again (after the reopen)", memU2, wantRestored: true)

        let pm = MemLevers.packReport
        if MemLevers.packMap {
            func mb(_ k: String) -> Double {
                Double(pm.components(separatedBy: " ").first { $0.hasPrefix(k + "=") }?.dropFirst(k.count + 1) ?? "") ?? 0
            }
            expect(pm.hasPrefix("on") && mb("mapped_mb") + mb("wrote_mb") > 0,
                   "pack map in use: \(pm) (a launch without the file writes it, then the reopen maps it)")
        }
        let cap = LLBEngine.residentReport.components(separatedBy: " ").first { $0.hasPrefix("cap=") } ?? "cap=?"
        say(String(format: "mem summary %@ model=%@ %@ fp_start=%d fp_open=%d fp_after_B=%d fp_released=%d fp_reopened=%d fp_end=%d fp_peak=%d avail_min=%d load_ms_open=%@ load_ms_reopen=%@ prefill_ms_A=%.0f prefill_ms_B=%.0f prefill_ms_B2=%.0f prefix_ids_B=%d prefix_ids_B2=%d hot_file_mb=%d packmap=\"%@\"",
                   MemLevers.line, model.lastPathComponent, cap, base, open, Int(b.num("footprint_mb")), released, reopened,
                   VisionCheck.footprint(), peak, availMin, loadOpen, loadReopen, a.num("prefill_ms"),
                   b.num("prefill_ms"), c.num("prefill_ms"), Int(b.num("prefix_ids")), Int(c.num("prefix_ids")), hotMB, pm))
        return ok
    }
}
