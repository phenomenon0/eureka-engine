#if BEND_EARS
import Foundation

/// G38: `-engineCheck ears` (it turns -voiceASR ears on): EARS through the app's own voice path, as a person's turn runs
/// it. Each clip goes Voice.listen(fixture:) → FixtureSource (real-time pace, 100 ms buffers, as the mic delivers) →
/// EarsListener → Voice.release → the turn's text, `-earsIters N` times (default 1). Clips: `-earsClips <dir>` | env
/// BEND_EARS_CLIPS | <ears dir>/clips — every `<name>.wav` (16 kHz mono) with its golden `<name>.txt` (the reference's
/// raw answer, "language English<asr_text>…") and, if present, `<name>.tsv` (EARS-G0 transcript.tsv: step, id, …).
/// One line per turn:
///   BEND check ears turn clip=… iter=1 text=MATCH ids=IDENTICAL(41/41) heard_by=… final_ms=… total_ms=… tower_ms=… …
/// then `ears summary …`. Without -visionWithChat chat stays cold (EngineCheck.visionOnly), so the footprint is the
/// voice + ears alone (swap mode); `-visionWithChat` warms the chat model first (the co-resident row).
/// PASS: every turn heard by EARS, its text equal to the golden's words (and its ids to the golden ids where given).
extension EngineCheck {
    static func ears(key: String) async -> Bool {
        let args = ProcessInfo.processInfo.arguments
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        guard EarsASR.enabled else { say("ears: not enabled (models in \(EarsASR.dir.path)?)"); return false }
        if args.contains("-visionWithChat") {
            let warmed = await chat(key: key)
            while LLBEngine.shared.runningBy != nil { try? await Task.sleep(for: .milliseconds(20)) }
            say("ears with chat warm (p2 \(warmed ? "answered" : "FAILED")) footprint_mb=\(EngineManager.footprint())")
            guard warmed else { return false }
        }
        let dir = URL(fileURLWithPath: value("-earsClips") ?? ProcessInfo.processInfo.environment["BEND_EARS_CLIPS"]
                      ?? EarsASR.dir.appendingPathComponent("clips").path)
        let clips = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "wav" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !clips.isEmpty else { say("ears: no clips in \(dir.path)"); return false }
        let iters = max(1, value("-earsIters").flatMap(Int.init) ?? 1)
        let base = VisionCheck.footprint()
        say("ears clips=\(clips.count) iters=\(iters) llm=\(EarsASR.llmOnGPU ? "gpu" : "cpu") dir=\(EarsASR.dir.path) baseline \(VisionCheck.mem) peak_mb=\(EarsASR.peakFootprint())")
        var ok = true, matched = 0, idsSame = 0, idsJudged = 0, turns = 0
        var finals: [Double] = [], totals: [Double] = [], rtf: [Double] = []
        for clip in clips {
            let stem = clip.deletingPathExtension()
            let golden = (try? String(contentsOf: stem.appendingPathExtension("txt"), encoding: .utf8)).map(EarsASR.words)
            let goldIds: [Int32]? = (try? String(contentsOf: stem.appendingPathExtension("tsv"), encoding: .utf8)).map { tsv in
                tsv.split(separator: "\n").dropFirst().compactMap { $0.split(separator: "\t").dropFirst().first.flatMap { Int32($0) } }
            }
            for iter in 1...iters {
                let receipt: Voice.Receipt
                do {
                    let turn = try await Voice.shared.listen(fixture: clip.path)
                    receipt = try await Voice.shared.release(turn)
                    await Voice.shared.cancel(reason: "ears check")
                } catch { say("ears turn clip=\(clip.lastPathComponent) error: \(error.localizedDescription)"); ok = false; continue }
                turns += 1
                let h = EarsASR.last
                let finalMs = (receipt.finalAt ?? 0) - (receipt.speechEndAt ?? 0)
                let textOK = golden.map { $0 == receipt.text }
                if textOK == true { matched += 1 }
                var idsNote = "-"
                if let goldIds, let h {
                    idsJudged += 1
                    let same = goldIds == h.ids
                    if same { idsSame += 1 }
                    idsNote = same ? "IDENTICAL(\(h.ids.count)/\(goldIds.count))" : "DIFFER(\(zip(goldIds, h.ids).prefix { $0 == $1 }.count)/\(goldIds.count))"
                    if !same { ok = false }
                }
                if receipt.heardBy != EarsASR.label || textOK == false || textOK == nil && receipt.text.isEmpty { ok = false }
                finals.append(finalMs)
                if let h { totals.append(h.totalMs); rtf.append(h.totalMs / 1000 / max(h.seconds, 0.01)) }
                say(String(format: "ears turn clip=%@ iter=%d text=%@ ids=%@ heard_by=%@ final_ms=%.1f %@",
                           clip.lastPathComponent, iter, textOK == nil ? "NO-GOLDEN" : textOK! ? "MATCH" : "DIFFER", idsNote, receipt.heardBy,
                           finalMs, h.map { String($0.line.dropFirst("ears heard ".count)) } ?? "-"))
                if iter == 1 { say("ears said clip=\(clip.lastPathComponent) \(receipt.text.debugDescription)") }
                if textOK == false { say("ears golden clip=\(clip.lastPathComponent) \((golden ?? "").debugDescription)") }
            }
        }
        say(String(format: "ears summary turns=%d text_match=%d/%d ids_identical=%d/%d final_ms_p50=%.1f final_ms_p90=%.1f total_ms_p50=%.1f rtf_p50=%.3f footprint_mb_base=%d peak_mb=%d available_mb=%d",
                   turns, matched, turns, idsSame, idsJudged, VisionCheck.percentile(finals, 0.5), VisionCheck.percentile(finals, 0.9),
                   VisionCheck.percentile(totals, 0.5), VisionCheck.percentile(rtf, 0.5), base, EarsASR.peakFootprint(), VisionCheck.available()))
        return ok && turns == clips.count * iters
    }
}
#endif
