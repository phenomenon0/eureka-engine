import Foundation

/// G27: `-engineCheck pocket` (it turns -visionPocket on): the pocket skeleton over the app's own HTTP surface, as the
/// web would call it. Each bundled photo (engine-check/look-news-1280.jpg, look-news-wire.jpg; or `-pocketPhoto <path>`)
/// goes `-pocketIters N` times (default 10; the first is the cold one: package load + first ANE dispatch) to
/// POST /v1/bend/pocket, one line each:
///   BEND check pocket answered photo=… iter=2 http_ms=… ms=… det_ms=… enc_ms=… prompt_ms=… dec_ms=… boxes=3 top=person:0.91
///        mask_area=0.212 prompt=point footprint_mb=… available_mb=…
/// then `pocket say photo=… "<the sentence>"` and the mask PNG in Caches/pocket/. Packages as `-engineCheck vision`
/// stages them (Documents/ml: rfdetr-small-int8.mlpackage + edgetam-box/; `-pocketSeg edgetam|sam21-tiny|none`). Without
/// `-visionWithChat` chat stays cold (EngineCheck.visionOnly). PASS: every request answered 200 with a box, a non-empty
/// mask (unless -pocketSeg none) and a sentence.
extension EngineCheck {
    static func pocket(key: String) async -> Bool {
        let args = ProcessInfo.processInfo.arguments
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        if args.contains("-visionWithChat") {
            let warmed = await chat(key: key)
            while LLBEngine.shared.runningBy != nil { try? await Task.sleep(for: .milliseconds(20)) }
            say("pocket with chat warm (p2 \(warmed ? "answered" : "FAILED")) footprint_mb=\(EngineManager.footprint())")
            guard warmed else { return false }
        }
        let iters = max(1, value("-pocketIters").flatMap(Int.init) ?? 10)
        // G38: -pocketPhoto <path> (one photo) or -pocketPhotos <dir> (every jpg/png in it, sorted)
        let photoDir = value("-pocketPhotos").map { d in
            ((try? FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: d), includingPropertiesForKeys: nil)) ?? [])
                .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        let photos = photoDir ?? value("-pocketPhoto").map { [URL(fileURLWithPath: $0)] }
            ?? ["look-news-1280.jpg", "look-news-wire.jpg"].map { fixtures.appendingPathComponent($0) }
        let out = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("pocket")
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let base = VisionCheck.footprint()
        say("pocket iters=\(iters) photos=\(photos.map(\.lastPathComponent).joined(separator: ",")) baseline \(VisionCheck.mem)")
        var ok = true, peak = base, minAvail = VisionCheck.available()
        var http: [Double] = [], engine: [Double] = [], cold: [Double] = []
        for photo in photos {
            guard let data = try? Data(contentsOf: photo) else { say("pocket: no photo at \(photo.path)"); return false }
            let mime = photo.pathExtension.lowercased() == "png" ? "png" : "jpeg"
            let body = try! JSONSerialization.data(withJSONObject: ["url": "data:image/\(mime);base64," + data.base64EncodedString(), "mask": true])
            for iter in 1...iters {
                let t0 = Date()
                let reply: (Data, URLResponse)
                do { reply = try await URLSession.shared.data(for: post("/v1/bend/pocket", body, key: key)) } catch {
                    say("pocket error: \(error.localizedDescription)"); return false
                }
                let httpMs = Date().timeIntervalSince(t0) * 1000
                let code = (reply.1 as? HTTPURLResponse)?.statusCode ?? -1
                guard code == 200, let j = try? JSONSerialization.jsonObject(with: reply.0) as? [String: Any] else {
                    say("pocket HTTP \(code): \(String(decoding: reply.0.prefix(300), as: UTF8.self))"); return false
                }
                let boxes = j["boxes"] as? [[String: Any]] ?? [], mask = j["mask"] as? [String: Any]
                let sentence = j["say"] as? String ?? "", ms = j["ms"] as? [String: Double] ?? [:]
                let top = boxes.first.map { "\(($0["label"] as? String ?? "?").replacingOccurrences(of: " ", with: "_")):\(String(format: "%.2f", $0["score"] as? Double ?? 0))" } ?? "-"
                let area = mask?["area"] as? Double ?? -1
                let fp = j["footprint_mb"] as? Int ?? -1, av = j["available_mb"] as? Int ?? -1
                peak = max(peak, fp)
                if av >= 0 { minAvail = min(minAvail, av) }
                let parts = ["pre", "det", "enc", "prompt", "dec", "post"].compactMap { k in ms[k].map { String(format: "%@_ms=%.1f", k, $0) } }
                say(String(format: "pocket answered photo=%@ iter=%d http_ms=%.1f ms=%.1f %@ boxes=%d top=%@ mask_area=%.3f prompt=%@ footprint_mb=%d available_mb=%d",
                           photo.lastPathComponent, iter, httpMs, ms["total"] ?? -1, parts.joined(separator: " "), boxes.count, top, area,
                           j["prompt"] as? String ?? "?", fp, av))
                if iter == 1 {
                    cold.append(httpMs)
                    say("pocket say photo=\(photo.lastPathComponent) \"\(sentence)\"")
                    // G38: who found them, and every box (label:score:x0,y0,x1,y1), so two runs (CoreML EP vs CPU EP) compare offline
                    say("pocket boxes photo=\(photo.lastPathComponent) det=\((j["det"] as? String ?? "?").replacingOccurrences(of: " ", with: "/")) " + boxes.map { b in
                        let xy = (b["box"] as? [Double] ?? []).map { String(format: "%.4f", $0) }.joined(separator: ",")
                        return "\((b["label"] as? String ?? "?").replacingOccurrences(of: " ", with: "_")):\(String(format: "%.4f", b["score"] as? Double ?? 0)):\(xy)"
                    }.joined(separator: " "))
                    if let png = (mask?["png"] as? String).flatMap({ Data(base64Encoded: $0) }) {
                        try? png.write(to: out.appendingPathComponent(photo.lastPathComponent + ".mask.png"))
                    }
                } else {
                    http.append(httpMs); engine.append(ms["total"] ?? -1)
                }
                let segOff = (j["seg"] as? String) == "none"
                if boxes.isEmpty || sentence.isEmpty || (!segOff && area <= 0) { say("pocket iter \(iter): no box, no mask or no sentence"); ok = false }
            }
        }
        say(String(format: "pocket summary photos=%d iters=%d cold_http_ms=%.1f http_ms_p50=%.1f http_ms_p90=%.1f ms_p50=%.1f ms_p90=%.1f footprint_mb_base=%d footprint_mb_peak=%d delta_mb=%d available_mb_min=%d masks=%@",
                   photos.count, iters, cold.first ?? -1, VisionCheck.percentile(http, 0.5), VisionCheck.percentile(http, 0.9),
                   VisionCheck.percentile(engine, 0.5), VisionCheck.percentile(engine, 0.9), base, peak, peak - base, minAvail, out.path))
        return ok
    }
}
