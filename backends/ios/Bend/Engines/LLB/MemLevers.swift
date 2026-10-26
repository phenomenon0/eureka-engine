import Foundation

/// G18: the engine's memory levers (G3, engine branch g18-memlevers-eng), each behind its own flag and OFF by default.
/// With every flag off, the app makes exactly the calls it made before (llb_open at cap 4096, llb_generate, llb_free):
/// the library's flags-off ids are byte-identical to the vendored imgfast library's (G3 + G18 receipts).
///
///   pack map    `-llbPackMap` / BEND_PACK_MAP=1 / UserDefaults bend.llb.packMap
///               LLB_PACK_MAP=<Caches>/llb-<program>/pack-<model dir>.pack is set before every llb_open of that
///               model: the first load writes the packed weights there (+0.3 s, +3.2 GB disk per model), later loads
///               map them (clean file pages). Mac: footprint 3949 → 846 MB. iOS: what the G18 device run measures.
///               Off when the volume lacks the room (the model's GGUF × 1.3 + 1 GB); the engine keys the file to the
///               GGUF's size + mtime and to its own path (G18), so the Qwen ↔ Kev swap keeps one file per model.
///   hot state   `-llbHotState` / BEND_HOT_STATE=1 / UserDefaults bend.llb.hotState
///               chat runs llb_generate_px with the system turn as the kept prefix (restored, not prefilled, on every
///               later turn); the session's text snapshots go to <cache>/hot-<model dir>.px before the session closes
///               (release, or a swap to the other model) and are read back after its next llb_open. A memory
///               warning drops them (llb_prefix_forget). Costs ~60 MB RAM per kept snapshot; saves prefill only.
///   reset       `-llbLeversReset` (launch flag only): deletes this program's pack and hot files at boot, before any
///               llb_open, so a measurement starts from a known state (scripts/mem-levers.sh: the first launch of a set).
///   KV cap      `-llbKvCap N` / BEND_KV_CAP=N / UserDefaults bend.llb.kvCap (1024 | 2048 | 4096 | 8192; default 4096)
///               the KV is f32, 64 KiB per id of cap, allocated whole at open: 2048 is −128 MB. A request longer than
///               the cap runs one-shot (the session closes for it).
enum MemLevers {
    private static let args = ProcessInfo.processInfo.arguments
    private static let env = ProcessInfo.processInfo.environment

    private static func flag(_ arg: String, _ envKey: String, _ defaultsKey: String) -> Bool {
        args.contains(arg) || env[envKey] == "1" || UserDefaults.standard.bool(forKey: defaultsKey)
    }

    static let packMap = flag("-llbPackMap", "BEND_PACK_MAP", "bend.llb.packMap")
    static let hotState = flag("-llbHotState", "BEND_HOT_STATE", "bend.llb.hotState")

    static let kvCap: Int32 = {
        var asked: Int?
        if let i = args.firstIndex(of: "-llbKvCap"), i + 1 < args.count { asked = Int(args[i + 1]) }
        asked = asked ?? env["BEND_KV_CAP"].flatMap(Int.init)
        if asked == nil, UserDefaults.standard.object(forKey: "bend.llb.kvCap") != nil {
            asked = UserDefaults.standard.integer(forKey: "bend.llb.kvCap")
        }
        guard let n = asked, [1024, 2048, 4096, 8192].contains(n) else { return 4096 }
        return Int32(n)
    }()

    /// `-llbLeversReset`: the pack (+ .idx) and hot files of every model in the cache dir, removed. Called at boot,
    /// before the lifecycle opens anything (a mapped pack file must never be removed under a live session).
    static func resetIfAsked() -> String? {
        guard args.contains("-llbLeversReset") else { return nil }
        let fm = FileManager.default
        let names = (try? fm.contentsOfDirectory(atPath: LLBEngine.cacheDir.path)) ?? []
        let gone = names.filter { ($0.hasPrefix("pack-") && ($0.hasSuffix(".pack") || $0.hasSuffix(".pack.idx")))
            || ($0.hasPrefix("hot-") && $0.hasSuffix(".px")) }
        for n in gone { try? fm.removeItem(at: LLBEngine.cacheDir.appendingPathComponent(n)) }
        return "levers reset: removed \(gone.count) file(s)\(gone.isEmpty ? "" : " (" + gone.sorted().joined(separator: " ") + ")")"
    }

    /// One line for the boot log and the checks.
    static var line: String {
        "levers pack_map=\(packMap ? "on" : "off") hot_state=\(hotState ? "on" : "off") kv_cap=\(kvCap)"
    }

    static func packFile(_ dir: URL) -> URL {
        LLBEngine.cacheDir.appendingPathComponent("pack-\(dir.lastPathComponent).pack")
    }

    static func hotFile(_ dir: URL) -> URL {
        LLBEngine.cacheDir.appendingPathComponent("hot-\(dir.lastPathComponent).px")
    }

    /// Before llb_open of `dir` (under the run lock): LLB_PACK_MAP names this model's pack file, or is unset. The
    /// engine reads it at every load (G18), so each model gets its own file.
    static func beforeOpen(_ dir: URL) -> String? {
        guard packMap else { return nil }
        let pack = packFile(dir)
        if !FileManager.default.fileExists(atPath: pack.path) {
            // The first load writes the whole pack: only with the room for it (and some to spare).
            let gguf = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey]))?
                .filter { $0.pathExtension == "gguf" }
                .compactMap { try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }.max() ?? 0
            let free = (try? LLBEngine.cacheDir.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
                .volumeAvailableCapacityForImportantUsage ?? 0
            let need = Int64(Double(gguf) * 1.3) + (1 << 30)
            if free < need {
                unsetenv("LLB_PACK_MAP")
                return "pack map off for \(dir.lastPathComponent): free_mb=\(free >> 20) < need_mb=\(need >> 20)"
            }
        }
        setenv("LLB_PACK_MAP", pack.path, 1)
        return nil
    }

    /// The pack map's counters (llb_packmap_report: on|off mapped_mb= copied_mb= wrote_mb=).
    static var packReport: String { String(cString: llb_packmap_report()) }

    /// The kept prefix for a chat prompt: the system turn's id count, when the levers ask for it and the prompt starts
    /// with exactly those ids; 0 otherwise (plain llb_generate).
    static func prefix(messages: [LLBTokenizer.Message], promptIds: [UInt32], tok: LLBTokenizer) -> Int {
        guard hotState, LLBEngine.residentOn, let sys = LLBTokenizer.systemTurn(messages) else { return 0 }
        let ids = tok.encode(sys)
        guard ids.count >= 1, ids.count < promptIds.count, ids.count <= 1024, promptIds.starts(with: ids) else { return 0 }
        return ids.count
    }
}
