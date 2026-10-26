import Foundation

/// PHONEUX: one warm, stage by stage (docs/ENGINES.md "Boot and warm stages"). The stages are what warm() does, in
/// order: `listeners` (the loopback surface answers) → `tokenizer` (chat's, resident) → `engine open` (llb_load: the
/// sidecar and the GGUF checked) → `weights` (llb_open: the GBs read into the GPU span; the Metal compile too on a
/// cache's first run) → `warm-up` (the decider's throw-away decide) → `ready`. The status reports the stage in flight and
/// its elapsed time; the console gets one `warm <id> phases …` line per warm.
struct WarmTrace {
    struct Phase { let name: String; let ms: Double }

    let startedAt: Date
    /// The last load of this model's weights on this device, ms (nil: the first load here, so no estimate).
    let expectWeightsMs: Double?
    private(set) var stage = "starting"
    private(set) var stageAt: Date
    private(set) var phases: [Phase] = []
    var note = ""
    var done = false

    init(startedAt: Date, expectWeightsMs: Double?) {
        self.startedAt = startedAt
        self.expectWeightsMs = expectWeightsMs
        stageAt = startedAt
    }

    mutating func enter(_ next: String, now: Date = Date()) {
        if stage != "starting" { phases.append(Phase(name: stage, ms: now.timeIntervalSince(stageAt) * 1000)) }
        stage = next
        stageAt = now
    }

    /// `listeners=3 tokenizer=41 engine_open=12 weights=2210 total=2270 (load_ms=… compile_ms=…)`
    var line: String {
        let parts = phases.map { "\($0.name.replacingOccurrences(of: " ", with: "_"))=\(Int($0.ms))" }
        let total = Int(phases.reduce(0) { $0 + $1.ms })
        return (parts + ["total=\(total)"]).joined(separator: " ") + (note.isEmpty ? "" : " (\(note))")
    }

    private static func key(_ id: EngineID) -> String { "bend.engine.weightsMs.\(id.dir)" }

    static func expectedWeightsMs(_ id: EngineID) -> Double? {
        let v = UserDefaults.standard.double(forKey: key(id))
        return v > 0 ? v : nil
    }

    static func rememberWeightsMs(_ id: EngineID, _ ms: Double) {
        UserDefaults.standard.set(ms, forKey: key(id))
    }
}

/// PHONEUX: launch-time facts the first screen needs: how long the process has been up, and which engine files are
/// missing (the "Engine files not set up" message names them).
enum Boot {
    /// The process's start (the kernel's), so "since launch" includes everything before Bend's own code ran.
    static let processStart: Date = {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        guard sysctl(&mib, 4, &info, &size, nil, 0) == 0 else { return Date() }
        let tv = info.kp_proc.p_starttime
        return Date(timeIntervalSince1970: Double(tv.tv_sec) + Double(tv.tv_usec) / 1_000_000)
    }()

    static func sinceStartMs(_ now: Date = Date()) -> Double { now.timeIntervalSince(processStart) * 1000 }

    /// The files the engines need, relative to Documents/models, and whether each is there.
    static func files() -> [(engine: String, path: String, present: Bool)] {
        var out: [(String, String, Bool)] = []
        for id in EngineID.allCases {
            for f in [id.model, "manifest.json", "tokenizer.bin"] {
                let rel = "\(id.dir)/\(f)"
                out.append((id.rawValue, rel, FileManager.default.fileExists(atPath: EngineManager.modelsDir.appendingPathComponent(rel).path)))
            }
        }
        #if BEND_CAMERA
        let tower = "\(CameraConfig.mmprojDir)/\(CameraConfig.mmprojFile)"
        out.append(("camera", tower, FileManager.default.fileExists(atPath: EngineManager.modelsDir.appendingPathComponent(tower).path)))
        #endif
        return out
    }

    /// `{chat, decision, camera: present?, missing: [paths], dir, command}` for the status and the shell info.
    static func setup() -> [String: Any] {
        let all = files()
        var out: [String: Any] = [:]
        for engine in Set(all.map(\.engine)) { out[engine] = all.filter { $0.engine == engine }.allSatisfy(\.present) }
        out["missing"] = all.filter { !$0.present }.map(\.path)
        out["dir"] = "Documents/models"
        // The one-time setup: copies the GGUFs + sidecars + the tower onto the phone over the cable (Mac side).
        out["command"] = "scripts/cam-device-run.sh models mmproj"
        return out
    }
}
