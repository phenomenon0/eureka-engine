import Foundation

/// VOICESHOP: the voices the phone can speak with, from any engine (docs/VOICES.md).
///
/// Three layers, so a new TTS engine is one adapter and nothing else changes:
/// - `VoiceEngine`: text → mono Float32 at `sampleRate` (what the Speaker plays). Supertonic's NeuralVoice and Kokoro's
///   KokoroVoice are two.
/// - `VoicePack`: one pickable voice on disk: an engine tag + its file + what the picker says about it.
/// - `VoiceAdapter`: per engine, which files are its packs and how a pack becomes a `VoiceEngine` (sharing the model
///   between voices where it can: every Kokoro voice runs on one loaded Kokoro).
/// `VoiceLibrary` lists the packs (bundled, bought in the shop, dropped into `voices/`) and remembers the choice.
protocol VoiceEngine: AnyObject, Sendable {
    var name: String { get }
    var sampleRate: Int { get }
    func synthesize(_ text: String) throws -> [Float]
}

struct VoicePack: Identifiable, Hashable, Sendable, Codable {
    enum Origin: String, Sendable, Codable {
        /// In the app bundle: always there, offline included.
        case bundled
        /// Downloaded from the voice shop (Documents/voices, with its catalog sidecar).
        case shop
        /// A file someone put in Documents/voices themselves (Files app, Finder, devicectl).
        case dropped
    }
    /// Engine-qualified and stable: "kokoro:af_heart". The choice is remembered by it.
    let id: String
    let engine: String
    let name: String
    /// "en-us", "en-gb".
    let lang: String
    var gender: String?
    /// The publisher's quality grade (hexgrad's VOICES.md for Kokoro), shown as is.
    var grade: String?
    var detail: String?
    var license: String
    var credit: String
    var origin: Origin
    /// The pack's file (or, for an engine whose voice is a folder, that folder).
    var path: String
    var bytes: Int64

    var url: URL { URL(fileURLWithPath: path) }
    var accent: String { lang == "en-gb" ? "British" : "American" }
    /// "American · female · A" (the picker's second line).
    var summary: String { ([accent, gender, grade.map { "grade \($0)" }] as [String?]).compactMap { $0 }.joined(separator: " · ") }
}

/// What a catalog entry / sidecar says about a pack (the same keys in both: docs/VOICES.md "catalog").
struct VoiceMeta: Codable, Sendable {
    var id: String
    var engine: String
    var name: String?
    var lang: String?
    var gender: String?
    var grade: String?
    var detail: String?
    var license: String?
    var credit: String?
    var speed: Float?
    /// Set by the shop when it installs the pack (ISO 8601); a sidecar without it is someone's own.
    var installed: String?
}

protocol VoiceAdapter: AnyObject, Sendable {
    /// The engine tag catalog entries and sidecars carry ("kokoro", "supertonic").
    var engine: String { get }
    /// Shown as the engine's name ("Kokoro 82M").
    var title: String { get }
    /// File extensions a dropped or downloaded pack of this engine may have (lower case, no dot).
    var packExtensions: [String] { get }
    /// Can this build run the engine at all (its model files are here)?
    var available: Bool { get }
    /// The bundle's own packs.
    func bundledPacks() -> [VoicePack]
    /// A file in Documents/voices this engine claims, with its sidecar's words if there is one.
    func pack(file: URL, meta: VoiceMeta?, origin: VoicePack.Origin) -> VoicePack?
    /// The pack, ready to speak. May share heavy state with the adapter's other voices.
    func load(_ pack: VoicePack) throws -> any VoiceEngine
    /// Let go of shared state (memory warning, or another engine took over).
    func release()
}

final class VoiceLibrary: @unchecked Sendable {
    static let shared = VoiceLibrary()

    /// THE house voice (owner's pick, PICK.md 2026-10-09): Kokoro Bella. A fresh install speaks with it; it is the
    /// fallback when the picked voice can't load. One line to change.
    static let defaultID = "kokoro:af_bella"
    static let choiceKey = "BendVoice"
    /// The same line for every preview (picker and shop), so voices compare on equal terms.
    static let demoLine = "Hi, I'm the voice Eureka will use to read its replies. Shall we plan your Saturday, or keep it simple?"

    let adapters: [any VoiceAdapter]
    private let lock = NSLock()
    private var cached: [VoicePack]?

    init(adapters: [any VoiceAdapter] = [KokoroAdapter(), SupertonicAdapter()]) { self.adapters = adapters }

    func adapter(_ engine: String) -> (any VoiceAdapter)? { adapters.first { $0.engine == engine } }

    /// Documents/voices: the drop-in folder (visible in Files › On My iPhone › Bend › voices, the app shares its
    /// Documents); the shop installs there too. BEND_VOICES_DIR overrides it (the Mac harness).
    static var userDir: URL {
        if let env = ProcessInfo.processInfo.environment["BEND_VOICES_DIR"] { return URL(fileURLWithPath: env) }
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("voices")
    }

    /// Bundled first (in their curated order), then shop and dropped packs by name. A user pack with a bundled id is
    /// skipped (the bundled one stays the one the id names).
    func packs(refresh: Bool = false) -> [VoicePack] {
        if !refresh, let c = lock.withLock({ cached }) { return c }
        var out: [VoicePack] = []
        var seen = Set<String>()
        for a in adapters where a.available {
            for p in a.bundledPacks() where seen.insert(p.id).inserted { out.append(p) }
        }
        var user: [VoicePack] = []
        let fm = FileManager.default
        try? fm.createDirectory(at: Self.userDir, withIntermediateDirectories: true)
        let files = (try? fm.contentsOfDirectory(at: Self.userDir, includingPropertiesForKeys: nil)) ?? []
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let ext = file.pathExtension.lowercased()
            guard ext != "json", ext != "part", !file.lastPathComponent.hasPrefix(".") else { continue }
            let sidecar = file.deletingPathExtension().appendingPathExtension("json")
            let meta = (try? Data(contentsOf: sidecar)).flatMap { try? JSONDecoder().decode(VoiceMeta.self, from: $0) }
            let origin: VoicePack.Origin = meta?.installed != nil ? .shop : .dropped
            let candidates = meta.flatMap { m in adapter(m.engine).map { [$0] } } ?? adapters.filter { $0.packExtensions.contains(ext) }
            for a in candidates where a.available && a.packExtensions.contains(ext) {
                if let p = a.pack(file: file, meta: meta, origin: origin), seen.insert(p.id).inserted { user.append(p); break }
            }
        }
        out += user.sorted { $0.name < $1.name }
        lock.withLock { cached = out }
        return out
    }

    func pack(_ id: String) -> VoicePack? { packs().first { $0.id == id } }

    /// The chosen voice's id: `-BendVoice <id>` at launch, BEND_VOICE (the Mac harness), else what was picked last.
    var choice: String {
        get { ProcessInfo.processInfo.environment["BEND_VOICE"] ?? UserDefaults.standard.string(forKey: Self.choiceKey) ?? Self.defaultID }
        set { UserDefaults.standard.set(newValue, forKey: Self.choiceKey) }
    }

    /// The pack to speak with: the choice, else the default, else the first that loads (the Speaker walks this list).
    func candidates() -> [VoicePack] {
        let all = packs()
        var ids = [choice, Self.defaultID]
        ids += all.map(\.id)
        var seen = Set<String>()
        return ids.compactMap { id in seen.insert(id).inserted ? all.first { $0.id == id } : nil }
    }

    static func size(_ b: Int64) -> String {
        b >= 1_000_000 ? String(format: "%.1f MB", Double(b) / 1e6) : String(format: "%.0f KB", Double(b) / 1e3)
    }
}

// MARK: - Kokoro

final class KokoroAdapter: VoiceAdapter, @unchecked Sendable {
    let engine = "kokoro"
    let title = "Kokoro 82M"
    let packExtensions = ["bin", "pt"]
    private let lock = NSLock()
    private var runtime: KokoroRuntime?
    private let dir = KokoroRuntime.directory()

    var available: Bool { dir != nil }

    func bundledPacks() -> [VoicePack] {
        guard let dir else { return [] }
        let packs = dir.appendingPathComponent("packs")
        // The curated order is the bundle's index (scripts/fetch-voice.sh writes it).
        let order = ((try? Data(contentsOf: packs.appendingPathComponent("index.json"))).flatMap { try? JSONDecoder().decode([String].self, from: $0) }) ?? []
        return order.compactMap { name in
            let file = packs.appendingPathComponent("\(name).bin")
            let meta = (try? Data(contentsOf: packs.appendingPathComponent("\(name).json"))).flatMap { try? JSONDecoder().decode(VoiceMeta.self, from: $0) }
            return pack(file: file, meta: meta, origin: .bundled)
        }
    }

    /// Kokoro's voice files are named <lang><gender>_<name> (af_heart: American female "Heart"); a sidecar's words win.
    func pack(file: URL, meta: VoiceMeta?, origin: VoicePack.Origin) -> VoicePack? {
        let stem = file.deletingPathExtension().lastPathComponent
        guard let bytes = (try? FileManager.default.attributesOfItem(atPath: file.path)[.size]) as? Int64,
              bytes >= KokoroRuntime.styleBytes, bytes < 4 * KokoroRuntime.styleBytes else { return nil }
        let parts = stem.split(separator: "_", maxSplits: 1)
        let code = parts.count == 2 && parts[0].count == 2 ? Array(parts[0]) : []
        let lang = meta?.lang ?? (code.first == "b" ? "en-gb" : "en-us")
        let gender = meta?.gender ?? (code.count == 2 ? (code[1] == "f" ? "female" : code[1] == "m" ? "male" : nil) : nil)
        let name = meta?.name ?? (parts.count == 2 ? String(parts[1]) : stem).replacingOccurrences(of: "_", with: " ").capitalized
        return VoicePack(id: "kokoro:\(meta?.id ?? stem)", engine: engine, name: name, lang: lang, gender: gender, grade: meta?.grade,
                         detail: meta?.detail, license: meta?.license ?? (origin == .dropped ? "as supplied" : "Apache-2.0"),
                         credit: meta?.credit ?? (origin == .dropped ? "added from Files" : "hexgrad/Kokoro-82M"), origin: origin,
                         path: file.path, bytes: bytes)
    }

    func load(_ pack: VoicePack) throws -> any VoiceEngine {
        guard let dir else { throw NeuralVoiceError("no Kokoro model in this build (scripts/fetch-voice.sh)") }
        let style = try KokoroVoice.style(at: pack.url)
        let rt = try lock.withLock { () throws -> KokoroRuntime in
            if let r = runtime { return r }
            let r = try KokoroRuntime(dir: dir)
            runtime = r
            return r
        }
        let sidecar = pack.url.deletingPathExtension().appendingPathExtension("json")
        let speed = (try? Data(contentsOf: sidecar)).flatMap { try? JSONDecoder().decode(VoiceMeta.self, from: $0) }?.speed ?? 1
        return KokoroVoice(runtime: rt, name: "Kokoro \(pack.name)", style: style, british: pack.lang == "en-gb", speed: speed)
    }

    func release() { lock.withLock { runtime = nil } }
}

// MARK: - Supertonic (VOICEREAL's voice, kept as the second engine)

final class SupertonicAdapter: VoiceAdapter, @unchecked Sendable {
    let engine = "supertonic"
    let title = "Supertonic 2"
    /// Supertonic voices are style JSONs (F1.json …) beside the model; a dropped one needs a sidecar naming the engine.
    let packExtensions = ["style"]
    private let dir = NeuralVoice.directory()

    var available: Bool { dir != nil }

    func bundledPacks() -> [VoicePack] {
        guard let dir else { return [] }
        return ["F1"].compactMap { v in
            let file = dir.appendingPathComponent("\(v).json")
            guard let bytes = (try? FileManager.default.attributesOfItem(atPath: file.path)[.size]) as? Int64 else { return nil }
            return VoicePack(id: "supertonic:\(v)", engine: engine, name: "Supertonic \(v)", lang: "en-us", gender: v.hasPrefix("F") ? "female" : "male",
                             grade: nil, detail: "VOICEREAL's voice: the fastest first word", license: "OpenRAIL-M", credit: "Supertone/supertonic-2",
                             origin: .bundled, path: file.path, bytes: bytes)
        }
    }

    func pack(file: URL, meta: VoiceMeta?, origin: VoicePack.Origin) -> VoicePack? { nil }

    func load(_ pack: VoicePack) throws -> any VoiceEngine {
        guard let dir else { throw NeuralVoiceError("no Supertonic files in this build") }
        return try NeuralVoice(dir: dir, voice: pack.url.deletingPathExtension().lastPathComponent)
    }

    func release() {}
}

extension NeuralVoice: VoiceEngine {}
