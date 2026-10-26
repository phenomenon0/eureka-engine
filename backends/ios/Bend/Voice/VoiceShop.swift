#if BEND_FETCH  // the shop downloads with IOSFETCH's ModelFetch (Models/, the Bend target)
import Foundation
import Observation

/// VOICESHOP: "More voices": a curated catalog → download (progress, Range resume, sha256: IOSFETCH's ModelFetch) →
/// install into Documents/voices with its sidecar → the picker lists it → preview → use. docs/VOICES.md.
///
/// Offline-first: the bundled voices never depend on this. With no network the shop says so ("Can't reach the voice
/// shop") and shows the last catalog it saw, if any; nothing else changes.
enum VoiceCatalog {
    /// THE policy point: where the voice shop's catalog lives. One constant; nothing else in the app names a voice host.
    /// Public HF mirror of the repo's `voices/` folder (jethrovic/bend-models, beside the models; docs/VOICES.md "decision
    /// gate"). Relocate the host by changing this one string. Unreachable host: "Can't reach the voice shop" with Retry,
    /// and the bundled voices as they are.
    /// Pack and preview URLs inside the catalog may be absolute (the canonical catalog fetches packs straight from
    /// hexgrad/Kokoro-82M at a pinned revision) or relative to the catalog.
    static let CATALOG_URL = "https://huggingface.co/jethrovic/bend-models/resolve/main/voices/catalog.json"

    /// The catalog this launch reads. DEBUG only, in order: `-voicesCatalogURL <url>` (remembered), BEND_VOICES_CATALOG_URL,
    /// the remembered one, then CATALOG_URL. Release reads CATALOG_URL only.
    static var url: URL {
        #if DEBUG
        let defaults = UserDefaults.standard
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-voicesCatalogURL"), i + 1 < args.count, let url = URL(string: args[i + 1]) {
            defaults.set(args[i + 1], forKey: overrideKey)
            return url
        }
        if let env = ProcessInfo.processInfo.environment["BEND_VOICES_CATALOG_URL"], let url = URL(string: env) { return url }
        if let saved = defaults.string(forKey: overrideKey), let url = URL(string: saved) { return url }
        #endif
        return URL(string: CATALOG_URL)!
    }

    static var overridden: Bool { url.absoluteString != CATALOG_URL }
    static let overrideKey = "bend.voices.catalogURL"

    struct Entry: Codable, Identifiable, Hashable, Sendable {
        let id: String
        let engine: String
        let name: String
        let lang: String
        var gender: String?
        var grade: String?
        var detail: String?
        let license: String
        let credit: String
        /// A short clip of `demo_line` in this voice (played before buying; streamed, not kept).
        var preview_url: String?
        let pack_url: String
        let sha256: String
        let bytes: Int64
        /// The pack's file name once installed (its extension tells the adapter the format: .pt, .bin …).
        let file: String

        var packID: String { "\(engine):\(id)" }
        var accent: String { lang == "en-gb" ? "British" : "American" }
        var summary: String { ([accent, gender, grade.map { "grade \($0)" }] as [String?]).compactMap { $0 }.joined(separator: " · ") }
        var meta: VoiceMeta {
            VoiceMeta(id: id, engine: engine, name: name, lang: lang, gender: gender, grade: grade, detail: detail, license: license, credit: credit)
        }
    }

    struct Document: Codable, Sendable {
        let version: Int
        /// The house voice's pack id ("kokoro:af_bella"): bundled, so not for sale; the shop names it.
        var house_voice: String?
        var demo_line: String?
        let voices: [Entry]
    }

    static func resolve(_ ref: String, against base: URL) -> URL? { URL(string: ref, relativeTo: base)?.absoluteURL }
}

@MainActor @Observable
final class VoiceShop {
    static let shared = VoiceShop()
    static var log: (String) -> Void = { print("BEND voices \($0)") }

    enum Listing: Equatable {
        case idle, loading
        case ready
        /// No catalog this time: the reason, said plainly; `cached` = the last good list is shown.
        case offline(String, cached: Bool)
    }

    enum Row: Equatable {
        case available
        case downloading(Double)
        case paused(Int64)
        case installed
        case failed(String)
    }

    private(set) var listing = Listing.idle
    private(set) var entries: [VoiceCatalog.Entry] = []
    /// What the catalog names as the house voice (else the app's own default).
    private(set) var houseVoice = VoiceLibrary.defaultID
    private(set) var rows: [String: Row] = [:]
    private(set) var bytesDone: [String: Int64] = [:]
    /// What happened, one line per file (the receipts read these).
    private(set) var receipts: [String] = []
    /// Called after an install, with the new pack's id (the picker rescans; the Speaker may switch to it).
    var onInstalled: ((String) -> Void)?

    private var tasks: [String: Task<Void, Never>] = [:]
    private var fetches: [String: ModelFetch] = [:]
    let dir: URL
    private let cacheFile: URL

    init(dir: URL = VoiceLibrary.userDir) {
        self.dir = dir
        let caches = ProcessInfo.processInfo.environment["BEND_CACHE_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        cacheFile = caches.appendingPathComponent("voice-catalog.json")
    }

    var host: String { VoiceCatalog.url.host() ?? VoiceCatalog.url.absoluteString }

    /// Fetches the catalog (8 s timeout); on failure, the cached copy if any, and a plain reason.
    func load() async {
        if listing == .loading { return }
        listing = .loading
        let url = VoiceCatalog.url
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: 8)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw ModelFetch.Failure.http((response as? HTTPURLResponse)?.statusCode ?? 0)
            }
            let doc = try JSONDecoder().decode(VoiceCatalog.Document.self, from: data)
            entries = doc.voices
            houseVoice = doc.house_voice ?? VoiceLibrary.defaultID
            try? FileManager.default.createDirectory(at: cacheFile.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? data.write(to: cacheFile)
            listing = .ready
            Self.log("catalog ok url=\(url.absoluteString) voices=\(doc.voices.map(\.id).joined(separator: ","))")
        } catch {
            let why: String
            switch error {
            case ModelFetch.Failure.http(let code): why = "The voice shop answered HTTP \(code)."
            case is DecodingError: why = "The voice shop’s list didn’t read as a catalog."
            default: why = "Can’t reach the voice shop (\(host))."
            }
            let cached = (try? Data(contentsOf: cacheFile)).flatMap { try? JSONDecoder().decode(VoiceCatalog.Document.self, from: $0) }
            if let cached { entries = cached.voices; houseVoice = cached.house_voice ?? VoiceLibrary.defaultID }
            listing = .offline(why, cached: cached != nil)
            Self.log("catalog offline url=\(url.absoluteString) cached=\(cached != nil) error=\(error.localizedDescription)")
        }
        refreshRows()
    }

    func refreshRows() {
        for e in entries where !isBusy(e.id) {
            if FileManager.default.fileExists(atPath: dest(e).path) { rows[e.id] = .installed }
            else if ModelFetch.partBytes(dest(e)) > 0 { rows[e.id] = .paused(ModelFetch.partBytes(dest(e))) }
            else if case .failed = rows[e.id] {} else { rows[e.id] = .available }
        }
    }

    func isBusy(_ id: String) -> Bool { if case .downloading = rows[id] { true } else { false } }
    func dest(_ e: VoiceCatalog.Entry) -> URL { dir.appendingPathComponent(e.file) }

    /// Download (the button), Resume and Retry are one call: what the part holds is kept and the rest asked for (Range).
    func download(_ e: VoiceCatalog.Entry) {
        guard !isBusy(e.id) else { return }
        guard let url = VoiceCatalog.resolve(e.pack_url, against: VoiceCatalog.url) else { rows[e.id] = .failed("The catalog’s link for \(e.name) is broken."); return }
        rows[e.id] = .downloading(Double(ModelFetch.partBytes(dest(e))) / Double(max(e.bytes, 1)))
        Self.log("download start id=\(e.id) url=\(url.absoluteString) have=\(ModelFetch.partBytes(dest(e)))")
        tasks[e.id] = Task { await self.run(e, url) }
    }

    func pause(_ e: VoiceCatalog.Entry) {
        tasks[e.id]?.cancel()
        fetches[e.id]?.cancel()
        rows[e.id] = .paused(ModelFetch.partBytes(dest(e)))
        Self.log("pause id=\(e.id) at=\(ModelFetch.partBytes(dest(e)))")
    }

    private func run(_ e: VoiceCatalog.Entry, _ url: URL) async {
        let file = ModelFile(dir: "voices", name: e.file, bytes: e.bytes, sha256: e.sha256)
        var attempt = 1
        while true {
            let t0 = Date(), from = ModelFetch.partBytes(dest(e))
            let fetch = ModelFetch(file: file, url: url, dest: dest(e)) { [weak self] _, have in
                Task { @MainActor in
                    guard let self, self.isBusy(e.id) else { return }
                    self.bytesDone[e.id] = have
                    self.rows[e.id] = .downloading(Double(have) / Double(max(e.bytes, 1)))
                }
            }
            fetches[e.id] = fetch
            do {
                try await fetch.run()
                try install(e)
                let line = String(format: "file %@ ok bytes=%lld resumed_from=%lld sha256=%@… %@ s=%.2f", e.file, e.bytes, from, String(e.sha256.prefix(8)), fetch.note, Date().timeIntervalSince(t0))
                receipts.append(line); Self.log(line)
                rows[e.id] = .installed
                bytesDone[e.id] = e.bytes
                onInstalled?(e.packID)
                return
            } catch let f as ModelFetch.Failure {
                if f == .cancelled || Task.isCancelled { return }  // pause(): its row already says so
                let line = "file \(e.file) FAILED attempt=\(attempt) \(f) \(fetch.note)"
                receipts.append(line); Self.log(line)
                if f.transient, attempt < 3 {  // as ModelStore: twice more (1 s, 3 s), then Retry is the person's
                    try? await Task.sleep(for: .seconds(attempt == 1 ? 1 : 3))
                    attempt += 1
                    continue
                }
                rows[e.id] = .failed(Self.message(f, e, host: url.host() ?? host))
                return
            } catch {
                rows[e.id] = .failed("Couldn’t install \(e.name): \(error.localizedDescription)")
                Self.log("install FAILED id=\(e.id) \(error.localizedDescription)")
                return
            }
        }
    }

    /// The verified file is in place (ModelFetch moved it only after size + sha256 matched): its sidecar makes it a
    /// shop pack, and the engine must be able to read it before it is offered (a pack that can't load is removed).
    private func install(_ e: VoiceCatalog.Entry) throws {
        var meta = e.meta
        meta.installed = ISO8601DateFormatter().string(from: Date())
        let sidecar = dest(e).deletingPathExtension().appendingPathExtension("json")
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try enc.encode(meta).write(to: sidecar)
        let lib = VoiceLibrary.shared
        guard let pack = lib.packs(refresh: true).first(where: { $0.id == e.packID }) else {
            try? FileManager.default.removeItem(at: dest(e)); try? FileManager.default.removeItem(at: sidecar)
            throw NeuralVoiceError("this build has no \(e.engine) engine for it")
        }
        if e.engine == "kokoro" { _ = try KokoroVoice.style(at: pack.url) }  // the table reads (the full load happens on use)
    }

    /// Removes a shop pack (file + sidecar); the picker falls back to the default if it was in use.
    func remove(_ e: VoiceCatalog.Entry) {
        try? FileManager.default.removeItem(at: dest(e))
        try? FileManager.default.removeItem(at: dest(e).deletingPathExtension().appendingPathExtension("json"))
        _ = VoiceLibrary.shared.packs(refresh: true)
        rows[e.id] = .available
        Self.log("removed id=\(e.id)")
    }

    static func message(_ f: ModelFetch.Failure, _ e: VoiceCatalog.Entry, host: String) -> String {
        switch f {
        case .unreachable: "Can’t reach \(host). Check the network, then Retry. Your voices on this phone still work."
        case .interrupted: "Connection lost partway through \(e.name). Retry picks up where it stopped."
        case .http(404): "The shop doesn’t have \(e.name) right now (HTTP 404)."
        case .http(let code): "The shop answered HTTP \(code) for \(e.name). Retry in a moment."
        case .hostSize(let n): "The shop’s copy of \(e.name) is \(n >= 0 ? VoiceLibrary.size(n) : "a different size"), not \(VoiceLibrary.size(e.bytes)). Nothing was kept."
        case .short: "\(e.name) arrived incomplete. Nothing was kept. Retry downloads it again."
        case .checksum: "\(e.name) arrived damaged (its checksum didn’t match), so it was deleted. Retry downloads it again."
        case .disk(let why): "Couldn’t save \(e.name): \(why)."
        case .cancelled: "Paused"
        }
    }
}
#endif
