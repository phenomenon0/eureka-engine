import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// IOSFETCH: the first-run setup's state (ModelSetupView shows it; scripts/models-test.sh drives it on the Mac).
///
/// The lifecycle rules of docs/ENGINES.md hold here too: progress is always visible, and no state is terminal.
///   missing ──Download──▶ running ──all verified──▶ done (the engine warms)
///                           │  ▲
///          network error ×3 │  │ Retry / Resume / back on screen / next launch (the parts are kept)
///          (retried twice,  ▼  │
///          1 s / 3 s)      failed or paused
/// A wrong file (size or hash) is deleted and said plainly; Retry fetches it again. "Not now" leaves the card for the
/// demo replies; the card comes back from the web's status line, bend://models, or the camera.
@MainActor @Observable
final class ModelStore {
    enum Row: Equatable {
        case missing, present, queued
        case checking(Double)       // hashing a kept part before resuming
        case downloading(Double)
        case failed
    }

    enum Phase: Equatable {
        case idle
        case running
        /// Waiting out a transient failure before the next automatic attempt (attempt n of 3).
        case retrying(String, attempt: Int)
        case paused(String)
        case failed(String)
        case done
    }

    struct Fit: Equatable {
        var ok: Bool
        var line: String
    }

    let items: [ModelItem]
    let modelsDir: URL
    private(set) var rows: [ModelItem.ID: Row] = [:]
    var wantVision = false
    private(set) var phase: Phase = .idle
    var isPresented = false
    /// Bytes of the current run's set on disk (verified files + parts) and in total.
    private(set) var bytesDone: Int64 = 0
    private(set) var bytesTotal: Int64 = 0
    /// Smoothed transfer rate (B/s) and the time left at that rate.
    private(set) var speed: Double = 0
    private(set) var eta: Double?
    private(set) var currentFile: String?
    /// The run's receipt, one line per file: how it went (also on the console as `BEND models …`).
    private(set) var receipts: [String] = []

    /// The engine side: told which items just became complete (EngineManager.modelsArrived).
    var onInstalled: ((Set<ModelItem.ID>) -> Void)?
    nonisolated(unsafe) static var log: (String) -> Void = { print("BEND models \($0)") }

    private var runTask: Task<Void, Never>?
    private var fetch: ModelFetch?
    private var doneBefore: Int64 = 0
    private var lastSample: (t: Date, bytes: Int64)?
    private var startedAt = Date()
    private var startedBytes: Int64 = 0
    private var pausedByBackground = false
    #if canImport(UIKit)
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    #endif

    init(items: [ModelItem], modelsDir: URL) {
        self.items = items
        self.modelsDir = modelsDir
        refresh()
        // Kept off iCloud/iTunes backups: GBs that the host can always give again.
        var dir = modelsDir
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? dir.setResourceValues(values)
    }

    // MARK: what is on disk

    func dest(_ f: ModelFile) -> URL { modelsDir.appendingPathComponent(f.dir).appendingPathComponent(f.name) }

    /// Present = every file there at the manifest's size. A downloaded file was hashed before it was moved into place;
    /// a pushed one (the developer path, scripts/engine-device-run.sh models) is trusted by size, so launches never
    /// re-hash GBs.
    func present(_ f: ModelFile) -> Bool {
        ((try? FileManager.default.attributesOfItem(atPath: dest(f).path)[.size]) as? Int64) == f.bytes
    }

    func present(_ item: ModelItem) -> Bool { item.files.allSatisfy(present) }

    var needsSetup: Bool { items.contains { $0.required && !present($0) } }
    var visionItem: ModelItem? { items.first { $0.id == .vision } }

    /// The items a Download fetches now: the missing required ones, and the camera upgrade if chosen.
    var wanted: [ModelItem] {
        items.filter { !present($0) && ($0.required || ($0.id == .vision && wantVision)) }
    }

    func refresh() {
        for item in items where !isActive(rows[item.id]) {
            rows[item.id] = present(item) ? .present : (rows[item.id] == .failed ? .failed : .missing)
        }
    }

    private func isActive(_ r: Row?) -> Bool {
        switch r {
        case .queued, .checking, .downloading: phase == .running || isRetrying
        default: false
        }
    }

    private var isRetrying: Bool { if case .retrying = phase { true } else { false } }
    var isBusy: Bool { phase == .running || isRetrying }

    /// Bytes still to fetch for these items (parts already kept count as fetched).
    func remaining(_ set: [ModelItem]) -> Int64 {
        set.flatMap(\.files).filter { !present($0) }.reduce(0) { $0 + $1.bytes - ModelFetch.partBytes(dest($1)) }
    }

    // MARK: fit

    nonisolated static let minimumMemory: UInt64 = 7_500_000_000  // the 8 GB iPhones and up: ~4.2 GB resident + the app
    nonisolated static let headroom: Int64 = 300_000_000

    func fit(for set: [ModelItem]? = nil) -> Fit {
        let need = remaining(set ?? wanted)
        let ram = ProcessInfo.processInfo.physicalMemory
        let free = (try? modelsDir.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
        let device = Self.deviceName
        if ram < Self.minimumMemory {
            return Fit(ok: false, line: "This \(device) has \(Self.gb(Int64(ram), digits: 0)) of memory; Eureka’s engine needs 8 GB · demo replies still work")
        }
        if let free, free < need + Self.headroom {
            return Fit(ok: false, line: "Needs \(Self.size(need + Self.headroom)) free · \(Self.size(free)) free now · free up space, then Retry")
        }
        let freeText = free.map { " · \(Self.size($0)) free" } ?? ""
        return Fit(ok: true, line: "Fits this \(device) · \(Self.gb(Int64(ram), digits: 0)) memory\(freeText)")
    }

    nonisolated static var deviceName: String {
        #if canImport(UIKit)
        return MainActor.assumeIsolated { UIDevice.current.model }
        #else
        return "Mac"
        #endif
    }

    // MARK: the run

    /// Download (the button), Retry, Resume: all the same: fetch what is missing, resuming every kept part.
    func start() {
        guard !isBusy else { return }
        let set = wanted
        guard !set.isEmpty else { phase = .done; refresh(); return }
        let fit = fit(for: set)
        if !fit.ok { phase = .failed(fit.line); Self.log("fit refused: \(fit.line)"); return }
        bytesTotal = set.reduce(0) { $0 + $1.bytes }
        bytesDone = bytesTotal - remaining(set)
        startedBytes = bytesDone
        startedAt = Date()
        speed = 0; eta = nil; lastSample = nil
        for item in set { rows[item.id] = .queued }
        phase = .running
        pausedByBackground = false
        keepAwake(true)
        Self.log("run start host=\(ModelManifest.baseURL.absoluteString) items=\(set.map(\.id.rawValue).joined(separator: ",")) total=\(bytesTotal) have=\(bytesDone)")
        runTask = Task { await self.run(set) }
    }

    /// Pause: the transfer stops, the part stays; Resume picks up from it (Range).
    func pause(_ why: String = "Paused") {
        guard isBusy else { return }
        runTask?.cancel()
        fetch?.cancel()
        Self.log("pause: \(why) at \(bytesDone)/\(bytesTotal)")
        phase = .paused(why)
        keepAwake(false)
        refresh()
    }

    private func run(_ set: [ModelItem]) async {
        for item in set {
            for f in item.files where !present(f) {
                currentFile = f.path
                doneBefore = bytesDone - ModelFetch.partBytes(dest(f))
                var attempt = 1
                while true {
                    if Task.isCancelled { return }
                    phase = .running
                    let t0 = Date(), from = ModelFetch.partBytes(dest(f))
                    let fetch = ModelFetch(file: f, url: ModelManifest.url(f), dest: dest(f)) { [weak self] p, have in
                        Task { @MainActor in self?.progress(item.id, f, p, have) }
                    }
                    self.fetch = fetch
                    do {
                        try await fetch.run()
                        let secs = Date().timeIntervalSince(t0)
                        let line = String(format: "file %@ ok bytes=%lld resumed_from=%lld sha256=%@… %@ s=%.1f MB/s=%.0f", f.path, f.bytes, from,
                                          String(f.sha256.prefix(8)), fetch.note, secs, Double(f.bytes - from) / max(secs, 0.001) / 1e6)
                        receipts.append(line); Self.log(line)
                        bytesDone = doneBefore + f.bytes
                        break
                    } catch let failure as ModelFetch.Failure {
                        if failure == .cancelled || Task.isCancelled { return }  // pause(): its phase already says so
                        let message = Self.message(failure, f)
                        let line = "file \(f.path) FAILED attempt=\(attempt) \(failure) \(fetch.note)"
                        receipts.append(line); Self.log(line)
                        bytesDone = doneBefore + ModelFetch.partBytes(dest(f))
                        if failure.transient, attempt < 3 {
                            // The lifecycle's bounded retries: twice more, 1 s then 3 s, then Retry is the person's.
                            phase = .retrying(message, attempt: attempt + 1)
                            try? await Task.sleep(for: .seconds(attempt == 1 ? 1 : 3))
                            attempt += 1
                            continue
                        }
                        rows[item.id] = .failed
                        phase = .failed(message)
                        keepAwake(false)
                        return
                    } catch {
                        rows[item.id] = .failed
                        phase = .failed(error.localizedDescription)
                        keepAwake(false)
                        return
                    }
                }
            }
            rows[item.id] = .present
            Self.log("item \(item.id.rawValue) installed (\(item.files.count) files, \(item.bytes) bytes)")
            onInstalled?([item.id])
        }
        currentFile = nil
        let secs = Date().timeIntervalSince(startedAt)
        Self.log(String(format: "run done bytes=%lld s=%.1f MB/s=%.0f", bytesTotal - startedBytes, secs, Double(bytesTotal - startedBytes) / max(secs, 0.001) / 1e6))
        phase = .done
        speed = 0; eta = nil
        keepAwake(false)
        refresh()
    }

    private func progress(_ id: ModelItem.ID, _ f: ModelFile, _ p: ModelFetch.Phase, _ have: Int64) {
        guard isBusy, currentFile == f.path else { return }  // a late report from the file before
        let item = items.first { $0.id == id }!
        let before = item.files.prefix { $0 != f }.reduce(0) { $0 + $1.bytes }
        let fraction = Double(before + have) / Double(item.bytes)
        switch p {
        case .checking:
            rows[id] = .checking(Double(have) / Double(f.bytes))
        case .downloading:
            rows[id] = .downloading(fraction)
            bytesDone = doneBefore + have
            let now = Date()
            if let s = lastSample, now.timeIntervalSince(s.t) >= 0.5 {
                let rate = Double(bytesDone - s.bytes) / now.timeIntervalSince(s.t)
                speed = speed == 0 ? rate : speed * 0.7 + rate * 0.3
                eta = speed > 0 ? Double(bytesTotal - bytesDone) / speed : nil
                lastSample = (now, bytesDone)
            } else if lastSample == nil {
                lastSample = (now, bytesDone)
            }
        }
    }

    // MARK: plain words

    static func message(_ f: ModelFetch.Failure, _ file: ModelFile) -> String {
        let host = ModelManifest.baseURL.host() ?? ModelManifest.baseURL.absoluteString
        switch f {
        case .unreachable:
            return "Can’t reach the model host (\(host)). Check the network, then Retry."
        case .interrupted(let have):
            return "Connection lost at \(size(have)) of \(size(file.bytes)) (\(file.name)). Retry picks up where it stopped."
        case .http(404):
            return "The model host doesn’t have \(file.path) (HTTP 404). Retry, or check the host."
        case .http(let code):
            return "The model host answered HTTP \(code) for \(file.name). Retry in a moment."
        case .hostSize(let n):
            return "The host’s \(file.name) is \(n >= 0 ? size(n) : "a different size"); Eureka expects \(size(file.bytes)). Nothing was kept. Retry once the host has the right file."
        case .short(let n):
            return "\(file.name) stopped at \(size(n)) of \(size(file.bytes)). Nothing was kept. Retry downloads it again."
        case .checksum:
            return "\(file.name) arrived damaged (its checksum didn’t match), so it was deleted. Retry downloads it again."
        case .disk(let why):
            return "Couldn’t save \(file.name): \(why). Free up space, then Retry."
        case .cancelled:
            return "Paused"
        }
    }

    nonisolated static func size(_ b: Int64) -> String {
        b >= 1_000_000_000 ? gb(b, digits: b >= 10_000_000_000 ? 0 : 1) : String(format: "%.0f MB", Double(b) / 1e6)
    }

    nonisolated static func gb(_ b: Int64, digits: Int) -> String {
        String(format: "%.\(digits)f GB", Double(b) / 1e9)
    }

    nonisolated static func duration(_ s: Double) -> String {
        if s < 60 { return "\(max(1, Int(s.rounded()))) s" }
        if s < 3600 { return "\(Int(s / 60)) min \(Int(s) % 60) s" }
        return "\(Int(s / 3600)) h \(Int(s / 60) % 60) min"
    }

    // MARK: the app around it

    private func keepAwake(_ on: Bool) {
        #if canImport(UIKit)
        // A multi-GB download with the screen asleep would be paused by iOS; keep the screen on while it runs.
        UIApplication.shared.isIdleTimerDisabled = on
        #endif
    }

    #if canImport(UIKit)
    /// Leaving the screen: carry on for the background time iOS gives, then pause (the part stays).
    func away() {
        guard isBusy, backgroundTask == .invalid else { return }
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "bend.models.download") {
            MainActor.assumeIsolated {
                let store = ModelStore.shared
                store.pausedByBackground = true
                store.pause("Paused while Eureka was in the background · resumes now that you’re back")
                store.endBackground()
            }
        }
    }

    /// Back on screen: a run iOS paused resumes by itself (never a dead end).
    func back() {
        endBackground()
        if pausedByBackground, case .paused = phase {
            Self.log("back on screen: resuming")
            start()
        }
    }

    private func endBackground() {
        guard backgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
    }
    #endif
}
