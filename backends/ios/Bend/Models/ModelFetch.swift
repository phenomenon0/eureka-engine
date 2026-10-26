import CryptoKit
import Foundation

/// IOSFETCH: one file of the manifest, from the host into `<models>/<dir>/<name>`.
///
/// - **Resume:** bytes land in `<name>.part`. A later attempt (Retry, the next launch, coming back from the background)
///   hashes what the part already holds and asks for the rest with `Range: bytes=<have>-`. A host that ignores Range
///   (200) restarts the file from 0; one that answers 416 for a part that is already complete goes straight to the check.
/// - **Verify:** sha256 runs over the bytes as they arrive (no second pass over GBs). The part becomes the real file only
///   when its size and hash both match the manifest; anything else is deleted and said plainly, never kept.
/// - **The host's size** (Content-Length / Content-Range) is checked before any body byte: a short or different file on
///   the host fails at once ("the host's copy is 1.0 GB; Bend expects 2.7 GB") instead of after the download.
/// Foundation + CryptoKit only: the app and the Mac test (scripts/models-test.sh) run the same code.
final class ModelFetch: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    enum Failure: Error, Equatable, Sendable {
        /// No connection to the host at all (offline, DNS, refused, timed out before any byte).
        case unreachable(String)
        /// The connection dropped mid-file; the part is kept and the next attempt resumes at `have`.
        case interrupted(have: Int64)
        case http(Int)
        /// The host's file is not the manifest's size (nothing kept).
        case hostSize(Int64)
        /// The body ended before the manifest's bytes without an error (nothing kept).
        case short(Int64)
        /// The bytes arrived but the hash is wrong (nothing kept).
        case checksum(String)
        case disk(String)
        /// Paused (the person, or the app leaving the screen); the part is kept.
        case cancelled

        /// Worth trying again by itself (bounded): the network, not the file.
        var transient: Bool {
            switch self {
            case .unreachable, .interrupted: true
            default: false
            }
        }
    }

    enum Phase: Sendable { case checking, downloading }

    let file: ModelFile
    let url: URL
    let dest: URL
    var part: URL { dest.appendingPathExtension("part") }

    private let onProgress: @Sendable (Phase, Int64) -> Void
    private let lock = NSLock()
    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var handle: FileHandle?
    private var hasher = SHA256()
    private var have: Int64 = 0
    private var started: Int64 = 0
    private var failure: Failure?
    private var cancelled = false
    /// 416 on a part that already holds every byte: nothing to fetch, straight to the check.
    private var rangeDone = false
    private var lastReport = Date.distantPast
    private var continuation: CheckedContinuation<Void, Error>?
    /// What the attempt did, for the log line: the Range asked for, the status answered.
    private(set) var note = ""

    init(file: ModelFile, url: URL, dest: URL, onProgress: @escaping @Sendable (Phase, Int64) -> Void) {
        self.file = file
        self.url = url
        self.dest = dest
        self.onProgress = onProgress
    }

    /// Bytes of this file already on disk (the part), without hashing: for the progress total before a run starts.
    static func partBytes(_ dest: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: dest.appendingPathExtension("part").path)[.size]) as? Int64) ?? 0
    }

    /// Downloads (or finishes) the file. Throws `Failure`.
    func run() async throws {
        let fm = FileManager.default
        try? fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        // What the part already holds: hashed now, so the hash can go on from it.
        if let size = (try? fm.attributesOfItem(atPath: part.path)[.size]) as? Int64 {
            if size > file.bytes {
                try? fm.removeItem(at: part)
            } else if size > 0 {
                onProgress(.checking, size)
                try hashPrefix(size)
            }
        }
        if !fm.fileExists(atPath: part.path) { fm.createFile(atPath: part.path, contents: nil) }
        if have == file.bytes { note = "part already complete"; return try finish() }

        guard let h = try? FileHandle(forWritingTo: part) else { throw Failure.disk("cannot write \(part.lastPathComponent)") }
        try? h.truncate(atOffset: UInt64(have))
        handle = h
        started = have

        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData, timeoutInterval: 30)
        if have > 0 { request.setValue("bytes=\(have)-", forHTTPHeaderField: "Range") }
        note = have > 0 ? "Range bytes=\(have)-" : "from 0"
        let config = URLSessionConfiguration.ephemeral
        config.urlCache = nil
        config.timeoutIntervalForRequest = 30
        config.waitsForConnectivity = false
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        let session = URLSession(configuration: config, delegate: self, delegateQueue: queue)

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                lock.withLock {
                    continuation = c
                    self.session = session
                    let t = session.dataTask(with: request)
                    task = t
                    if cancelled { failure = .cancelled; t.cancel() }
                    t.resume()
                }
            }
        } onCancel: {
            self.cancel()
        }
    }

    /// Stops the transfer; the part stays for the next attempt.
    func cancel() {
        lock.withLock {
            cancelled = true
            if failure == nil { failure = .cancelled }
            task?.cancel()
        }
    }

    private func hashPrefix(_ size: Int64) throws {
        guard let h = try? FileHandle(forReadingFrom: part) else { throw Failure.disk("cannot read \(part.lastPathComponent)") }
        defer { try? h.close() }
        var left = size
        while left > 0 {
            let chunk = try autoreleasepool { try h.read(upToCount: Int(min(left, 16 << 20))) } ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
            left -= Int64(chunk.count)
            have += Int64(chunk.count)
            if Date().timeIntervalSince(lastReport) > 0.2 { lastReport = Date(); onProgress(.checking, have) }
        }
    }

    /// Size and hash against the manifest; the part becomes the file only if both match.
    private func finish() throws {
        let fm = FileManager.default
        guard have == file.bytes else {
            try? fm.removeItem(at: part)
            throw Failure.short(have)
        }
        let hex = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard hex == file.sha256 else {
            try? fm.removeItem(at: part)
            throw Failure.checksum(hex)
        }
        try? fm.removeItem(at: dest)
        do { try fm.moveItem(at: part, to: dest) } catch { throw Failure.disk("cannot move \(file.name) into place: \(error.localizedDescription)") }
    }

    private func fail(_ f: Failure) {
        lock.withLock { if failure == nil { failure = f } }
    }

    // MARK: URLSessionDataDelegate (the session's serial queue)

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else { fail(.http(0)); return completionHandler(.cancel) }
        note += " → \(http.statusCode)"
        switch http.statusCode {
        case 206:
            // Content-Range: bytes <first>-<last>/<total>
            let range = http.value(forHTTPHeaderField: "Content-Range") ?? ""
            let nums = range.split(whereSeparator: { !$0.isNumber }).compactMap { Int64($0) }
            if nums.count == 3, nums[2] != file.bytes { fail(.hostSize(nums[2])); return completionHandler(.cancel) }
            if nums.count == 3, nums[0] != have {
                // A range that doesn't start where the part ends: start over cleanly next time.
                try? handle?.truncate(atOffset: 0)
                fail(.interrupted(have: 0)); return completionHandler(.cancel)
            }
        case 200:
            if http.expectedContentLength > 0, http.expectedContentLength != file.bytes {
                fail(.hostSize(http.expectedContentLength)); return completionHandler(.cancel)
            }
            if have > 0 {
                // The host ignored Range: the whole file is coming, so the part starts over.
                note += " (Range ignored: from 0)"
                try? handle?.truncate(atOffset: 0)
                hasher = SHA256()
                have = 0
                started = 0
            }
        case 416:
            // Nothing past the part: complete if the host's size says so (Content-Range: bytes */<total>).
            let total = (http.value(forHTTPHeaderField: "Content-Range") ?? "").split(separator: "/").last.flatMap { Int64($0) }
            if let total, total != file.bytes { fail(.hostSize(total)) } else if have != file.bytes { fail(.hostSize(total ?? -1)) } else { rangeDone = true }
            return completionHandler(.cancel)
        default:
            fail(.http(http.statusCode)); return completionHandler(.cancel)
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard lock.withLock({ failure == nil }) else { return }
        if have + Int64(data.count) > file.bytes {
            fail(.hostSize(have + Int64(data.count))); dataTask.cancel(); return
        }
        do { try handle?.write(contentsOf: data) } catch {
            fail(.disk(error.localizedDescription)); dataTask.cancel(); return
        }
        hasher.update(data: data)
        have += Int64(data.count)
        if Date().timeIntervalSince(lastReport) > 0.1 { lastReport = Date(); onProgress(.downloading, have) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        try? handle?.synchronize()
        try? handle?.close()
        handle = nil
        session.finishTasksAndInvalidate()
        onProgress(.downloading, have)
        var outcome: Result<Void, Error>
        if rangeDone {
            outcome = Result { try finish() }
        } else if let f = lock.withLock({ failure }) {
            if case .hostSize = f { try? FileManager.default.removeItem(at: part) }
            outcome = .failure(f)
        } else if let error = error as? URLError {
            outcome = .failure(Self.map(error, have: have, started: started))
        } else if let error {
            outcome = .failure(Failure.interrupted(have: have)); _ = error
        } else {
            outcome = Result { try finish() }
        }
        let c = lock.withLock { () -> CheckedContinuation<Void, Error>? in defer { continuation = nil }; return continuation }
        c?.resume(with: outcome)
    }

    static func map(_ e: URLError, have: Int64, started: Int64) -> Failure {
        switch e.code {
        case .cancelled: return .cancelled
        case .notConnectedToInternet, .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed, .appTransportSecurityRequiresSecureConnection,
             .secureConnectionFailed, .serverCertificateUntrusted, .serverCertificateHasBadDate, .internationalRoamingOff, .dataNotAllowed:
            return .unreachable(e.localizedDescription)
        case .timedOut where have == started:
            return .unreachable(e.localizedDescription)
        default:
            return .interrupted(have: have)
        }
    }
}
