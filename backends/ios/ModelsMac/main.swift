import CryptoKit
import Foundation

// IOSFETCH: the app's download path (Models/ModelFetch.swift + ModelStore.swift, compiled as is) against
// scripts/model-host.py on this Mac, in each of the host's modes. scripts/models-test.sh builds and runs it.
//   bend-models-mac <work dir> <path to model-host.py>
// Small generated files stand in for the GBs (same code path: Range, 206/416, sha256 as bytes arrive).

let work = URL(fileURLWithPath: CommandLine.arguments[1])
let hostScript = CommandLine.arguments[2]
let hostRoot = work.appendingPathComponent("host")
let phone = work.appendingPathComponent("phone")
var port = 8970
var passed = 0, failed = 0

@MainActor func check(_ name: String, _ ok: Bool, _ detail: String = "") {
    if ok { passed += 1 } else { failed += 1 }
    print("CHECK \(ok ? "PASS" : "FAIL") \(name)\(detail.isEmpty ? "" : " · \(detail)")")
}

nonisolated func sha(_ url: URL) -> String {
    SHA256.hash(data: try! Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
}

// The host's files: a model dir (a big file last, as the real ones) and the camera upgrade.
let fm = FileManager.default
try? fm.removeItem(at: work)
for d in ["m1", "cam"] { try! fm.createDirectory(at: hostRoot.appendingPathComponent(d), withIntermediateDirectories: true) }
func make(_ path: String, _ bytes: Int) -> ModelFile {
    var g = SystemRandomNumberGenerator()
    var data = Data(count: bytes)
    data.withUnsafeMutableBytes { p in for i in stride(from: 0, to: bytes, by: 8) { var r = g.next(); memcpy(p.baseAddress! + i, &r, min(8, bytes - i)) } }
    let url = hostRoot.appendingPathComponent(path)
    try! data.write(to: url)
    let parts = path.split(separator: "/")
    return ModelFile(dir: String(parts[0]), name: String(parts[1]), bytes: Int64(bytes), sha256: sha(url))
}
let small = make("m1/tokenizer.bin", 70_001)
let big = make("m1/model.gguf", 48_000_000)
let cam = make("cam/tower.gguf", 6_000_000)
let items = [
    ModelItem(id: .chat, title: "Test model", detail: "chat", files: [small, big], required: true),
    ModelItem(id: .vision, title: "Camera vision", detail: "opt-in", files: [cam], required: false),
]

@MainActor final class Host {
    let proc = Process()
    let log: URL
    init(_ name: String, _ extra: [String]) {
        port += 1
        log = work.appendingPathComponent("host-\(name).log")
        fm.createFile(atPath: log.path, contents: nil)
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        proc.arguments = [hostScript, "--root", hostRoot.path, "--bind", "127.0.0.1", "--port", "\(port)"] + extra
        proc.standardOutput = try! FileHandle(forWritingTo: log)
        proc.standardError = proc.standardOutput
        try! proc.run()
        for _ in 0..<50 where !((try? String(contentsOf: log, encoding: .utf8)) ?? "").contains("HOST serving") { usleep(100_000) }
        setenv("BEND_MODELS_BASE_URL", "http://127.0.0.1:\(port)", 1)
    }
    var lines: [String] { ((try? String(contentsOf: log, encoding: .utf8)) ?? "").split(separator: "\n").map(String.init) }
    func stop() { proc.terminate(); proc.waitUntilExit() }
}

@MainActor func settle(_ s: ModelStore, timeout: Double = 60) async {
    let t0 = Date()
    while s.isBusy, Date().timeIntervalSince(t0) < timeout { try? await Task.sleep(for: .milliseconds(50)) }
}

@MainActor func fresh(vision: Bool = false) -> ModelStore {
    let s = ModelStore(items: items, modelsDir: phone)
    s.wantVision = vision
    return s
}

@MainActor func run() async {
    ModelStore.log = { print("  BEND models \($0)") }
    var installed: [ModelItem.ID] = []

    print("== 1 fresh: setup needed, Download → verified, moved into place; the camera row opt-in")
    var host = Host("fresh", [])
    var s = fresh(vision: true)
    s.onInstalled = { installed += $0 }
    check("fresh: needsSetup before", s.needsSetup)
    check("fresh: fit line", s.fit().ok, s.fit().line)
    s.start(); await settle(s)
    check("fresh: done", s.phase == .done, "\(s.phase)")
    check("fresh: files byte-exact", sha(phone.appendingPathComponent(big.path)) == big.sha256 && sha(phone.appendingPathComponent(cam.path)) == cam.sha256)
    check("fresh: no parts left", !fm.fileExists(atPath: phone.appendingPathComponent(big.path + ".part").path))
    check("fresh: onInstalled chat + vision", installed == [.chat, .vision], "\(installed)")
    check("fresh: needsSetup after", !s.needsSetup)
    let gets = host.lines.filter { $0.contains("HOST GET") }.count
    host.stop()

    print("== 2 present: a second store finds them by size and fetches nothing")
    host = Host("present", [])
    s = fresh(vision: true); s.start(); await settle(s)
    check("present: done with no request", s.phase == .done && host.lines.filter { $0.contains("HOST GET") }.isEmpty, "first run GETs=\(gets)")
    host.stop()
    try? fm.removeItem(at: phone)

    print("== 3 connection lost mid-file: the automatic retry resumes with Range")
    host = Host("drop", ["--drop", "m1/model.gguf:20000000"])
    s = fresh(); s.start(); await settle(s)
    check("drop: done", s.phase == .done, "\(s.phase)")
    check("drop: resumed at the cut (206)", host.lines.contains { $0.contains("/m1/model.gguf range=bytes=20000000- -> 206") },
          host.lines.filter { $0.contains("model.gguf") }.joined(separator: " | "))
    check("drop: sha ok after resume", sha(phone.appendingPathComponent(big.path)) == big.sha256)
    check("drop: receipt says resumed_from", s.receipts.contains { $0.contains("resumed_from=20000000") })
    host.stop(); try? fm.removeItem(at: phone)

    print("== 4 pause / resume (the person's Pause, or iOS taking the background time)")
    host = Host("pause", ["--rate", "20"])
    s = fresh(); s.start()
    try? await Task.sleep(for: .milliseconds(1200))
    s.pause()
    try? await Task.sleep(for: .milliseconds(300))
    let part = ModelFetch.partBytes(phone.appendingPathComponent(big.path))
    check("pause: paused with a part kept", { if case .paused = s.phase { true } else { false } }() && part > 0, "part=\(part) bytesDone=\(s.bytesDone)")
    s.start(); await settle(s)
    check("pause: resumed to done", s.phase == .done)
    try? await Task.sleep(for: .milliseconds(1500))  // the throttled host logs after its last paced write
    check("pause: resumed with Range from the part", s.receipts.contains { $0.contains("resumed_from=\(part)") && $0.contains("Range bytes=\(part)- → 206") }
          && host.lines.contains { $0.contains("range=bytes=\(part)- -> 206") },
          host.lines.filter { $0.contains("model.gguf") }.joined(separator: " | "))
    check("pause: sha ok", sha(phone.appendingPathComponent(big.path)) == big.sha256)
    host.stop(); try? fm.removeItem(at: phone)

    print("== 5 truncated file on the host: a clean error before any body byte, nothing kept; Retry once fixed")
    host = Host("short", ["--short", "m1/model.gguf:30000000"])
    s = fresh(); s.start(); await settle(s)
    if case .failed(let m) = s.phase {
        check("short: failed with the size said plainly", m.contains("30.0 MB") || m.contains("30 MB"), m)
    } else { check("short: failed", false, "\(s.phase)") }
    check("short: no auto-retry (not the network)", s.receipts.filter { $0.contains("FAILED") }.count == 1)
    check("short: nothing kept", !fm.fileExists(atPath: phone.appendingPathComponent(big.path).path)
          && !fm.fileExists(atPath: phone.appendingPathComponent(big.path + ".part").path))
    check("short: the small file before it is kept", fm.fileExists(atPath: phone.appendingPathComponent(small.path).path))
    check("short: still needs setup (card stays, Retry)", s.needsSetup && s.rows[.chat] == .failed)
    host.stop()
    host = Host("short-fixed", [])
    s.start(); await settle(s)
    check("short: Retry after the host is fixed → done", s.phase == .done)
    host.stop(); try? fm.removeItem(at: phone)

    print("== 6 damaged bytes (same size, wrong sha256): deleted, said, Retry fetches again")
    host = Host("corrupt", ["--corrupt", "m1/model.gguf"])
    s = fresh(); s.start(); await settle(s)
    if case .failed(let m) = s.phase { check("corrupt: checksum error", m.contains("checksum"), m) } else { check("corrupt: failed", false, "\(s.phase)") }
    check("corrupt: nothing kept", !fm.fileExists(atPath: phone.appendingPathComponent(big.path).path)
          && !fm.fileExists(atPath: phone.appendingPathComponent(big.path + ".part").path))
    host.stop()
    host = Host("corrupt-fixed", [])
    s.start(); await settle(s)
    check("corrupt: Retry → done, sha ok", s.phase == .done && sha(phone.appendingPathComponent(big.path)) == big.sha256)
    host.stop(); try? fm.removeItem(at: phone)

    print("== 7 a host that ignores Range: the part starts over (still verified)")
    host = Host("norange-drop", ["--drop", "m1/model.gguf:10000000", "--no-range"])
    s = fresh(); s.start(); await settle(s)
    check("no-range: done", s.phase == .done, "\(s.phase)")
    check("no-range: second GET was a 200 from 0", s.receipts.contains { $0.contains("Range ignored") }, s.receipts.joined(separator: " | "))
    check("no-range: sha ok", sha(phone.appendingPathComponent(big.path)) == big.sha256)
    host.stop(); try? fm.removeItem(at: phone)

    print("== 8 the host down: three attempts (1 s, 3 s), then Retry; up again → done")
    port += 1
    setenv("BEND_MODELS_BASE_URL", "http://127.0.0.1:\(port)", 1)
    s = fresh(); s.start()
    var sawRetrying = false
    let t0 = Date()
    while s.isBusy, Date().timeIntervalSince(t0) < 30 {
        if case .retrying = s.phase { sawRetrying = true }
        try? await Task.sleep(for: .milliseconds(50))
    }
    if case .failed(let m) = s.phase { check("down: Can't reach, after retries", m.hasPrefix("Can’t reach") && sawRetrying, m) } else { check("down: failed", false, "\(s.phase)") }
    check("down: 3 attempts", s.receipts.filter { $0.contains("FAILED") }.count == 3, "\(s.receipts.count)")
    port -= 1
    host = Host("up", [])
    s.start(); await settle(s)
    check("down: Retry once up → done", s.phase == .done)
    host.stop(); try? fm.removeItem(at: phone)

    print("== 9 a part already complete: 416 → straight to the check")
    try! fm.createDirectory(at: phone.appendingPathComponent("m1"), withIntermediateDirectories: true)
    try! fm.copyItem(at: hostRoot.appendingPathComponent(big.path), to: phone.appendingPathComponent(big.path + ".part"))
    host = Host("416", [])
    s = fresh(); s.start(); await settle(s)
    check("416: done without re-downloading", s.phase == .done && !host.lines.contains { $0.contains("model.gguf") && $0.contains("sent=48000000") },
          host.lines.filter { $0.contains("model.gguf") }.joined(separator: " | "))
    host.stop(); try? fm.removeItem(at: phone)

    print("== 10 404 and a host-size mismatch say what is wrong")
    host = Host("404", [])
    try? fm.removeItem(at: hostRoot.appendingPathComponent(cam.path))
    s = fresh(vision: true); s.start(); await settle(s)
    if case .failed(let m) = s.phase { check("404: plain", m.contains("HTTP 404") && s.rows[.chat] == .present, m) } else { check("404", false, "\(s.phase)") }
    host.stop()

    print("RESULT models \(failed == 0 ? "PASS" : "FAIL") \(passed)/\(passed + failed)")
    exit(failed == 0 ? 0 : 1)
}

Task { @MainActor in await run() }
RunLoop.main.run()
