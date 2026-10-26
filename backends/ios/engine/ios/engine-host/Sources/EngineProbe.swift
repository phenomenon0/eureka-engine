import Foundation
import Darwin
import Metal
import os

// The engine probe: llb_load + one llb_generate over the Bend-emitted engine
// (build/LLBEngine.xcframework), with the Mac's tokens as the parity reference;
// (ENGINE-PHONE2) or one llb_decide (DecideProbe.swift) judged against R1 and the Mac.
//
// Launch arguments (NSArgumentDomain, e.g. `-- -autorun 1 -n 16`):
//   -model DIR   model dir (one .gguf + llb sidecar); absolute, or relative to
//                Documents/. Default: Documents/models/qwen35-4b-q4km, else the
//                first directory in Documents/models/.
//   -prompt p0|p2  p0 = "The capital of France is" (5 ids), p2 = engine-4b's two-part
//                ChatML request (31 ids). Default p0. Or -ids "760 6511 ..." for any ids.
//   -prompt dec|dec-rot  (ENGINE-PHONE2) llb_decide: dec = the systemone travel request (3 questions,
//                292 ids), dec-rot = the 8 cyclic rotations in one call; needs a Kev-4B model dir
//                (default Documents/models/kev-4b-q4km). -fastpf K sets LLB_DECIDE_FASTPF (default
//                512 = the fast chunk tier; 0 = the exact path).
//   -n N         new ids (default 16 for p0, 32 for p2)
//   -spanmb N    GPU span in MB (default GGUF + 3 GiB; floors measured on the Mac:
//                4B Q4_K_M 4608, Q8_0 7168; see REPORT.md "Memory")
//   -kernels 0   skip the threadgroup table (default: printed before the run)
//   -probeonly 1 device limits + span plan + threadgroup table, then stop before
//                llb_generate (the safe first launch: no weights, no jetsam)
//   -autorun 1   run on launch;  -exit 1  exit(0) after an autorun
// Every result line starts with ENGINEPROBE (console) and is also written to
// Documents/engine-probe.txt.

struct ProbePrompt {
  let name: String
  let ids: [UInt32]
  let n: Int
  let mac: [UInt32]   // the Mac's ids for this model (engine-4b receipts = this dir's Mac API), M3 Ultra Metal
}

// (ENGINE-PHONE) the 4B prompts and the Mac reference ids per quant: engine-4b's recorded
// receipts (receipts/parity/{q4kmgguf,q8gguf}-{p0,p2}.out, first 16 / 32 ids), which this
// dir's Mac-slice API reproduces byte for byte (receipts/mac/summary.txt). The quant is
// told from the model dir's name ("q80"/"q8" -> Q8_0, else Q4_K_M).
enum EngineProbe {
  // (ENGINE-PHONE2) the first 8 hex of the emitted C build-ios.sh pins (SRC_SHA); names the Metal program cache dir
  static let programTag = "c13b9e28"
  static let p0Ids: [UInt32] = [760, 6511, 314, 9338, 369]
  static let p2Ids: [UInt32] = [248045, 846, 198, 814, 20139, 303, 1330, 22157, 3069, 279, 12515, 369, 6105, 11, 1179, 1103, 2250, 7736, 314, 264, 41564, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
  static let macQ4: [String: [UInt32]] = [
    "p0": [11751, 13, 198, 32, 13, 2912, 198, 33, 13, 3439, 198, 15666, 25, 198, 32, 271],
    "p2": [760, 12515, 7701, 6105, 1521, 33470, 303, 279, 8964, 579, 16078, 43000, 2716, 89661, 314, 37728, 11, 1680, 430, 6105, 321, 76990, 11, 777, 13057, 1056, 975, 7736, 13, 10022, 76990, 3000]]
  static let macQ8: [String: [UInt32]] = [
    "p0": [11751, 13, 198, 32, 13, 2912, 198, 33, 13, 3439, 198, 15666, 25, 198, 32, 271],
    "p2": [760, 12515, 7701, 6105, 1521, 33470, 303, 279, 16078, 43000, 22602, 89661, 314, 37728, 318, 11855, 321, 76990, 8, 777, 13057, 1056, 4860, 89661, 13, 10022, 76990, 3000, 369, 35742, 1442, 777]]

  static var platform: String {
    #if targetEnvironment(simulator)
    return "SIMULATOR/Metal"
    #else
    return "DEVICE/Metal"
    #endif
  }

  static var documents: URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
  }

  static func modelDir() -> URL? {
    let fm = FileManager.default
    if let m = UserDefaults.standard.string(forKey: "model") {
      return m.hasPrefix("/") ? URL(fileURLWithPath: m) : documents.appendingPathComponent(m)
    }
    let models = documents.appendingPathComponent("models")
    let dec = (UserDefaults.standard.string(forKey: "prompt") ?? "").hasPrefix("dec")
    let dflt = models.appendingPathComponent(dec ? "kev-4b-q4km" : "qwen35-4b-q4km")
    if fm.fileExists(atPath: dflt.path) { return dflt }
    let dirs = (try? fm.contentsOfDirectory(at: models, includingPropertiesForKeys: nil)) ?? []
    return dirs.sorted { $0.path < $1.path }.first
  }

  static func prompt(_ dir: URL) -> ProbePrompt {
    let d = UserDefaults.standard
    let name = dir.lastPathComponent.lowercased()
    let mac = name.contains("q80") || name.contains("q8") ? macQ8 : macQ4
    var p = d.string(forKey: "prompt") == "p2"
      ? ProbePrompt(name: "p2", ids: p2Ids, n: 32, mac: mac["p2"]!)
      : ProbePrompt(name: "p0", ids: p0Ids, n: 16, mac: mac["p0"]!)
    if let s = d.string(forKey: "ids") {
      let ids = s.split(separator: " ").compactMap { UInt32($0) }
      p = ProbePrompt(name: "custom", ids: ids, n: p.n, mac: [])
    }
    let n = d.integer(forKey: "n")
    return n > 0 ? ProbePrompt(name: p.name, ids: p.ids, n: n, mac: p.mac) : p
  }

  static func footprint() -> (now: UInt64, peak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    return kr == KERN_SUCCESS ? (info.phys_footprint, UInt64(info.ledger_phys_footprint_peak)) : (0, 0)
  }

  static func availableMemory() -> UInt64 {
    #if os(iOS)
    return UInt64(os_proc_available_memory())
    #else
    return 0
    #endif
  }

  static func maxBuffer() -> UInt64 {
    MTLCreateSystemDefaultDevice().map { UInt64($0.maxBufferLength) } ?? 0
  }

  static func emit(_ line: String, _ out: inout [String]) {
    let l = "ENGINEPROBE " + line
    out.append(l)
    print(l)
    fflush(stdout)
  }

  /// One probe; returns the result lines. One llb_generate per process (R1).
  static func run() -> [String] {
    var out: [String] = []
    let mb = { (b: UInt64) in String(format: "%.0f", Double(b) / 1048576) }
    emit("platform=\(platform) device: \(String(cString: llb_device_info()))", &out)
    let fp0 = footprint()
    emit("memory at start: footprint_mb=\(mb(fp0.now)) proc_available_mb=\(mb(availableMemory()))", &out)
    if UserDefaults.standard.object(forKey: "kernels") == nil || UserDefaults.standard.bool(forKey: "kernels") {
      emit("kernels \(String(cString: llb_kernel_report()))", &out)
    }
    guard let dir = modelDir() else {
      emit("ERROR no model dir (Documents/models/<name>/ with a .gguf + sidecar)", &out)
      return save(out)
    }
    let p = prompt(dir)
    // (ENGINE-PHONE2) the cache dir is keyed by the emitted program: the engine reuses <cache>/sys1mac.gpu whenever it
    // exists, and a stale archive from an older program (engine-phone's 3ba355ea in Caches/llb) only misses
    // (MTLPipelineOptionFailOnBinaryArchiveMiss -> an uncached compile at every launch), never refreshes
    let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("llb-" + programTag)
    try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
    let mode = UserDefaults.standard.string(forKey: "prompt") ?? "p0"
    let decide = mode == "dec" || mode == "dec-rot"
    emit(decide ? "model=\(dir.path) cache=\(cache.path) prompt=\(mode) (llb_decide)"
      : "model=\(dir.path) cache=\(cache.path) prompt=\(p.name) n_ids=\(p.ids.count) n_new=\(p.n)", &out)
    // -spanmb N: the GPU span in MB (LLB_GPU_SPAN_MB), e.g. 4608 = the 4B Q4_K_M's measured floor
    let spanmb = UserDefaults.standard.integer(forKey: "spanmb")
    if spanmb > 0 { setenv("LLB_GPU_SPAN_MB", String(spanmb), 1) }
    let t0 = Date()
    if llb_load(dir.path, cache.path) != 0 {
      emit("ERROR load: \(String(cString: llb_error()))", &out)
      return save(out)
    }
    // (ENGINE-PHONE) the 4B span vs this device, and the threadgroup table, before the
    // engine runs (a refusal or a jetsam still leaves these lines)
    let plan = llb_span_plan()
    emit("span plan_mb=\(mb(plan)) maxBufferLength_mb=\(mb(maxBuffer())) proc_available_mb=\(mb(availableMemory())) "
      + (plan > maxBuffer() ? "SPAN>maxBufferLength" : "span fits one MTLBuffer"), &out)
    if UserDefaults.standard.bool(forKey: "probeonly") {
      emit("probeonly: stopping before llb_generate", &out)
      return save(out)
    }
    if decide {
      _ = DecideProbe.run(mode) { emit($0, &out) }
      llb_free()
      return save(out)
    }
    var ids = [UInt32](repeating: 0, count: p.n)
    var st = llb_stats()
    let got = p.ids.withUnsafeBufferPointer { ip in
      ids.withUnsafeMutableBufferPointer { op in
        llb_generate(ip.baseAddress, Int32(p.ids.count), Int32(p.n), op.baseAddress, &st)
      }
    }
    let wall = Date().timeIntervalSince(t0) * 1000
    let fp = footprint()
    if got < 0 {
      emit("ERROR generate: \(String(cString: llb_error())) exit=\(st.exit_code)", &out)
      emit("memory span_mb=\(mb(st.span_bytes)) footprint_mb=\(mb(fp.now)) footprint_peak_mb=\(mb(fp.peak)) proc_available_mb=\(mb(availableMemory()))", &out)
      let log = String(cString: llb_log()).split(separator: "\n").filter { !$0.hasPrefix("logits") }
      for l in log.suffix(12) { emit("log \(l)", &out) }
      return save(out)
    }
    let res = Array(ids.prefix(Int(got)))
    emit("ids \(res.map(String.init).joined(separator: " "))", &out)
    if !p.mac.isEmpty {
      let n = min(res.count, p.mac.count)
      let div = (0..<n).first { res[$0] != p.mac[$0] }
      if let i = div {
        emit("PARITY DIFF vs Mac: first divergence at new token \(i): ios=\(res[i]) mac=\(p.mac[i])", &out)
      } else {
        emit("PARITY OK vs Mac (\(n)/\(p.mac.count) tokens identical)", &out)
      }
    }
    let dec = st.steps > 0 ? st.step_ms_sum / Double(st.steps) : 0
    emit(String(format: "stats compile_ms=%.0f load_ms=%.0f prefill_ms=%.0f decode_ms_per_tok=%.2f tok_s=%.1f steps=%d total_ms=%.0f wall_ms=%.0f",
      st.compile_ms, st.load_ms, st.prefill_ms, dec, dec > 0 ? 1000 / dec : 0, st.steps, st.total_ms, wall), &out)
    emit("memory span_mb=\(mb(st.span_bytes)) gguf_mb=\(mb(st.gguf_bytes)) footprint_mb=\(mb(fp.now)) footprint_peak_mb=\(mb(fp.peak)) proc_available_mb=\(mb(availableMemory()))", &out)
    llb_free()
    return save(out)
  }

  static func save(_ out: [String]) -> [String] {
    let f = documents.appendingPathComponent("engine-probe.txt")
    try? (out.joined(separator: "\n") + "\n").write(to: f, atomically: true, encoding: .utf8)
    return out
  }
}
