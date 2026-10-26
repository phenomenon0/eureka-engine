import Foundation

// (ENGINE-PHONE2) the decide probe: llb_decide (engine-one's fused head, engine-decidechunk's fast chunk tier by
// default) on the travel request / the 8 cyclic rotations, judged in the app by engine-one's check.py rules:
//   vs R1 (llama-server Metal, the J3 reference): same picks, same layout (answer, field, option, legend order),
//     same usage, every field |d| <= 0.03                                           -> "J3 ... PASS|FAIL"
//   vs the Mac's own fast decide reply (engine-decidechunk receipts/d2):  max |d|, byte-identical or not
//   rot5..7 vs rot0..2 in the same run: byte-identical (fresh state per request)
// The fixtures are the bundle folder "decide" (ios/engine-host/Resources/decide).

// JSON with key order kept (JSONSerialization drops it; the layout check needs it)
indirect enum OJ {
  case obj([(String, OJ)]), arr([OJ]), num(Double), str(String), bool(Bool), null

  subscript(_ k: String) -> OJ? {
    if case .obj(let kv) = self { return kv.first { $0.0 == k }?.1 }
    return nil
  }
  var keys: [String] { if case .obj(let kv) = self { return kv.map { $0.0 } }; return [] }
  var num: Double? { if case .num(let d) = self { return d }; return nil }
  var str: String? { if case .str(let s) = self { return s }; return nil }

  static func parse(_ s: String) -> OJ? {
    var p = OJParser(Array(s.utf8))
    guard let v = p.value() else { return nil }
    p.ws()
    return p.i == p.b.count ? v : nil
  }

  // a canonical text for equality of small sub-objects (legend)
  var text: String {
    switch self {
    case .obj(let kv): return "{" + kv.map { "\"\($0.0)\":\($0.1.text)" }.joined(separator: ",") + "}"
    case .arr(let a): return "[" + a.map { $0.text }.joined(separator: ",") + "]"
    case .num(let d): return String(d)
    case .str(let s): return "\"\(s)\""
    case .bool(let b): return String(b)
    case .null: return "null"
    }
  }
}

struct OJParser {
  let b: [UInt8]
  var i = 0
  init(_ b: [UInt8]) { self.b = b }

  mutating func ws() { while i < b.count, [0x20, 0x0a, 0x0d, 0x09].contains(b[i]) { i += 1 } }

  mutating func lit(_ s: String) -> Bool {
    let u = Array(s.utf8)
    guard i + u.count <= b.count, Array(b[i..<i + u.count]) == u else { return false }
    i += u.count
    return true
  }

  mutating func string() -> String? {
    guard i < b.count, b[i] == 0x22 else { return nil }
    i += 1
    var out: [UInt8] = []
    while i < b.count, b[i] != 0x22 {
      if b[i] == 0x5c, i + 1 < b.count {
        i += 1
        switch b[i] {
        case 0x6e: out.append(0x0a)
        case 0x74: out.append(0x09)
        case 0x72: out.append(0x0d)
        case 0x62: out.append(0x08)
        case 0x66: out.append(0x0c)
        case 0x75:
          guard i + 4 < b.count, let v = UInt32(String(decoding: b[i + 1...i + 4], as: UTF8.self), radix: 16)
          else { return nil }
          i += 4
          out.append(contentsOf: Array(String(UnicodeScalar(v).map(Character.init) ?? "?").utf8))
        default: out.append(b[i])
        }
      } else {
        out.append(b[i])
      }
      i += 1
    }
    guard i < b.count else { return nil }
    i += 1
    return String(decoding: out, as: UTF8.self)
  }

  mutating func value() -> OJ? {
    ws()
    guard i < b.count else { return nil }
    switch b[i] {
    case 0x7b:
      i += 1
      var kv: [(String, OJ)] = []
      ws()
      if i < b.count, b[i] == 0x7d { i += 1; return .obj(kv) }
      while true {
        ws()
        guard let k = string() else { return nil }
        ws()
        guard i < b.count, b[i] == 0x3a else { return nil }
        i += 1
        guard let v = value() else { return nil }
        kv.append((k, v))
        ws()
        guard i < b.count else { return nil }
        if b[i] == 0x2c { i += 1; continue }
        if b[i] == 0x7d { i += 1; return .obj(kv) }
        return nil
      }
    case 0x5b:
      i += 1
      var a: [OJ] = []
      ws()
      if i < b.count, b[i] == 0x5d { i += 1; return .arr(a) }
      while true {
        guard let v = value() else { return nil }
        a.append(v)
        ws()
        guard i < b.count else { return nil }
        if b[i] == 0x2c { i += 1; continue }
        if b[i] == 0x5d { i += 1; return .arr(a) }
        return nil
      }
    case 0x22: return string().map { .str($0) }
    case 0x74: return lit("true") ? .bool(true) : nil
    case 0x66: return lit("false") ? .bool(false) : nil
    case 0x6e: return lit("null") ? .null : nil
    default:
      let s = i
      while i < b.count, "+-0123456789.eE".utf8.contains(b[i]) { i += 1 }
      return Double(String(decoding: b[s..<i], as: UTF8.self)).map { .num($0) }
    }
  }
}

enum DecideProbe {
  static let tol = 0.03   // engine-one's J3 field rule

  static func fixture(_ name: String) -> String? {
    guard let u = Bundle.main.url(forResource: name, withExtension: nil, subdirectory: "decide") else { return nil }
    return (try? String(contentsOf: u, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
  }

  // check.py fields(): every number the gate compares, in reply order
  static func fields(_ a: OJ) -> [(String, Double)] {
    var out: [(String, Double)] = []
    for k in ["noul", "score"] { if let v = a[k]?.num { out.append((k, v)) } }
    if case .obj(let kv)? = a["probabilities"] { for (o, p) in kv { out.append(("p[\(o)]", p.num ?? .nan)) } }
    if let c = a["confidence"]?.num { out.append(("confidence", c)) }
    return out
  }

  // check.py pick(): the choice, the most likely score, or the noul's side
  static func pick(_ a: OJ) -> String {
    switch a["type"]?.str {
    case "choice": return a["choice"]?.str ?? "?"
    case "score":
      if case .obj(let kv)? = a["probabilities"] {
        return kv.max { ($0.1.num ?? -1) < ($1.1.num ?? -1) }?.0 ?? "?"
      }
      return "?"
    default: return (a["noul"]?.num ?? 0) >= 0.5 ? "yes" : "no"
    }
  }

  struct Cmp { var maxd = 0.0; var layout: [String] = []; var tops: [String] = []; var usage = true }

  // check.py compare(a = ours, b = reference)
  static func compare(_ a: OJ, _ b: OJ) -> Cmp {
    var c = Cmp()
    let aa = a["answers"] ?? .obj([]), bb = b["answers"] ?? .obj([])
    if aa.keys != bb.keys { c.layout.append("answer order") }
    for q in bb.keys {
      guard let ra = aa[q], let rb = bb[q] else { c.layout.append("\(q) missing"); continue }
      if ra.keys != rb.keys { c.layout.append("\(q): fields") }
      if (ra["probabilities"]?.keys ?? []) != (rb["probabilities"]?.keys ?? []) { c.layout.append("\(q): option order") }
      if ra["legend"]?.text != rb["legend"]?.text { c.layout.append("\(q): legend") }
      if ra["type"]?.str != rb["type"]?.str { c.layout.append("\(q): type") }
      if pick(ra) != pick(rb) { c.tops.append("\(q): top \(pick(ra)) != \(pick(rb))") }
      let fa = Dictionary(fields(ra), uniquingKeysWith: { x, _ in x })
      for (n, vb) in fields(rb) { c.maxd = max(c.maxd, abs((fa[n] ?? .nan) - vb)) }
      if c.maxd.isNaN { c.maxd = .infinity }
    }
    c.usage = a["usage"]?.text == b["usage"]?.text
    return c
  }

  static func verdict(_ c: Cmp) -> Bool { c.layout.isEmpty && c.tops.isEmpty && c.usage && c.maxd <= tol }

  static func describe(_ c: Cmp) -> String {
    String(format: "max|d|=%.6f", c.maxd) + " picks " + (c.tops.isEmpty ? "same" : c.tops.joined(separator: "; "))
      + " layout " + (c.layout.isEmpty ? "same" : c.layout.joined(separator: "; ")) + " usage " + (c.usage ? "same" : "DIFFERS")
  }

  /// -prompt dec (the travel request, one call) or dec-rot (8 rotations, rot k mod 5, one call)
  static func run(_ mode: String, _ emit: (String) -> Void) -> Bool {
    let rot = mode == "dec-rot"
    var bodies: [String] = [], refs: [String] = [], macs: [String] = []
    if rot {
      let mac = (fixture("rotations.mac.jsonl") ?? "").split(separator: "\n").map(String.init)
      for k in 0..<8 {
        guard let b = fixture("rot\(k % 5).json"), let r = fixture("rot\(k % 5).ref.json") else {
          emit("ERROR decide: missing bundle fixture decide/rot\(k % 5){,.ref}.json"); return false
        }
        bodies.append(b); refs.append(r); macs.append(k < mac.count ? mac[k] : "")
      }
    } else {
      guard let b = fixture("travel.json"), let r = fixture("travel.ref.json") else {
        emit("ERROR decide: missing bundle fixture decide/travel{,.ref}.json"); return false
      }
      bodies = [b]; refs = [r]; macs = [fixture("travel.mac.json") ?? ""]
    }
    // -fastpf K: llb_decide's prefill tier (LLB_DECIDE_FASTPF; default 512 = fast chunk, 0 = exact)
    if let f = UserDefaults.standard.string(forKey: "fastpf") { setenv("LLB_DECIDE_FASTPF", f, 1) }
    let t0 = Date()
    var st = llb_dstats()
    var replies = [UnsafeMutablePointer<CChar>?](repeating: nil, count: bodies.count)
    let cbodies = bodies.map { strdup($0) }
    defer { cbodies.forEach { free($0) } }
    let got = cbodies.map { $0.map { UnsafePointer($0) } }.withUnsafeBufferPointer { bp in
      replies.withUnsafeMutableBufferPointer { rp in
        llb_decide(bp.baseAddress, Int32(bodies.count), rp.baseAddress, &st)
      }
    }
    let wall = Date().timeIntervalSince(t0) * 1000
    if got < 0 {
      emit("ERROR decide: \(String(cString: llb_error())) exit=\(st.exit_code)")
      let log = String(cString: llb_log()).split(separator: "\n")
      for l in log.suffix(12) { emit("log \(l)") }
      return false
    }
    let lines = replies.map { $0.map { String(cString: $0) } ?? "" }
    replies.forEach { free($0) }
    var all = true
    for (k, line) in lines.enumerated() {
      let tag = rot ? "rot\(k)" : "travel"
      emit("reply \(tag) \(line)")
      guard let ours = OJ.parse(line), let ref = OJ.parse(refs[k]), ours["answers"] != nil else {
        emit("J3 \(tag) vs R1: FAIL (reply is not an answer)"); all = false; continue
      }
      if let d = ours["answers"]?["destination"] {
        let p = d["probabilities"]?[pick(d)]?.num ?? .nan
        emit(String(format: "DECIDE %@ destination=%@ p=%.6f", tag, pick(d), p))
      }
      let c = compare(ours, ref)
      let ok = verdict(c)
      all = all && ok
      emit("J3 \(tag) vs R1 llama-server: \(ok ? "PASS" : "FAIL") \(describe(c))")
      if let mac = OJ.parse(macs[k]) {
        let m = compare(ours, mac)
        emit("MACREF \(tag) vs Mac fast decide: " + (line == macs[k] ? "BYTE-IDENTICAL" : "differs") + " \(describe(m))")
      }
    }
    if rot {
      for k in 5..<min(8, lines.count) {
        let same = lines[k] == lines[k - 5]
        all = all && same
        emit("J3 rot\(k) vs rot\(k - 5) (same request, later in the same run): reply \(same ? "BYTE-IDENTICAL" : "DIFFERS")")
      }
    }
    emit("J3 \(mode): \(all ? "PASS" : "FAIL") (\(got)/\(bodies.count) answered)")
    emit(String(format: "stats_decide fastpf=%d compile_ms=%.0f load_ms=%.0f head_open_ms=%.1f forward_ms=%.1f head_ms=%.2f "
      + "chunks=%d chunk_tokens=%d chunk_ms_sum=%.0f steps=%d step_ms_sum=%.0f questions=%d input_tokens=%d total_ms=%.0f wall_ms=%.0f",
      st.fastpf, st.compile_ms, st.load_ms, st.head_open_ms, st.forward_ms, st.head_ms, st.chunks, st.chunk_tokens,
      st.chunk_ms_sum, st.steps, st.step_ms_sum, st.questions, st.input_tokens, st.total_ms, wall))
    let mb = { (b: UInt64) in String(format: "%.0f", Double(b) / 1048576) }
    let fp = EngineProbe.footprint()
    emit("memory span_mb=\(mb(st.span_bytes)) gguf_mb=\(mb(st.gguf_bytes)) footprint_mb=\(mb(fp.now)) footprint_peak_mb=\(mb(fp.peak)) proc_available_mb=\(mb(EngineProbe.availableMemory()))")
    return all
  }
}
