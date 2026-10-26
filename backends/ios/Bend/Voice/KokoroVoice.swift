import CoreML
import Foundation
import onnxruntime

/// VOICESHOP: Kokoro-82M v1.0 (hexgrad, Apache-2.0) as one of the phone's voice engines (docs/VOICES.md).
///
/// One runtime, many voices: the model (kokoro-v1.0.int8.onnx, 92 MB, kokoro-onnx's int8 export) loads once; a voice is
/// only its style table (510 × 256 float32 = 522,240 bytes), so changing voice is swapping that table (no reload).
/// Text → phonemes is `KokoroG2P`: misaki's own gold + silver lexicons (Apache-2.0) and its rules for -s/-ed/-ing,
/// numbers, stress and the/a/to, in Swift. No espeak-ng (GPL) and no POS tagger in the app: an out-of-lexicon word
/// that no rule covers is split into known words or spelled, never sent anywhere.
/// One chunk at a time (the web's chunker already cuts the reply); not thread-safe: the Speaker calls it from one queue.
///
/// Files (`dir`, bundled as /voices/kokoro by scripts/fetch-voice.sh): kokoro-v1.0.int8.onnx, vocab.json (the model's
/// phoneme → id table, from hexgrad's config.json), us_gold.json, us_silver.json, gb_gold.json, gb_silver.json
/// (misaki), packs/<id>.bin + <id>.json (the bundled voices).
final class KokoroRuntime: @unchecked Sendable {
    /// Where runtime lines go: stderr here; the app points it at Voice.log (the console the receipts read).
    nonisolated(unsafe) static var log: (String) -> Void = { FileHandle.standardError.write(Data("BEND voice \($0)\n".utf8)) }
    static let modelFile = "kokoro-v1.0.int8.onnx"
    static let fileNames = [modelFile, "vocab.json", "us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json"]
    static let sampleRate = 24_000
    /// A style table: one 256-float row per phoneme count (0…509).
    static let styleRows = 510, styleDim = 256
    static let styleBytes = styleRows * styleDim * 4

    let dir: URL
    private let threads: Int32
    private var ort: ORT?
    private var session: ORT.Session?
    /// VOICESHOP2: Kokoro's Core ML chain (KokoroANE), when bundled and not switched off: ~15× less time per chunk than
    /// onnxruntime's int8 path on the CPU (work/ane/RESULT.md). onnx is the fallback (load error, a chunk it can't take)
    /// and loads only then, so the two models are never both resident unless the fallback was needed.
    private var coreml: KokoroANE?
    private let vocab: [Character: Int64]
    private var g2p: [Bool: KokoroG2P] = [:]  // british → its lexicon, loaded on first use

    /// The bundled engine dir; BEND_KOKORO_DIR overrides it (the Mac harness).
    static func directory() -> URL? {
        var candidates: [URL] = []
        if let env = ProcessInfo.processInfo.environment["BEND_KOKORO_DIR"] { candidates.append(URL(fileURLWithPath: env)) }
        if let bundled = Bundle.main.url(forResource: "kokoro", withExtension: nil, subdirectory: "voices") { candidates.append(bundled) }
        return candidates.first { dir in fileNames.allSatisfy { FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path) } }
    }

    /// Which runtime rendered the last chunk, for the receipts' timer ("coreml-cpu", "onnx-cpu-2t").
    private(set) var label: String

    /// `coreml: false` = onnxruntime only (the bench, the kill switch).
    init(dir: URL, threads: Int32 = KokoroRuntime.threads, coreml: Bool = KokoroRuntime.useCoreML) throws {
        self.dir = dir
        self.threads = threads
        label = "onnx-cpu-\(threads)t"
        let table = try JSONDecoder().decode([String: Int64].self, from: Data(contentsOf: dir.appendingPathComponent("vocab.json")))
        vocab = Dictionary(uniqueKeysWithValues: table.compactMap { k, v in k.count == 1 ? (k.first!, v) : nil })
        if coreml, let ane = KokoroANE.directory() {
            do {
                let t0 = Date()
                self.coreml = try KokoroANE(dir: ane, units: Self.coremlUnits, bucket: Self.bucket)
                label = Self.coremlUnits == .cpuOnly ? "coreml-cpu" : "coreml-ane"
                KokoroRuntime.log(String(format: "kokoro runtime %@ loaded in %.0f ms (onnx stays unloaded unless needed)", label, Date().timeIntervalSince(t0) * 1000))
            } catch {
                KokoroRuntime.log("kokoro runtime coreml failed to load (\(error.localizedDescription)); onnx renders")
            }
        }
        if self.coreml == nil { try openONNX() }
    }

    private func openONNX() throws {
        guard session == nil else { return }
        let o = try ORT(threads: threads)
        session = try o.session(dir.appendingPathComponent(Self.modelFile))
        ort = o
    }

    /// The Core ML chain unless `BendKokoroRuntime=onnx` (UserDefaults / `-BendKokoroRuntime onnx`) or
    /// BEND_KOKORO_RUNTIME=onnx: the kill switch, should the chain misbehave on some iOS build.
    static var useCoreML: Bool {
        (ProcessInfo.processInfo.environment["BEND_KOKORO_RUNTIME"] ?? UserDefaults.standard.string(forKey: "BendKokoroRuntime") ?? "coreml") != "onnx"
    }
    /// .cpuOnly by default: as fast as .cpuAndNeuralEngine here (the chain's dynamic shapes keep the ANE out), and no
    /// multi-second ANE compile on the first load after an install. `-kokoroUnits ane` to try the Neural Engine.
    static var coremlUnits: MLComputeUnits {
        switch unitsSetting {
        case "ane": return .cpuAndNeuralEngine
        case "auto": return hasNeuralEngine ? .cpuAndNeuralEngine : .cpuOnly
        default: return .cpuOnly
        }
    }
    /// G28: `cpu` (default, as shipped) | `ane` | `auto` (the ANE where the device has one, with the bucket).
    /// `-kokoroUnits <v>` at launch (also UserDefaults BendKokoroUnits) or BEND_KOKORO_UNITS on the Mac harness.
    static var unitsSetting: String {
        let a = ProcessInfo.processInfo.arguments
        if let i = a.firstIndex(of: "-kokoroUnits"), i + 1 < a.count { return a[i + 1] }
        return ProcessInfo.processInfo.environment["BEND_KOKORO_UNITS"] ?? UserDefaults.standard.string(forKey: "BendKokoroUnits") ?? "cpu"
    }
    /// G28: run first chunks at the models' default shapes, the only ones the ANE executes (KokoroANE.bucket). On with
    /// `auto`, or `-kokoroBucket` / BEND_KOKORO_BUCKET=1 / UserDefaults BendKokoroBucket for any units.
    static var bucket: Bool {
        unitsSetting == "auto" || ProcessInfo.processInfo.arguments.contains("-kokoroBucket")
            || ProcessInfo.processInfo.environment["BEND_KOKORO_BUCKET"] == "1" || UserDefaults.standard.bool(forKey: "BendKokoroBucket")
    }
    static var hasNeuralEngine: Bool {
        MLModel.availableComputeDevices.contains { if case .neuralEngine = $0 { return true }; return false }
    }

    /// 2 by default (as the Supertonic voice); BEND_KOKORO_THREADS / `-kokoroThreads N` for measurements.
    static var threads: Int32 {
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-kokoroThreads"), i + 1 < args.count, let n = Int32(args[i + 1]) { return n }
        return ProcessInfo.processInfo.environment["BEND_KOKORO_THREADS"].flatMap { Int32($0) } ?? 2
    }

    func phonemizer(british: Bool) throws -> KokoroG2P {
        if let g = g2p[british] { return g }
        let p = british ? "gb" : "us"
        let g = try KokoroG2P(gold: dir.appendingPathComponent("\(p)_gold.json"), silver: dir.appendingPathComponent("\(p)_silver.json"), british: british)
        g2p[british] = g
        return g
    }

    /// Phonemes → ids (unknown symbols dropped), at most 510 with the two pads.
    func ids(_ phonemes: String) -> [Int64] { phonemes.compactMap { vocab[$0] } }

    /// One run: [0, ids…, 0] with the style row for ids.count → 24 kHz mono.
    func run(_ ids: [Int64], style: [Float], speed: Float) throws -> [Float] {
        guard !ids.isEmpty else { return [] }
        if let coreml {
            do {
                // The chain's output sits 3.2 dB under onnx's on the same ids (measured): matched here, so a voice is
                // equally loud whichever runtime rendered the chunk.
                let out = try coreml.run(ids, style: style, speed: speed).map { $0 * 1.45 }
                label = (Self.coremlUnits == .cpuOnly ? "coreml-cpu" : "coreml-ane") + (coreml.lastBucketed ? "-f120" : "")
                return out
            } catch {
                KokoroRuntime.log("kokoro coreml failed on a chunk (\(error.localizedDescription)); onnx renders it")
            }
        }
        try openONNX()
        guard let ort, let session else { throw NeuralVoiceError("no Kokoro runtime") }
        label = "onnx-cpu-\(threads)t"
        let row = min(ids.count, Self.styleRows - 1) * Self.styleDim
        let tokens = try ort.tensor([0] + ids + [0], [1, Int64(ids.count + 2)])
        let s = try ort.tensor(Array(style[row ..< row + Self.styleDim]), [1, Int64(Self.styleDim)])
        let sp = try ort.tensor([speed], [1])
        return try session.run(["tokens": tokens, "style": s, "speed": sp], ["audio"])[0].floats()
    }
}

/// One Kokoro voice: the shared runtime + this pack's style table.
final class KokoroVoice: VoiceEngine, @unchecked Sendable {
    let name: String
    let sampleRate = KokoroRuntime.sampleRate
    let runtime: KokoroRuntime
    let style: [Float]
    let british: Bool
    var speed: Float

    init(runtime: KokoroRuntime, name: String, style: [Float], british: Bool, speed: Float = 1) {
        self.runtime = runtime
        self.name = name
        self.style = style
        self.british = british
        self.speed = speed
    }

    func synthesize(_ text: String) throws -> [Float] {
        let phonemes = try runtime.phonemizer(british: british).phonemize(text)
        var out: [Float] = []
        // Over ~500 ids the model rushes (hexgrad's VOICES.md); the chunker's clauses never get near, a pasted page might.
        for piece in Self.split(phonemes, max: 400) {
            out += Self.trim(try runtime.run(runtime.ids(piece), style: style, speed: speed), rate: sampleRate)
        }
        return Self.lift(out)
    }

    /// VOICE-LATENCY (KOKOROWIRE's trim): Kokoro pads every render with ~0.42 s of silence before the voice and ~0.5 s
    /// after (measured on the Core ML chain, results/voice-latency/). The Speaker plays a chunk from its first sample, so
    /// the lead is pure time-to-first-word and lead + tail is a ~1 s gap between chunks: keep 50 ms / 200 ms. Samples
    /// under 0.01 (-40 dBFS, before the lift) count as silence.
    static func trim(_ x: [Float], rate: Int, threshold: Float = 0.01) -> [Float] {
        guard let first = x.firstIndex(where: { abs($0) > threshold }), let last = x.lastIndex(where: { abs($0) > threshold }) else { return x }
        return Array(x[max(0, first - rate / 20)..<min(x.count, last + rate / 5)])
    }

    /// VOICESHOP2: Kokoro speaks at ~-24 dBFS RMS, peaks ~0.5: quiet on a phone speaker next to other apps. A fixed
    /// +4 dB (the same for every chunk, so loudness never jumps between chunks), held under 0.98 at the peak.
    static let gain: Float = 1.58
    static func lift(_ samples: [Float]) -> [Float] {
        let peak = samples.reduce(0) { max($0, abs($1)) }
        let g = peak * gain > 0.98 ? 0.98 / max(peak, 1e-6) : gain
        return samples.map { $0 * g }
    }

    /// Cuts at the last space before `max` symbols (punctuation first), so no run is over the model's window.
    static func split(_ ps: String, max: Int) -> [String] {
        var rest = Substring(ps), parts: [String] = []
        while rest.count > max {
            let window = rest.prefix(max)
            let cut = window.lastIndex(where: { ".!?;:,—…".contains($0) }).map { window.index(after: $0) }
                ?? window.lastIndex(of: " ") ?? window.endIndex
            parts.append(String(rest[..<cut]).trimmingCharacters(in: .whitespaces))
            rest = rest[cut...].drop { $0 == " " }
        }
        if !rest.isEmpty { parts.append(String(rest)) }
        return parts
    }

    // MARK: pack files

    /// A Kokoro style table from a pack file: `.bin` (raw little-endian float32, 510 × 256: kokoro-onnx's and
    /// sherpa-onnx's per-voice layout) or `.pt` (hexgrad's own file, as published: a stored zip holding `<id>/data/0`).
    static func style(at url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let raw: Data
        switch url.pathExtension.lowercased() {
        case "bin": raw = data
        case "pt": raw = try ptTensor(data)
        default: throw NeuralVoiceError("\(url.lastPathComponent): not a Kokoro voice (.bin or .pt)")
        }
        guard raw.count == KokoroRuntime.styleBytes else {
            throw NeuralVoiceError("\(url.lastPathComponent): \(raw.count) bytes of style; a Kokoro voice is \(KokoroRuntime.styleBytes) (510 × 256 float32)")
        }
        return raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// The tensor bytes inside a torch .pt (zip, stored): the central directory's `…/data/0` entry.
    static func ptTensor(_ zip: Data) throws -> Data {
        func u16(_ o: Int) -> Int { Int(zip[zip.startIndex + o]) | Int(zip[zip.startIndex + o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        let bad = NeuralVoiceError("not a torch voice file (.pt zip)")
        guard zip.count > 22 else { throw bad }
        var eocd = zip.count - 22
        while eocd >= 0, u32(eocd) != 0x0605_4b50 { eocd -= 1 }
        guard eocd >= 0 else { throw bad }
        var at = u32(eocd + 16)
        for _ in 0 ..< u16(eocd + 10) {
            guard at + 46 <= zip.count, u32(at) == 0x0201_4b50 else { throw bad }
            let method = u16(at + 10), size = u32(at + 20), nameLen = u16(at + 28), extraLen = u16(at + 30), commentLen = u16(at + 32)
            let local = u32(at + 42)
            let name = String(decoding: zip[(zip.startIndex + at + 46) ..< (zip.startIndex + at + 46 + nameLen)], as: UTF8.self)
            if name.hasSuffix("/data/0") {
                guard method == 0 else { throw NeuralVoiceError("compressed .pt (method \(method)): save it with torch.save") }
                guard local + 30 <= zip.count, u32(local) == 0x0403_4b50 else { throw bad }
                let start = local + 30 + u16(local + 26) + u16(local + 28)
                guard start + size <= zip.count else { throw bad }
                return zip.subdata(in: (zip.startIndex + start) ..< (zip.startIndex + start + size))
            }
            at += 46 + nameLen + extraLen + commentLen
        }
        throw bad
    }
}

// MARK: - text → phonemes

/// misaki's English G2P (hexgrad/misaki `en.py`, Apache-2.0) without spaCy and without its neural/espeak fallback:
/// gold then silver lexicon, capital stress, the -s/-ed/-ing stems, the/a/an/to/in by the next sound, numbers (cardinal,
/// year, ordinal, decimal, money, clock), a few abbreviations. Out-of-lexicon words: split into known parts, else spelled.
final class KokoroG2P: @unchecked Sendable {
    let british: Bool
    private let gold: [String: String]
    private let silver: [String: String]
    /// Words the last `phonemize` could not find (for receipts).
    private(set) var unknown: [String] = []

    init(gold: URL, silver: URL, british: Bool) throws {
        self.british = british
        self.gold = try Self.load(gold)
        self.silver = try Self.load(silver)
    }

    /// The lexicon's value is a string, or {POS: string|null} with a DEFAULT: without a tagger, DEFAULT.
    private static func load(_ url: URL) throws -> [String: String] {
        guard let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else { throw NeuralVoiceError("\(url.lastPathComponent): not a lexicon") }
        var out: [String: String] = [:]
        out.reserveCapacity(raw.count)
        for (k, v) in raw {
            if let s = v as? String { out[k] = s } else if let d = v as? [String: Any], let s = d["DEFAULT"] as? String { out[k] = s }
        }
        return out
    }

    static let stresses: Set<Character> = ["ˌ", "ˈ"]
    static let primary: Character = "ˈ", secondary: Character = "ˌ"
    static let vowels = Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
    static let consonants = Set("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
    static let usTaus = Set("AIOWYiuæɑəɛɪɹʊʌ")
    static let puncts = Set(";:,.!?—…\"“”")
    static let nonQuotePuncts = Set(";:,.!?—…")
    static let symbols = ["%": "percent", "&": "and", "+": "plus", "@": "at"]
    static let abbreviations = ["Dr": "doctor", "Mr": "mister", "Mrs": "missus", "Ms": "miz", "St": "saint", "vs": "versus", "etc": "et cetera",
                                "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "Jun": "June", "Jul": "July", "Aug": "August",
                                "Sep": "September", "Sept": "September", "Oct": "October", "Nov": "November", "Dec": "December",
                                "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"]

    private enum Token { case word(String), number(String, currency: String?), punct(Character) }

    func phonemize(_ input: String) -> String {
        unknown = []
        let tokens = Self.tokenize(Self.normalize(input))
        // Right to left, as misaki: a word's sound depends on the next one's first sound (the → ði before a vowel).
        var futureVowel: Bool? = nil
        var out: [(String, Bool)] = []  // (phonemes, is punctuation)
        for token in tokens.reversed() {
            var ps: String
            switch token {
            case .punct(let c):
                out.append((String(c), true))
                if Self.nonQuotePuncts.contains(c) { futureVowel = nil }
                continue
            case .word(let w): ps = word(w, futureVowel: futureVowel)
            case .number(let n, let currency): ps = number(n, currency: currency)
            }
            guard !ps.isEmpty else { continue }
            out.append((ps, false))
            if let c = ps.first(where: { Self.vowels.contains($0) || Self.consonants.contains($0) || Self.nonQuotePuncts.contains($0) }) {
                futureVowel = Self.nonQuotePuncts.contains(c) ? nil : Self.vowels.contains(c)
            }
        }
        var s = ""
        for (ps, isPunct) in out.reversed() {
            if isPunct, "“(".contains(ps) { s += (s.isEmpty ? "" : " ") + ps; continue }
            if isPunct || s.isEmpty || s.hasSuffix("“") { s += ps } else { s += " " + ps }
        }
        return s.trimmingCharacters(in: .whitespaces)
    }

    // MARK: normalize + tokenize

    static func normalize(_ text: String) -> String {
        var t = text.precomposedStringWithCompatibilityMapping
        for (a, b) in [("\u{2018}", "'"), ("\u{2019}", "'"), ("´", "'"), ("`", "'"), ("...", "…"), (" - ", " — "), ("–", "—"), ("\u{201C}", "“"), ("\u{201D}", "”")] {
            t = t.replacingOccurrences(of: a, with: b)
        }
        // a.m. / p.m. → letters; "e.g." / "i.e." → words (before the tokenizer reads the dots as sentence ends).
        for (a, b) in [("a.m.", "A M"), ("p.m.", "P M"), ("A.M.", "A M"), ("P.M.", "P M"), ("e.g.", "for example"), ("i.e.", "that is")] {
            t = t.replacingOccurrences(of: a, with: b)
        }
        return t
    }

    private static func tokenize(_ text: String) -> [Token] {
        var tokens: [Token] = []
        let chars = Array(text)
        var i = 0
        var currency: String? = nil
        func isWordChar(_ c: Character) -> Bool { c.isLetter }
        while i < chars.count {
            let c = chars[i]
            if "$£€".contains(c), i + 1 < chars.count, chars[i + 1].isNumber { currency = String(c); i += 1; continue }
            if c.isNumber {
                var j = i
                while j < chars.count, chars[j].isNumber || (",.:".contains(chars[j]) && j + 1 < chars.count && chars[j + 1].isNumber) { j += 1 }
                // ordinals / plurals glued on: 8th, 1990s, 3rd
                var k = j
                while k < chars.count, chars[k].isLetter || chars[k] == "'" { k += 1 }
                let suffix = String(chars[j ..< k]).lowercased()
                if ["st", "nd", "rd", "th", "s", "'s"].contains(suffix) { j = k }
                tokens.append(.number(String(chars[i ..< j]), currency: currency))
                currency = nil
                i = j
                continue
            }
            if isWordChar(c) || (c == "'" && i + 1 < chars.count && chars[i + 1].isLetter && (i == 0 || !chars[i - 1].isLetter)) {
                var j = i + 1
                while j < chars.count, isWordChar(chars[j]) || ("'-".contains(chars[j]) && j + 1 < chars.count && chars[j + 1].isLetter) { j += 1 }
                var w = String(chars[i ..< j])
                // Abbreviations with their dot: "Dr." → doctor (the dot is not a sentence end then).
                if j < chars.count, chars[j] == ".", let full = abbreviations[w] { w = full; j += 1 }
                for part in w.split(separator: " ") { tokens.append(.word(String(part))) }
                i = j
                continue
            }
            if let s = symbols[String(c)] { tokens.append(.word(s)); i += 1; continue }
            if c == "\"" {
                // A straight quote opens after nothing or punctuation, else it closes.
                let opens: Bool = { guard let last = tokens.last else { return true }; if case .punct = last { return true }; return false }()
                tokens.append(.punct(opens ? "“" : "”"))
            } else if puncts.contains(c) { tokens.append(.punct(c)) }
            else if "()[]".contains(c) { tokens.append(.punct(",")) }
            i += 1
        }
        // A run of commas from brackets collapses to one.
        var out: [Token] = []
        for t in tokens {
            if case .punct(",") = t, let last = out.last, case .punct = last { continue }
            out.append(t)
        }
        return out
    }

    // MARK: words

    private func word(_ raw: String, futureVowel: Bool?) -> String {
        let w = raw
        let stress: Double? = w == w.lowercased() ? nil : (w == w.uppercased() && w.count > 1 ? 2 : 0.5)
        if let special = special(w, futureVowel: futureVowel) { return special }
        if let ps = lookupWord(w, stress: stress) { return ps }
        // hyphenated: each part
        if w.contains("-") {
            let parts = w.split(separator: "-").map { word(String($0), futureVowel: nil) }
            if !parts.contains(where: \.isEmpty) { return parts.joined(separator: " ") }
        }
        // compound of two known words ("voicepack" → voice + pack)
        let lower = w.lowercased()
        if lower.count >= 6 {
            for cut in stride(from: lower.count - 3, through: 3, by: -1) {
                let a = String(lower.prefix(cut)), b = String(lower.dropFirst(cut))
                if let pa = known(a), let pb = known(b) { return pa + Self.applyStress(pb, -0.5) }
            }
        }
        unknown.append(w)
        // A name or a new word: sounded out by letter rules (KokoroLTS), not spelled; acronyms were spelled above.
        if lower.hasSuffix("'s"), let ps = KokoroLTS.sound(String(lower.dropLast(2)), british: british) { return s(ps) }
        return KokoroLTS.sound(lower, british: british) ?? spell(w)
    }

    private func known(_ w: String) -> String? { gold[w] ?? silver[w] }

    private func special(_ w: String, futureVowel: Bool?) -> String? {
        switch w {
        case "a", "A": return futureVowel == nil && w == "A" ? "ˈA" : "ɐ"
        case "an", "An": return "ɐn"
        case "I": return "ˌI"
        case "the", "The": return futureVowel == true ? "ði" : "ðə"
        case "to", "To": return futureVowel == nil ? gold["to"] : (futureVowel! ? "tʊ" : "tə")
        case "in", "In": return (futureVowel == nil ? "ˈ" : "") + "ɪn"
        case "am", "Am": return futureVowel == nil ? gold["am"] : "ɐm"
        default: return nil
        }
    }

    /// misaki's get_word without tags: the word as written, its lower-case form for capitalised words the lexicon
    /// knows lower-case, possessives, then the -s / -ed / -ing stems.
    private func lookupWord(_ w0: String, stress: Double?) -> String? {
        var w = w0
        let wl = w.lowercased()
        if w.count > 1, w != wl, gold[w] == nil, silver[w] == nil, w == w.uppercased() || w.dropFirst() == w.dropFirst().lowercased(),
           known(wl) != nil || stemS(wl) != nil || stemEd(wl) != nil || stemIng(wl) != nil {
            w = wl
        }
        if let ps = lookup(w, stress: stress) { return ps }
        if w.hasSuffix("s'"), let ps = lookup(String(w.dropLast(2)) + "'s", stress: stress) { return ps }
        if w.hasSuffix("'"), let ps = lookup(String(w.dropLast()), stress: stress) { return ps }
        if let ps = stemS(w, stress: stress) { return ps }
        if let ps = stemEd(w, stress: stress) { return ps }
        if let ps = stemIng(w, stress: stress ?? 0.5) { return ps }
        return nil
    }

    private func lookup(_ w0: String, stress: Double?) -> String? {
        var w = w0
        if w == w.uppercased(), gold[w] == nil { w = w.lowercased() }
        let caps = w.prefix(1).uppercased() + w.dropFirst()
        guard let ps = gold[w] ?? silver[w] ?? (w == w.lowercased() ? (gold[caps] ?? silver[caps]) : (gold[w.lowercased()] ?? silver[w.lowercased()])) else {
            if w0.count > 1, w0 == w0.uppercased(), w0.allSatisfy(\.isLetter) { return spell(w0) }  // an acronym: its letters
            return nil
        }
        return Self.applyStress(ps, stress)
    }

    private func isKnown(_ w: String) -> Bool {
        if known(w) != nil || known(w.prefix(1).uppercased() + w.dropFirst()) != nil { return true }
        return w.count == 1 && w.first!.isLetter
    }

    private func s(_ stem: String) -> String {
        guard let last = stem.last else { return stem }
        if "ptkfθ".contains(last) { return stem + "s" }
        if "szʃʒʧʤ".contains(last) { return stem + (british ? "ɪ" : "ᵻ") + "z" }
        return stem + "z"
    }

    private func stemS(_ w: String, stress: Double? = nil) -> String? {
        guard w.count >= 3, w.hasSuffix("s") else { return nil }
        let stem: String
        if !w.hasSuffix("ss"), isKnown(String(w.dropLast())) { stem = String(w.dropLast()) }
        else if (w.hasSuffix("'s") || (w.count > 4 && w.hasSuffix("es") && !w.hasSuffix("ies"))), isKnown(String(w.dropLast(2))) { stem = String(w.dropLast(2)) }
        else if w.count > 4, w.hasSuffix("ies"), isKnown(String(w.dropLast(3)) + "y") { stem = String(w.dropLast(3)) + "y" }
        else { return nil }
        return lookup(stem, stress: stress).map(s)
    }

    private func ed(_ stem: String) -> String {
        guard let last = stem.last else { return stem }
        if "pkfθʃsʧ".contains(last) { return stem + "t" }
        if last == "d" { return stem + (british ? "ɪ" : "ᵻ") + "d" }
        if last != "t" { return stem + "d" }
        if british || stem.count < 2 { return stem + "ɪd" }
        if Self.usTaus.contains(stem[stem.index(stem.endIndex, offsetBy: -2)]) { return String(stem.dropLast()) + "ɾᵻd" }
        return stem + "ᵻd"
    }

    private func stemEd(_ w: String, stress: Double? = nil) -> String? {
        guard w.count >= 4, w.hasSuffix("d") else { return nil }
        let stem: String
        if !w.hasSuffix("dd"), isKnown(String(w.dropLast())) { stem = String(w.dropLast()) }
        else if w.count > 4, w.hasSuffix("ed"), !w.hasSuffix("eed"), isKnown(String(w.dropLast(2))) { stem = String(w.dropLast(2)) }
        else { return nil }
        return lookup(stem, stress: stress).map(ed)
    }

    private func ing(_ stem: String) -> String? {
        guard let last = stem.last else { return nil }
        if british { return "əː".contains(last) ? nil : stem + "ɪŋ" }
        if stem.count > 1, last == "t", Self.usTaus.contains(stem[stem.index(stem.endIndex, offsetBy: -2)]) { return String(stem.dropLast()) + "ɾɪŋ" }
        return stem + "ɪŋ"
    }

    private func stemIng(_ w: String, stress: Double? = nil) -> String? {
        guard w.count >= 5, w.hasSuffix("ing") else { return nil }
        let base = String(w.dropLast(3))
        let stem: String
        if w.count > 5, isKnown(base) { stem = base }
        else if isKnown(base + "e") { stem = base + "e" }
        else if w.count > 5, let l = base.last, base.dropLast().last == l, "bcdgklmnprstvxz".contains(l), isKnown(String(base.dropLast())) { stem = String(base.dropLast()) }
        else if w.hasSuffix("cking"), isKnown(String(base.dropLast())) { stem = String(base.dropLast()) }
        else { return nil }
        return lookup(stem, stress: stress).flatMap(ing)
    }

    /// misaki's get_NNP: each letter's name from the gold lexicon, one primary stress on the last.
    private func spell(_ w: String) -> String {
        let ps = w.filter(\.isLetter).compactMap { gold[String($0).uppercased()] }
        guard !ps.isEmpty else { return "" }
        var joined = Self.applyStress(ps.joined(), 0)
        if let r = joined.lastIndex(of: Self.secondary) { joined.replaceSubrange(r ... r, with: String(Self.primary)) }
        return joined
    }

    // MARK: numbers

    private func number(_ raw: String, currency: String?) -> String {
        var n = raw.replacingOccurrences(of: ",", with: "")
        var suffix = ""
        for s in ["'s", "st", "nd", "rd", "th", "s"] where n.lowercased().hasSuffix(s) && n.count > s.count { suffix = s; n = String(n.dropLast(s.count)); break }
        var words: [String]
        if n.contains(":"), case let p = n.split(separator: ":"), p.count == 2, let h = Int(p[0]), let m = Int(p[1]) {
            words = Self.cardinal(h) + (m == 0 ? ["o'clock"] : m < 10 ? ["oh"] + Self.cardinal(m) : Self.cardinal(m))
        } else if ["st", "nd", "rd", "th"].contains(suffix), let v = Int(n) {
            words = Self.ordinal(v); suffix = ""
        } else if let cur = currency {
            let units = ["$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")][cur]!
            let p = n.split(separator: ".", omittingEmptySubsequences: false)
            let whole = Int(p[0]) ?? 0, cents = p.count > 1 ? Int(p[1].prefix(2).padding(toLength: 2, withPad: "0", startingAt: 0)) ?? 0 : 0
            words = []
            if whole > 0 || cents == 0 { words += Self.cardinal(whole) + [whole == 1 ? units.0 : units.0 + "s"] }
            if cents > 0 { if !words.isEmpty { words.append("and") }; words += Self.cardinal(cents) + [cents == 1 || units.1 == "pence" ? units.1 : units.1 + "s"] }
        } else if n.contains(".") {
            let p = n.split(separator: ".", omittingEmptySubsequences: false)
            words = (p[0].isEmpty ? ["zero"] : Self.cardinal(Int(p[0]) ?? 0)) + ["point"] + p[1].compactMap { Int(String($0)) }.flatMap { Self.cardinal($0) }
        } else if n.count == 4, let v = Int(n), v >= 1000, v < 3000 {
            words = Self.year(v)
        } else if n.count > 1, n.hasPrefix("0") || n.count > 12 {
            words = n.compactMap { Int(String($0)) }.flatMap { Self.cardinal($0) }
        } else {
            words = Self.cardinal(Int(n) ?? 0)
        }
        var ps = words.compactMap { w -> String? in w == "point" ? gold["point"].map { Self.applyStress($0, -2) } : lookup(w, stress: nil) }.joined(separator: " ")
        if suffix == "s" || suffix == "'s" { ps = s(ps) }
        return ps
    }

    static let ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
                       "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    static let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    static func cardinal(_ v: Int) -> [String] {
        if v < 0 { return ["minus"] + cardinal(-v) }
        if v < 20 { return [ones[v]] }
        if v < 100 { return [tens[v / 10]] + (v % 10 > 0 ? [ones[v % 10]] : []) }
        if v < 1000 { return [ones[v / 100], "hundred"] + (v % 100 > 0 ? cardinal(v % 100) : []) }
        for (scale, name) in [(1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")] where v >= scale {
            return cardinal(v / scale) + [name] + (v % scale > 0 ? cardinal(v % scale) : [])
        }
        return []
    }

    static func ordinal(_ v: Int) -> [String] {
        var w = cardinal(v)
        let irregular = ["one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"]
        let last = w.removeLast()
        if let o = irregular[last] { w.append(o) }
        else if last.hasSuffix("y") { w.append(String(last.dropLast()) + "ieth") }
        else { w.append(last + "th") }
        return w
    }

    static func year(_ v: Int) -> [String] {
        let hi = v / 100, lo = v % 100
        if v % 1000 < 10 && hi % 10 == 0 { return cardinal(v) }  // 2000, 2005: two thousand (five)
        if lo == 0 { return cardinal(hi) + ["hundred"] }
        return cardinal(hi) + (lo < 10 ? ["oh"] + cardinal(lo) : cardinal(lo))
    }

    // MARK: stress (misaki's apply_stress)

    static func applyStress(_ ps: String, _ stress: Double?) -> String {
        guard let stress else { return ps }
        let hasPrimary = ps.contains(primary), hasAny = ps.contains(where: { stresses.contains($0) })
        let hasVowel = ps.contains(where: { vowels.contains($0) })
        if stress < -1 { return ps.filter { !stresses.contains($0) } }
        if stress == -1 || ((stress == 0 || stress == -0.5) && hasPrimary) {
            return String(ps.filter { $0 != secondary }.map { $0 == primary ? secondary : $0 })
        }
        if [0, 0.5, 1].contains(stress), !hasAny { return hasVowel ? restress(String(secondary) + ps) : ps }
        if stress >= 1, !hasPrimary, ps.contains(secondary) { return String(ps.map { $0 == secondary ? primary : $0 }) }
        if stress > 1, !hasAny { return hasVowel ? restress(String(primary) + ps) : ps }
        return ps
    }

    /// Moves each stress mark to just before the vowel it governs.
    private static func restress(_ ps: String) -> String {
        let chars = Array(ps)
        var keyed: [(Double, Character)] = chars.enumerated().map { (Double($0.offset), $0.element) }
        for (i, c) in chars.enumerated() where stresses.contains(c) {
            if let j = chars[i...].firstIndex(where: { vowels.contains($0) }) { keyed[i] = (Double(j) - 0.5, c) }
        }
        return String(keyed.sorted { $0.0 < $1.0 }.map(\.1))
    }
}

// MARK: - out-of-lexicon words

/// VOICESHOP: a small letter-to-sound fallback for words misaki's lexicons don't have (mostly names: "Sinek", "Okafor").
/// Ordered grapheme rules (longest match first, a little context for silent e, soft c/g and open vowels) → misaki's
/// phoneme symbols, primary stress on the first vowel. A best guess by design: it beats spelling a name letter by letter;
/// a real G2P model (kokorowire's phone kit) replaces it behind `KokoroRuntime.phonemizer`.
enum KokoroLTS {
    private static let vowelLetters = Set("aeiouy")

    static func sound(_ word: String, british: Bool) -> String? {
        // café → cafe, with the accent remembered: a final é is sounded (A), not silent
        let accentedEnd = word.lowercased().hasSuffix("é")
        let folded = word.lowercased().folding(options: .diacriticInsensitive, locale: nil)
        let w = Array(folded.filter { $0.isLetter && $0.isASCII })
        guard w.count >= 2 else { return nil }
        var out = "", i = 0
        func at(_ k: Int) -> Character? { k >= 0 && k < w.count ? w[k] : nil }
        func isV(_ c: Character?) -> Bool { c.map { vowelLetters.contains($0) } ?? false }
        /// A vowel is long ("open") before one consonant + vowel, or before consonant + final silent e.
        func open(_ k: Int) -> Bool {
            guard let c1 = at(k + 1), !isV(c1) else { return false }
            if k + 2 == w.count - 1, at(k + 2) == "e" { return true }
            return isV(at(k + 2)) && at(k + 2) != "e" && k == 0
        }
        let r: Character = british ? "ɹ" : "ɹ"
        while i < w.count {
            let rest = String(w[i...])
            func take(_ g: String, _ ps: String) -> Bool { if rest.hasPrefix(g) { out += ps; i += g.count; return true }; return false }
            // silent final e (after a consonant), and the -le / -es endings
            if w[i] == "e", i == w.count - 1, i > 1 {
                // silent only in the CVCe shape (Jane, Kolawole); after a cluster or accented it is sounded (Tunde, café)
                if !accentedEnd, isV(at(i - 2)), !isV(at(i - 1)) { i += 1; continue }
                out += "A"; i += 1; continue
            }
            if i == w.count - 2, rest == "le", i > 0, !isV(at(i - 1)) { out += "əl"; i += 2; continue }
            if take("tion", "ʃən") || take("sion", "ʒən") || take("ture", "ʧəɹ") { continue }
            if take("igh", "I") || take("tch", "ʧ") || take("dge", "ʤ") || take("sch", "sk") || take("ough", "O") { continue }
            if take("ch", "ʧ") || take("sh", "ʃ") || take("th", "θ") || take("ph", "f") || take("ck", "k") || take("wh", "w")
                || take("qu", "kw") || take("nk", "ŋk") || take("ng", "ŋ") || take("gh", "ɡ") || take("kn", "n") || take("wr", "ɹ") || take("ps", i == 0 ? "s" : "ps") { continue }
            if take("ee", "i") || take("ea", "i") || take("oo", "u") || take("ou", "W") || take("ow", "O") || take("ai", "A") || take("ay", "A")
                || take("oa", "O") || take("oi", "Y") || take("oy", "Y") || take("au", "ɔ") || take("aw", "ɔ") || take("ie", "i") || take("ei", "A")
                || take("ey", "A") || take("ue", "u") || take("ew", "ju") { continue }
            if take("ar", "ɑɹ") || take("or", "ɔɹ") || take("er", "əɹ") || take("ir", "ɜɹ") || take("ur", "ɜɹ") { continue }
            let c = w[i], next = at(i + 1)
            switch c {
            case "a": out += open(i) ? "A" : (i == w.count - 1 ? "ə" : "æ")
            case "e": out += open(i) ? "i" : (i == w.count - 1 ? "i" : "ɛ")
            case "i": out += open(i) ? "I" : (i == w.count - 1 || next == "y" || isV(next) ? "i" : "ɪ")
            case "o": out += open(i) || i == w.count - 1 ? "O" : "ɑ"
            case "u": out += open(i) ? "ju" : "ʌ"
            case "y": out += i == 0 || isV(next) ? "j" : (i == w.count - 1 ? "i" : "ɪ")
            case "c": out += "eiy".contains(next ?? " ") ? "s" : "k"
            case "g": out += ("eiy".contains(next ?? " ") && i > 0) ? "ʤ" : "ɡ"
            case "j": out += "ʤ"
            case "q": out += "k"
            case "x": out += i == 0 ? "z" : "ks"
            case "r": out.append(r)
            case "s": out += (i > 0 && isV(at(i - 1)) && isV(next)) ? "z" : "s"
            default: out.append(c)  // b d f h k l m n p t v w z: the same symbol
            }
            // a doubled consonant sounds once
            if let n = next, n == c, !isV(c) { i += 1 }
            i += 1
        }
        guard out.contains(where: { KokoroG2P.vowels.contains($0) }) else { return nil }
        // primary stress before the first vowel; unstressed later "æ/ɑ/ʌ" relax a little toward ə
        guard let v = out.firstIndex(where: { KokoroG2P.vowels.contains($0) }) else { return out }
        var ps = out
        ps.insert(KokoroG2P.primary, at: v)
        return ps
    }
}
