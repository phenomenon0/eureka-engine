#!/usr/bin/env python3
"""tools/tts/kokoro_tts.py: G41. Kokoro-82M v1.0 speech on the Linux engine path (docs: tools/tts/README.md).

The phone's voice layer (bend app, ios/Bend/Voice/KokoroVoice.swift + docs/VOICES.md, VOICESHOP) on Linux, the same
choices except the model file: kokoro-onnx's fp32 export (325 MB; the phone's int8 export is ~9x slower on x86
onnxruntime CPU, G41 bench; BEND_KOKORO_MODEL=int8 / fetch --model int8 for byte-parity with the phone), a voice = its 510 x 256 float32 style table (522,240 B),
misaki 0.7.4's gold + silver lexicons as the G2P (no espeak-ng, no spaCy), the app's letter rules for words the
lexicons lack, house voice af_bella, the app's bundled set (hexgrad/Kokoro-82M @ f3ff357) + the voice shop's catalog
(HF jethrovic/bend-models voices/catalog.json, pinned revision). KokoroG2P / KokoroLTS below are line-for-line ports of
the Swift, so a Linux client hears what the phone says.

  kokoro_tts.py fetch [DIR] [--model fp32|int8|both]   pinned + sha256-checked assets into DIR (default $BEND_KOKORO_DIR or
                                             ~/.local/share/models/voice/kokoro); idempotent
  kokoro_tts.py say --text T [--voice V] [--speed S] [--out F.wav] [--dir DIR]
                                             one render; prints a receipt line (bytes, sha256, latency) as JSON
  kokoro_tts.py phonemes --text T [--voice V]  the G2P only
  kokoro_tts.py worker [--dir DIR]           llb-serve's resident TTS child (serve.c --tts): JSON lines on stdin,
                                             a JSON header line on stdout (+ "bytes" raw bytes of audio for speak)
Worker ops: {"op":"voices"} | {"op":"select","id":V} | {"op":"speak","text":T,"voice":V?,"speed":S?,"format":"wav"|"pcm"}
Env: BEND_KOKORO_DIR, BEND_KOKORO_MODEL (fp32|int8|file; default kokoro.json's), BEND_KOKORO_THREADS (default 4), BEND_VOICE (the starting voice; default the house voice).
Needs: python3, numpy, onnxruntime (CPU). The fetch needs curl-free urllib only.
"""
import hashlib, io, json, os, struct, sys, time, unicodedata, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.environ.get("BEND_KOKORO_DIR") or os.path.expanduser("~/.local/share/models/voice/kokoro")
SAMPLE_RATE = 24000
STYLE_ROWS, STYLE_DIM = 510, 256
STYLE_BYTES = STYLE_ROWS * STYLE_DIM * 4
MODELS = {"fp32": "kokoro-v1.0.onnx", "int8": "kokoro-v1.0.int8.onnx"}


def log(msg):
    sys.stderr.write(f"kokoro: {msg}\n")
    sys.stderr.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---- assets (fetch) -------------------------------------------------------------------------------------------------

def _download(url, sha, dest):
    if os.path.exists(dest) and sha256_file(dest) == sha:
        return False
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "bend-kokoro-fetch/1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            break
        except OSError as e:
            if attempt == 2:
                raise SystemExit(f"fetch: {url}: {e}")
            time.sleep(1 + 2 * attempt)
    got = sha256_file(part)
    if got != sha:
        os.unlink(part)
        raise SystemExit(f"fetch: {url}: sha256 {got}, expected {sha}")
    os.replace(part, dest)
    return True


def pt_tensor(data):
    """The style bytes inside a hexgrad .pt (torch zip, stored): the `<id>/data/0` entry (KokoroVoice.ptTensor)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.filename.endswith("/data/0"):
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError(f"compressed .pt (method {info.compress_type}): save it with torch.save")
                return z.read(info)
    raise ValueError("not a torch voice file (.pt zip)")


def style_bytes(path):
    data = open(path, "rb").read()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        data = pt_tensor(data)
    elif ext != ".bin":
        raise ValueError(f"{os.path.basename(path)}: not a Kokoro voice (.bin or .pt)")
    if len(data) != STYLE_BYTES:
        raise ValueError(f"{os.path.basename(path)}: {len(data)} bytes of style; a Kokoro voice is {STYLE_BYTES} (510 x 256 float32)")
    return data


def fetch(dest, model="fp32"):
    pins = json.load(open(os.path.join(HERE, "pins.json")))
    os.makedirs(os.path.join(dest, "packs"), exist_ok=True)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "bend-kokoro-fetch")
    os.makedirs(tmp, exist_ok=True)
    want = ["fp32", "int8"] if model == "both" else [model]
    for f in [pins["models"][m] for m in want] + pins["files"]:
        if _download(f["url"], f["sha256"], os.path.join(dest, f["file"])):
            log(f"fetched {f['file']}")
    # the model's phoneme -> id table, from hexgrad's config.json (as fetch-voice.sh)
    cfg = os.path.join(tmp, "kokoro-config.json")
    _download(pins["config"]["url"], pins["config"]["sha256"], cfg)
    vocab = json.load(open(cfg, encoding="utf-8"))["vocab"]
    with open(os.path.join(dest, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, sort_keys=True)
    order = []

    def install(v, url, sha, file, origin, extra):
        src = os.path.join(tmp, file)
        _download(url, sha, src)
        raw = style_bytes(src)
        with open(os.path.join(dest, "packs", v["id"] + ".bin"), "wb") as f:
            f.write(raw)
        meta = {k: v[k] for k in ("id", "name", "lang", "gender", "grade", "detail", "license", "credit") if k in v}
        meta.update({"engine": "kokoro", "origin": origin, "source_sha256": sha, "source_url": url}, **extra)
        meta.setdefault("license", "Apache-2.0")
        meta.setdefault("credit", "hexgrad/Kokoro-82M")
        with open(os.path.join(dest, "packs", v["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        order.append(v["id"])

    b = pins["bundled"]
    for v in b["voices"]:  # the app bundle's set, in the picker's order (house voice first)
        install(v, f"{b['base']}/voices/{v['id']}.pt", v["pt_sha256"], v["id"] + ".pt", "bundled", {})
    c = pins["catalog"]  # the voice shop's catalog, HF jethrovic/bend-models voices/, pinned revision
    cat_path = os.path.join(tmp, "catalog.json")
    _download(c["url"], c["sha256"], cat_path)
    cat = json.load(open(cat_path, encoding="utf-8"))
    base = c["url"].rsplit("/", 1)[0]
    for v in cat["voices"]:
        if v.get("engine", "kokoro") != "kokoro" or v["id"] in order:
            continue
        url = v["pack_url"] if "://" in v["pack_url"] else f"{base}/{v['pack_url']}"
        install(v, url, v["sha256"], v["file"], "shop", {"catalog": c["url"]})
    with open(os.path.join(dest, "packs", "index.json"), "w") as f:
        json.dump(order, f)
    with open(os.path.join(dest, "kokoro.json"), "w") as f:
        json.dump({"model": MODELS[want[0]], "models": [MODELS[m] for m in want], "sampleRate": SAMPLE_RATE, "house_voice": cat.get("house_voice", "kokoro:af_bella"),
                   "demo_line": cat.get("demo_line", ""), "catalog": c["url"], "bundled_revision": b["revision"],
                   "licence": "Kokoro-82M v1.0 Apache-2.0 (hexgrad); ONNX export thewh1teagle/kokoro-onnx (MIT tooling); "
                              "lexicons misaki Apache-2.0; voices Apache-2.0; no espeak-ng"}, f, indent=1)
    log(f"Kokoro-82M v1.0 {'+'.join(want)} + {len(order)} voices in {dest}")
    return order


# ---- text -> phonemes: KokoroG2P (misaki en.py without spaCy / espeak), ported from KokoroVoice.swift --------------

PRIMARY, SECONDARY = "ˈ", "ˌ"
STRESSES = {PRIMARY, SECONDARY}
VOWELS = set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
CONSONANTS = set("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
US_TAUS = set("AIOWYiuæɑəɛɪɹʊʌ")
PUNCTS = set(";:,.!?—…\"“”")
NON_QUOTE_PUNCTS = set(";:,.!?—…")
SYMBOLS = {"%": "percent", "&": "and", "+": "plus", "@": "at"}
ABBREVIATIONS = {"Dr": "doctor", "Mr": "mister", "Mrs": "missus", "Ms": "miz", "St": "saint", "vs": "versus", "etc": "et cetera",
                 "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "Jun": "June", "Jul": "July", "Aug": "August",
                 "Sep": "September", "Sept": "September", "Oct": "October", "Nov": "November", "Dec": "December",
                 "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int(s):
    try:
        return int(s)
    except ValueError:
        return None


def apply_stress(ps, stress):
    if stress is None:
        return ps
    has_primary = PRIMARY in ps
    has_any = any(c in STRESSES for c in ps)
    has_vowel = any(c in VOWELS for c in ps)
    if stress < -1:
        return "".join(c for c in ps if c not in STRESSES)
    if stress == -1 or (stress in (0, -0.5) and has_primary):
        return "".join(SECONDARY if c == PRIMARY else c for c in ps if c != SECONDARY)
    if stress in (0, 0.5, 1) and not has_any:
        return _restress(SECONDARY + ps) if has_vowel else ps
    if stress >= 1 and not has_primary and SECONDARY in ps:
        return ps.replace(SECONDARY, PRIMARY)
    if stress > 1 and not has_any:
        return _restress(PRIMARY + ps) if has_vowel else ps
    return ps


def _restress(ps):
    keyed = [(float(i), c) for i, c in enumerate(ps)]
    for i, c in enumerate(ps):
        if c in STRESSES:
            j = next((k for k in range(i, len(ps)) if ps[k] in VOWELS), None)
            if j is not None:
                keyed[i] = (j - 0.5, c)
    return "".join(c for _, c in sorted(keyed, key=lambda t: t[0]))


def cardinal(v):
    if v < 0:
        return ["minus"] + cardinal(-v)
    if v < 20:
        return [ONES[v]]
    if v < 100:
        return [TENS[v // 10]] + ([ONES[v % 10]] if v % 10 else [])
    if v < 1000:
        return [ONES[v // 100], "hundred"] + (cardinal(v % 100) if v % 100 else [])
    for scale, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if v >= scale:
            return cardinal(v // scale) + [name] + (cardinal(v % scale) if v % scale else [])
    return []


def ordinal(v):
    w = cardinal(v)
    irregular = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
    last = w.pop()
    if last in irregular:
        w.append(irregular[last])
    elif last.endswith("y"):
        w.append(last[:-1] + "ieth")
    else:
        w.append(last + "th")
    return w


def year(v):
    hi, lo = v // 100, v % 100
    if v % 1000 < 10 and hi % 10 == 0:
        return cardinal(v)
    if lo == 0:
        return cardinal(hi) + ["hundred"]
    return cardinal(hi) + (["oh"] + cardinal(lo) if lo < 10 else cardinal(lo))


def normalize(text):
    t = unicodedata.normalize("NFKC", text)
    for a, b in (("‘", "'"), ("’", "'"), ("´", "'"), ("`", "'"), ("...", "…"), (" - ", " — "), ("–", "—"), ("“", "“"), ("”", "”")):
        t = t.replace(a, b)
    for a, b in (("a.m.", "A M"), ("p.m.", "P M"), ("A.M.", "A M"), ("P.M.", "P M"), ("e.g.", "for example"), ("i.e.", "that is")):
        t = t.replace(a, b)
    return t


def tokenize(text):
    """Tokens: ("w", word) | ("n", number, currency) | ("p", char)."""
    tokens = []
    chars = text
    n = len(chars)
    i = 0
    currency = None
    while i < n:
        c = chars[i]
        if c in "$£€" and i + 1 < n and chars[i + 1].isdigit():
            currency = c
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < n and (chars[j].isdigit() or (chars[j] in ",.:" and j + 1 < n and chars[j + 1].isdigit())):
                j += 1
            k = j
            while k < n and (chars[k].isalpha() or chars[k] == "'"):
                k += 1
            if chars[j:k].lower() in ("st", "nd", "rd", "th", "s", "'s"):
                j = k
            tokens.append(("n", chars[i:j], currency))
            currency = None
            i = j
            continue
        if c.isalpha() or (c == "'" and i + 1 < n and chars[i + 1].isalpha() and (i == 0 or not chars[i - 1].isalpha())):
            j = i + 1
            while j < n and (chars[j].isalpha() or (chars[j] in "'-" and j + 1 < n and chars[j + 1].isalpha())):
                j += 1
            w = chars[i:j]
            if j < n and chars[j] == "." and w in ABBREVIATIONS:
                w = ABBREVIATIONS[w]
                j += 1
            for part in w.split(" "):
                if part:
                    tokens.append(("w", part))
            i = j
            continue
        if c in SYMBOLS:
            tokens.append(("w", SYMBOLS[c]))
            i += 1
            continue
        if c == '"':
            opens = not tokens or tokens[-1][0] == "p"
            tokens.append(("p", "“" if opens else "”"))
        elif c in PUNCTS:
            tokens.append(("p", c))
        elif c in "()[]":
            tokens.append(("p", ","))
        i += 1
    out = []
    for t in tokens:
        if t == ("p", ",") and out and out[-1][0] == "p":
            continue
        out.append(t)
    return out


def lts_sound(word, british):
    """KokoroLTS: ordered letter rules for out-of-lexicon words (mostly names). A best guess by design."""
    accented_end = word.lower().endswith("é")
    folded = "".join(ch for ch in unicodedata.normalize("NFD", word.lower()) if not unicodedata.combining(ch))
    w = [ch for ch in folded if ch.isalpha() and ch.isascii()]
    if len(w) < 2:
        return None
    n = len(w)
    vl = set("aeiouy")
    at = lambda k: w[k] if 0 <= k < n else None
    is_v = lambda ch: ch is not None and ch in vl

    def open_(k):
        c1 = at(k + 1)
        if c1 is None or is_v(c1):
            return False
        if k + 2 == n - 1 and at(k + 2) == "e":
            return True
        return is_v(at(k + 2)) and at(k + 2) != "e" and k == 0

    out = []
    i = 0
    while i < n:
        rest = "".join(w[i:])

        def take(g, ps):
            nonlocal i
            if rest.startswith(g):
                out.append(ps)
                i += len(g)
                return True
            return False

        if w[i] == "e" and i == n - 1 and i > 1:
            if not accented_end and is_v(at(i - 2)) and not is_v(at(i - 1)):
                i += 1
                continue
            out.append("A")
            i += 1
            continue
        if i == n - 2 and rest == "le" and i > 0 and not is_v(at(i - 1)):
            out.append("əl")
            i += 2
            continue
        if take("tion", "ʃən") or take("sion", "ʒən") or take("ture", "ʧəɹ"):
            continue
        if take("igh", "I") or take("tch", "ʧ") or take("dge", "ʤ") or take("sch", "sk") or take("ough", "O"):
            continue
        if (take("ch", "ʧ") or take("sh", "ʃ") or take("th", "θ") or take("ph", "f") or take("ck", "k") or take("wh", "w")
                or take("qu", "kw") or take("nk", "ŋk") or take("ng", "ŋ") or take("gh", "ɡ") or take("kn", "n") or take("wr", "ɹ")
                or take("ps", "s" if i == 0 else "ps")):
            continue
        if (take("ee", "i") or take("ea", "i") or take("oo", "u") or take("ou", "W") or take("ow", "O") or take("ai", "A") or take("ay", "A")
                or take("oa", "O") or take("oi", "Y") or take("oy", "Y") or take("au", "ɔ") or take("aw", "ɔ") or take("ie", "i") or take("ei", "A")
                or take("ey", "A") or take("ue", "u") or take("ew", "ju")):
            continue
        if take("ar", "ɑɹ") or take("or", "ɔɹ") or take("er", "əɹ") or take("ir", "ɜɹ") or take("ur", "ɜɹ"):
            continue
        c, nxt = w[i], at(i + 1)
        if c == "a":
            out.append("A" if open_(i) else ("ə" if i == n - 1 else "æ"))
        elif c == "e":
            out.append("i" if open_(i) else ("i" if i == n - 1 else "ɛ"))
        elif c == "i":
            out.append("I" if open_(i) else ("i" if i == n - 1 or nxt == "y" or is_v(nxt) else "ɪ"))
        elif c == "o":
            out.append("O" if open_(i) or i == n - 1 else "ɑ")
        elif c == "u":
            out.append("ju" if open_(i) else "ʌ")
        elif c == "y":
            out.append("j" if i == 0 or is_v(nxt) else ("i" if i == n - 1 else "ɪ"))
        elif c == "c":
            out.append("s" if (nxt or " ") in "eiy" else "k")
        elif c == "g":
            out.append("ʤ" if (nxt or " ") in "eiy" and i > 0 else "ɡ")
        elif c == "j":
            out.append("ʤ")
        elif c == "q":
            out.append("k")
        elif c == "x":
            out.append("z" if i == 0 else "ks")
        elif c == "r":
            out.append("ɹ")
        elif c == "s":
            out.append("z" if i > 0 and is_v(at(i - 1)) and is_v(nxt) else "s")
        else:
            out.append(c)
        if nxt is not None and nxt == c and not is_v(c):
            i += 1
        i += 1
    s = "".join(out)
    v = next((k for k, ch in enumerate(s) if ch in VOWELS), None)
    if v is None:
        return None
    return s[:v] + PRIMARY + s[v:]


def _load_lexicon(path):
    out = {}
    for k, v in json.load(open(path, encoding="utf-8")).items():
        if isinstance(v, str):
            out[k] = v
        elif isinstance(v, dict) and isinstance(v.get("DEFAULT"), str):
            out[k] = v["DEFAULT"]
    return out


class G2P:
    def __init__(self, gold, silver, british):
        self.british = british
        self.gold = _load_lexicon(gold)
        self.silver = _load_lexicon(silver)
        self.unknown = []

    def phonemize(self, text):
        self.unknown = []
        future_vowel = None
        out = []
        for tok in reversed(tokenize(normalize(text))):
            if tok[0] == "p":
                out.append((tok[1], True))
                if tok[1] in NON_QUOTE_PUNCTS:
                    future_vowel = None
                continue
            ps = self.word(tok[1], future_vowel) if tok[0] == "w" else self.number(tok[1], tok[2])
            if not ps:
                continue
            out.append((ps, False))
            c = next((ch for ch in ps if ch in VOWELS or ch in CONSONANTS or ch in NON_QUOTE_PUNCTS), None)
            if c is not None:
                future_vowel = None if c in NON_QUOTE_PUNCTS else (c in VOWELS)
        s = ""
        for ps, is_punct in reversed(out):
            if is_punct and ps in "“(":
                s += ("" if not s else " ") + ps
                continue
            if is_punct or not s or s.endswith("“"):
                s += ps
            else:
                s += " " + ps
        return s.strip(" ")

    def known(self, w):  # gold[w] ?? silver[w]
        return self.gold[w] if w in self.gold else self.silver.get(w)

    def word(self, w, future_vowel):
        stress = None if w == w.lower() else (2 if w == w.upper() and len(w) > 1 else 0.5)
        sp = self.special(w, future_vowel)
        if sp is not None:
            return sp
        ps = self.lookup_word(w, stress)
        if ps is not None:
            return ps
        if "-" in w:
            parts = [self.word(p, None) for p in w.split("-") if p]
            if all(parts):
                return " ".join(parts)
        lower = w.lower()
        if len(lower) >= 6:
            for cut in range(len(lower) - 3, 2, -1):
                pa, pb = self.known(lower[:cut]), self.known(lower[cut:])
                if pa is not None and pb is not None:
                    return pa + apply_stress(pb, -0.5)
        self.unknown.append(w)
        if lower.endswith("'s"):
            ps = lts_sound(lower[:-2], self.british)
            if ps:
                return self.s(ps)
        return lts_sound(lower, self.british) or self.spell(w)

    def special(self, w, fv):
        if w in ("a", "A"):
            return "ˈA" if fv is None and w == "A" else "ɐ"
        if w in ("an", "An"):
            return "ɐn"
        if w == "I":
            return "ˌI"
        if w in ("the", "The"):
            return "ði" if fv is True else "ðə"
        if w in ("to", "To"):
            return self.gold.get("to") if fv is None else ("tʊ" if fv else "tə")
        if w in ("in", "In"):
            return ("ˈ" if fv is None else "") + "ɪn"
        if w in ("am", "Am"):
            return self.gold.get("am") if fv is None else "ɐm"
        return None

    def lookup_word(self, w0, stress):
        w = w0
        wl = w.lower()
        if (len(w) > 1 and w != wl and w not in self.gold and w not in self.silver
                and (w == w.upper() or w[1:] == w[1:].lower())
                and (self.known(wl) is not None or self.stem_s(wl) is not None or self.stem_ed(wl) is not None
                     or self.stem_ing(wl) is not None)):
            w = wl
        ps = self.lookup(w, stress)
        if ps is not None:
            return ps
        if w.endswith("s'"):
            ps = self.lookup(w[:-2] + "'s", stress)
            if ps is not None:
                return ps
        if w.endswith("'"):
            ps = self.lookup(w[:-1], stress)
            if ps is not None:
                return ps
        for f, st in ((self.stem_s, stress), (self.stem_ed, stress), (self.stem_ing, 0.5 if stress is None else stress)):
            ps = f(w, st)
            if ps is not None:
                return ps
        return None

    def lookup(self, w0, stress):
        w = w0
        if w == w.upper() and w not in self.gold:
            w = w.lower()
        caps = w[:1].upper() + w[1:]
        ps = self.known(w)
        if ps is None:
            ps = self.known(caps) if w == w.lower() else self.known(w.lower())
        if ps is None:
            if len(w0) > 1 and w0 == w0.upper() and w0.isalpha():
                return self.spell(w0)
            return None
        return apply_stress(ps, stress)

    def is_known(self, w):
        if self.known(w) is not None or self.known(w[:1].upper() + w[1:]) is not None:
            return True
        return len(w) == 1 and w.isalpha()

    def s(self, stem):
        if not stem:
            return stem
        last = stem[-1]
        if last in "ptkfθ":
            return stem + "s"
        if last in "szʃʒʧʤ":
            return stem + ("ɪ" if self.british else "ᵻ") + "z"
        return stem + "z"

    def stem_s(self, w, stress=None):
        if len(w) < 3 or not w.endswith("s"):
            return None
        if not w.endswith("ss") and self.is_known(w[:-1]):
            stem = w[:-1]
        elif (w.endswith("'s") or (len(w) > 4 and w.endswith("es") and not w.endswith("ies"))) and self.is_known(w[:-2]):
            stem = w[:-2]
        elif len(w) > 4 and w.endswith("ies") and self.is_known(w[:-3] + "y"):
            stem = w[:-3] + "y"
        else:
            return None
        ps = self.lookup(stem, stress)
        return None if ps is None else self.s(ps)

    def ed(self, stem):
        if not stem:
            return stem
        last = stem[-1]
        if last in "pkfθʃsʧ":
            return stem + "t"
        if last == "d":
            return stem + ("ɪ" if self.british else "ᵻ") + "d"
        if last != "t":
            return stem + "d"
        if self.british or len(stem) < 2:
            return stem + "ɪd"
        if stem[-2] in US_TAUS:
            return stem[:-1] + "ɾᵻd"
        return stem + "ᵻd"

    def stem_ed(self, w, stress=None):
        if len(w) < 4 or not w.endswith("d"):
            return None
        if not w.endswith("dd") and self.is_known(w[:-1]):
            stem = w[:-1]
        elif len(w) > 4 and w.endswith("ed") and not w.endswith("eed") and self.is_known(w[:-2]):
            stem = w[:-2]
        else:
            return None
        ps = self.lookup(stem, stress)
        return None if ps is None else self.ed(ps)

    def ing(self, stem):
        if not stem:
            return None
        last = stem[-1]
        if self.british:
            return None if last in "əː" else stem + "ɪŋ"
        if len(stem) > 1 and last == "t" and stem[-2] in US_TAUS:
            return stem[:-1] + "ɾɪŋ"
        return stem + "ɪŋ"

    def stem_ing(self, w, stress=None):
        if len(w) < 5 or not w.endswith("ing"):
            return None
        base = w[:-3]
        if len(w) > 5 and self.is_known(base):
            stem = base
        elif self.is_known(base + "e"):
            stem = base + "e"
        elif len(w) > 5 and base and len(base) > 1 and base[-2] == base[-1] and base[-1] in "bcdgklmnprstvxz" and self.is_known(base[:-1]):
            stem = base[:-1]
        elif w.endswith("cking") and self.is_known(base[:-1]):
            stem = base[:-1]
        else:
            return None
        ps = self.lookup(stem, stress)
        return None if ps is None else self.ing(ps)

    def spell(self, w):
        ps = [self.gold[ch.upper()] for ch in w if ch.isalpha() and ch.upper() in self.gold]
        if not ps:
            return ""
        joined = apply_stress("".join(ps), 0)
        r = joined.rfind(SECONDARY)
        if r >= 0:
            joined = joined[:r] + PRIMARY + joined[r + 1:]
        return joined

    def number(self, raw, currency):
        n = raw.replace(",", "")
        suffix = ""
        for sx in ("'s", "st", "nd", "rd", "th", "s"):
            if n.lower().endswith(sx) and len(n) > len(sx):
                suffix = sx
                n = n[:-len(sx)]
                break
        p = n.split(":")
        if ":" in n and len(p) == 2 and _int(p[0]) is not None and _int(p[1]) is not None:
            h, m = int(p[0]), int(p[1])
            words = cardinal(h) + (["o'clock"] if m == 0 else (["oh"] + cardinal(m) if m < 10 else cardinal(m)))
        elif suffix in ("st", "nd", "rd", "th") and _int(n) is not None:
            words = ordinal(int(n))
            suffix = ""
        elif currency:
            unit, sub = {"$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")}[currency]
            p = n.split(".")
            whole = _int(p[0]) or 0
            cents = (_int(p[1][:2].ljust(2, "0")) or 0) if len(p) > 1 else 0
            words = []
            if whole > 0 or cents == 0:
                words += cardinal(whole) + [unit if whole == 1 else unit + "s"]
            if cents > 0:
                if words:
                    words.append("and")
                words += cardinal(cents) + [sub if cents == 1 or sub == "pence" else sub + "s"]
        elif "." in n:
            p = n.split(".")
            words = (["zero"] if not p[0] else cardinal(_int(p[0]) or 0)) + ["point"] + [x for d in p[1] if d.isdigit() for x in cardinal(int(d))]
        elif len(n) == 4 and _int(n) is not None and 1000 <= int(n) < 3000:
            words = year(int(n))
        elif len(n) > 1 and (n.startswith("0") or len(n) > 12):
            words = [x for d in n if d.isdigit() for x in cardinal(int(d))]
        else:
            words = cardinal(_int(n) or 0)
        parts = []
        for w in words:
            if w == "point":
                if "point" in self.gold:
                    parts.append(apply_stress(self.gold["point"], -2))
            else:
                ps = self.lookup(w, None)
                if ps is not None:
                    parts.append(ps)
        ps = " ".join(parts)
        if suffix in ("s", "'s"):
            ps = self.s(ps)
        return ps


def split_phonemes(ps, mx=400):
    """KokoroVoice.split: cut at the last punctuation (else space) before `mx` symbols."""
    parts = []
    rest = ps
    while len(rest) > mx:
        window = rest[:mx]
        cut = max((window.rfind(ch) for ch in ".!?;:,—…"), default=-1)
        if cut >= 0:
            cut += 1
        else:
            cut = window.rfind(" ")
            if cut < 0:
                cut = mx
        parts.append(rest[:cut].strip(" "))
        rest = rest[cut:].lstrip(" ")
    if rest:
        parts.append(rest)
    return parts


# ---- runtime ----------------------------------------------------------------------------------------------------------

def trim(np, x, rate=SAMPLE_RATE, threshold=0.01):
    """VOICE-LATENCY (KokoroVoice.trim, release-b5): Kokoro pads each render with ~0.42 s of silence before the voice
    and ~0.5 s after; keep 50 ms / 200 ms. Samples under 0.01 (-40 dBFS, before the lift) count as silence."""
    loud = np.flatnonzero(np.abs(x) > threshold)
    if loud.size == 0:
        return x
    return x[max(0, loud[0] - rate // 20):min(x.size, loud[-1] + rate // 5)]


GAIN = 1.58  # VOICESHOP2: Kokoro speaks at ~-24 dBFS; a fixed +4 dB, held under 0.98 at the peak (KokoroVoice.lift)


class Kokoro:
    def __init__(self, d=DEFAULT_DIR, threads=None):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        self.dir = d
        meta = json.load(open(os.path.join(d, "kokoro.json"))) if os.path.exists(os.path.join(d, "kokoro.json")) else {}
        m = os.environ.get("BEND_KOKORO_MODEL") or meta.get("model") or MODELS["fp32"]
        self.model = MODELS.get(m, m)
        missing = [f for f in (self.model, "vocab.json", "us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json", "packs/index.json")
                   if not os.path.exists(os.path.join(d, f))]
        if missing:
            raise SystemExit(f"kokoro: {d} lacks {', '.join(missing)}: run `kokoro_tts.py fetch {d}`")
        t0 = time.monotonic()
        self.threads = int(threads or os.environ.get("BEND_KOKORO_THREADS", "4"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(os.path.join(d, self.model), so, providers=["CPUExecutionProvider"])
        vocab = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8"))
        self.vocab = {k: v for k, v in vocab.items() if len(k) == 1}
        self.g2p = {}
        self.house = meta.get("house_voice", "kokoro:af_bella").split(":", 1)[-1]
        self.demo_line = meta.get("demo_line", "")
        self.packs = {}
        self.order = json.load(open(os.path.join(d, "packs", "index.json")))
        for vid in self.order:
            mp = os.path.join(d, "packs", vid + ".json")
            self.packs[vid] = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {"id": vid}
        self.styles = {}
        self.selected = None
        self.select(os.environ.get("BEND_VOICE") or self.house)
        self.load_ms = (time.monotonic() - t0) * 1000
        self.label = f"onnx-cpu-{self.threads}t-{'int8' if 'int8' in self.model else 'fp32'}"

    @staticmethod
    def voice_id(v):
        return (v or "").split(":", 1)[-1]

    def style(self, vid):
        vid = self.voice_id(vid)
        if vid not in self.packs:
            raise KeyError(vid)
        if vid not in self.styles:
            raw = style_bytes(os.path.join(self.dir, "packs", vid + ".bin"))
            self.styles[vid] = self.np.frombuffer(raw, dtype="<f4").reshape(STYLE_ROWS, STYLE_DIM)
        return self.styles[vid]

    def select(self, vid):
        vid = self.voice_id(vid)
        self.style(vid)
        self.selected = vid
        return vid

    def british(self, vid):
        p = self.packs.get(vid, {})
        return p.get("lang", "").lower() == "en-gb" or vid[:1] == "b"

    def phonemizer(self, british):
        if british not in self.g2p:
            p = "gb" if british else "us"
            self.g2p[british] = G2P(os.path.join(self.dir, f"{p}_gold.json"), os.path.join(self.dir, f"{p}_silver.json"), british)
        return self.g2p[british]

    def ids(self, ps):
        return [self.vocab[c] for c in ps if c in self.vocab]

    def run(self, ids, style, speed):
        np = self.np
        if not ids:
            return np.zeros(0, dtype=np.float32)
        row = min(len(ids), STYLE_ROWS - 1)
        out = self.session.run(["audio"], {"tokens": np.array([[0] + ids + [0]], dtype=np.int64),
                                           "style": style[row:row + 1].astype(np.float32),
                                           "speed": np.array([speed], dtype=np.float32)})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def synthesize(self, text, voice=None, speed=1.0):
        np = self.np
        vid = self.voice_id(voice) if voice else self.selected
        style = self.style(vid)
        t0 = time.monotonic()
        g = self.phonemizer(self.british(vid))
        ps = g.phonemize(text)
        t1 = time.monotonic()
        pieces = [trim(np, self.run(self.ids(p), style, float(speed))) for p in split_phonemes(ps, 400)]
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        gain = 0.98 / max(peak, 1e-6) if peak * GAIN > 0.98 else GAIN
        audio = audio * gain
        t2 = time.monotonic()
        return audio, {"voice": vid, "phonemes": ps, "unknown": list(g.unknown), "pieces": len(pieces),
                       "g2p_ms": round((t1 - t0) * 1000, 3), "synth_ms": round((t2 - t1) * 1000, 3),
                       "samples": int(audio.size), "audio_s": round(audio.size / SAMPLE_RATE, 4), "runtime": self.label}

    def voices(self):
        return [{"id": "kokoro:" + v, "engine": "kokoro", "name": self.packs[v].get("name", v), "lang": self.packs[v].get("lang"),
                 "gender": self.packs[v].get("gender"), "grade": self.packs[v].get("grade"), "origin": self.packs[v].get("origin"),
                 "house": v == self.house} for v in self.order]


def pcm16(np, audio):
    return (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def wav_bytes(np, audio):
    data = pcm16(np, audio)
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


# ---- worker (llb-serve --tts) ----------------------------------------------------------------------------------------

def worker(d):
    k = Kokoro(d)
    out = sys.stdout.buffer
    # warm the graph once so the first served request doesn't pay onnxruntime's first-run allocations
    tw = time.monotonic()
    k.synthesize("Ready.")
    log(f"ready dir={d} model={k.model} runtime={k.label} voices={len(k.order)} voice={k.selected} "
        f"load_ms={k.load_ms:.0f} warm_ms={(time.monotonic() - tw) * 1000:.0f}")

    def reply(obj, payload=b""):
        obj["bytes"] = len(payload)
        out.write(json.dumps(obj, ensure_ascii=False).encode() + b"\n")
        if payload:
            out.write(payload)
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "voices":
                reply({"ok": True, "voices": k.voices(), "selected": "kokoro:" + k.selected, "house_voice": "kokoro:" + k.house})
            elif op == "select":
                vid = k.select(req.get("id", ""))
                reply({"ok": True, "selected": "kokoro:" + vid})
            elif op == "speak":
                text = req.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    reply({"ok": False, "status": 400, "error": "input must be a non-empty string"})
                    continue
                if len(text) > 4096:
                    reply({"ok": False, "status": 400, "error": "input is longer than 4096 characters"})
                    continue
                speed = float(req.get("speed", 1.0) or 1.0)
                if not 0.25 <= speed <= 4.0:
                    reply({"ok": False, "status": 400, "error": "speed must be between 0.25 and 4.0"})
                    continue
                fmt = req.get("format") or "wav"
                if fmt not in ("wav", "pcm"):
                    reply({"ok": False, "status": 400, "error": f"response_format {fmt!r}: this server speaks wav or pcm (s16le 24 kHz mono)"})
                    continue
                audio, info = k.synthesize(text, req.get("voice") or None, speed)
                payload = wav_bytes(k.np, audio) if fmt == "wav" else pcm16(k.np, audio)
                info.update({"ok": True, "format": fmt, "sample_rate": SAMPLE_RATE, "voice": "kokoro:" + info["voice"],
                             "sha256": hashlib.sha256(payload).hexdigest()})
                reply(info, payload)
            else:
                reply({"ok": False, "status": 400, "error": f"unknown op {op!r}"})
        except KeyError as e:
            reply({"ok": False, "status": 400, "error": f"unknown voice {e.args[0]!r}; GET /v1/audio/voices lists them"})
        except Exception as e:  # one bad request must not take the worker down
            reply({"ok": False, "status": 500, "error": f"{type(e).__name__}: {e}"})


# ---- CLI --------------------------------------------------------------------------------------------------------------

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("dir", nargs="?", default=DEFAULT_DIR)
    f.add_argument("--model", choices=["fp32", "int8", "both"], default="fp32")
    for name in ("say", "phonemes"):
        s = sub.add_parser(name)
        s.add_argument("--text", required=True)
        s.add_argument("--voice")
        s.add_argument("--speed", type=float, default=1.0)
        s.add_argument("--dir", default=DEFAULT_DIR)
        s.add_argument("--threads", type=int)
        if name == "say":
            s.add_argument("--out")
    w = sub.add_parser("worker")
    w.add_argument("--dir", default=DEFAULT_DIR)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.dir, a.model)
    elif a.cmd == "worker":
        worker(a.dir)
    elif a.cmd == "phonemes":
        k = Kokoro(a.dir, a.threads)
        g = k.phonemizer(k.british(k.voice_id(a.voice) if a.voice else k.selected))
        print(json.dumps({"phonemes": g.phonemize(a.text), "unknown": g.unknown}, ensure_ascii=False))
    else:
        t0 = time.monotonic()
        k = Kokoro(a.dir, a.threads)
        t1 = time.monotonic()
        audio, info = k.synthesize(a.text, a.voice, a.speed)
        wav = wav_bytes(k.np, audio)
        if a.out:
            with open(a.out, "wb") as fh:
                fh.write(wav)
        info.update({"text": a.text, "bytes": len(wav), "sha256": hashlib.sha256(wav).hexdigest(), "out": a.out,
                     "load_ms": round((t1 - t0) * 1000, 1), "total_ms": round((time.monotonic() - t0) * 1000, 1)})
        print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
