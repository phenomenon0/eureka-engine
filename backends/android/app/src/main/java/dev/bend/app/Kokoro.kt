package dev.bend.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.channels.FileChannel
import java.text.Normalizer

/**
 * KOKORODROID: Kokoro-82M v1.0 (hexgrad, Apache-2.0), the Android twin of iOS voiceshop's KokoroVoice.swift.
 *
 * One runtime, many voices: the model (assets/voices/kokoro/kokoro-v1.0.int8.onnx, 92 MB, kokoro-onnx's int8 export, stored
 * uncompressed and mmapped) loads once; a voice is only its style table (510 × 256 float32 = 522,240 bytes), so changing
 * voice is swapping that table. Text → phonemes is [KokoroG2P] (misaki's lexicons + rules, line for line the Swift port;
 * no espeak-ng). Not thread-safe: Voice calls it from one executor.
 */
class KokoroRuntime(private val assets: AssetManager, val threads: Int) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val vocab: Map<Char, Long>
    private val g2p = HashMap<Boolean, KokoroG2P>()

    init {
        val v = JSONObject(text("vocab.json"))
        vocab = v.keys().asSequence().filter { it.length == 1 }.associate { it[0] to v.getLong(it) }
        val fd = assets.openFd("$DIR/$MODEL") // noCompress "onnx" (app/build.gradle.kts): mappable in place
        val model = FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.length)
        // ORT's default CPU kernels: the XNNPACK provider measured slower on the Pixel 6a (receipts/bench-xnnpack.json).
        session = env.createSession(model, OrtSession.SessionOptions().apply { setIntraOpNumThreads(threads) })
    }

    private fun text(name: String) = assets.open("$DIR/$name").reader().use { it.readText() }

    fun phonemizer(british: Boolean): KokoroG2P = g2p.getOrPut(british) {
        val p = if (british) "gb" else "us"
        KokoroG2P(KokoroG2P.lexicon(text("${p}_gold.json")), KokoroG2P.lexicon(text("${p}_silver.json")), british)
    }

    /** Phonemes → ids (unknown symbols dropped). */
    fun ids(phonemes: String): LongArray = phonemes.mapNotNull { vocab[it] }.toLongArray()

    /** One run: [0, ids…, 0] with the style row for ids.size → 24 kHz mono. */
    fun run(ids: LongArray, style: FloatArray, speed: Float): FloatArray {
        if (ids.isEmpty()) return FloatArray(0)
        val row = minOf(ids.size, STYLE_ROWS - 1) * STYLE_DIM
        val tokens = LongArray(ids.size + 2).also { ids.copyInto(it, 1) }
        OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong())).use { t ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(style, row, STYLE_DIM), longArrayOf(1, STYLE_DIM.toLong())).use { s ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(speed)), longArrayOf(1)).use { sp ->
                    session.run(mapOf("tokens" to t, "style" to s, "speed" to sp)).use { out ->
                        val a = (out[0] as OnnxTensor).floatBuffer
                        return FloatArray(a.remaining()).also { a.get(it) }
                    }
                }
            }
        }
    }

    fun close() = session.close()

    companion object {
        const val DIR = "voices/kokoro"
        const val MODEL = "kokoro-v1.0.int8.onnx"
        const val SAMPLE_RATE = 24_000
        const val STYLE_ROWS = 510
        const val STYLE_DIM = 256
        const val STYLE_BYTES = STYLE_ROWS * STYLE_DIM * 4

        /** A style table from pack bytes: `bin` (raw little-endian float32, 510 × 256) or `pt` (hexgrad's torch zip, stored). */
        fun style(bytes: ByteArray, ext: String, name: String): FloatArray {
            val raw = when (ext.lowercase()) {
                "bin" -> ByteBuffer.wrap(bytes)
                "pt" -> ptTensor(bytes)
                else -> throw IllegalArgumentException("$name: not a Kokoro voice (.bin or .pt)")
            }
            require(raw.remaining() == STYLE_BYTES) { "$name: ${raw.remaining()} bytes of style; a Kokoro voice is $STYLE_BYTES (510 × 256 float32)" }
            return FloatArray(STYLE_ROWS * STYLE_DIM).also { raw.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
        }

        fun style(f: File) = style(f.readBytes(), f.extension, f.name)

        /** The tensor bytes inside a torch .pt (zip, stored): the central directory's `…/data/0` entry. */
        fun ptTensor(zip: ByteArray): ByteBuffer {
            fun u16(o: Int) = (zip[o].toInt() and 0xff) or ((zip[o + 1].toInt() and 0xff) shl 8)
            fun u32(o: Int) = u16(o) or (u16(o + 2) shl 16)
            val bad = IllegalArgumentException("not a torch voice file (.pt zip)")
            if (zip.size <= 22) throw bad
            var eocd = zip.size - 22
            while (eocd >= 0 && u32(eocd) != 0x06054b50) eocd--
            if (eocd < 0) throw bad
            var at = u32(eocd + 16)
            repeat(u16(eocd + 10)) {
                if (at + 46 > zip.size || u32(at) != 0x02014b50) throw bad
                val method = u16(at + 10); val size = u32(at + 20); val nameLen = u16(at + 28)
                val extraLen = u16(at + 30); val commentLen = u16(at + 32); val local = u32(at + 42)
                if (String(zip, at + 46, nameLen).endsWith("/data/0")) {
                    if (method != 0) throw IllegalArgumentException("compressed .pt (method $method): save it with torch.save")
                    if (local + 30 > zip.size || u32(local) != 0x04034b50) throw bad
                    val start = local + 30 + u16(local + 26) + u16(local + 28)
                    if (start + size > zip.size) throw bad
                    return ByteBuffer.wrap(zip, start, size).slice()
                }
                at += 46 + nameLen + extraLen + commentLen
            }
            throw bad
        }
    }
}

/** One Kokoro voice: the shared runtime + this pack's style table. */
class KokoroVoice(val runtime: KokoroRuntime, val name: String, val style: FloatArray, val british: Boolean, val speed: Float = 1f) {
    fun synthesize(text: String): FloatArray {
        val phonemes = runtime.phonemizer(british).phonemize(text)
        // Over ~500 ids the model rushes (hexgrad's VOICES.md); the chunker's clauses never get near, a pasted page might.
        val parts = split(phonemes, 400).map { runtime.run(runtime.ids(it), style, speed) }
        return FloatArray(parts.sumOf { it.size }).also { var o = 0; for (p in parts) { p.copyInto(it, o); o += p.size } }
    }

    companion object {
        /** Cuts at the last punctuation (else space) before `max` symbols, so no run is over the model's window. */
        fun split(ps: String, max: Int): List<String> {
            var rest = ps; val parts = ArrayList<String>()
            while (rest.length > max) {
                val window = rest.take(max)
                val p = window.indexOfLast { it in ".!?;:,—…" }
                val cut = if (p >= 0) p + 1 else window.lastIndexOf(' ').let { if (it >= 0) it else window.length }
                parts.add(rest.substring(0, cut).trim(' '))
                rest = rest.substring(cut).trimStart(' ')
            }
            if (rest.isNotEmpty()) parts.add(rest)
            return parts
        }
    }
}

/**
 * misaki's English G2P (hexgrad/misaki `en.py`, Apache-2.0) without spaCy and without its neural/espeak fallback, ported
 * line for line from iOS voiceshop's KokoroG2P (Swift) so both phones say the same phonemes (receipts: g2p parity).
 * Out-of-lexicon words: split into known parts, else spelled.
 */
class KokoroG2P(private val gold: Map<String, String>, private val silver: Map<String, String>, val british: Boolean) {
    /** Words the last `phonemize` could not find (for receipts). */
    var unknown = mutableListOf<String>(); private set

    private sealed interface Token
    private data class Word(val w: String) : Token
    private data class Num(val n: String, val currency: String?) : Token
    private data class Punct(val c: Char) : Token

    fun phonemize(input: String): String {
        unknown = mutableListOf()
        val tokens = tokenize(normalize(input))
        // Right to left, as misaki: a word's sound depends on the next one's first sound (the → ði before a vowel).
        var futureVowel: Boolean? = null
        val out = ArrayList<Pair<String, Boolean>>()
        for (token in tokens.asReversed()) {
            val ps = when (token) {
                is Punct -> { out.add(token.c.toString() to true); if (token.c in NON_QUOTE_PUNCTS) futureVowel = null; continue }
                is Word -> word(token.w, futureVowel)
                is Num -> number(token.n, token.currency)
            }
            if (ps.isEmpty()) continue
            out.add(ps to false)
            ps.firstOrNull { it in VOWELS || it in CONSONANTS || it in NON_QUOTE_PUNCTS }?.let { c ->
                futureVowel = if (c in NON_QUOTE_PUNCTS) null else c in VOWELS
            }
        }
        val s = StringBuilder()
        for ((ps, isPunct) in out.asReversed()) {
            if (isPunct && ps in listOf("“", "(")) { s.append(if (s.isEmpty()) "" else " ").append(ps); continue }
            if (isPunct || s.isEmpty() || s.endsWith("“")) s.append(ps) else s.append(' ').append(ps)
        }
        return s.toString().trim(' ')
    }

    // MARK: words

    private fun word(w: String, futureVowel: Boolean?): String {
        val stress: Double? = if (w == w.lowercase()) null else if (w == w.uppercase() && w.length > 1) 2.0 else 0.5
        special(w, futureVowel)?.let { return it }
        lookupWord(w, stress)?.let { return it }
        if ('-' in w) {
            val parts = w.split('-').filter { it.isNotEmpty() }.map { word(it, null) }
            if (parts.none { it.isEmpty() }) return parts.joinToString(" ")
        }
        // compound of two known words ("voicepack" → voice + pack)
        val lower = w.lowercase()
        if (lower.length >= 6) {
            for (cut in lower.length - 3 downTo 3) {
                val pa = known(lower.substring(0, cut)); val pb = known(lower.substring(cut))
                if (pa != null && pb != null) return pa + applyStress(pb, -0.5)
            }
        }
        unknown.add(w)
        return spell(w)
    }

    private fun known(w: String) = gold[w] ?: silver[w]

    private fun special(w: String, futureVowel: Boolean?): String? = when (w) {
        "a", "A" -> if (futureVowel == null && w == "A") "ˈA" else "ɐ"
        "an", "An" -> "ɐn"
        "I" -> "ˌI"
        "the", "The" -> if (futureVowel == true) "ði" else "ðə"
        "to", "To" -> if (futureVowel == null) gold["to"] else if (futureVowel) "tʊ" else "tə"
        "in", "In" -> (if (futureVowel == null) "ˈ" else "") + "ɪn"
        "am", "Am" -> if (futureVowel == null) gold["am"] else "ɐm"
        else -> null
    }

    private fun caps(w: String) = w.take(1).uppercase() + w.drop(1)

    /** misaki's get_word without tags: as written, lower-cased if the lexicon knows that, possessives, then -s/-ed/-ing stems. */
    private fun lookupWord(w0: String, stress: Double?): String? {
        var w = w0
        val wl = w.lowercase()
        if (w.length > 1 && w != wl && gold[w] == null && silver[w] == null && (w == w.uppercase() || w.drop(1) == w.drop(1).lowercase()) &&
            (known(wl) != null || stemS(wl) != null || stemEd(wl) != null || stemIng(wl) != null)) w = wl
        lookup(w, stress)?.let { return it }
        if (w.endsWith("s'")) lookup(w.dropLast(2) + "'s", stress)?.let { return it }
        if (w.endsWith("'")) lookup(w.dropLast(1), stress)?.let { return it }
        stemS(w, stress)?.let { return it }
        stemEd(w, stress)?.let { return it }
        stemIng(w, stress ?: 0.5)?.let { return it }
        return null
    }

    private fun lookup(w0: String, stress: Double?): String? {
        var w = w0
        if (w == w.uppercase() && gold[w] == null) w = w.lowercase()
        val ps = gold[w] ?: silver[w] ?: (if (w == w.lowercase()) gold[caps(w)] ?: silver[caps(w)] else gold[w.lowercase()] ?: silver[w.lowercase()])
        if (ps == null) {
            if (w0.length > 1 && w0 == w0.uppercase() && w0.all { it.isLetter() }) return spell(w0) // an acronym: its letters
            return null
        }
        return applyStress(ps, stress)
    }

    private fun isKnown(w: String) = known(w) != null || known(caps(w)) != null || (w.length == 1 && w[0].isLetter())

    private fun s(stem: String): String {
        val last = stem.lastOrNull() ?: return stem
        if (last in "ptkfθ") return stem + "s"
        if (last in "szʃʒʧʤ") return stem + (if (british) "ɪ" else "ᵻ") + "z"
        return stem + "z"
    }

    private fun stemS(w: String, stress: Double? = null): String? {
        if (w.length < 3 || !w.endsWith("s")) return null
        val stem = when {
            !w.endsWith("ss") && isKnown(w.dropLast(1)) -> w.dropLast(1)
            (w.endsWith("'s") || (w.length > 4 && w.endsWith("es") && !w.endsWith("ies"))) && isKnown(w.dropLast(2)) -> w.dropLast(2)
            w.length > 4 && w.endsWith("ies") && isKnown(w.dropLast(3) + "y") -> w.dropLast(3) + "y"
            else -> return null
        }
        return lookup(stem, stress)?.let(::s)
    }

    private fun ed(stem: String): String {
        val last = stem.lastOrNull() ?: return stem
        if (last in "pkfθʃsʧ") return stem + "t"
        if (last == 'd') return stem + (if (british) "ɪ" else "ᵻ") + "d"
        if (last != 't') return stem + "d"
        if (british || stem.length < 2) return stem + "ɪd"
        if (stem[stem.length - 2] in US_TAUS) return stem.dropLast(1) + "ɾᵻd"
        return stem + "ᵻd"
    }

    private fun stemEd(w: String, stress: Double? = null): String? {
        if (w.length < 4 || !w.endsWith("d")) return null
        val stem = when {
            !w.endsWith("dd") && isKnown(w.dropLast(1)) -> w.dropLast(1)
            w.length > 4 && w.endsWith("ed") && !w.endsWith("eed") && isKnown(w.dropLast(2)) -> w.dropLast(2)
            else -> return null
        }
        return lookup(stem, stress)?.let(::ed)
    }

    private fun ing(stem: String): String? {
        val last = stem.lastOrNull() ?: return null
        if (british) return if (last in "əː") null else stem + "ɪŋ"
        if (stem.length > 1 && last == 't' && stem[stem.length - 2] in US_TAUS) return stem.dropLast(1) + "ɾɪŋ"
        return stem + "ɪŋ"
    }

    private fun stemIng(w: String, stress: Double? = null): String? {
        if (w.length < 5 || !w.endsWith("ing")) return null
        val base = w.dropLast(3)
        val stem = when {
            w.length > 5 && isKnown(base) -> base
            isKnown(base + "e") -> base + "e"
            w.length > 5 && base.length >= 2 && base.last() == base[base.length - 2] && base.last() in "bcdgklmnprstvxz" && isKnown(base.dropLast(1)) -> base.dropLast(1)
            w.endsWith("cking") && isKnown(base.dropLast(1)) -> base.dropLast(1)
            else -> return null
        }
        return lookup(stem, stress)?.let(::ing)
    }

    /** misaki's get_NNP: each letter's name from the gold lexicon, one primary stress on the last. */
    private fun spell(w: String): String {
        val ps = w.filter { it.isLetter() }.mapNotNull { gold[it.toString().uppercase()] }
        if (ps.isEmpty()) return ""
        val joined = applyStress(ps.joinToString(""), 0.0)
        val r = joined.lastIndexOf(SECONDARY)
        return if (r >= 0) joined.substring(0, r) + PRIMARY + joined.substring(r + 1) else joined
    }

    // MARK: numbers

    private fun number(raw: String, currency: String?): String {
        var n = raw.replace(",", "")
        var suffix = ""
        for (s in listOf("'s", "st", "nd", "rd", "th", "s")) if (n.lowercase().endsWith(s) && n.length > s.length) { suffix = s; n = n.dropLast(s.length); break }
        val colon = n.split(':').filter { it.isNotEmpty() }
        val words: List<String> = when {
            ':' in n && colon.size == 2 && colon[0].toLongOrNull() != null && colon[1].toLongOrNull() != null -> {
                val h = colon[0].toLong(); val m = colon[1].toLong()
                cardinal(h) + (if (m == 0L) listOf("o'clock") else if (m < 10L) listOf("oh") + cardinal(m) else cardinal(m))
            }
            suffix in listOf("st", "nd", "rd", "th") && n.toLongOrNull() != null -> ordinal(n.toLong()).also { suffix = "" }
            currency != null -> {
                val (unit, sub) = mapOf("$" to ("dollar" to "cent"), "£" to ("pound" to "pence"), "€" to ("euro" to "cent"))[currency]!!
                val p = n.split('.')
                val whole = p[0].toLongOrNull() ?: 0L
                val cents = if (p.size > 1) p[1].take(2).padEnd(2, '0').toLongOrNull() ?: 0L else 0L
                val w = ArrayList<String>()
                if (whole > 0 || cents == 0L) { w += cardinal(whole); w += if (whole == 1L) unit else unit + "s" }
                if (cents > 0) { if (w.isNotEmpty()) w += "and"; w += cardinal(cents); w += if (cents == 1L || sub == "pence") sub else sub + "s" }
                w
            }
            '.' in n -> {
                val p = n.split('.')
                (if (p[0].isEmpty()) listOf("zero") else cardinal(p[0].toLongOrNull() ?: 0L)) + "point" + p[1].mapNotNull { it.digitToIntOrNull()?.toLong() }.flatMap { cardinal(it) }
            }
            n.length == 4 && n.toLongOrNull()?.let { it in 1000..2999 } == true -> year(n.toLong())
            n.length > 1 && (n.startsWith("0") || n.length > 12) -> n.mapNotNull { it.digitToIntOrNull()?.toLong() }.flatMap { cardinal(it) }
            else -> cardinal(n.toLongOrNull() ?: 0L)
        }
        var ps = words.mapNotNull { w -> if (w == "point") gold["point"]?.let { applyStress(it, -2.0) } else lookup(w, null) }.joinToString(" ")
        if (suffix == "s" || suffix == "'s") ps = s(ps)
        return ps
    }

    companion object {
        const val PRIMARY = 'ˈ'
        const val SECONDARY = 'ˌ'
        private const val STRESSES = "ˌˈ"
        private const val VOWELS = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ"
        private const val CONSONANTS = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ"
        private const val US_TAUS = "AIOWYiuæɑəɛɪɹʊʌ"
        private const val PUNCTS = ";:,.!?—…\"“”"
        private const val NON_QUOTE_PUNCTS = ";:,.!?—…"
        private val SYMBOLS = mapOf('%' to "percent", '&' to "and", '+' to "plus", '@' to "at")
        private val ABBREVIATIONS = mapOf("Dr" to "doctor", "Mr" to "mister", "Mrs" to "missus", "Ms" to "miz", "St" to "saint", "vs" to "versus", "etc" to "et cetera",
            "Jan" to "January", "Feb" to "February", "Mar" to "March", "Apr" to "April", "Jun" to "June", "Jul" to "July", "Aug" to "August",
            "Sep" to "September", "Sept" to "September", "Oct" to "October", "Nov" to "November", "Dec" to "December",
            "Mon" to "Monday", "Tue" to "Tuesday", "Wed" to "Wednesday", "Thu" to "Thursday", "Fri" to "Friday", "Sat" to "Saturday", "Sun" to "Sunday")

        /** The lexicon's value is a string, or {POS: string|null} with a DEFAULT: without a tagger, DEFAULT. */
        fun lexicon(json: String): Map<String, String> {
            val raw = JSONObject(json)
            val out = HashMap<String, String>(raw.length() * 2)
            for (k in raw.keys()) when (val v = raw.get(k)) {
                is String -> out[k] = v
                is JSONObject -> v.optString("DEFAULT", "").takeIf { v.has("DEFAULT") && !v.isNull("DEFAULT") }?.let { out[k] = it }
            }
            return out
        }

        fun normalize(text: String): String {
            var t = Normalizer.normalize(text, Normalizer.Form.NFKC)
            for ((a, b) in listOf("‘" to "'", "’" to "'", "´" to "'", "`" to "'", "..." to "…", " - " to " — ", "–" to "—", "“" to "“", "”" to "”")) t = t.replace(a, b)
            // a.m. / p.m. → letters; "e.g." / "i.e." → words (before the tokenizer reads the dots as sentence ends).
            for ((a, b) in listOf("a.m." to "A M", "p.m." to "P M", "A.M." to "A M", "P.M." to "P M", "e.g." to "for example", "i.e." to "that is")) t = t.replace(a, b)
            return t
        }

        private fun isNumber(c: Char) = c.category == CharCategory.DECIMAL_DIGIT_NUMBER || c.category == CharCategory.LETTER_NUMBER || c.category == CharCategory.OTHER_NUMBER

        private fun tokenize(text: String): List<Token> {
            val tokens = ArrayList<Token>()
            val chars = text
            var i = 0
            var currency: String? = null
            while (i < chars.length) {
                val c = chars[i]
                if (c in "$£€" && i + 1 < chars.length && isNumber(chars[i + 1])) { currency = c.toString(); i++; continue }
                if (isNumber(c)) {
                    var j = i
                    while (j < chars.length && (isNumber(chars[j]) || (chars[j] in ",.:" && j + 1 < chars.length && isNumber(chars[j + 1])))) j++
                    // ordinals / plurals glued on: 8th, 1990s, 3rd
                    var k = j
                    while (k < chars.length && (chars[k].isLetter() || chars[k] == '\'')) k++
                    if (chars.substring(j, k).lowercase() in listOf("st", "nd", "rd", "th", "s", "'s")) j = k
                    tokens.add(Num(chars.substring(i, j), currency))
                    currency = null; i = j; continue
                }
                if (c.isLetter() || (c == '\'' && i + 1 < chars.length && chars[i + 1].isLetter() && (i == 0 || !chars[i - 1].isLetter()))) {
                    var j = i + 1
                    while (j < chars.length && (chars[j].isLetter() || (chars[j] in "'-" && j + 1 < chars.length && chars[j + 1].isLetter()))) j++
                    var w = chars.substring(i, j)
                    // Abbreviations with their dot: "Dr." → doctor (the dot is not a sentence end then).
                    if (j < chars.length && chars[j] == '.') ABBREVIATIONS[w]?.let { w = it; j++ }
                    for (part in w.split(' ').filter { it.isNotEmpty() }) tokens.add(Word(part))
                    i = j; continue
                }
                val sym = SYMBOLS[c]
                if (sym != null) { tokens.add(Word(sym)); i++; continue }
                if (c == '"') {
                    // A straight quote opens after nothing or punctuation, else it closes.
                    tokens.add(Punct(if (tokens.lastOrNull().let { it == null || it is Punct }) '“' else '”'))
                } else if (c in PUNCTS) tokens.add(Punct(c))
                else if (c in "()[]") tokens.add(Punct(','))
                i++
            }
            // A run of commas from brackets collapses to one.
            val out = ArrayList<Token>()
            for (t in tokens) { if (t == Punct(',') && out.lastOrNull() is Punct) continue; out.add(t) }
            return out
        }

        private val ONES = listOf("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
        private val TENS = listOf("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

        fun cardinal(v: Long): List<String> {
            if (v < 0) return listOf("minus") + cardinal(-v)
            if (v < 20) return listOf(ONES[v.toInt()])
            if (v < 100) return listOf(TENS[(v / 10).toInt()]) + (if (v % 10 > 0) listOf(ONES[(v % 10).toInt()]) else emptyList())
            if (v < 1000) return listOf(ONES[(v / 100).toInt()], "hundred") + (if (v % 100 > 0) cardinal(v % 100) else emptyList())
            for ((scale, name) in listOf(1_000_000_000L to "billion", 1_000_000L to "million", 1000L to "thousand"))
                if (v >= scale) return cardinal(v / scale) + name + (if (v % scale > 0) cardinal(v % scale) else emptyList())
            return emptyList()
        }

        fun ordinal(v: Long): List<String> {
            val w = cardinal(v).toMutableList()
            val irregular = mapOf("one" to "first", "two" to "second", "three" to "third", "five" to "fifth", "eight" to "eighth", "nine" to "ninth", "twelve" to "twelfth")
            val last = w.removeAt(w.size - 1)
            w.add(irregular[last] ?: if (last.endsWith("y")) last.dropLast(1) + "ieth" else last + "th")
            return w
        }

        fun year(v: Long): List<String> {
            val hi = v / 100; val lo = v % 100
            if (v % 1000 < 10 && hi % 10 == 0L) return cardinal(v) // 2000, 2005: two thousand (five)
            if (lo == 0L) return cardinal(hi) + "hundred"
            return cardinal(hi) + (if (lo < 10) listOf("oh") + cardinal(lo) else cardinal(lo))
        }

        /** misaki's apply_stress. */
        fun applyStress(ps: String, stress: Double?): String {
            if (stress == null) return ps
            val hasPrimary = PRIMARY in ps; val hasAny = ps.any { it in STRESSES }; val hasVowel = ps.any { it in VOWELS }
            if (stress < -1) return ps.filter { it !in STRESSES }
            if (stress == -1.0 || ((stress == 0.0 || stress == -0.5) && hasPrimary)) return ps.filter { it != SECONDARY }.map { if (it == PRIMARY) SECONDARY else it }.joinToString("")
            if (stress in listOf(0.0, 0.5, 1.0) && !hasAny) return if (hasVowel) restress(SECONDARY + ps) else ps
            if (stress >= 1 && !hasPrimary && SECONDARY in ps) return ps.map { if (it == SECONDARY) PRIMARY else it }.joinToString("")
            if (stress > 1 && !hasAny) return if (hasVowel) restress(PRIMARY + ps) else ps
            return ps
        }

        /** Moves each stress mark to just before the vowel it governs. */
        private fun restress(ps: String): String {
            val keyed = ps.mapIndexed { i, c -> i.toDouble() to c }.toMutableList()
            for ((i, c) in ps.withIndex()) if (c in STRESSES) {
                val j = (i until ps.length).firstOrNull { ps[it] in VOWELS }
                if (j != null) keyed[i] = (j - 0.5) to c
            }
            return keyed.sortedBy { it.first }.map { it.second }.joinToString("")
        }
    }
}
