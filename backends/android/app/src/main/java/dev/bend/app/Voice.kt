package dev.bend.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread
import kotlin.math.log10
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * The native half of phoneone's voice mode (src/voice/types.ts), over Android's own services: hearing is the platform
 * mic through [Whisper] (ANDROIDVOICE: whisper.cpp on the device; the platform SpeechRecognizer needed a SODA pack the 6a lacks), speaking is Kokoro (KOKORODROID: [Voices]' chosen pack, on-device onnxruntime; android/voice/VOICES.md), else [TextToSpeech], or, on a phone with no TTS engine (the Pixel 6a has none), the bundled espeak-ng (android/espeak/build.sh); the reply text
 * is still OUR engine's (the web app streams it through /v1/chat/completions). Routes: the voice paths in Server.kt.
 */
class Voice(private val ctx: Context, private val log: (String) -> Unit, private val mic: () -> Boolean, val voices: Voices, private val keepAudio: Boolean = false) {
    private val ttsUp = CountDownLatch(1)
    private var ttsOk = false
    private val tts = TextToSpeech(ctx) { ttsOk = it == TextToSpeech.SUCCESS; ttsUp.countDown() }
    private val whisper = Whisper(ctx, log)
    @Volatile private var rec: AudioRecord? = null
    @Volatile private var micDone = CountDownLatch(0)
    @Volatile private var heardPcm = ByteArrayOutputStream()
    @Volatile private var level = -120.0
    @Volatile private var peak = -120.0
    @Volatile private var speech = false
    @Volatile private var voiced = false
    @Volatile private var noise = -120.0
    /** TTSMODE: "fast" = bundled espeak-ng (the pre-Kokoro latency, ~100 ms a clause); "kokoro" = the neural voice (RTF ~1.7 here); anything else = Kokoro when a pack is chosen. */
    @Volatile var ttsMode = "fast"
    @Volatile private var auto = false
    @Volatile private var endpointed = 0
    @Volatile private var finalMs = -1L
    @Volatile private var firstText = -1L
    @Volatile private var phase = "idle"
    @Volatile private var partial = ""
    @Volatile private var text = ""
    @Volatile private var turn = 0
    @Volatile private var pending = 0
    @Volatile private var spoken = CountDownLatch(0)
    @Volatile private var t0 = 0L
    @Volatile private var tRelease = 0L
    @Volatile private var firstAudio = -1L
    @Volatile private var chunks = 0
    @Volatile private var reply = StringBuilder()
    private val say = Executors.newSingleThreadExecutor()
    @Volatile private var gen = 0
    @Volatile private var player: MediaPlayer? = null
    private val play = Executors.newSingleThreadExecutor()
    @Volatile private var track: AudioTrack? = null
    private val kokoro = voices.available
    /** One ORT session: /v1/speak renders and voice-mode clauses take turns (KokoroVoice is not thread-safe). */
    private val kokoroLock = Any()

    init {
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String?) { if (firstAudio < 0) firstAudio = System.currentTimeMillis() - tRelease }
            override fun onDone(id: String?) = done()
            @Deprecated("api") override fun onError(id: String?) = done()
        })
    }

    private fun done() { synchronized(this) { if (--pending <= 0) spoken.countDown() } }

    fun status() = JSONObject().put("available", true).put("assets", "installed") // the whisper model ships in the APK
        .put("phase", phase).put("line", auto && rec != null).put("speech", speech).put("level", level).put("endpointed", endpointed)
        .put("mic", if (ctx.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) "granted" else "undetermined").put("partial", partial).put("turn", turn).put("prepared", true)
        .put("voice", voices.current()?.takeIf { kokoro }?.let { "Kokoro ${it.name}" } ?: if (ttsOk) tts.defaultVoice?.name ?: "system" else "espeak-ng")
        .put("voiceID", voices.current()?.takeIf { kokoro }?.id ?: JSONObject.NULL).put("voiceChoice", voices.choice)

    fun prepare(): JSONObject {
        val t = System.currentTimeMillis()
        if (!mic()) throw IllegalStateException("Microphone permission was denied")
        ttsUp.await(10, TimeUnit.SECONDS)
        whisper.ensure()
        if (kokoro) runCatching { voices.voice(voices.current()!!) }.onFailure { log("voice kokoro load failed: $it") }
        else if (!ttsOk) espeakDir()
        return JSONObject().put("ms", System.currentTimeMillis() - t).put("ttsReady", ttsOk || kokoro)
    }

    /**
     * A turn starts: the mic, 16 kHz mono. Hold-to-talk ends at [release]; hands-free ([auto], VOICE-R2) ends itself after
     * [silenceMs] of quiet following speech (`endpointed`; the web loop then calls release). The line is an energy gate,
     * not a model: a 20 ms frame is speech above [SPEECH_DB]. ponytail: no barge-in (the mic is closed while the reply
     * plays; talking over it needs echo cancellation first), and no partials (whisper hears the whole turn at once).
     */
    fun listen(auto: Boolean = false, silenceMs: Int = 700): Int {
        cancel()
        if (!mic()) throw IllegalStateException("Microphone permission was denied")
        turn++; text = ""; partial = ""; firstAudio = -1; firstText = -1; finalMs = -1; chunks = 0; reply = StringBuilder()
        val r = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT), RATE))
        if (r.state != AudioRecord.STATE_INITIALIZED) { r.release(); throw IllegalStateException("The microphone did not open") }
        val pcm = ByteArrayOutputStream(); val me = turn; val done = CountDownLatch(1)
        heardPcm = pcm; micDone = done; voiced = false; speech = false; level = -120.0; peak = -120.0; this.auto = auto
        r.startRecording(); rec = r; t0 = System.currentTimeMillis(); phase = "listening"
        thread(name = "mic", isDaemon = true) {
            val buf = ShortArray(RATE / 50); val bytes = ByteBuffer.allocate(buf.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            var lastVoice = 0L; var floor = Double.NaN; var voicedMs = 0
            try {
                while (rec === r) {
                    val n = r.read(buf, 0, buf.size)
                    if (n <= 0) break
                    bytes.clear(); var sum = 0.0
                    for (i in 0 until n) { bytes.putShort(buf[i]); sum += buf[i].toDouble() * buf[i] }
                    pcm.write(bytes.array(), 0, n * 2)
                    val db = 20 * log10(maxOf(sqrt(sum / n) / 32768, 1e-6))
                    val now = System.currentTimeMillis()
                    // Speech = well over the room's own noise (tracked while quiet), and over an absolute minimum.
                    level = db; peak = maxOf(peak, db); speech = db > maxOf(SPEECH_DB, (if (floor.isNaN()) db else floor) + OVER_FLOOR_DB)
                    // The floor drops to any quieter frame and creeps up otherwise (fast while quiet, ~10 s while loud, so
                    // steady music stops reading as speech). AudioRecord's leading digital zeros are not the room.
                    if (db > -90) floor = if (floor.isNaN() || db < floor) db else floor + (db - floor) * (if (speech) 0.002 else 0.05)
                    if (!speech && !voiced && now - lastVoice > silenceMs) voicedMs = 0 // scattered clicks never add up to a turn
                    noise = floor
                    if (speech) { voicedMs += 20; lastVoice = now; if (voicedMs >= MIN_VOICED_MS) voiced = true }
                    val full = pcm.size() >= MAX_S * RATE * 2
                    if (auto && voiced && (now - lastVoice >= silenceMs || full) && endpointed < me) { tRelease = lastVoice; endpointed = me; break }
                    if (full) { if (auto && !voiced) pcm.reset() else break } // a quiet open line keeps listening
                }
            } finally {
                if (rec === r) rec = null
                speech = false; level = -120.0
                runCatching { r.stop() }; r.release(); done.countDown()
            }
        }
        return turn
    }

    /** Hands-free: end the turn now (the person tapped send). */
    fun endpoint() { if (rec != null && endpointed < turn) { tRelease = System.currentTimeMillis(); endpointed = turn; rec = null } }

    /** Speech ended: the mic closes and whisper hears the turn. */
    fun release(): JSONObject {
        if (endpointed < turn) tRelease = System.currentTimeMillis() // hold-to-talk: the release is the end of speech
        rec = null; micDone.await(2, TimeUnit.SECONDS)
        phase = "thinking"
        val raw = heardPcm.toByteArray()
        val shorts = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        // whisper wants >= 1 s; a short tap is padded with silence.
        val a = FloatArray(maxOf(shorts.remaining(), RATE + RATE / 10)).also { for (i in 0 until shorts.remaining()) it[i] = shorts.get(i) / 32768f }
        val audioS = shorts.remaining().toDouble() / RATE
        val t = System.currentTimeMillis()
        // Silence is not sent to whisper (it would guess words: "you", "Thank you.").
        text = if (peak < SILENCE_DB) "" else NOISE.replace(whisper.transcribe(a), " ").replace(Regex("\\s+"), " ").trim()
        val asrMs = System.currentTimeMillis() - t
        finalMs = System.currentTimeMillis() - tRelease
        log("voice asr turn=$turn model=${whisper.model} audio_s=${"%.2f".format(audioS)} asr_ms=$asrMs rtf=${"%.3f".format(asrMs / 1000.0 / maxOf(audioS, 0.01))} final_ms=$finalMs peak_db=${"%.1f".format(peak)} floor_db=${"%.1f".format(noise)} auto=$auto text=\"$text\"")
        if (keepAudio) File(ctx.getExternalFilesDir(null), "voice").apply { mkdirs() }.let { wav(File(it, "turn-$turn.wav"), raw) }
        return receipt(false).put("finalMs", finalMs)
    }

    /** The heard turn as a 16 kHz mono wav (receipts only: `--ez keep_audio true`). */
    private fun wav(f: File, pcm: ByteArray) = f.outputStream().use { o ->
        val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        h.put("RIFF".toByteArray()).putInt(36 + pcm.size).put("WAVEfmt ".toByteArray()).putInt(16).putShort(1).putShort(1)
            .putInt(RATE).putInt(RATE * 2).putShort(2).putShort(16).put("data".toByteArray()).putInt(pcm.size)
        o.write(h.array()); o.write(pcm)
    }

    fun speak(t: String) {
        synchronized(this) { if (pending++ == 0) spoken = CountDownLatch(1) }
        if (firstText < 0) firstText = System.currentTimeMillis() - tRelease
        chunks++; reply.append(if (reply.isEmpty()) "" else " ").append(t); phase = "speaking"
        val g = gen; val n = chunks
        // Kokoro: synth on `say`, play on `play`, so clause n+1 renders while clause n is heard. The pack is read per
        // clause, so a pick in voices.html is heard from the next clause on (hot swap).
        val pack = voices.current()
        if (ttsMode != "fast" && kokoro && pack != null) {
            say.execute {
                val pcm = if (g == gen) try { synchronized(kokoroLock) { voices.voice(pack).synthesize(t) } } catch (e: Exception) { log("voice kokoro failed: $e"); null } else null
                play.execute { if (pcm != null && g == gen) pcm(pcm, g); done() }
            }
            return
        }
        if (ttsOk) { tts.speak(t, TextToSpeech.QUEUE_ADD, null, "t$turn-$chunks"); return }
        say.execute { if (g == gen) try { espeak(t, n) } catch (e: Exception) { log("voice espeak failed: $e") }; done() }
    }

    /** 24 kHz mono float PCM to the speaker; returns once it has been heard (or a cancel stopped it). */
    private fun pcm(a: FloatArray, g: Int) {
        val t = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ASSISTANT).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT).setSampleRate(KokoroRuntime.SAMPLE_RATE).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setTransferMode(AudioTrack.MODE_STREAM).build()
        track = t
        try {
            t.play()
            if (firstAudio < 0) firstAudio = System.currentTimeMillis() - tRelease
            var o = 0
            while (o < a.size && g == gen) { val n = t.write(a, o, minOf(4800, a.size - o), AudioTrack.WRITE_BLOCKING); if (n <= 0) break; o += n }
            val until = System.currentTimeMillis() + a.size * 1000L / KokoroRuntime.SAMPLE_RATE + 1000
            while (g == gen && t.playbackHeadPosition < o && System.currentTimeMillis() < until) Thread.sleep(10)
        } finally { track = null; runCatching { t.stop() }; t.release() }
    }

    /**
     * voices.html's Preview (and the receipts' bench): [text] in pack [id], clause by clause as voice mode chunks a reply,
     * rendering clause n+1 while n plays. Times: first_clause_ms = request → first clause rendered; rtf = render s / audio s.
     */
    fun preview(id: String, text: String, playIt: Boolean): JSONObject {
        val pack = voices.pack(id) ?: throw IllegalStateException("no voice $id")
        cancel()
        val g = gen; val t0 = System.nanoTime()
        val v = voices.voice(pack)
        val loadMs = (System.nanoTime() - t0) / 1_000_000
        val clauses = text.split(Regex("(?<=[.!?;])\\s+")).filter { it.isNotBlank() }
        val rows = JSONArray(); var synth = 0.0; var audio = 0.0; var first = -1L
        val heard = java.util.concurrent.CountDownLatch(clauses.size)
        tRelease = System.currentTimeMillis(); firstAudio = -1
        for (c in clauses) {
            val s = System.nanoTime()
            val pcm = synchronized(kokoroLock) { v.synthesize(c) }
            val ms = (System.nanoTime() - s) / 1e6
            if (first < 0) first = (System.nanoTime() - t0) / 1_000_000
            val sec = pcm.size.toDouble() / KokoroRuntime.SAMPLE_RATE
            synth += ms / 1000; audio += sec
            rows.put(JSONObject().put("text", c).put("synth_ms", ms.toLong()).put("audio_s", "%.2f".format(sec).toDouble()))
            if (playIt) play.execute { try { if (g == gen) pcm(pcm, g) } finally { heard.countDown() } } else heard.countDown()
        }
        val r = JSONObject().put("id", pack.id).put("name", "Kokoro ${pack.name}").put("threads", voices.threads).put("load_ms", loadMs)
            .put("first_clause_ms", first).put("synth_s", "%.3f".format(synth).toDouble()).put("audio_s", "%.2f".format(audio).toDouble())
            .put("rtf", "%.3f".format(synth / audio).toDouble()).put("clauses", rows)
        if (playIt) { heard.await(60, TimeUnit.SECONDS); r.put("first_audio_ms", firstAudio) }
        log("voice preview $r")
        return r
    }

    /**
     * SPEAK route: text -> 24 kHz mono PCM16 RIFF/WAVE bytes, rendered by [Voices]' pack (Kokoro ONNX, on-device).
     * Returns the bytes plus what a receipt needs: the pack id, the SHA-256 of the style bank the runtime read, and
     * the render time. `g2p` is the app's misaki port (the same one voice mode speaks through).
     */
    fun speakWav(text: String, id: String?): Spoken {
        if (text.isBlank()) throw IllegalArgumentException("text is empty")
        if (text.length > 2000) throw IllegalArgumentException("text is ${text.length} chars; the route takes 1-2000")
        val want = id?.trim()?.takeIf { it.isNotEmpty() }
        val pack = if (want == null) voices.current() else voices.pack(if (want.startsWith("kokoro:")) want else "kokoro:$want")
        if (pack == null) throw IllegalStateException(
            "no voice $want on this phone; bundled: ${voices.packs().joinToString(", ") { it.id.removePrefix("kokoro:") }}")
        if (!kokoro) throw IllegalStateException("this build carries no Kokoro runtime")
        val t0 = System.currentTimeMillis()
        val pcm = synchronized(kokoroLock) { voices.voice(pack).synthesize(text) }
        if (pcm.isEmpty()) throw IllegalStateException("the voice rendered no audio (no known phonemes in that text)")
        val wav = wav24(pcm)
        val ms = System.currentTimeMillis() - t0
        val sha = sha256(wav)
        log("speak id=${pack.id} sha=${sha.take(8)}… text_len=${text.length} samples=${pcm.size} bytes=${wav.size} ms=$ms")
        return Spoken(wav, pack.id.removePrefix("kokoro:"), voices.sha(pack), ms, pcm.size, sha)
    }

    /**
     * HEAR route: WAV bytes -> transcript. The clip is decoded here (PCM16/float32, mono or stereo), resampled to the
     * 16 kHz [Whisper] expects, padded to whisper's 1 s floor, then heard by the same model voice mode uses.
     */
    fun hearWav(bytes: ByteArray): Heard {
        val t0 = System.currentTimeMillis()
        val clip = readWav(bytes)
        val pcm16 = if (clip.rate == RATE) clip.pcm else resample(clip.pcm, clip.rate, RATE)
        val padded = if (pcm16.size < RATE) pcm16.copyOf(RATE) else pcm16
        val t1 = System.currentTimeMillis()
        val raw = whisper.transcribe(padded)
        val text = NOISE.replace(raw, " ").replace(Regex("\\s+"), " ").trim()
        val ms = System.currentTimeMillis() - t1
        val secs = clip.pcm.size.toDouble() / clip.rate
        log("hear model=${whisper.model} sha=${whisper.sha.take(8)}… audio_s=${"%.2f".format(secs)} resample_ms=${t1 - t0} decode_ms=$ms rtf=${"%.3f".format(ms / 1000.0 / maxOf(secs, 0.01))} text=\"$text\"")
        return Heard(text, whisper.model, whisper.sha, ms, t1 - t0, secs)
    }

    class Spoken(val wav: ByteArray, val voice: String, val bankSha: String, val ms: Long, val samples: Int, val wavSha: String)
    class Heard(val text: String, val model: String, val modelSha: String, val decodeMs: Long, val resampleMs: Long, val durationS: Double)

    /** 24 kHz mono PCM16 little-endian RIFF/WAVE (float -1..1 -> int16, round half away from zero). */
    private fun wav24(pcm: FloatArray): ByteArray {
        val n = pcm.size
        val b = ByteBuffer.allocate(44 + n * 2).order(ByteOrder.LITTLE_ENDIAN)
        b.put("RIFF".toByteArray()).putInt(36 + n * 2).put("WAVE".toByteArray())
        b.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1)
            .putInt(KokoroRuntime.SAMPLE_RATE).putInt(KokoroRuntime.SAMPLE_RATE * 2).putShort(2).putShort(16)
        b.put("data".toByteArray()).putInt(n * 2)
        for (x in pcm) b.putShort((x.coerceIn(-1f, 1f) * 32767f).roundToInt().toShort())
        return b.array()
    }

    private fun sha256(b: ByteArray): String =
        java.security.MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

    private class Clip(val pcm: FloatArray, val rate: Int)

    /** RIFF/WAVE -> mono float PCM. Accepts PCM16 and IEEE float32 (mono or stereo); anything else is refused by name. */
    private fun readWav(b: ByteArray): Clip {
        if (b.size < 44 || String(b, 0, 4, Charsets.ISO_8859_1) != "RIFF" || String(b, 8, 4, Charsets.ISO_8859_1) != "WAVE")
            throw IllegalArgumentException("audio must be a RIFF/WAVE file (a base64 WAV or data:audio/wav)")
        fun u16(o: Int) = (b[o].toInt() and 0xff) or ((b[o + 1].toInt() and 0xff) shl 8)
        fun u32(o: Int) = u16(o) or (u16(o + 2) shl 16)
        var pos = 12
        var fmt = 0; var ch = 0; var rate = 0; var bits = 0
        var off = -1; var len = 0
        while (pos + 8 <= b.size) {
            val id = String(b, pos, 4, Charsets.ISO_8859_1)
            val size = u32(pos + 4)
            if (id == "fmt " && pos + 24 <= b.size) {
                fmt = u16(pos + 8); ch = u16(pos + 10); rate = u32(pos + 12); bits = u16(pos + 22)
            } else if (id == "data") { off = pos + 8; len = minOf(size, b.size - off) }
            pos += 8 + size + (size and 1)
        }
        if (off < 0 || len <= 0) throw IllegalArgumentException("the WAV has no data chunk")
        if (rate <= 0 || ch <= 0) throw IllegalArgumentException("the WAV has no valid fmt chunk")
        val bytesPer = bits / 8
        if (bytesPer <= 0 || (fmt != 1 && fmt != 3 && fmt != 0xFFFE))
            throw IllegalArgumentException("unsupported WAV format $fmt (${bits}-bit); send PCM16 or float32")
        val frames = len / (bytesPer * ch)
        val out = FloatArray(frames)
        for (i in 0 until frames) {
            var acc = 0f
            for (c in 0 until ch) {
                val o = off + (i * ch + c) * bytesPer
                acc += when {
                    fmt == 3 || (fmt == 0xFFFE && bits == 32) -> java.lang.Float.intBitsToFloat(u32(o))
                    bits == 16 -> (u16(o).toShort()).toFloat() / 32768f
                    bits == 8 -> (b[o].toInt() and 0xff) / 128f - 1f
                    bits == 32 -> u32(o).toFloat() / 2147483648f
                    else -> throw IllegalArgumentException("unsupported ${bits}-bit PCM")
                }
            }
            out[i] = acc / ch
        }
        return Clip(out, rate)
    }

    /** Linear resample; only whisper's 16 kHz input needs it (the mic already records there). */
    private fun resample(src: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || src.isEmpty()) return src
        val n = (src.size.toLong() * to / from).toInt().coerceAtLeast(1)
        val step = from.toDouble() / to
        val out = FloatArray(n)
        for (i in 0 until n) {
            val x = i * step
            val i0 = x.toInt().coerceIn(0, src.size - 1)
            val i1 = minOf(i0 + 1, src.size - 1)
            val f = (x - i0).toFloat()
            out[i] = src[i0] * (1f - f) + src[i1] * f
        }
        return out
    }

    private fun espeak(t: String, n: Int) {
        val wav = File(ctx.cacheDir, "say-$turn-$n.wav")
        val p = ProcessBuilder(File(ctx.applicationInfo.nativeLibraryDir, "libespeak.so").path, "--path=${espeakDir()}", "-s", "160", "-w", wav.path, "--", t)
            .redirectErrorStream(true).start()
        p.inputStream.readBytes(); p.waitFor()
        val end = CountDownLatch(1)
        val m = MediaPlayer().apply { setDataSource(wav.path); setOnCompletionListener { end.countDown() }; prepare() }
        player = m
        if (firstAudio < 0) firstAudio = System.currentTimeMillis() - tRelease
        m.start(); end.await(60, TimeUnit.SECONDS)
        player = null; m.release(); wav.delete()
    }

    /** filesDir/espeak/espeak-ng-data, unpacked from the APK assets once (the binary reads real files). */
    private fun espeakDir(): File {
        val root = File(ctx.filesDir, "espeak")
        if (File(root, "espeak-ng-data/phondata").exists()) return root
        fun copy(a: String) {
            val kids = ctx.assets.list(a) ?: emptyArray()
            if (kids.isEmpty()) File(root, a).apply { parentFile!!.mkdirs() }.outputStream().use { o -> ctx.assets.open(a).use { it.copyTo(o) } }
            else kids.forEach { copy("$a/$it") }
        }
        copy("espeak-ng-data")
        return root
    }

    fun finish(): JSONObject {
        if (pending > 0) spoken.await(60, TimeUnit.SECONDS)
        val r = receipt(true).put("playedMs", System.currentTimeMillis() - tRelease)
        log("voice turn=$turn final_ms=$finalMs first_text_ms=$firstText first_audio_ms=$firstAudio played_ms=${r.getLong("playedMs")}")
        phase = "idle"
        return r
    }

    fun cancel() {
        gen++; player?.let { runCatching { it.stop() } }; track?.let { runCatching { it.pause(); it.flush() } }
        tts.stop(); synchronized(this) { pending = 0; spoken.countDown() }
        rec = null; micDone.await(1, TimeUnit.SECONDS)
        phase = "idle"
    }

    fun close() { cancel(); tts.shutdown(); say.shutdownNow(); play.shutdownNow() }

    private fun receipt(played: Boolean) = JSONObject().put("turn", turn).put("source", "mic").put("text", text)
        .put("heardBy", "whisper ${whisper.model} (on-device)").put("reply", reply.toString()).put("chunks", chunks).put("cancelled", false)
        .put("firstTextMs", if (firstText >= 0) firstText else JSONObject.NULL)
        .put("firstAudioMs", if (firstAudio >= 0) firstAudio else JSONObject.NULL).put("auto", auto).put("listenMs", tRelease - t0)
        .put("audioMs", 0).put("file", JSONObject.NULL).put("fileBytes", 0)

    private companion object {
        const val RATE = 16000
        /**
         * The hands-free line's gate (calibration knobs; the turn's peak and floor are in the log): a 20 ms frame is speech
         * when it is [OVER_FLOOR_DB] over the room's noise floor and over [SPEECH_DB]; a turn needs [MIN_VOICED_MS] of it.
         * A fixed -45 dB gate opened on the Pixel's room noise (-38..-42 dB peaks) and whisper captioned it "you".
         */
        const val SPEECH_DB = -50.0
        const val OVER_FLOOR_DB = 10.0
        const val MIN_VOICED_MS = 200
        /** A turn whose loudest frame is under this was silence: whisper does not hear it. */
        const val SILENCE_DB = -60.0
        /** whisper's window: a longer turn ends there. */
        const val MAX_S = 30
        /** Non-speech annotations whisper may still write: [BLANK_AUDIO], (music), *laughs*. */
        val NOISE = Regex("\\[[^\\]]*\\]|\\([^)]*\\)|\\*[^*]*\\*|[♪♫]+")
    }
}
