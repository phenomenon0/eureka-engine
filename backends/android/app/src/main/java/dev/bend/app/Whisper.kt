package dev.bend.app

import android.content.Context
import java.io.File
import java.security.MessageDigest

/**
 * ANDROIDVOICE: the app's ears, on the device: whisper.cpp (ggml, CPU) via libbendwhisper.so (android/asr/build.sh), so
 * hearing needs neither the platform recognizer (its SODA pack is missing on the Pixel 6a: LANGUAGE_PACK_ERROR 12) nor
 * the network. The bundled model is assets/asr/ggml-base.en-q8_0.bin (numbers: android/asr/README.md); a .bin dropped in
 * <external files>/asr/ replaces it (how tiny.en and base.en were compared). Loaded once, on voice mode's prepare or the
 * first /v1/hear.
 */
class Whisper(private val ctx: Context, private val log: (String) -> Unit) {
    private var handle = 0L
    var model = ""; private set
    /** SHA-256 of the .bin the handle was loaded from (receipts; computed once at load). */
    var sha = ""; private set

    @Synchronized fun ensure() {
        if (handle != 0L) return
        val t = System.currentTimeMillis()
        val f = modelFile()
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { i -> val b = ByteArray(1 shl 20); while (true) { val n = i.read(b); if (n < 0) break; md.update(b, 0, n) } }
        sha = md.digest().joinToString("") { "%02x".format(it) }
        handle = load(f.path)
        if (handle == 0L) throw IllegalStateException("whisper could not load ${f.name}")
        model = f.name.removePrefix("ggml-").removeSuffix(".bin")
        log("voice asr load model=$model sha=${sha.take(8)}… ms=${System.currentTimeMillis() - t}")
    }

    /** 16 kHz mono float PCM → text. */
    @Synchronized fun transcribe(pcm: FloatArray): String {
        ensure()
        return transcribe(handle, pcm, THREADS) ?: throw IllegalStateException("whisper failed")
    }

    private fun modelFile(): File {
        File(ctx.getExternalFilesDir(null), "asr").listFiles { f -> f.name.endsWith(".bin") }?.firstOrNull()?.let { return it }
        val out = File(ctx.filesDir, "asr/$BUNDLED")
        val size = ctx.assets.openFd("asr/$BUNDLED").use { it.length }
        if (out.length() != size) { // whisper reads a real file; the asset is copied out once (stored, not deflated)
            out.parentFile!!.mkdirs()
            out.parentFile!!.listFiles()?.forEach { if (it != out) it.delete() } // a previous build's model (0.5 shipped tiny)
            ctx.assets.open("asr/$BUNDLED").use { i -> out.outputStream().use { i.copyTo(it) } }
        }
        return out
    }

    companion object {
        const val BUNDLED = "ggml-base.en-q8_0.bin"
        /** The 6a's big cores (2× X1 + 2× A76); the engine process is idle while a turn is heard. */
        const val THREADS = 4
        init { System.loadLibrary("bendwhisper") }
        @JvmStatic private external fun load(path: String): Long
        @JvmStatic private external fun transcribe(ctx: Long, pcm: FloatArray, threads: Int): String?
    }
}
