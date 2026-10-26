package dev.bend.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.security.MessageDigest
import java.time.Instant
import java.util.concurrent.ConcurrentHashMap
import kotlin.concurrent.thread

/**
 * KOKORODROID: the voices this phone can speak with, and the voice shop (android/voice/VOICES.md). Mirrors iOS voiceshop's
 * VoiceLibrary + VoiceShop: bundled packs (assets/voices/kokoro/packs, curated order in index.json), shop packs and dropped
 * packs (both in the drop-in folder [userDir], a shop pack has a sidecar with `installed`); the choice is remembered by
 * its id ("kokoro:af_bella"), default [DEFAULT_ID].
 */
class Voices(private val ctx: Context, private val log: (String) -> Unit) {
    private val prefs = ctx.getSharedPreferences("voices", Context.MODE_PRIVATE)

    /** getExternalFilesDir/voices: `adb push x.bin /sdcard/Android/data/dev.bend.app/files/voices/`. Made by the app
     *  (a dir adb creates under Android/data is not readable by the app). */
    val userDir = File(ctx.getExternalFilesDir(null), "voices").apply { mkdirs() }

    class Pack(val id: String, val name: String, val lang: String, val gender: String?, val grade: String?, val detail: String?,
               val license: String, val credit: String, val origin: String, val path: String, val bytes: Long, val speed: Float) {
        fun json() = JSONObject().put("id", id).put("name", name).put("lang", lang).put("gender", gender ?: JSONObject.NULL)
            .put("grade", grade ?: JSONObject.NULL).put("detail", detail ?: JSONObject.NULL).put("license", license)
            .put("credit", credit).put("origin", origin).put("bytes", bytes)
    }

    // MARK: library

    @Volatile private var cached: List<Pack>? = null

    fun packs(refresh: Boolean = false): List<Pack> {
        if (!refresh) cached?.let { return it }
        val out = ArrayList<Pack>(); val seen = HashSet<String>()
        val dir = "${KokoroRuntime.DIR}/packs"
        val order = runCatching { JSONArray(ctx.assets.open("$dir/index.json").reader().use { it.readText() }) }.getOrNull()
        for (i in 0 until (order?.length() ?: 0)) {
            val stem = order!!.getString(i)
            val meta = runCatching { JSONObject(ctx.assets.open("$dir/$stem.json").reader().use { it.readText() }) }.getOrNull()
            val bytes = runCatching { ctx.assets.openFd("$dir/$stem.bin").use { it.length } }.getOrElse { KokoroRuntime.STYLE_BYTES.toLong() }
            pack(stem, meta, "bundled", "asset:$dir/$stem.bin", bytes)?.let { if (seen.add(it.id)) out.add(it) }
        }
        val user = ArrayList<Pack>()
        for (f in (userDir.listFiles() ?: emptyArray()).sortedBy { it.name }) {
            val ext = f.extension.lowercase()
            if (ext !in listOf("bin", "pt") || f.name.startsWith(".")) continue
            val meta = File(f.parentFile, f.nameWithoutExtension + ".json").takeIf { it.exists() }?.let { runCatching { JSONObject(it.readText()) }.getOrNull() }
            if (meta != null && meta.optString("engine", "kokoro") != "kokoro") continue
            // A pack's size: 510 × 256 float32, plus the zip around it for a .pt.
            if (f.length() < KokoroRuntime.STYLE_BYTES || f.length() >= 4L * KokoroRuntime.STYLE_BYTES) continue
            pack(f.nameWithoutExtension, meta, if (meta?.has("installed") == true) "shop" else "dropped", f.path, f.length())?.let { if (seen.add(it.id)) user.add(it) }
        }
        out += user.sortedBy { it.name }
        cached = out
        return out
    }

    /** Kokoro's files are named <lang><gender>_<name> (af_bella: American female "Bella"); a sidecar's words win. */
    private fun pack(stem: String, meta: JSONObject?, origin: String, path: String, bytes: Long): Pack? {
        fun m(k: String) = meta?.optString(k)?.takeIf { meta.has(k) && !meta.isNull(k) }
        val parts = stem.split('_', limit = 2)
        val code = if (parts.size == 2 && parts[0].length == 2) parts[0] else ""
        val name = m("name") ?: (if (parts.size == 2) parts[1] else stem).replace('_', ' ').split(' ').joinToString(" ") { it.replaceFirstChar(Char::uppercase) }
        val dropped = origin == "dropped"
        return Pack("kokoro:${m("id") ?: stem}", name, m("lang") ?: if (code.startsWith("b")) "en-gb" else "en-us",
            m("gender") ?: when (code.getOrNull(1)) { 'f' -> "female"; 'm' -> "male"; else -> null }, m("grade"), m("detail"),
            m("license") ?: if (dropped) "as supplied" else "Apache-2.0", m("credit") ?: if (dropped) "added by hand" else "hexgrad/Kokoro-82M",
            origin, path, bytes, meta?.optDouble("speed", 1.0)?.toFloat() ?: 1f)
    }

    fun pack(id: String) = packs().firstOrNull { it.id == id }

    var choice: String
        get() = prefs.getString("choice", null) ?: DEFAULT_ID
        set(v) { prefs.edit().putString("choice", v).apply() }

    /** The pack to speak with: the choice, else the default, else the first listed. */
    fun current(): Pack? = pack(choice) ?: pack(DEFAULT_ID) ?: packs().firstOrNull()

    // MARK: engine

    /** Intra-op threads for the runtime: 6 by default (2×X1 + 2×A76 + 2×A55 on the 6a; 4 left the small cores idle while
     *  Kokoro ran slower than real time); `adb shell am start … --ei kokoro_threads N` overrides. */
    var threads: Int
        get() = prefs.getInt("threads", 6)
        set(v) { if (v != threads) { prefs.edit().putInt("threads", v).apply(); reset() } }

    @Synchronized private fun reset() { runtime?.close(); runtime = null; loaded = null; loadedId = null }

    private var runtime: KokoroRuntime? = null
    private var loaded: KokoroVoice? = null
    private var loadedId: String? = null
    @Volatile var loadMs = -1L; private set

    val available get() = runCatching { ctx.assets.openFd("${KokoroRuntime.DIR}/${KokoroRuntime.MODEL}").close() }.isSuccess

    /** The pack ready to speak (style tables are small: loaded per pack; the model once). */
    @Synchronized fun voice(p: Pack): KokoroVoice {
        if (loadedId == p.id) loaded?.let { return it }
        val rt = runtime ?: run {
            val t = System.currentTimeMillis()
            KokoroRuntime(ctx.assets, threads).also { runtime = it; loadMs = System.currentTimeMillis() - t; log("kokoro runtime threads=$threads load_ms=$loadMs") }
        }
        val style = if (p.path.startsWith("asset:")) KokoroRuntime.style(ctx.assets.open(p.path.removePrefix("asset:")).use { it.readBytes() }, "bin", p.name)
            else KokoroRuntime.style(File(p.path))
        return KokoroVoice(rt, "Kokoro ${p.name}", style, p.lang == "en-gb", p.speed).also { loaded = it; loadedId = p.id }
    }

    /** SHA-256 of the exact pack bytes the runtime reads — the fixture id a receipt records (on-device hash). */
    fun sha(p: Pack): String {
        val md = MessageDigest.getInstance("SHA-256")
        val input = if (p.path.startsWith("asset:")) ctx.assets.open(p.path.removePrefix("asset:")) else File(p.path).inputStream()
        input.use { i -> val b = ByteArray(1 shl 16); while (true) { val n = i.read(b); if (n < 0) break; md.update(b, 0, n) } }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    @Synchronized fun g2p(text: String, british: Boolean): JSONObject {
        val rt = runtime ?: KokoroRuntime(ctx.assets, threads).also { runtime = it }
        val g = rt.phonemizer(british)
        return JSONObject().put("phonemes", g.phonemize(text)).put("unknown", JSONArray(g.unknown))
    }

    // MARK: shop

    /** The catalog this launch reads: the developer override (voices.html "Source", kept in prefs), else [CATALOG_URL]. */
    var catalogUrl: String
        get() = prefs.getString("catalog", null)?.takeIf { it.isNotBlank() } ?: CATALOG_URL
        set(v) { prefs.edit().putString("catalog", v.trim().ifBlank { null }).apply() }

    private val cacheFile = File(ctx.cacheDir, "voice-catalog.json")
    @Volatile private var entries: List<JSONObject> = emptyList()
    @Volatile private var listing = "idle"
    @Volatile private var why: String? = null
    private val rows = ConcurrentHashMap<String, JSONObject>() // id → {state, have, error}
    private val stop = ConcurrentHashMap<String, Boolean>()
    val receipts = ArrayList<String>()

    private fun entry(id: String) = entries.firstOrNull { it.getString("id") == id } ?: throw IllegalStateException("no voice $id in the catalog")
    private fun dest(e: JSONObject) = File(userDir, e.getString("file"))
    private fun part(e: JSONObject) = File(userDir, e.getString("file") + ".part")
    private fun resolve(ref: String) = URI(catalogUrl).resolve(ref).toString()

    /** Fetches the catalog (8 s); on failure the cached copy if any and a plain reason. Offline-first: nothing else waits on it. */
    fun load(): JSONObject {
        listing = "loading"
        try {
            val c = URL(catalogUrl).openConnection() as HttpURLConnection
            c.connectTimeout = 8000; c.readTimeout = 8000
            val code = c.responseCode
            if (code != 200) { c.disconnect(); throw ShopError("The voice shop answered HTTP $code.") }
            val text = c.inputStream.reader().use { it.readText() }
            val doc = try { JSONObject(text).getJSONArray("voices") } catch (e: Exception) { throw ShopError("The voice shop’s list didn’t read as a catalog.") }
            entries = (0 until doc.length()).map { doc.getJSONObject(it) }
            cacheFile.writeText(text)
            listing = "ready"; why = null
            log("voices catalog ok url=$catalogUrl voices=${entries.joinToString(",") { it.getString("id") }}")
        } catch (e: Exception) {
            why = (e as? ShopError)?.message ?: "Can’t reach the voice shop (${runCatching { URI(catalogUrl).host }.getOrNull() ?: catalogUrl})."
            val cached = runCatching { JSONObject(cacheFile.readText()).getJSONArray("voices") }.getOrNull()
            if (cached != null) entries = (0 until cached.length()).map { cached.getJSONObject(it) }
            listing = "offline"
            log("voices catalog offline url=$catalogUrl cached=${cached != null} error=$e")
        }
        return shop()
    }

    fun shop(): JSONObject {
        val list = JSONArray()
        for (e in entries) {
            val id = e.getString("id")
            val r = rows[id]
            val state = when {
                r?.optString("state") == "downloading" -> "downloading"
                dest(e).exists() -> "installed"
                r?.optString("state") == "failed" -> "failed"
                part(e).length() > 0 -> "paused"
                else -> "available"
            }
            list.put(JSONObject(e.toString()).put("state", state).put("have", if (state == "installed") e.getLong("bytes") else r?.optLong("have") ?: part(e).length())
                .put("error", if (state == "failed") r!!.optString("error") else JSONObject.NULL)
                .put("preview", e.optString("preview_url").takeIf { it.isNotEmpty() }?.let { runCatching { resolve(it) }.getOrNull() } ?: JSONObject.NULL))
        }
        return JSONObject().put("listing", listing).put("why", why ?: JSONObject.NULL).put("catalog", catalogUrl)
            .put("overridden", catalogUrl != CATALOG_URL).put("voices", list)
    }

    /** Download, Resume and Retry are one call: the .part is kept and the rest asked for with Range. */
    fun download(id: String) {
        val e = entry(id)
        if (rows[id]?.optString("state") == "downloading") return
        rows[id] = JSONObject().put("state", "downloading").put("have", part(e).length())
        stop[id] = false
        thread(name = "voice-dl-$id", isDaemon = true) {
            var attempt = 1
            while (true) {
                val from = part(e).length(); val t0 = System.currentTimeMillis()
                try {
                    fetch(e, id)
                    install(e)
                    rows.remove(id)
                    line("file ${e.getString("file")} ok bytes=${e.getLong("bytes")} resumed_from=$from sha256=${e.getString("sha256").take(8)}… s=${"%.2f".format((System.currentTimeMillis() - t0) / 1000.0)}")
                    return@thread
                } catch (x: Paused) {
                    rows.remove(id); line("pause id=$id at=${part(e).length()}"); return@thread
                } catch (x: Exception) {
                    val transient = x is IOException && x !is ShopError
                    line("file ${e.getString("file")} FAILED attempt=$attempt $x")
                    if (transient && attempt < 3) { Thread.sleep(if (attempt == 1) 1000 else 3000); attempt++; continue } // as iOS: twice more, then Retry is the person's
                    rows[id] = JSONObject().put("state", "failed").put("error", message(x, e))
                    return@thread
                }
            }
        }
    }

    fun pause(id: String) { stop[id] = true }

    /** Streams into <file>.part from what it already holds; renamed into place only when size and sha256 match. */
    private fun fetch(e: JSONObject, id: String) {
        val want = e.getLong("bytes"); val p = part(e)
        val sha = MessageDigest.getInstance("SHA-256")
        if (p.length() > want) p.delete()
        if (p.exists()) p.inputStream().use { i -> val b = ByteArray(1 shl 16); while (true) { val n = i.read(b); if (n < 0) break; sha.update(b, 0, n) } }
        var have = p.length()
        if (have < want) {
            val c = URL(resolve(e.getString("pack_url"))).openConnection() as HttpURLConnection
            c.connectTimeout = 15000; c.readTimeout = 20000
            if (have > 0) c.setRequestProperty("Range", "bytes=$have-")
            val code = try { c.responseCode } catch (x: IOException) { throw x }
            if (code == 200 && have > 0) { have = 0; sha.reset(); p.delete() } // no Range support: start over
            else if (code != 200 && code != 206) { c.disconnect(); throw ShopError("http $code", code) }
            val total = if (code == 206) c.getHeaderField("Content-Range")?.substringAfter('/')?.toLongOrNull() else c.contentLengthLong.takeIf { it >= 0 }
            if (total != null && total != want) { c.disconnect(); p.delete(); throw ShopError("host size $total", hostSize = total) }
            RandomAccessFile(p, "rw").use { f ->
                f.seek(have)
                c.inputStream.use { i ->
                    val b = ByteArray(1 shl 16)
                    while (true) {
                        if (stop[id] == true) throw Paused()
                        val n = i.read(b); if (n < 0) break
                        f.write(b, 0, n); sha.update(b, 0, n); have += n
                        rows[id]?.put("have", have)
                    }
                }
            }
        }
        if (have != want) throw IOException("short: $have of $want bytes") // a cut connection: transient, resumes
        val hex = sha.digest().joinToString("") { "%02x".format(it) }
        if (hex != e.getString("sha256")) { p.delete(); throw ShopError("checksum $hex", checksum = true) }
        if (!p.renameTo(dest(e))) throw ShopError("Couldn’t move the file into place.")
    }

    /** The verified file is in place: its sidecar makes it a shop pack; it must read as a voice, else it is removed. */
    private fun install(e: JSONObject) {
        val meta = JSONObject()
        for (k in listOf("id", "engine", "name", "lang", "gender", "grade", "detail", "license", "credit")) if (e.has(k)) meta.put(k, e.get(k))
        meta.put("installed", Instant.now().toString())
        val sidecar = File(userDir, dest(e).nameWithoutExtension + ".json")
        sidecar.writeText(meta.toString(1))
        try { KokoroRuntime.style(dest(e)) } catch (x: Exception) { dest(e).delete(); sidecar.delete(); throw ShopError("Couldn’t install ${e.getString("name")}: ${x.message}") }
        packs(refresh = true)
    }

    /** Removes a shop pack (file + sidecar); the choice falls back to the default if it was this one. */
    fun remove(id: String) {
        val e = entry(id)
        dest(e).delete(); part(e).delete(); File(userDir, dest(e).nameWithoutExtension + ".json").delete()
        rows.remove(id); packs(refresh = true)
        line("removed id=$id")
    }

    private fun line(s: String) { synchronized(receipts) { receipts.add(s) }; log("voices $s") }

    private fun message(x: Exception, e: JSONObject): String {
        val name = e.getString("name"); val host = runCatching { URI(resolve(e.getString("pack_url"))).host }.getOrNull() ?: "the shop"
        if (x is ShopError) return when {
            x.checksum -> "$name arrived damaged (its checksum didn’t match), so it was deleted. Retry downloads it again."
            x.hostSize != null -> "The shop’s copy of $name is ${size(x.hostSize)}, not ${size(e.getLong("bytes"))}. Nothing was kept."
            x.code == 404 -> "The shop doesn’t have $name right now (HTTP 404)."
            x.code != null -> "The shop answered HTTP ${x.code} for $name. Retry in a moment."
            else -> x.message ?: "Couldn’t install $name."
        }
        if (x.message?.startsWith("short") == true) return "Connection lost partway through $name. Retry picks up where it stopped."
        return "Can’t reach $host. Check the network, then Retry. Your voices on this phone still work."
    }

    private class ShopError(msg: String, val code: Int? = null, val hostSize: Long? = null, val checksum: Boolean = false) : IOException(msg)
    private class Paused : Exception()

    companion object {
        /** The house voice (owner pick, voice-2026/PICK): bundle default and playback default. */
        const val DEFAULT_ID = "kokoro:af_bella"
        /** The same line for every preview, so voices compare on equal terms (iOS VoiceLibrary.demoLine). */
        const val DEMO_LINE = "Hi, I'm the voice Bend will use to read its replies. Shall we plan your Saturday, or keep it simple?"

        /**
         * THE policy point: where the voice shop's catalog lives. One constant; nothing else in the app names a voice host.
         * `.invalid` until the owner picks a host (decision gate: android/voice/VOICES.md; the same constant as iOS
         * VoiceCatalog.CATALOG_URL): the shop then says "Can't reach the voice shop" and the bundled voices work as they are.
         * pack_url / preview_url in the catalog: absolute, or relative to it.
         */
        const val CATALOG_URL = "https://models.bend.invalid/v0/voices/catalog.json"

        fun size(b: Long) = if (b >= 1_000_000) "%.1f MB".format(b / 1e6) else "%.0f KB".format(b / 1e3)
    }
}
