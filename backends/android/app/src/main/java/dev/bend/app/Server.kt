package dev.bend.app

import android.content.res.AssetManager
import android.util.Base64
import org.json.JSONArray
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.OutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import kotlin.concurrent.thread

/**
 * Same-origin loopback HTTP on 127.0.0.1 (ephemeral port): the web dist from assets/web, and the routes
 * phoneone's native transport calls (src/engines/transport.ts). Control and engine share this one port.
 * POSTs need the per-launch X-Api-Key. One CPU engine serves chat; decide only on the Kev-4B pack (see Models).
 */
class Server(private val assets: AssetManager, private val engine: Engine, private val voice: Voice, private val key: String, private val autorun: Boolean,
             private val cpuOnly: Boolean = false, private val models: Models, private val detector: Detector, private val onGpu: (String, String) -> Unit = { _, _ -> },
             /** VOICEFAST: the voice profile trades length for latency: replies cap at 48 tokens (see chatTurn). */ private val voiceFast: Boolean = false,
             /** VOICEFAST: lazily built second engine for voice turns (a small trunk beside the 4B). Null = single engine. */
             private val voiceEngineFactory: (() -> Engine)? = null) {
    /** PHONEFAST: the WebGPU model path (assets/web/gpu-shim.js) reports here: probing | loading | live | failed | off. */
    @Volatile var gpu = if (cpuOnly) "off" else "probing"
    /** VOICEFAST: the voice trunk, built on the first voice turn (max_tokens <= 64). A failed load falls back to the main engine. */
    @Volatile private var voiceEng: Engine? = null
    @Volatile private var voiceEngTried = false
    private fun voiceEngine(): Engine? {
        val f = voiceEngineFactory ?: return null
        if (!voiceEngTried) { voiceEngTried = true; voiceEng = try { f() } catch (e: Exception) { engine.log("voice engine build failed: ${e.message}"); null } }
        return voiceEng
    }
    /** PHONEFAST: false = the GPU path opens the LLM only (no WGSL tower; photos answered text-only). Intent extra gpu_vision. */
    @Volatile var gpuVision = true
    private val gpuOn get() = gpu == "probing" || gpu == "loading" || gpu == "live"
    private val socket = ServerSocket(0, 16, InetAddress.getByName("127.0.0.1"))
    val port get() = socket.localPort
    private val model get() = models.active.name

    fun start() = thread(name = "bend-http", isDaemon = true) {
        while (!socket.isClosed) {
            val s = try { socket.accept() } catch (e: Exception) { break }
            thread(isDaemon = true) { s.use { handle(it) } }
        }
    }

    fun close() = socket.close()

    private fun handle(s: Socket) {
        val inp = BufferedInputStream(s.getInputStream())
        val head = readHead(inp) ?: return
        val lines = head.split("\r\n")
        val (method, target) = lines[0].split(" ").let { it[0] to it.getOrElse(1) { "/" } }
        val headers = lines.drop(1).filter { ':' in it }
            .associate { it.substringBefore(':').trim().lowercase() to it.substringAfter(':').trim() }
        val body = ByteArray(headers["content-length"]?.toIntOrNull() ?: 0).also { var n = 0; while (n < it.size) { val r = inp.read(it, n, it.size - n); if (r < 0) break; n += r } }
        val path = target.substringBefore('?')
        val query = target.substringAfter('?', "")
        val out = s.getOutputStream()
        try {
            if (method == "GET" && path == "/voice/status") return send(out, 200, json(voice.status()))
            if (method == "GET" && path == "/voices") return send(out, 200, json(voices()))
            if (method == "GET" && path == "/voices/shop") return send(out, 200, json(voice.voices.shop()))
            if (method == "POST" && headers["x-api-key"] != key) return send(out, 401, json(err("bad key")))
            when {
                method == "GET" && path == "/engines/status" -> send(out, 200, json(status()))
                method == "POST" && path == "/engines/ensure" -> { if (!gpuOn) engine.ensure(); send(out, 200, json(status())) }
                method == "POST" && path == "/gpu/state" -> { val r = JSONObject(String(body)); gpu = r.optString("state"); engine.log("gpu state=$gpu ${r.optString("reason")}"); onGpu(gpu, r.optString("reason")); send(out, 200, json(status())) }
                method == "POST" && path == "/gpu/log" -> { engine.log("gpu ${JSONObject(String(body)).optString("line")}"); send(out, 200, "{}".toByteArray()) }
                method == "GET" && path.startsWith("/data/") -> data(out, path.removePrefix("/data/"), headers["range"])
                method == "POST" && path == "/engines/visibility" -> { engine.log("visibility $query"); send(out, 200, json(status())) }
                method == "POST" && path.startsWith("/voices/") -> voicesOp(out, path.removePrefix("/voices/"), JSONObject(String(body).ifBlank { "{}" }))
                method == "POST" && path.startsWith("/voice/") -> voice(out, path.removePrefix("/voice/"), JSONObject(String(body).ifBlank { "{}" }))
                method == "POST" && path == "/v1/chat/completions" -> chat(out, JSONObject(String(body)))
                method == "POST" && path == "/v1/systemone" -> decide(out, String(body))
                method == "POST" && path == "/v1/see" -> see(out, JSONObject(String(body).ifBlank { "{}" }))
                method == "POST" && path == "/v1/speak" -> speak(out, JSONObject(String(body).ifBlank { "{}" }))
                method == "POST" && path == "/v1/hear" -> hear(out, JSONObject(String(body).ifBlank { "{}" }))
                method == "GET" -> file(out, if (path == "/") "/index.html" else path)
                else -> send(out, 404, json(err("not found")))
            }
        } catch (e: Exception) {
            engine.log("http $method $path failed: $e")
            try { send(out, 500, json(err(e.message ?: e.toString()))) } catch (_: Exception) {}
        }
    }

    /** phoneone's voice protocol (src/voice/types.ts); a refusal is a 409 with the reason, like the iOS shell. */
    private fun voice(out: OutputStream, op: String, req: JSONObject) {
        val ok = JSONObject().put("ok", true)
        try {
            val reply = when (op) {
                "prepare" -> voice.prepare()
                "install" -> JSONObject().put("assets", voice.status().getString("assets"))
                "listen" -> JSONObject().put("turn", voice.listen(req.optBoolean("auto"), req.optInt("silenceMs", 700)))
                "endpoint" -> { voice.endpoint(); ok }
                "release" -> voice.release()
                "speak" -> { voice.speak(req.optString("text")); ok }
                "finish" -> voice.finish()
                "cancel" -> { voice.cancel(); ok }
                "log" -> { engine.log("voice web ${req.optString("line")}"); ok }
                else -> return send(out, 404, json(err("no route POST /voice/$op")))
            }
            send(out, 200, json(reply))
        } catch (e: IllegalStateException) {
            engine.log("voice $op refused: ${e.message}")
            send(out, 409, json(err(e.message ?: "voice $op failed")))
        }
    }

    /** KOKORODROID: the picker's list (voices.html). */
    private fun voices(): JSONObject {
        val v = voice.voices
        return JSONObject().put("choice", v.choice).put("current", v.current()?.id ?: JSONObject.NULL).put("default", Voices.DEFAULT_ID)
            .put("demo", Voices.DEMO_LINE).put("dropIn", v.userDir.path).put("threads", v.threads).put("engine", v.available)
            .put("voices", JSONArray(v.packs(refresh = true).map { it.json() }))
    }

    /** The picker + shop (voices.html); g2p is the parity receipt against iOS (voice/g2p-golden-ios.tsv, voice/test-voices.py). */
    private fun voicesOp(out: OutputStream, op: String, req: JSONObject) {
        val v = voice.voices
        val reply = when (op) {
            "select" -> { val id = req.getString("id"); if (v.pack(id) == null) return send(out, 404, json(err("no voice $id"))); v.choice = id; engine.log("voices select $id"); voices() }
            "preview" -> voice.preview(req.optString("id", v.choice), req.optString("text").ifBlank { Voices.DEMO_LINE }, req.optBoolean("play", true))
            "threads" -> { v.threads = req.getInt("threads"); voices() }
            "g2p" -> v.g2p(req.getString("text"), req.optBoolean("british"))
            "shop/load" -> v.load()
            "shop/download" -> { v.download(req.getString("id")); v.shop() }
            "shop/pause" -> { v.pause(req.getString("id")); v.shop() }
            "shop/remove" -> { v.remove(req.getString("id")); v.shop() }
            "shop/source" -> { v.catalogUrl = req.optString("url"); v.load() }
            else -> return send(out, 404, json(err("no route POST /voices/$op")))
        }
        send(out, 200, json(reply))
    }

    private fun readHead(inp: BufferedInputStream): String? {
        val b = ByteArrayOutputStream()
        while (true) {
            val c = inp.read()
            if (c < 0) return null
            b.write(c)
            if (b.size() >= 4 && b.toString(Charsets.ISO_8859_1.name()).endsWith("\r\n\r\n")) return b.toString(Charsets.ISO_8859_1.name())
        }
    }

    /** decide runs only on the Kev-4B pack (kev.c's head math is 2560-wide); other packs report off, never live. */
    private fun decisionOf(state: JSONObject): JSONObject =
        if (models.active.decide) JSONObject(state.toString())
        else JSONObject().put("state", "off").put("model", model).put("reason", "decide needs the Kev-4B pack")

    private fun status(): JSONObject {
        if (gpu == "live" || gpu == "loading") {
            val g = JSONObject().put("state", if (gpu == "live") "live" else "loading").put("model", model)
            return JSONObject().put("port", port).put("backend", "bend-webgpu (q35 turbo + WGSL tower, $model)")
                .put("warm", gpu == "live").put("active", if (gpu == "live") "chat" else JSONObject.NULL)
                .put("chat", g).put("decision", decisionOf(g)).put("footprintMB", 0).put("decisionIdleSeconds", 0)
                .put("verbs", verbs())
        }
        val e = JSONObject().put("state", engine.state).put("model", model)
        engine.loadSeconds?.let { e.put("loadSeconds", it) }
        engine.error?.let { e.put("error", it) }
        engine.rss()["VmRSS"]?.let { e.put("footprintMB", it) }
        return JSONObject().put("port", port).put("backend", "bend-engine (qwen35 CPU, ${models.active.pack})")
            .put("warm", engine.state == "live").put("active", if (engine.state == "live") "chat" else JSONObject.NULL)
            .put("chat", e).put("decision", decisionOf(e))
            .put("footprintMB", engine.rss()["VmRSS"] ?: 0).put("decisionIdleSeconds", 0)
            .put("vision", JSONObject().put("available", visionReason() == null).put("model", "qwen3vl_merger (mmproj-F16)")
                .put("canvas", "aspect-preserving, up to 256 image tokens").put("reason", visionReason() ?: JSONObject.NULL))
            .put("verbs", verbs())
    }

    /** Why a photo can't be looked at right now (null = it can). */
    private fun visionReason(): String? =
        if (!models.active.vision) "photos need the Kev-4B pack (the vision tower is the 4B mmproj); the active model is $model"
        else if (!engine.mmproj.exists()) "mmproj-F16.gguf is not on this phone (expected ${engine.mmproj.path}); push it to enable photos"
        else if (engine.mmproj.length() < 600_000_000L) "mmproj-F16.gguf is incomplete (${engine.mmproj.length() / 1_000_000} MB of ~641 MB); re-push the file"
        else null

    /** ChatML, no-think (Qwen3.5 template with enable_thinking=false). Markers in content are defused. */
    private fun prompt(messages: JSONArray): String {
        val b = StringBuilder()
        var lastImg = -1 // one photo per turn: the engine looks at the newest, older ones are dropped
        for (i in 0 until messages.length()) if (imageOf(messages.getJSONObject(i)) != null) lastImg = i
        for (i in 0 until messages.length()) {
            val m = messages.getJSONObject(i)
            val c = m.opt("content")
            var text = if (c is JSONArray) (0 until c.length()).mapNotNull { c.optJSONObject(it)?.takeIf { p -> p.optString("type") == "text" }?.optString("text") }.joinToString("\n") else m.optString("content")
            if (i == lastImg) text = "<|vision_start|><|image_pad|><|vision_end|>" + text.ifBlank { "Describe this photo in one or two sentences." }
            val content = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "\u0001").replace("<|", "<¦").replace("|>", "¦>").replace("\u0001", "<|vision_start|><|image_pad|><|vision_end|>")
            b.append("<|im_start|>").append(m.optString("role", "user")).append('\n').append(content).append("<|im_end|>\n")
        }
        return b.append("<|im_start|>assistant\n<think>\n\n</think>\n\n").toString()
    }

    private fun imageOf(m: JSONObject): String? {
        val c = m.opt("content") as? JSONArray ?: return null
        for (k in 0 until c.length()) {
            val p = c.optJSONObject(k) ?: continue
            if (p.optString("type") == "image_url") return p.optJSONObject("image_url")?.optString("url") ?: p.optString("image_url")
        }
        return null
    }

    private fun chat(out: OutputStream, req: JSONObject) {
        val messages = req.getJSONArray("messages")
        val url = (messages.length() - 1 downTo 0).firstNotNullOfOrNull { imageOf(messages.getJSONObject(it)) }
        var img: File? = null
        if (url != null) {
            visionReason()?.let { return send(out, 503, json(err("no-vision: $it"))) }
            if (!url.startsWith("data:") || ',' !in url) return send(out, 400, json(err("no-vision: the photo must be a data: URL")))
            img = File(File(engine.dataDir, "vis").apply { mkdirs() }, "img-${System.currentTimeMillis()}.bin")
            try { img.writeBytes(Base64.decode(url.substringAfter(','), Base64.DEFAULT)) } catch (e: Exception) { return send(out, 400, json(err("no-vision: the photo could not be decoded (${e.message})"))) }
        }
        try { chatTurn(out, req, messages, img) } finally { img?.delete() }
    }

    /** VOICEFAST: the last user utterance as text (a content array joins its text parts). */
    private fun lastUserText(messages: JSONArray): String {
        for (i in messages.length() - 1 downTo 0) {
            val m = messages.optJSONObject(i) ?: continue
            if (m.optString("role") != "user") continue
            val c = m.opt("content") ?: continue
            if (c is String) return c
            if (c is JSONArray) return (0 until c.length()).mapNotNull { j ->
                val p = c.optJSONObject(j) ?: return@mapNotNull null
                if (p.optString("type") == "text") p.optString("text") else null
            }.joinToString("\n")
        }
        return ""
    }

    /** VOICEFAST: instant spoken intents — no engine, no 3 s. Deliberately tiny: time, hello, thanks, stop. */
    private fun fastAnswer(raw: String): String? {
        val t = raw.trim().lowercase(Locale.ROOT).trim('.', '!', '?', ' ')
        if (t.isEmpty() || t.length > 48) return null
        if (Regex("^(what('s| is)? the time|what time is it|time is it|got the time)$").matches(t))
            return "It's " + SimpleDateFormat("h:mm a", Locale.US).format(Date()) + "."
        if (Regex("^(hi|hello|hey|yo|good (morning|afternoon|evening)|how are you|how's it going)( there)?$").matches(t)) {
            val h = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
            return when { h < 12 -> "Morning."; h < 18 -> "Afternoon."; else -> "Evening." }
        }
        if (Regex("^(thanks|thank you|thx|ty|appreciate it|cheers)( so much| a lot)?$").matches(t)) return "Any time."
        if (Regex("^(stop|quiet|cancel|never ?mind|shut up|enough)$").matches(t)) return "Okay."
        return null
    }

    /** VOICEFAST: answer in the shape the caller asked for (SSE frames or one JSON body). */
    private fun sendText(out: OutputStream, stream: Boolean, text: String) {
        if (!stream) return send(out, 200, json(JSONObject().put("choices", JSONArray().put(JSONObject().put("message", JSONObject().put("role", "assistant").put("content", text))))))
        out.write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n".toByteArray()); out.flush()
        out.write("data: ${JSONObject().put("choices", JSONArray().put(JSONObject().put("delta", JSONObject().put("content", text))))}\n\n".toByteArray())
        out.write("data: [DONE]\n\n".toByteArray()); out.flush()
    }

    private fun chatTurn(out: OutputStream, req: JSONObject, messages: JSONArray, img: File?) {
        if (gpuOn) return send(out, 503, json(err("the on-device GPU engine is still loading ($gpu); try again in a moment")))
        // VOICEFAST: the voice client asks for 64 tokens; that turn goes to the voice trunk when one is configured,
        // and falls back to the main engine if it cannot load (memory, missing pack) — the answer is then just slower.
        val small = req.optInt("max_tokens", 256) <= 64 && img == null
        if (small && voiceFast) { // VOICEFAST: trivial intents answer instantly, before any engine work
            val said = lastUserText(messages)
            fastAnswer(said)?.let { answer ->
                engine.log("fastpath \"${said.take(40)}\" -> ${answer.take(40)}")
                return sendText(out, req.optBoolean("stream"), answer)
            }
        }
        val eng = (if (small) voiceEngine() else null) ?: engine
        val active = if (eng === engine || eng.ensure()) eng else engine.also { engine.log("voice engine unavailable (${eng.error}); main engine serves the turn") }
        if (!active.ensure()) return send(out, 503, json(err(active.error ?: "engine not live")))
        // ponytail: n capped at 256 (KV ctx 2048, ~0.2 s/token); raise with a longer ctx. Temperature ignored: greedy.
        // VOICEFAST: the voice profile caps replies at 48 tokens; a spoken answer is a sentence or two, not an essay.
        val cap = if (voiceFast) 24 else 256
        val n = minOf(req.optInt("max_tokens", cap), cap)
        val line = JSONObject().put("mode", "generate").put("prompt", prompt(messages)).put("n", n).also { if (img != null) it.put("image", img.path) }.toString()
        if (active !== engine) engine.log("voice turn on ${active.model.name}")
        val t0 = System.nanoTime()
        if (!req.optBoolean("stream")) {
            val reply = JSONObject(active.request(line))
            if (reply.has("error")) return send(out, 500, json(err(reply.get("error").toString())))
            val text = reply.optString("text")
            engine.log("chat vision=${active.lastVision} run=${active.lastRun} engine=${active.model.name}")
            engine.log("chat n=${reply.optJSONArray("ids")?.length()} ms=${(System.nanoTime() - t0) / 1_000_000} stream=0")
            return send(out, 200, json(JSONObject().put("choices", JSONArray().put(JSONObject().put("message", JSONObject().put("role", "assistant").put("content", text))))))
        }
        out.write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n".toByteArray())
        out.flush()
        val pending = ByteArrayOutputStream()
        var first = 0L
        var gone = false
        val frame = { text: String ->
            if (!gone) try {
                val d = JSONObject().put("choices", JSONArray().put(JSONObject().put("delta", JSONObject().put("content", text))))
                out.write("data: $d\n\n".toByteArray()); out.flush()
            } catch (e: Exception) { gone = true } // the page left; the engine finishes this generate anyway
        }
        val reply = JSONObject(active.request(line) { bytes ->
            if (first == 0L) first = System.nanoTime()
            pending.write(bytes)
            val all = pending.toByteArray()
            val cut = completeUtf8(all)
            if (cut > 0) { frame(String(all, 0, cut, Charsets.UTF_8)); pending.reset(); pending.write(all, cut, all.size - cut) }
        })
        if (reply.has("error") && !gone) {
            engine.log("chat error ${reply.get("error")}")
            try { out.write("data: ${err(reply.get("error").toString())}\n\n".toByteArray()) } catch (_: Exception) {}
        }
        val ms = (System.nanoTime() - t0) / 1_000_000
        val ttft = if (first > 0) (first - t0) / 1_000_000 else -1
        engine.log("chat n=${reply.optJSONArray("ids")?.length()} ms=$ms ttft_ms=$ttft stream=1 rss=${engine.rss()}")
        if (img != null && !gone && !reply.has("error")) try {
            val v = engine.lastVision?.let { JSONObject(it).optJSONObject("vision") }
            val run = engine.lastRun?.let { JSONObject(it).optJSONObject("kevcpu") }
            val tower = v?.optDouble("tower_ms") ?: 0.0
            val llb = JSONObject().put("look", true).put("image_tokens", v?.optInt("tokens") ?: 0).put("canvas", v?.optJSONArray("grid") ?: JSONObject.NULL)
                .put("first_ms", ttft).put("total_ms", ms).put("tower_ms_sum", tower).put("llm_ms_sum", run?.optLong("run_ms") ?: 0)
            out.write("data: ${JSONObject().put("llb", llb)}\n\n".toByteArray())
        } catch (_: Exception) {}
        if (!gone) try { out.write("data: [DONE]\n\n".toByteArray()); out.flush() } catch (_: Exception) {}
    }

    /** The body goes to kev.c verbatim: option order is the criteria's text order (transport.ts systemoneBody). */
    private fun decide(out: OutputStream, body: String) {
        // PHONEFAST: decide is kev.c (CPU) only; starting it next to the resident GPU weights gets the app killed by lmkd.
        if (gpuOn) return send(out, 503, json(err("decide runs on the CPU engine, which is off while the GPU engine holds the model")))
        // MODEL SELECT: kev.c's decide head math is 2560-wide (Kev-4B); other packs serve chat only.
        if (!models.active.decide) return send(out, 503, json(err("decide needs the Kev-4B pack; the active model is $model")))
        if (!engine.ensure()) return send(out, 503, json(err(engine.error ?: "engine not live")))
        val t0 = System.nanoTime()
        val reply = engine.request(body)
        val ms = (System.nanoTime() - t0) / 1_000_000
        engine.log("decide ms=$ms body=$body reply=$reply")
        send(out, if (JSONObject(reply).has("error")) 500 else 200, reply.toByteArray())
    }

    /**
     * SEE: D-FINE-N through the onnxruntime AAR the app already ships (CPU EP, no child process).
     * Request {"image"|"image_b64": raw base64 or data: URL}. Boxes are xyxy in original image pixels.
     */
    private fun see(out: OutputStream, req: JSONObject) {
        val s = req.optString("image").ifBlank { req.optString("image_b64") }
        if (s.isBlank()) return send(out, 400, json(err("image must be a base64 image string (image or image_b64)")))
        val b64 = if (s.startsWith("data:")) s.substringAfter(',', "") else s
        val bytes = try { Base64.decode(b64, Base64.DEFAULT) } catch (e: Exception) { return send(out, 400, json(err("image is not valid base64"))) }
        if (bytes.isEmpty()) return send(out, 400, json(err("image is empty")))
        if (bytes.size > 8 shl 20) return send(out, 400, json(err("image is ${bytes.size} bytes; the route takes up to 8 MiB")))
        val t = System.nanoTime()
        val r = try { detector.run(bytes) } catch (e: IllegalArgumentException) { return send(out, 400, json(err(e.message ?: "$e"))) }
            catch (e: Exception) { engine.log("see failed: $e"); return send(out, 503, json(err("see failed: ${e.message}"))) }
        engine.log("see dets=${r.optJSONArray("detections")?.length()} ms=${(System.nanoTime() - t) / 1_000_000} sha=${detector.sha.take(8)}")
        send(out, 200, json(r))
    }

    /** SPEAK: text -> 24 kHz mono PCM16 WAV (Kokoro af_heart by default), the SCHEMAS §5 shape + X-LLB-* headers. */
    private fun speak(out: OutputStream, req: JSONObject) {
        val text = req.optString("text")
        if (text.isBlank()) return send(out, 400, json(err("text is required")))
        val style = req.optString("style", "neutral")
        if (style != "neutral") return send(out, 400, json(err("style=$style is not implemented on device: the app renders the bank as trained (neutral)")))
        val sp = try { voice.speakWav(text, req.optString("voice").ifBlank { "af_heart" }) }
        catch (e: IllegalArgumentException) { return send(out, 400, json(err(e.message ?: "$e"))) }
        catch (e: Exception) { engine.log("speak failed: $e"); return send(out, 503, json(err("speak failed: ${e.message}"))) }
        out.write(("HTTP/1.1 200 X\r\nContent-Type: audio/wav\r\nContent-Length: ${sp.wav.size}\r\n" +
            "X-LLB-Voice: ${sp.voice}\r\nX-LLB-Sample-Rate: 24000\r\nX-LLB-G2P: misaki\r\nX-LLB-Style: neutral\r\n" +
            "X-LLB-Speak-Ms: ${sp.ms}\r\nX-LLB-Model-Sha256: ${sp.bankSha}\r\nX-LLB-Wav-Sha256: ${sp.wavSha}\r\n" +
            "Cache-Control: no-store\r\nConnection: close\r\n\r\n").toByteArray())
        out.write(sp.wav)
        out.flush()
        engine.log("speak bytes=${sp.wav.size} voice=${sp.voice} sha=${sp.wavSha.take(8)} ms=${sp.ms}")
    }

    /** HEAR: WAV bytes -> transcript (whisper.cpp on device; 16 kHz is native, other rates are resampled here). */
    private fun hear(out: OutputStream, req: JSONObject) {
        val s = req.optString("audio").ifBlank { req.optString("audio_b64") }
        if (s.isBlank()) return send(out, 400, json(err("audio must be a base64 WAV (audio or audio_b64)")))
        val b64 = if (s.startsWith("data:")) s.substringAfter(',', "") else s
        val bytes = try { Base64.decode(b64, Base64.DEFAULT) } catch (e: Exception) { return send(out, 400, json(err("audio is not valid base64"))) }
        if (bytes.isEmpty()) return send(out, 400, json(err("audio is empty")))
        if (bytes.size > 16 shl 20) return send(out, 400, json(err("audio is ${bytes.size} bytes; the route takes up to 16 MiB")))
        val h = try { voice.hearWav(bytes) } catch (e: IllegalArgumentException) { return send(out, 400, json(err(e.message ?: "$e"))) }
            catch (e: Exception) { engine.log("hear failed: $e"); return send(out, 503, json(err("hear failed: ${e.message}"))) }
        send(out, 200, json(JSONObject().put("model", h.model).put("text", h.text)
            .put("duration_s", "%.4f".format(h.durationS).toDouble())
            .put("llb", JSONObject().put("decode_ms", h.decodeMs).put("resample_ms", h.resampleMs)
                .put("runtime", "whisper.cpp (libbendwhisper.so, CPU)").put("model_sha256", h.modelSha))))
    }

    /** What this build serves, for /engines/status (a route that is listed here is wired; see M3-ROUTES.md). */
    private fun verbs() = JSONObject()
        .put("see", JSONObject().put("route", "POST /v1/see").put("available", detector.available).put("model", "dfine-n")
            .put("loaded", detector.loaded).put("sha256", detector.sha.takeIf { it.isNotEmpty() } ?: JSONObject.NULL))
        .put("speak", JSONObject().put("route", "POST /v1/speak").put("available", voice.voices.available)
            .put("model", "kokoro-v1.0.int8").put("voice", voice.voices.current()?.id ?: JSONObject.NULL))
        .put("hear", JSONObject().put("route", "POST /v1/hear").put("available", true).put("model", "whisper.cpp"))

    private fun file(out: OutputStream, path: String) {
        val name = "web" + path.replace("..", "")
        val bytes = try { assets.open(name).use { it.readBytes() } } catch (e: Exception) { return send(out, 404, json(err("no $path"))) }
        val type = when (path.substringAfterLast('.')) {
            "html" -> "text/html; charset=utf-8"; "js" -> "text/javascript"; "css" -> "text/css"; "svg" -> "image/svg+xml"
            "png" -> "image/png"; "webp" -> "image/webp"; "json", "webmanifest" -> "application/json"; "woff2" -> "font/woff2"; "ico" -> "image/x-icon"
            "wasm" -> "application/wasm"; "wgsl" -> "text/plain; charset=utf-8"
            else -> "application/octet-stream"
        }
        if (path == "/index.html" || path == "/voices.html") {
            val inject = "<script>window.__BEND_NATIVE__=${JSONObject().put("control", "http://127.0.0.1:$port")
                .put("engine", "http://127.0.0.1:$port").put("key", key).put("autorun", autorun).put("cpuOnly", !gpuOn).put("gpuVision", gpuVision)
                .put("model", model).put("models", JSONArray(models.packs.map { it.name }))};" +
                "window.__BEND_SHELL__=Object.freeze(${shell()});" +
                "window.__BEND_VOICE__=Object.freeze(${JSONObject().put("url", "http://127.0.0.1:$port").put("key", key).put("check", false)});" +
                // KOKORODROID: the voice screen's "speech on device · Kokoro Bella" line opens the picker (iOS: bridge voices.open).
                "addEventListener('click',e=>{const t=e.target.closest&&e.target.closest('*');if(t&&/speech on device/.test(t.textContent)&&t.textContent.length<80)location.href='/voices.html'},true)</script>"
            val shim = if (path == "/index.html" && gpuOn) "<script type=\"module\" src=\"/gpu-shim.js\"></script>" else ""
            return send(out, 200, String(bytes).replaceFirst("<head>", "<head>$inject$shim").toByteArray(), type)
        }
        send(out, 200, bytes, type)
    }

    /**
     * ANDROIDVOICE: what the iOS shell says at document start (ios/Bend/BootOverlay.swift ShellInfo, src/engines/native.ts
     * NativeShell): the surfaces this build has. Voice is real here (Voice.kt: whisper + Kokoro), so the switcher's Voice tab
     * opens it; Camera stays off: the in-app camera is the iOS shell's (bridge camera.open); Android's camera is the
     * composer's photo picker. No setup command: the engine files arrive by the model pack, not a Mac.
     */
    private fun shell(): JSONObject {
        val missing = engine.missing()
        val start = System.currentTimeMillis() - (android.os.SystemClock.elapsedRealtime() - android.os.Process.getStartElapsedRealtime())
        return JSONObject().put("platform", "device").put("engines", "on")
            .put("setup", JSONObject().put("chat", missing == null).put("decision", missing == null && models.active.decide)
                .put("missing", JSONArray().apply { missing?.let { put(it) } }).put("dir", engine.dataDir.path).put("command", ""))
            .put("model", model).put("models", JSONArray(models.packs.map { it.name }))
            .put("surfaces", JSONObject().put("camera", false).put("photo", false).put("voice", true))
            .put("processStart", start)
    }

    /** PHONEFAST: the GPU path's weights, read in place (Range) from the app's files dir. Only these names. */
    private fun data(out: OutputStream, name: String, range: String?) {
        if (name !in setOf(models.active.gguf, "mmproj-F16.gguf", "qwen35-tokenizer.bin")) return send(out, 404, json(err("no /data/$name")))
        val f = File(engine.dataDir, name)
        if (!f.exists()) return send(out, 404, json(err("$name is not on this phone (${f.path})")))
        val size = f.length()
        val m = range?.let { Regex("bytes=(\\d+)-(\\d*)").find(it) }
        val start = m?.groupValues?.get(1)?.toLong() ?: 0L
        val end = minOf(m?.groupValues?.get(2)?.takeIf { it.isNotEmpty() }?.toLong() ?: (size - 1), size - 1)
        if (start > end) return send(out, 416, ByteArray(0))
        val len = end - start + 1
        out.write(("HTTP/1.1 ${if (m != null) "206" else "200"} X\r\nContent-Type: application/octet-stream\r\nContent-Length: $len\r\n" +
            (if (m != null) "Content-Range: bytes $start-$end/$size\r\n" else "") + "Accept-Ranges: bytes\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n").toByteArray())
        java.io.RandomAccessFile(f, "r").use { r ->
            r.seek(start); val buf = ByteArray(1 shl 20); var left = len
            while (left > 0) { val k = r.read(buf, 0, minOf(buf.size.toLong(), left).toInt()); if (k < 0) break; out.write(buf, 0, k); left -= k }
        }
        out.flush()
    }

    private fun err(m: String) = JSONObject().put("error", JSONObject().put("message", m))
    private fun json(o: JSONObject) = o.toString().toByteArray()

    private fun send(out: OutputStream, code: Int, body: ByteArray, type: String = "application/json") {
        out.write("HTTP/1.1 $code X\r\nContent-Type: $type\r\nContent-Length: ${body.size}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n".toByteArray())
        out.write(body); out.flush()
    }

    companion object {
        /** Length of the longest prefix of `b` that ends on a complete UTF-8 sequence. */
        fun completeUtf8(b: ByteArray): Int {
            for (back in 1..minOf(3, b.size)) {
                val c = b[b.size - back].toInt() and 0xff
                if (c and 0xc0 == 0x80) continue // continuation byte: keep looking for the lead
                val need = when { c >= 0xf0 -> 4; c >= 0xe0 -> 3; c >= 0xc0 -> 2; else -> 1 }
                return if (need > back) b.size - back else b.size
            }
            return b.size
        }
    }
}
