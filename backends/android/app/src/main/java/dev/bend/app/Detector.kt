package dev.bend.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.floor

/**
 * SEE: D-FINE-N through the onnxruntime-android AAR this app already ships (Kokoro's runtime).
 *
 * Model: dfine_n_coco.onnx (15,534,295 B, sha256 4daf426f…): sigmoid -> top-300 -> rescale by
 * orig_target_sizes are baked into the graph, so the graph's boxes are already xyxy in ORIGINAL pixels.
 * Inputs  images float32 [1,3,640,640], orig_target_sizes int64 [1,2] = (w,h).
 * Outputs labels int64 [1,Q], boxes float32 [1,Q,4], scores float32 [1,Q].
 *
 * Preprocessing is the measured contract (cuda-5of5/C2-SEE-PREP, the reference that produced
 * dfine_n_ref.json): RGB, plain stretch to 640x640, bilinear with PIL's half-pixel filter, /255, NCHW.
 * The resampler below reproduces PIL's ImagingResample (two passes, 8-bit intermediate) so the phone's
 * pixels match the reference's, not just "some bilinear".
 *
 * Not thread-safe: one ORT session, one request at a time ([run] is synchronized).
 */
class Detector(private val assets: AssetManager, private val data: File, private val log: (String) -> Unit) {
    /** The pushed export: files/see/dfine_n_coco.onnx, else files/dfine_n_coco.onnx (where M0/M2 staged it). */
    private val model = listOf(File(File(data, "see"), MODEL), File(data, MODEL)).firstOrNull { it.exists() } ?: File(File(data, "see"), MODEL)
    private val labels: List<String> = loadLabels()
    private var session: OrtSession? = null
    private val env = OrtEnvironment.getEnvironment()

    @Volatile var sha = ""
        private set
    @Volatile var loadMs = -1L
        private set
    @Volatile var loaded = false
        private set

    val available get() = model.exists() && model.length() == MODEL_BYTES

    private fun loadLabels(): List<String> = runCatching {
        val a = org.json.JSONArray(assets.open("see/coco80.json").reader().use { it.readText() })
        (0 until a.length()).map { a.getString(it) }
    }.getOrElse { COCO }

    @Synchronized private fun ensure(): OrtSession {
        session?.let { return it }
        if (!model.exists()) throw IllegalStateException(
            "see needs ${MODEL} at ${model.path} (push the 15.5 MB 4daf426f… export)")
        if (model.length() != MODEL_BYTES) throw IllegalStateException(
            "${model.path} is ${model.length()} bytes; the D-FINE-N fixture is $MODEL_BYTES — re-push it")
        val t = System.currentTimeMillis()
        val digest = MessageDigest.getInstance("SHA-256")
        model.inputStream().use { i -> val b = ByteArray(1 shl 20); while (true) { val n = i.read(b); if (n < 0) break; digest.update(b, 0, n) } }
        sha = digest.digest().joinToString("") { "%02x".format(it) }
        val s = env.createSession(model.path, OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4) // the 6a's 2×X1 + 2×A76; the detector is one-shot conv/DETR work
            setInterOpNumThreads(1)
        })
        loadMs = System.currentTimeMillis() - t
        session = s
        loaded = true
        log("see dfine-n ready load_ms=$loadMs ep=CPU sha=${sha.take(8)}… bytes=${model.length()}")
        return s
    }

    /** Image bytes (JPEG/PNG/WebP) -> the route's JSON: detections in original pixels, threshold 0.5. */
    @Synchronized fun run(bytes: ByteArray, threshold: Double = 0.5, maxDet: Int = 100): JSONObject {
        val t0 = System.nanoTime()
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            ?: throw IllegalArgumentException("the image could not be decoded (JPEG/PNG/WebP bytes expected)")
        val w = bmp.width
        val h = bmp.height
        if (w <= 0 || h <= 0) throw IllegalArgumentException("the image has no pixels")
        val x = prepare(bmp)
        bmp.recycle()
        val t1 = System.nanoTime()
        val s = ensure()
        val out = JSONObject()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 3, IN.toLong(), IN.toLong())).use { img ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(w.toLong(), h.toLong())), longArrayOf(1, 2)).use { sz ->
                s.run(mapOf("images" to img, "orig_target_sizes" to sz)).use { r ->
                    val byName = HashMap<String, OnnxTensor>()
                    for (e in r) byName[e.key] = e.value as OnnxTensor
                    val lab = byName["labels"]?.longBuffer ?: throw IllegalStateException("model returned no labels")
                    val box = byName["boxes"]?.floatBuffer ?: throw IllegalStateException("model returned no boxes")
                    val sco = byName["scores"]?.floatBuffer ?: throw IllegalStateException("model returned no scores")
                    val dets = ArrayList<Det>()
                    val q = minOf(lab.capacity(), sco.capacity(), box.capacity() / 4)
                    for (i in 0 until q) {
                        val sc = sco.get(i)
                        if (sc.toDouble() < threshold) continue
                        val ci = lab.get(i).toInt()
                        dets.add(Det(ci, sc.toDouble(), floatArrayOf(box.get(i * 4), box.get(i * 4 + 1), box.get(i * 4 + 2), box.get(i * 4 + 3))))
                    }
                    dets.sortByDescending { it.score }
                    val arr = JSONArray()
                    for (d in dets.take(maxDet)) {
                        arr.put(JSONObject().put("box", JSONArray().apply { for (v in d.box) put(round(v.toDouble(), 1)) })
                            .put("label", labels.getOrElse(d.cls) { d.cls.toString() })
                            .put("class_id", d.cls)
                            .put("score", round(d.score, 4)))
                    }
                    out.put("model", "dfine-n").put("task", "detect").put("image_wh", JSONArray().put(w).put(h))
                        .put("detections", arr).put("usage", JSONObject().put("images", 1))
                        .put("llb", JSONObject().put("preprocess_ms", round((t1 - t0) / 1e6, 2))
                            .put("infer_ms", round((System.nanoTime() - t1) / 1e6, 2))
                            .put("ep", "CPUExecutionProvider")
                            .put("runtime", "onnxruntime 1.24.2 (app AAR)")
                            .put("model_sha256", sha).put("threshold", threshold))
                }
            }
        }
        return out
    }

    private class Det(val cls: Int, val score: Double, val box: FloatArray)

    /** RGB, stretch 640x640, PIL bilinear, /255, NCHW float32. */
    private fun prepare(bmp: Bitmap): FloatArray {
        val sw = bmp.width
        val sh = bmp.height
        val px = IntArray(sw * sh)
        bmp.getPixels(px, 0, sw, 0, 0, sw, sh)
        val hx = taps(sw)
        val vy = taps(sh)
        // horizontal pass into an 8-bit plane, exactly as PIL's 8bpc resample does between passes
        val tmp = ByteArray(IN * sh * 3)
        for (y in 0 until sh) {
            val row = y * sw
            for (x in 0 until IN) {
                val t = hx[x]
                var r = 0f; var g = 0f; var b = 0f
                for (k in t.w.indices) {
                    val p = px[row + t.i + k]
                    val w = t.w[k]
                    r += ((p shr 16) and 0xff) * w
                    g += ((p shr 8) and 0xff) * w
                    b += (p and 0xff) * w
                }
                val o = (y * IN + x) * 3
                tmp[o] = clamp8(r); tmp[o + 1] = clamp8(g); tmp[o + 2] = clamp8(b)
            }
        }
        val hw = IN * IN
        val out = FloatArray(3 * hw)
        for (y in 0 until IN) {
            val t = vy[y]
            for (x in 0 until IN) {
                var r = 0f; var g = 0f; var b = 0f
                for (k in t.w.indices) {
                    val o = ((t.i + k) * IN + x) * 3
                    val w = t.w[k]
                    r += (tmp[o].toInt() and 0xff) * w
                    g += (tmp[o + 1].toInt() and 0xff) * w
                    b += (tmp[o + 2].toInt() and 0xff) * w
                }
                val i = y * IN + x
                out[i] = r / 255f
                out[hw + i] = g / 255f
                out[2 * hw + i] = b / 255f
            }
        }
        return out
    }

    private class Taps(val i: Int, val w: FloatArray)

    /** PIL ImagingResample coefficients: center=(d+0.5)*scale, support=1*filterscale, weight=1-|(s+0.5-center)/filterscale|. */
    private fun taps(src: Int): Array<Taps> {
        val scale = src.toDouble() / IN
        val filterscale = if (scale < 1.0) 1.0 else scale
        val support = filterscale
        return Array(IN) { d ->
            val center = (d + 0.5) * scale
            val xmin = floor(center - support + 0.5).toInt().coerceIn(0, src - 1)
            val xmax = floor(center + support + 0.5).toInt().coerceIn(xmin + 1, src)
            val n = xmax - xmin
            val w = FloatArray(n)
            var sum = 0.0
            for (i in 0 until n) {
                val v = 1.0 - abs((xmin + i + 0.5 - center) / filterscale)
                val ww = if (v < 0) 0.0 else v
                w[i] = ww.toFloat()
                sum += ww
            }
            if (sum <= 0.0) { w[0] = 1f } else for (i in 0 until n) w[i] = (w[i] / sum).toFloat()
            Taps(xmin, w)
        }
    }

    private fun clamp8(v: Float): Byte = (v + 0.5f).toInt().coerceIn(0, 255).toByte()

    private fun round(v: Double, dp: Int): Double {
        var m = 1.0
        repeat(dp) { m *= 10.0 }
        return kotlin.math.round(v * m) / m
    }

    companion object {
        const val IN = 640
        const val MODEL = "dfine_n_coco.onnx"
        const val MODEL_BYTES = 15_534_295L

        /** COCO-80 in class-id order (assets/see/coco80.json is the same list; this is the fallback). */
        val COCO = listOf(
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush")
    }
}
