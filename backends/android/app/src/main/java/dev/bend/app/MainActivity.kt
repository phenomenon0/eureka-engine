package dev.bend.app

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.webkit.ConsoleMessage
import android.webkit.PermissionRequest
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import java.io.File
import java.security.SecureRandom
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

/** One WebView on the bundled phoneone dist, served same-origin by [Server]; the engine warms at launch. */
class MainActivity : Activity() {
    private lateinit var engine: Engine
    private lateinit var server: Server
    private lateinit var voice: Voice
    private var asking: CountDownLatch? = null
    private var granted = false
    private var chooser: ValueCallback<Array<Uri>>? = null
    private var shot: File? = null
    private lateinit var web: WebView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val data = getExternalFilesDir(null)!!
        val models = Models(data) // MODEL SELECT: files/models.json; absent = the Kev-4B pack
        engine = Engine("${applicationInfo.nativeLibraryDir}/libbendengine.so", data, File(cacheDir, "engine"), File(data, "engine.log"), models.active)
        val key = intent.getStringExtra("test_key") ?: ByteArray(16).also { SecureRandom().nextBytes(it) }.joinToString("") { "%02x".format(it) } // test_key: automation only
        val voices = Voices(this, engine::log)
        // SEE: D-FINE-N through the same ORT runtime (the AAR Kokoro already links); the model arrives by adb push.
        val detector = Detector(assets, data, engine::log)
        if (intent.hasExtra("kokoro_threads")) voices.threads = intent.getIntExtra("kokoro_threads", 4)
        voice = Voice(this, engine::log, { ensure(Manifest.permission.RECORD_AUDIO) }, voices, intent.getBooleanExtra("keep_audio", false))
        voice.ttsMode = intent.getStringExtra("tts") ?: "kokoro" // TTSMODE: kokoro = neural (default), fast = espeak-ng CLI (optional, not shipped: GPL-3.0)
        engine.log("tts mode=${voice.ttsMode}")
        // PHONEFAST: GPU first. The WebView's gpu-shim.js opens the WebGPU model path; the CPU engine (2.6 GB resident) starts
        // only when that fails, never next to it (both together = lmkd kills the app). cpu_only extra / a renderer crash marker = CPU.
        val crashed = File(data, "gpu-crashed")
        // PHONEFAST measured (6a): WebGPU 2.6-3.2 tok/s vs CPU ~5, ~260 MB headroom, no room for the tower -> GPU is opt-in (extra gpu=true)
        val cpuOnly = intent.getBooleanExtra("cpu_only", !intent.getBooleanExtra("gpu", false)) || crashed.exists()
        if (crashed.exists()) crashed.delete() // one CPU launch after a crash, then GPU is tried again
        val cpuWarm = { thread(name = "engine-warm", isDaemon = true) { engine.ensure() } }
        // VOICEFAST: voice turns (max_tokens <= 64) go to a small trunk loaded lazily beside the main one.
        // voice_model picks it (default Pocket 0.8B); absent on disk or equal to the active model = single engine.
        val voiceFast = intent.getBooleanExtra("voice_fast", true) // default ON: the phone's voice mode is the point
        val voiceModelName = intent.getStringExtra("voice_model") ?: "Qwen3.5-0.8B-Q4_K_M"
        val voicePack = models.packs.firstOrNull { it.name == voiceModelName && it.name != models.active.name }
        val voiceFactory: (() -> Engine)? = if (voiceFast && voicePack != null) {
            { Engine("${applicationInfo.nativeLibraryDir}/libbendengine.so", data, File(cacheDir, "engine-voice"), File(data, "engine.log"), voicePack) }
        } else null
        if (voiceFast) engine.log("voicefast on; voice trunk=${voicePack?.name ?: "none (single engine)"}")
        server = Server(assets, engine, voice, key, intent.getBooleanExtra("autorun", false), cpuOnly, models, detector, { state, _ -> if (state == "failed") cpuWarm() }, voiceFast, voiceFactory)
        server.gpuVision = intent.getBooleanExtra("gpu_vision", true)
        server.start()
        engine.log("app start port=${server.port} autorun=${intent.getBooleanExtra("autorun", false)} ${models.describe()} model_path=${if (cpuOnly) "cpu" else "gpu-first"}")
        if (cpuOnly) cpuWarm()
        else thread(name = "gpu-watchdog", isDaemon = true) { // no word at all from the shim in 240 s: the GPU path is not coming, fall back
            Thread.sleep(240_000); if (server.gpu == "probing") { engine.log("gpu watchdog: state=${server.gpu} after 240 s, CPU fallback"); server.gpu = "failed"; cpuWarm() }
        }
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG || intent.getBooleanExtra("devtools", false)) // debug build only: devtools reads window.__bendTimings
        web = WebView(this)
        setContentView(web.apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            // KOKORODROID: pages of our own server (voices.html) stay in the app; any other URL opens outside, as before.
            webViewClient = object : WebViewClient() {
                // PHONEFAST: the renderer died (lmkd under the GPU path's memory): don't take the app down; next launch is CPU.
                override fun onRenderProcessGone(w: WebView, d: android.webkit.RenderProcessGoneDetail): Boolean {
                    engine.log("webview renderer gone crashed=${d.didCrash()} gpu=${server.gpu}: CPU fallback")
                    if (server.gpu != "off") File(data, "gpu-crashed").writeText("${System.currentTimeMillis()}")
                    runOnUiThread { recreate() }
                    return true
                }

                override fun shouldOverrideUrlLoading(w: WebView, r: WebResourceRequest): Boolean {
                    if (r.url.host == "127.0.0.1") return false
                    try { startActivity(Intent(Intent.ACTION_VIEW, r.url)) } catch (e: Exception) { engine.log("open failed: $e") }
                    return true
                }

                // ANDROIDREFRESH: the voice lumen is WebGPU-only; say in engine.log/logcat whether this WebView has it.
                override fun onPageFinished(w: WebView, url: String) = w.evaluateJavascript(
                    "(async()=>{let r='navigator.gpu='+!!navigator.gpu;try{const a=navigator.gpu&&await navigator.gpu.requestAdapter();r+=' adapter='+!!a+(a&&a.info?' vendor='+a.info.vendor+' arch='+a.info.architecture:'')}catch(e){r+=' adapter-error='+e}console.info('webgpu-probe '+r+' ua='+navigator.userAgent)})()", null)
            }
            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(m: ConsoleMessage): Boolean {
                    if (m.message().startsWith("webgpu-probe")) engine.log(m.message())
                    return false
                }

                // getUserMedia: the voice screen does not use it (hearing is Voice.kt), but a page that does gets the mic/camera once allowed.
                override fun onPermissionRequest(r: PermissionRequest) = runOnUiThread {
                    val need = r.resources.mapNotNull {
                        when (it) { PermissionRequest.RESOURCE_AUDIO_CAPTURE -> Manifest.permission.RECORD_AUDIO; PermissionRequest.RESOURCE_VIDEO_CAPTURE -> Manifest.permission.CAMERA; else -> null }
                    }
                    thread { if (need.all { ensure(it) }) runOnUiThread { r.grant(r.resources) } else runOnUiThread { r.deny() } }
                }

                // The composer's attach button: take a photo with the camera, or pick one.
                override fun onShowFileChooser(w: WebView, cb: ValueCallback<Array<Uri>>, params: FileChooserParams): Boolean {
                    chooser?.onReceiveValue(null); chooser = cb
                    thread {
                        val cam = ensure(Manifest.permission.CAMERA)
                        runOnUiThread {
                            val pick = Intent(Intent.ACTION_GET_CONTENT).addCategory(Intent.CATEGORY_OPENABLE).setType("image/*")
                            val menu = Intent.createChooser(pick, "Add a photo")
                            if (cam) {
                                val f = File(File(cacheDir, "shots").apply { mkdirs() }, "shot-${System.currentTimeMillis()}.jpg").also { shot = it }
                                val capture = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                                    .putExtra(MediaStore.EXTRA_OUTPUT, Uri.parse("content://${packageName}.shots/${f.name}"))
                                    .addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION)
                                f.createNewFile()
                                menu.putExtra(Intent.EXTRA_INITIAL_INTENTS, arrayOf(capture))
                            }
                            try { startActivityForResult(menu, PHOTO) } catch (e: Exception) { engine.log("chooser failed: $e"); finishChooser(null) }
                        }
                    }
                    return true
                }
            }
            loadUrl("http://127.0.0.1:${server.port}/index.html")
        })
    }

    /** Runtime permission, asked on the UI thread, awaited on the caller's (never the UI thread). */
    private fun ensure(perm: String): Boolean {
        if (checkSelfPermission(perm) == PackageManager.PERMISSION_GRANTED) return true
        val wait = CountDownLatch(1); asking = wait
        runOnUiThread { requestPermissions(arrayOf(perm), PERM) }
        wait.await(120, TimeUnit.SECONDS)
        return checkSelfPermission(perm) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        asking?.countDown()
    }

    private fun finishChooser(uris: Array<Uri>?) { chooser?.onReceiveValue(uris); chooser = null }

    @Deprecated("api")
    override fun onActivityResult(code: Int, result: Int, data: Intent?) {
        if (code != PHOTO) return super.onActivityResult(code, result, data)
        val taken = shot?.takeIf { it.length() > 0 }?.let { Uri.fromFile(it) }
        finishChooser(if (result == RESULT_OK) (data?.data ?: taken)?.let { arrayOf(it) } else null)
    }

    @Deprecated("api")
    override fun onBackPressed() { if (web.canGoBack()) web.goBack() else super.onBackPressed() }

    override fun onDestroy() {
        engine.log("app destroy finishing=$isFinishing")
        server.close()
        voice.close()
        engine.stop()
        super.onDestroy()
    }

    private companion object { const val PERM = 1; const val PHOTO = 2 }
}
