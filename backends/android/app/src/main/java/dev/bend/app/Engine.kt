package dev.bend.app

import android.util.Log
import java.io.BufferedReader
import java.io.File
import java.io.OutputStream
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * The on-device engine: libbendengine.so (our qwen35 CPU engine + kev.c, `--decide CTX`) as ONE resident child
 * process speaking kevserve's JSON lines (one request line in, one reply line out). Loaded once, kept warm,
 * killed on exit. Pieces of a generate arrive on stderr as {"piece":"<hex>"} (KEV_EOS=1, patches/21-app-eos.patch).
 */
class Engine(private val exe: String, val dataDir: File, private val tmp: File, private val log: File, val model: Models.Pack) {
    @Volatile var state = "loading" // no-models | loading | live | error
    @Volatile var error: String? = null
    @Volatile var loadSeconds: Double? = null
    @Volatile var pid = -1L
    private var proc: Process? = null
    private var stdin: OutputStream? = null
    private var stdout: BufferedReader? = null
    @Volatile private var onPiece: ((ByteArray) -> Unit)? = null
    private val ends = LinkedBlockingQueue<String>() // stderr {"seq":..} lines: kev.c writes one per request

    val packDir get() = File(dataDir, model.pack)
    private val head get() = File(dataDir, model.head)
    val mmproj get() = File(dataDir, "mmproj-F16.gguf")
    @Volatile var lastVision: String? = null // stderr {"vision":{tokens,grid,tower_ms}} of the latest image request
    @Volatile var lastRun: String? = null    // stderr {"kevcpu":{..run_ms}} of the latest request

    fun missing(): String? =
        listOf("${model.pack}/pack.bin", "${model.pack}/manifest.json", "${model.pack}/tokenizer.bin", "${model.pack}/index.tsv", model.head)
            .firstOrNull { !File(dataDir, it).exists() }

    fun log(line: String) {
        Log.i("BendEngine", line)
        synchronized(log) { log.appendText("${System.currentTimeMillis()} $line\n") }
    }

    /** Starts the process if needed and warms it (one throwaway token pages the pack in). Serialized with requests. */
    @Synchronized fun ensure(): Boolean {
        if (proc?.isAlive == true && state == "live") return true
        missing()?.let { state = "no-models"; error = "no-models: $it missing in $dataDir"; log(error!!); return false }
        state = "loading"
        val t0 = System.nanoTime()
        try {
            stop()
            tmp.mkdirs()
            val pb = ProcessBuilder("/system/bin/taskset", "f0", exe, packDir.path, "--decide", "2048")
            pb.environment().putAll(mapOf(
                "OMP_NUM_THREADS" to "4", "OMP_SCHEDULE" to "dynamic,8", "LLB_RESIDENT" to "1",
                "KEV_EOS" to "1", "LLB_GGUF" to head.path, "LLB_MMPROJ" to mmproj.path, "LLB_VIS_TOKENS" to "256", "TMPDIR" to tmp.path))
            val p = pb.start()
            proc = p; pid = pidOf(p)
            stdin = p.outputStream
            stdout = p.inputStream.bufferedReader()
            log("spawn pid=$pid exe=$exe model=${model.name} pack=${packDir.path}")
            Thread({ drainStderr(p) }, "engine-stderr").apply { isDaemon = true }.start()
            val warm = request("""{"mode":"generate","prompt":"<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n","n":1}""")
            if (warm.contains("\"error\"")) throw IllegalStateException(warm)
            loadSeconds = (System.nanoTime() - t0) / 1e9
            state = "live"; error = null
            log("live pid=$pid load_s=${"%.1f".format(loadSeconds)} rss=${rss()}")
            return true
        } catch (e: Exception) {
            state = "error"; error = "The engine didn't load (${e.message}). Retry to try again."
            log("load failed: ${e.message}")
            stop()
            return false
        }
    }

    /** One request line -> its reply line. `piece` gets each generated token's bytes as they land. */
    @Synchronized fun request(line: String, piece: ((ByteArray) -> Unit)? = null): String {
        val p = proc ?: throw IllegalStateException("engine not running")
        ends.clear(); lastVision = null; lastRun = null
        onPiece = piece
        try {
            stdin!!.write((line.replace('\n', ' ') + "\n").toByteArray())
            stdin!!.flush()
            val reply = stdout!!.readLine() ?: run {
                state = "error"; error = "The engine exited. Retry to try again."
                log("engine exited during a request (alive=${p.isAlive})")
                throw IllegalStateException("engine exited")
            }
            ends.poll(10, TimeUnit.SECONDS) // its pieces are all in before this line
            return reply
        } finally {
            onPiece = null
        }
    }

    private fun drainStderr(p: Process) {
        p.errorStream.bufferedReader().forEachLine { line ->
            if (line.startsWith("{\"piece\":\"")) {
                val hex = line.substring(10, line.length - 2)
                onPiece?.invoke(ByteArray(hex.length / 2) { hex.substring(2 * it, 2 * it + 2).toInt(16).toByte() })
            } else {
                log("engine: $line")
                if (line.startsWith("{\"vision\":")) lastVision = line
                if (line.startsWith("{\"kevcpu\":")) lastRun = line
                if (line.startsWith("{\"seq\":")) ends.add(line)
            }
        }
    }

    /** VmRSS / VmHWM / RssAnon / RssFile of the engine process, in MB. */
    fun rss(): Map<String, Long> = try {
        File("/proc/$pid/status").readLines()
            .filter { it.startsWith("VmRSS") || it.startsWith("VmHWM") || it.startsWith("RssAnon") || it.startsWith("RssFile") }
            .associate { it.substringBefore(':') to it.substringAfter(':').trim().removeSuffix(" kB").trim().toLong() / 1024 }
    } catch (e: Exception) { emptyMap() }

    // Android's Process has no pid(); its UNIXProcess/ProcessImpl keeps it in a private field.
    private fun pidOf(p: Process) = try { p.javaClass.getDeclaredField("pid").apply { isAccessible = true }.getInt(p).toLong() } catch (e: Exception) { -1L }

    fun stop() {
        proc?.let { log("stop pid=$pid"); it.destroy(); if (!it.waitFor(3, TimeUnit.SECONDS)) it.destroyForcibly() }
        proc = null
    }
}
