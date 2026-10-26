package dev.bend.app

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * MODEL SELECT: which weights this install loads.
 *
 * <external files dir>/models.json (adb push-able; no rebuild to switch):
 *   {
 *     "active": "Qwen3.5-0.8B-Q4_K_M",
 *     "models": [
 *       {"name": "Kev-4B-Q4_K_M", "gguf": "Kev-4B-Q4_K_M.gguf", "pack": "pack", "head": "kev-head-q8.gguf", "decide": true},
 *       {"name": "Qwen3.5-0.8B-Q4_K_M", "gguf": "Qwen3.5-0.8B-Q4_K_M.gguf", "pack": "pack-0.8b", "decide": false, "vision": false}
 *     ]
 *   }
 * A plain string entry ("Qwen3.5-0.8B-Q4_K_M") means gguf <name>.gguf and pack dir "pack" for the built-in
 * default, "pack-<slug>" for anything else. Pack dirs hold the CPU engine's pack.bin/index.tsv/manifest.json/
 * tokenizer.bin; the gguf is what the WebGPU path reads in place. "head" is the Kev decision head kev.c loads
 * at startup (kev.c needs one; its decide math is 2560-wide, so decide stays Kev-4B-only), "decide" says
 * whether /v1/systemone is offered for this model.
 *
 * Missing/unreadable models.json, an unknown active name, or an active entry whose pack dir or gguf is not on
 * disk = the built-in Kev-4B pack: the behaviour every build before this selector had.
 */
class Models(private val dir: File) {
    data class Pack(val name: String, val gguf: String, val pack: String, val head: String, val decide: Boolean, val vision: Boolean)

    val packs: List<Pack>
    val active: Pack

    init {
        val builtIn = Pack(DEFAULT, "$DEFAULT.gguf", "pack", HEAD, true, true)
        val listed = ArrayList<Pack>()
        val f = File(dir, "models.json")
        var wanted = DEFAULT
        if (f.isFile) try {
            val o = JSONObject(f.readText())
            wanted = o.optString("active", DEFAULT)
            val a = o.optJSONArray("models") ?: JSONArray()
            for (i in 0 until a.length()) when (val e = a.opt(i)) {
                is String -> if (e.isNotBlank()) listed += fromName(e)
                is JSONObject -> if (e.optString("name").isNotBlank()) listed += fromJson(e)
            }
        } catch (e: Exception) {
            Log.w(TAG, "models.json unreadable (${e.message}); using $DEFAULT")
        }
        if (listed.none { it.name == DEFAULT }) listed.add(0, builtIn)
        packs = listed
        val pick = packs.firstOrNull { it.name == wanted && File(dir, it.pack).isDirectory && File(dir, it.gguf).isFile }
        active = pick ?: packs.first { it.name == DEFAULT }
        if (pick == null && wanted != DEFAULT) Log.w(TAG, "active '$wanted' is not on disk; using $DEFAULT")
        Log.i(TAG, describe())
    }

    fun describe() = "model=${active.name} gguf=${active.gguf} pack=${active.pack} decide=${active.decide} vision=${active.vision}"

    private fun fromName(n: String) = Pack(n, "$n.gguf", if (n == DEFAULT) "pack" else "pack-" + slug(n), HEAD, n == DEFAULT, n == DEFAULT)

    private fun fromJson(o: JSONObject): Pack {
        val n = o.getString("name")
        return Pack(n,
            o.optString("gguf", "$n.gguf"),
            o.optString("pack", if (n == DEFAULT) "pack" else "pack-" + slug(n)),
            o.optString("head", HEAD).ifBlank { HEAD },
            o.optBoolean("decide", n == DEFAULT),
            o.optBoolean("vision", n == DEFAULT))
    }

    companion object {
        const val TAG = "BendModels"
        const val DEFAULT = "Kev-4B-Q4_K_M"
        const val HEAD = "kev-head-q8.gguf"
        fun slug(n: String) = n.lowercase().replace(Regex("[^a-z0-9.]+"), "-").trim('-')
    }
}
