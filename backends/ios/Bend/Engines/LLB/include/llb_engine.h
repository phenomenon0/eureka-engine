// llb_engine.h: the in-process API over the Bend-emitted Qwen3.5 engine
// (sys1mac.c, M41/sys1mac lineage + engine-4b's 4B shape; ENGINE-PHONE; ENGINE-ONE: + Kev-4B's Q4_K
// embedding and llb_decide, the Jev (Kev) decision head). One engine per process; not thread-safe.
//
//   llb_load(dir, cache)   dir holds the model: one *.gguf + its llb sidecar
//                          (manifest.json, table.bin, tokenizer.bin, chat.flags).
//                          cache is a writable directory (the app's Caches/) for
//                          the compiled Metal program (<cache>/sys1mac.gpu).
//   llb_generate(ids, n)   greedy: prefill ids, then up to n new ids into out. (ENGINE-EARLYSTOP) n is a
//                          maximum: the run ends at the model's first stop id (tokenizer.bin's eos =
//                          <|im_end|>, and <|endoftext|>), which is the last id returned (the return value
//                          counts it). LLB_EARLYSTOP=0 in the environment: exactly n ids, as before.
//   llb_decide(bodies, n)  (ENGINE-ONE) /v1/systemone requests -> reply lines, the decision head
//                          on the engine's final-normed rows, in process (no server, no HTTP).
//   llb_free()             forget the model.
//
// (KEVPHONE) llb_open / llb_close: the resident session (below): one load, then any number of runs on it.
// R1 limits (see REPORT.md): llb_generate (or llb_decide) runs the engine's whole program
// (weights into the GPU span, prefill, decode), so the model is not resident
// between calls; a fatal engine error ends the process (the engine's _exit).
// (ENGINE-REPEAT) Any number of runs per process: each run releases everything it set up when it
// returns (the span unmapped, the engine's threads joined, its statics reset), so a request is
// llb_load -> llb_generate | llb_decide, and the next request (same or another model) loads again.
// One run per llb_load; llb_free also re-arms. Only a run whose release failed leaves the engine
// spent (every later call refuses with "R1: ... could not be released").
#ifndef LLB_ENGINE_H
#define LLB_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double   compile_ms;   // Metal program compiled + written to the cache (0 when cached)
  double   total_ms;     // the whole llb_generate call
  double   load_ms;      // engine's "resident ... in N ms" (weights into the span)
  double   prefill_ms;   // engine's chunkf ms (fast tier), else 0
  double   step_ms_sum;  // sum of the engine's per-step ms
  int      steps;        // per-step lines seen
  uint64_t span_bytes;   // the GPU span given to the engine (--gpu)
  uint64_t gguf_bytes;
  int      exit_code;    // the engine program's exit code
} llb_stats;

// (ENGINE-ONE) one llb_decide call
typedef struct {
  double   compile_ms;   // as llb_stats
  double   total_ms;     // the whole llb_decide call
  double   head_open_ms; // the head's GGUF tensors + the tokenizer (first call)
  double   load_ms;      // engine's "resident ... in N ms"
  double   forward_ms;   // sum over questions: the prefill (Sys1.next -> Sys1.done)
  double   head_ms;      // sum over questions: rows -> the head (+ the reply)
  double   step_ms_sum;  // sum of the engine's per-step ms (every prompt position)
  int      steps;
  int      questions;    // prefills run
  int      input_tokens; // sum of their prompt tokens
  uint64_t span_bytes;
  uint64_t gguf_bytes;
  int      exit_code;
  // (ENGINE-DECIDECHUNK) the prefill tier: the chunk K each question's prefill asked for (0 = exact)
  // and the engine's chunkf lines (fast tier: all but the last prompt position)
  int      fastpf;
  int      chunks;
  int      chunk_tokens;
  double   chunk_ms_sum;
} llb_dstats;

int         llb_load(const char* model_dir, const char* cache_dir);
int         llb_generate(const uint32_t* ids, int n_ids, int n_new, uint32_t* out,
              llb_stats* st);
// (ENGINE-ONE) n request bodies (each one /v1/systemone JSON object) -> replies[i], malloc'd: the
// answer line (llama-server's /v1/systemone layout) or {"error":"..."}. The model dir's GGUF must
// carry the Kev head (qwen35.decision.type = kev). Returns the number answered, -1 on an engine error.
int         llb_decide(const char* const* bodies, int n, char** replies, llb_dstats* st);
// (ENGINE-DECIDECHUNK) decide's prefill tier: LLB_DECIDE_FASTPF=K (default 512, the fast chunk tier;
// 0 = the exact path). llb_rows: the rows-parity harness, any pack (no head): item i = ids[i] (n_ids[i])
// with the hook on at pos[i] (npos[i]) through the same srv loop + hook as llb_decide; its rows to
// <outdir>/rows-<i>.hidden ([pos u32][width u32][width f32] each). Returns the items written, -1 on error.
int         llb_rows(const uint32_t* const* ids, const int* n_ids, const uint32_t* const* pos,
              const int* npos, int n, const char* outdir, llb_dstats* st);
void        llb_free(void);
const char* llb_error(void);         // last error text ("" if none)
const char* llb_log(void);           // the engine's stderr from the last llb_generate
const char* llb_device_info(void);   // Metal device + memory limits, one line
// (ENGINE-PHONE) after llb_load: the GPU span llb_generate will ask for, in bytes
uint64_t    llb_span_plan(void);
// (ENGINE-PHONE) compiles the engine's Metal library and builds a pipeline for
// every kernel the 4B decode path binds; one line: each kernel's
// maxTotalThreadsPerThreadgroup vs the engine's own requirement (its m9_init /
// warp rms checks), "fits" or the list that does not. A compile error is the
// engine's _exit (its stderr). Call before llb_generate.
const char* llb_kernel_report(void);
// (ENGINE-REPEAT) the last run's release, one line: released=yes|NO release_ms=
// footprint_mb_before= (main returned) footprint_mb_after= (after the release)
const char* llb_release_report(void);

// (KEVPHONE) the resident session: the weights stay loaded across requests.
// llb_open(dir, cache, ctx, flags, st): llb_load + one load of dir's weights into the span, kept: main runs on an engine thread
//   of its own in the SYS1 srv loop, its KV capacity ctx (rounded up to a power of two, 64 .. 131072). Returns once
//   the weights are resident (st: compile_ms, load_ms, total_ms, span_bytes), 0, or -1 (llb_error, llb_log). Already
//   open on dir: 0 at once, nothing loaded. Open on another model: that one is closed first (one engine per process).
// While open, llb_generate / llb_decide on dir run as jobs on it (load_ms 0: nothing loaded, nothing released), the
//   same ids / reply lines as a one-shot run's. A request longer than the KV capacity closes the session and runs
//   one-shot. llb_load of another dir and llb_free close it too.
// llb_close(): the loop ends, main returns and the release runs (llb_release_report). Safe when nothing is open.
// flags: LLB_OPEN_NOWARMUP skips the throw-away 16-id chat after the load (a request is waiting on this open; its own
//   first job pays that one-time set-up instead, which costs less than the warm-up)
#define LLB_OPEN_NOWARMUP 1
int         llb_open(const char* model_dir, const char* cache_dir, int ctx, int flags, llb_stats* st);
void        llb_close(void);
// the open session's model dir, or NULL
const char* llb_resident(void);
// one line: resident=yes|no model= cap= loads= jobs= load_ms= open_ms= compile_ms= span_mb= footprint_mb=
const char* llb_resident_report(void);

// (PHONECAM) the camera's verdict, in process, on the resident session (a VISION build: LLB_VISION; elsewhere every
// call refuses). One frame = the tower on its RGB canvas -> the image rows into the session's image area -> one
// resident job (the prompt pre + n <|image_pad|> + post, one new id) -> the head's logits[id_a], logits[id_b] at the
// last prompt position (the readout: verdict = a - b > 0, as vlive1's Yes/No logit readout).
// llb_vision_open(mmproj, d, cap): BEFORE llb_open: the tower (mmproj, backbone width d) loaded once and kept, and the
//   session's image area of cap rows (reserved at the session's first pass); 0 or -1 (llb_error).
// llb_vision_close(): the tower freed (the session keeps its area until it closes).
// llb_classify_rgb: rgb is w x h x 3 bytes (row stride w*3). Needs llb_open on the model llb_vision_open was given
//   for. 0, or -1 (llb_error; nothing held afterwards).
// (IMGFAST) the classify and the look share the kept rows (below): a fire keeps its canvas's rows, so a tap on that
//   frame (classify again) or Describe on it (llb_generate_rgb on the same canvas bytes) runs no tower; a kept canvas
//   needs no open tower either. The rows are the tower's own (deterministic), so the bits / ids are the same.
typedef struct {
  double   prep_ms;      // canvas -> planar (vpre)
  double   tower_ms;     // the tower's encode (host wall), and its GPU time
  double   tower_gpu_ms;
  double   llm_ms;       // the resident job: prefill (fast chunk) + the exact last step + the readout
  double   total_ms;     // the whole call
  int      n_img, nx, ny, n_ids;
  uint32_t first_id;     // the argmax at the last prompt position (the one new id)
  float    logit_a, logit_b;
  uint32_t bits_a, bits_b;  // their f32 bits
  // (IMGPERF) llb_generate_rgb: the job's prefill and decode split (as llb_stats), and whether the photo's rows were
  // kept from an earlier look (no tower ran: prep_ms = tower_ms = 0)
  double   prefill_ms;
  double   step_ms_sum;
  int      steps;
  int      rows_kept;
  // (IMGFAST) the job's prompt prefix restored from a snapshot (ids it did not prefill; 0: none) and the host ms the
  // snapshots cost it (restore, or the split's wait + the copy out when it kept one)
  int      prefix_ids;
  double   prefix_ms;
} llb_vstats;
// llb_vision_area(cap): only the session image area (no tower): call once before any llb_open (e.g. at app start), so
//   every session reserves it; 0 = none. Refused while a session with another cap is resident.
int         llb_vision_area(int cap);
int         llb_vision_open(const char* mmproj, int d, int cap);
void        llb_vision_close(void);
int         llb_classify_rgb(const uint8_t* rgb, int w, int h, const uint32_t* pre, int n_pre, const uint32_t* post,
              int n_post, uint32_t id_a, uint32_t id_b, llb_vstats* st);
// (IMGPERF) a look: the photo + its question -> an answer, greedy, at chat's decode speed. The tower runs once on the
//   canvas (or not at all: the rows of the last 4 photos are kept, by their exact bytes), the rows go into the session's
//   image area, and ONE resident job prefills pre + n <|image_pad|> + post once and decodes up to n_new ids, ending at
//   the first stop id (the last id returned, counted), as llb_generate. The ids stream as llb_generate's do (the
//   session's err file: `logits <pos> <argmax>` from pos n_pre + n + n_post - 1). Returns the count or -1 (llb_error).
//   st: prep/tower (0 when kept) / llm / total ms, prefill_ms / step_ms_sum / steps, n_img, n_ids, first_id, rows_kept.
//   Needs the resident session on the model llb_vision_open was given for; the tower open unless the rows are kept.
int         llb_generate_rgb(const uint8_t* rgb, int w, int h, const uint32_t* pre, int n_pre, const uint32_t* post,
              int n_post, int n_new, uint32_t* out, llb_vstats* st);
// (IMGPERF) 1 when this canvas's rows are kept (a look at it needs no tower), else 0. llb_vision_forget drops them all.
int         llb_vision_has(const uint8_t* rgb, int w, int h);
// (IMGPERF) the tower on this canvas now, its rows kept (no job, no session needed): a photo encoded while the person
//   is still typing, so its look needs no tower. 0 (st: prep/tower ms, n_img; rows_kept = 1 when they already were), or
//   -1 (llb_error: the tower is closed and the rows are not kept).
int         llb_vision_keep(const uint8_t* rgb, int w, int h, llb_vstats* st);
void        llb_vision_forget(void);
// (IMGFAST) prefix snapshots: the session keeps the layer state at the end of a vision prompt's prefix (the camera's
//   33-id system turn; the look's; the look's + an image's rows, up to 3, ~50 MB each on the 4B) and a later job that
//   starts with the same ids (and the same kept canvas) prefills only the rest; the ids / bits are the same (receipts:
//   the imgfast lane). llb_prefix_forget drops them (memory pressure). LLB_PREFIX=0: never kept or restored.
void        llb_prefix_forget(void);
// (G3) a text prompt on the prefix snapshots: llb_generate on the resident session with its first n_pre ids (1 ..
//   min(n_ids - 1, 1024): the system turn, or a conversation so far) kept as a snapshot the first time and restored
//   on every later prompt that starts with the same ids: those are not prefilled again. Same ids out as llb_generate.
//   *prefix_ids (may be NULL): the ids this job restored (0: none). Only the 4B (d 2560); LLB_PREFIX=0: plain.
int         llb_generate_px(const uint32_t* ids, int n_ids, int n_pre, int n_new, uint32_t* out, llb_stats* st,
                            int* prefix_ids);
// (G3) the hot state across launches: the session's text snapshots written to a file (llb_state_save, at background /
//   exit) and read back into a new session (llb_state_load, after llb_open on the same model and ctx), so the first
//   prompt after a cold launch restores its system turn instead of prefilling it. Both need the session idle. They
//   return the snapshots written / read, or -1 (llb_error). A file from another engine build, model or KV capacity is
//   refused, never half-loaded. A text snapshot at the 4B is 64 KiB an id (8 KV layers x K+V x 4 KiB f32) + the 24
//   GDN layers' state (~2.2 MiB each): 62.9 MB for a 143-id system turn (G3 receipts), held in RAM while kept
//   (llb_prefix_forget drops it).
int         llb_state_save(const char* path);
int         llb_state_load(const char* path);
// (G3) LLB_PACK_MAP=<file> (env, read at the first weight load): the mmap-class load. The packed weights are written
//   to <file> (+ <file>.idx) on the first load, then mapped from it (clean file pages) on later loads. One line:
//   "on|off mapped_mb= copied_mb= wrote_mb=" for this process.
const char* llb_packmap_report(void);

#ifdef __cplusplus
}
#endif

#endif
