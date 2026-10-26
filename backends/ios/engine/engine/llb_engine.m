// llb_engine.m: the in-process API (llb_engine.h) over the Bend-emitted engine.
// (ENGINE-ONE copy of engine-phone's: + llb_decide, the Jev (Kev) decision head on the same engine run.)
// (ENGINE-DECIDECHUNK: llb_decide's prefills take the FAST chunk tier, "--fastpf K" on each srv command
//  (patches/{metal,engine}-dchunk.patch: the chunk rms-norms the hook-marked rows itself);
//  LLB_DECIDE_FASTPF=0 is the exact path as before. + llb_rows, the rows-parity harness.)
//
// The emitted C is compiled unmodified, as part of this translation unit:
//   - its main becomes llb_bend_main, driven with the same argv the Mac CLI takes
//     (--gpu <span> --threads T -- <sidecar> ids... --steps n);
//   - _NSGetExecutablePath (used only by gpu_path, to put the compiled Metal
//     program next to the binary) answers <cache>/sys1mac, so the program cache
//     is <cache>/sys1mac.gpu in the app's own storage, not the read-only bundle;
//   - (ENGINE-ONE) the decide hooks dual_next/dual_done are this file's, not
//     tools/sys1dual/dual.c's: no stdin, no CLEF-3, no OpenBLAS. llb_decide runs
//     the engine's own SYS1 srv loop (LLB_SYS1: one load, then one fresh-state
//     exact prefill per question) and feeds each question's final-normed rows to
//     the head (head/jevsession.h, jev-head 8736c62) in this process.
// The engine prints ids to stdout and diagnostics to stderr; llb_generate and
// llb_decide point fds 1/2 at files in <cache>/run/ for the call, then parse them.
// (ENGINE-REPEAT) N runs per process: each run is load -> run -> release. After main returns, bend_close
// (patches/metal-reopen.patch) joins the pool and io helpers, drops every Metal object, unmaps the span and the
// loop's stack and puts every engine static back to its initializer; gguf_close (engine-reopen.patch) closes the
// GGUF. llb_load / llb_free then admit the next run; only a run whose release failed leaves the engine spent.
// (KEVPHONE) the resident session: llb_open runs main on an engine thread of its own in the SYS1 srv loop (one load:
// the weights stay in the span), parked in dual_next until a job comes. llb_generate / llb_decide on the open model
// hand their commands to that loop (a chat is one command with its --stop ids, patches/engine-resident.patch; a
// decide is its questions, as before) and wait for them: no load, no release per request. llb_close (or llb_free,
// or llb_load of another model) ends the loop: main returns and the release above runs on the engine thread. While
// a session is open the engine's stdout / stderr are its own FILEs (<cache>/run/out.txt, err.txt), not the
// process's fds 1 and 2, emptied after each job.

#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static char llb_exe[1024];  // "<cache>/sys1mac": gpu_path() appends ".gpu"

static int llb_exe_path(char* buf, uint32_t* n) {
  size_t len = strlen(llb_exe);
  if (len + 1 > *n) {
    *n = (uint32_t)(len + 1);
    return -1;
  }
  memcpy(buf, llb_exe, len + 1);
  return 0;
}

// (KEVPHONE) the engine's standard streams: the resident session's files while one is open, else the process's
// (a one-shot run points fds 1 and 2 at its files, as before)
static FILE* llb_fout;
static FILE* llb_ferr;
#undef stdout
#undef stderr
#define stdout (llb_fout ? llb_fout : __stdoutp)
#define stderr (llb_ferr ? llb_ferr : __stderrp)

#define _NSGetExecutablePath llb_exe_path
#define main llb_bend_main
#include "sys1mac.c"
#undef main
#undef _NSGetExecutablePath

#undef stdout
#undef stderr
#define stdout __stdoutp
#define stderr __stderrp

#include "llb_engine.h"
#include <dirent.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif

#include "jevsession.h"
#include "qwen35_tokenizer.h"

// (ENGINE-ONE) the decide queue: one item per question of every request, in order
typedef struct {
  int       r, k;      // request, question
  int*      ids;
  int       n;         // prompt tokens
  uint32_t* pos;       // the layout: option markers ..., the pointer (n - 1)
  int       npos;
} llb_dq;

static llb_dq*       llb_q;
static int           llb_qn, llb_qi;
static jvh_job**     llb_jobs;
static char**        llb_replies;
static char          llb_hpath[1200];
static double        llb_t_fwd, llb_fwd_ms, llb_head_ms;
static const char*   llb_model;
// (ENGINE-DECIDECHUNK) the fast tier's chunk K for every srv command (0 = the exact path), and
// llb_rows' output dir (set: dual_done keeps each item's rows there, no head)
static int           llb_fastpf;
static const char*   llb_rawdir;
static char llb_err[512];
static char llb_dir[1024];
static char llb_gguf[1024];
static char llb_cache[1024];
static char* llb_errlog;
static u64   llb_gguf_bytes;
static int   llb_loaded;
static int   llb_runs;    // runs since the last llb_load / llb_free (the R1 guards: at most one)
static int   llb_spent;   // (ENGINE-REPEAT) a run's release failed: no more runs in this process
static char  llb_rel[256];
static void  rs_reap(void);  // (KEVPHONE) the resident session, below

static double llb_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

// (ENGINE-REPEAT) the process's phys_footprint (what jetsam counts), in MB
static long llb_footprint_mb(void) {
  task_vm_info_data_t v;
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&v, &n) != KERN_SUCCESS) {
    return -1;
  }
  return (long)(v.phys_footprint >> 20);
}

const char* llb_release_report(void) { return llb_rel; }

// the R1 guards' refusal: a used-up engine (its release failed), or a second run without llb_load / llb_free
static int llb_refuse(const char* what) {
  if (llb_spent) {
    snprintf(llb_err, sizeof llb_err, "R1: %s: the previous run could not be released (%s); no more runs in this "
      "process", what, llb_rel);
  } else {
    snprintf(llb_err, sizeof llb_err, "R1: %s: one run per llb_load (llb_load or llb_free before the next run)",
      what);
  }
  return -1;
}

// (ENGINE-REPEAT) one engine run: main inside an autorelease pool (its command buffers and encoders, which hold the
// span's buffer, are gone when it returns), then the release: bend_close + gguf_close, the argv copy freed, and the
// calling thread's signal stack and the SIGSEGV / SIGBUS actions as they were before the run (pool_stack set them)
static int llb_run_main(int argc, char** argv) {
  stack_t oss;
  struct sigaction osegv, obus;
  sigaltstack(NULL, &oss);
  sigaction(SIGSEGV, NULL, &osegv);
  sigaction(SIGBUS, NULL, &obus);
  // main keeps argv + 1 as its io_argv and shifts the words in place: give it a copy
  char** av = calloc((size_t)argc + 1, sizeof(char*));
  memcpy(av, argv, (size_t)argc * sizeof(char*));
  int code;
  @autoreleasepool {
    code = llb_bend_main(argc, av);
  }
  long fp0 = llb_footprint_mb();
  double t0 = llb_ms();
  int rc = bend_close();
#ifdef CID_GGUF_READ
  if (rc == 0) {
    rc = gguf_close();
  }
#endif
  if (rc == 0) {
    free(av);  // io_argv pointed into it; bend_close set it back to NULL
    if (!(oss.ss_flags & SS_DISABLE)) {
      sigaltstack(&oss, NULL);
    }
    sigaction(SIGSEGV, &osegv, NULL);
    sigaction(SIGBUS, &obus, NULL);
  }
  llb_spent = rc != 0;
  snprintf(llb_rel, sizeof llb_rel, "released=%s release_ms=%.1f footprint_mb_before=%ld footprint_mb_after=%ld",
    rc == 0 ? "yes" : "NO (bend_close refused: CUDA build or --workers)", llb_ms() - t0, fp0, llb_footprint_mb());
  return code;
}

const char* llb_error(void) { return llb_err; }
const char* llb_log(void) { return llb_errlog ? llb_errlog : ""; }

static int llb_fail(const char* a, const char* b) {
  snprintf(llb_err, sizeof llb_err, "%s%s", a, b ? b : "");
  return -1;
}

// The span the engine runs in: weights + the engine's tables, KV and scratch.
// Measured on the Mac for the 9B Q4_K_M (5.67 GB GGUF): 8 GiB fails, 9 GiB runs;
// the default is GGUF + 3 GiB rounded up to 256 MiB (9B: 8.5 GiB, see engine-ios).
// LLB_GPU_SPAN_MB overrides; LLB_GPU_EXTRA_MB replaces the 3 GiB.
// (ENGINE-PHONE: the 4B's measured floor is in REPORT.md "Memory".)
// (METAL-ANYSHAPE) the default extra is sized from the model: GGUF / 2 + 1 GiB (the head and embedding
// regions the engine copies, the layer blobs' power-of-two classes, the Bend heap). The fused step keeps its
// KV, attention planes and fast-chunk area in the engine's own Metal buffer, sized from the requested ctx
// (llb_open's cap) and refused by the engine past the device's largest buffer, so the span does not grow
// with ctx. Floors measured on an M3 Ultra at ctx 4096 (docs/limits.md): 0.8B 1.6 GB, 2B 2.6 GB, 4B 4.5 GB,
// 9B 8.8 GB; this default gives 1.8, 2.9, 5.0, 9.6 GB. Not measured on a phone.
static u64 llb_span(void) {
  const char* s = getenv("LLB_GPU_SPAN_MB");
  if (s && atoll(s) > 0) {
    return (u64)atoll(s) << 20;
  }
  const char* x = getenv("LLB_GPU_EXTRA_MB");
  u64 extra = x && atoll(x) > 0 ? (u64)atoll(x) << 20 : llb_gguf_bytes / 2 + (1ull << 30);
  u64 q = 256ull << 20;
  return (llb_gguf_bytes + extra + q - 1) / q * q;
}

const char* llb_device_info(void) {
  static char info[512];
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    u64 avail = 0;
#if TARGET_OS_IPHONE
    avail = (u64)os_proc_available_memory();
#endif
    snprintf(info, sizeof info,
      "gpu=%s maxThreadsPerThreadgroup=%lu maxBuffer=%.2fGiB workingSet=%.2fGiB unified=%d ram=%.2fGiB proc_avail=%.2fGiB cpus=%ld",
      d ? [[d name] UTF8String] : "none",
      d ? (unsigned long)[d maxThreadsPerThreadgroup].width : 0ul,
      d ? (double)[d maxBufferLength] / (1ull << 30) : 0.0,
      d ? (double)[d recommendedMaxWorkingSetSize] / (1ull << 30) : 0.0,
      d ? (int)[d hasUnifiedMemory] : 0,
      (double)[[NSProcessInfo processInfo] physicalMemory] / (1ull << 30),
      (double)avail / (1ull << 30),
      (long)[[NSProcessInfo processInfo] activeProcessorCount]);
  }
  return info;
}

uint64_t llb_span_plan(void) { return llb_loaded ? llb_span() : 0; }

// (ENGINE-PHONE) the kernels the 4B decode (exact tier) binds, with the threadgroup
// each needs: the emitted engine's own checks (m9_init, the warp rms check)
static const struct { const char* name; unsigned need; } llb_kneed[] = {
  { "llb_rms", 1024 }, { "llb_warp", 128 }, { "llb_warp_q8", 128 },
  { "llb_warp_q8_split", 128 }, { "llb_warp_q8_v2", 128 }, { "llb_warp_q8_v2d11_r2", 256 },
  { "llb_m_rmsnorm", 1024 }, { "llb_m_amax", 1024 }, { "llb_m_slot", 1024 },
  { "llb_m_qkr", 256 }, { "llb_m_attn4", 256 }, { "llb_m_attn5", 1024 },
  { "llb_m_attn5s", 256 }, { "llb_m_attn5p", 32 }, { "llb_m_attn5v", 256 },
  { "llb_m_attn5c", 256 }, { "llb_m_embq8", 256 }, { "llb_m_embk4", 256 },
  { "llb_m_embk6", 256 }, { "llb_m_gdl2", 1024 }, { "llb_m_ropecs", 32 },
  { "llb_m_qkr2", 256 }, { "llb_m_l2nr", 64 }, { "llb_m_addrms", 1024 },
  { "llb_m_elem", 256 }, { "llb_m_conv1d", 256 }, { "llb_m_gdl", 128 },
  { "llb_m_l2norm", 64 }, { "llb_m_grn", 64 }, { "llb_m_mid32a", 48 }, { "llb_m_mid32b", 48 },
  { "bd2_f4_n4_s4_m0", 128 }, { "bd2_f4_n4_s2_m0", 64 }, { "bd2_f4_n16_s8_m0", 256 },
  { "bd2_f9_n4_s4_m0", 128 }, { "bd2_f9_n4_s2_m0", 64 }, { "bd2_f9_n16_s8_m0", 256 },
  { "bd2_f6_n4_s4_m0", 128 }, { "bd2_f6_n4_s2_m0", 64 }, { "bd2_f6_n16_s8_m0", 256 },
  { "bd2_f4_n8_s8_m0", 256 }, { "bd2_f6_n8_s8_m0", 256 }, { "rd2_f4_n32_p17_m0", 256 },
  { "rd2_f6_n8_p5_m0", 128 },
};

const char* llb_kernel_report(void) {
  static char rep[2048];
  @autoreleasepool {
    if (gpu_dev == nil && !gpu_probe()) {
      return "no Metal device";
    }
    double t0 = llb_ms();
    if (gpu_mlib == nil) {
      (void)gpu_desc();  // the engine's own compile (its options and macros)
    }
    double tc = llb_ms() - t0;
    unsigned n = 0, bad = 0, lo = 1u << 30, nall = (unsigned)[[gpu_mlib functionNames] count];
    char miss[1400] = "";
    size_t ml = 0;
    for (size_t i = 0; i < sizeof llb_kneed / sizeof llb_kneed[0]; i += 1) {
      NSError* err = nil;
      id<MTLFunction> f = [gpu_mlib newFunctionWithName:@(llb_kneed[i].name)];
      id<MTLComputePipelineState> p = f ? [gpu_dev newComputePipelineStateWithFunction:f error:&err] : nil;
      unsigned got = p ? (unsigned)p.maxTotalThreadsPerThreadgroup : 0;
      n += 1;
      lo = got < lo ? got : lo;
      if (got < llb_kneed[i].need) {
        bad += 1;
        if (ml < sizeof miss - 64) {
          ml += (size_t)snprintf(miss + ml, sizeof miss - ml, " %s(%u<%u)", llb_kneed[i].name,
            got, llb_kneed[i].need);
        }
      }
    }
    snprintf(rep, sizeof rep, "library_functions=%u checked=%u compile_ms=%.0f pipelines_ms=%.0f "
      "min_maxTotalThreads=%u %s%s", nall, n, tc, llb_ms() - t0 - tc, lo,
      bad ? "DO-NOT-FIT:" : "all fit", miss);
  }
  return rep;
}

static Q35Tokenizer  llb_tok;
static jvh_session*  llb_head;
static int           llb_tok_on;

// the decision head and the tokenizer of the model in llb_dir (llb_decide opens them on first use)
static void llb_head_close(void) {
  if (llb_head) {
    jvh_session_close(llb_head);
    llb_head = NULL;
  }
  if (llb_tok_on) {
    q35_tok_free(&llb_tok);
    llb_tok_on = 0;
  }
}

// (ENGINE-EARLYSTOP) the ids that end a chat reply: the tokenizer's eos (<|im_end|>) and <|endoftext|>, the pair
// Bend cuts its reply at. Read from <dir>/tokenizer.bin (its header, then the vocab records until the
// <|endoftext|> text), once per model dir. n = 0: no stop ids (no tokenizer.bin): llb_generate runs n_new ids.
// (KEVPHONE) + the <|im_end|> text when the eos is not it: Kev-4B's tokenizer.bin names <|endoftext|> (248044) as its
// eos, so a Kev chat ran past its <|im_end|>. Qwen3.5's pair is unchanged (eos = <|im_end|>, then <|endoftext|>).
static char     llb_stop_dir[1024];
static uint32_t llb_stop[3];
static int      llb_nstop;

static void llb_stop_add(uint32_t id) {
  for (int i = 0; i < llb_nstop; i += 1) {
    if (llb_stop[i] == id) {
      return;
    }
  }
  if (llb_nstop < 3) {
    llb_stop[llb_nstop++] = id;
  }
}

static void llb_stop_read(const char* dir) {
  if (strcmp(llb_stop_dir, dir) == 0) {
    return;
  }
  snprintf(llb_stop_dir, sizeof llb_stop_dir, "%s", dir);
  llb_nstop = 0;
  char p[1200];
  snprintf(p, sizeof p, "%s/tokenizer.bin", dir);
  FILE* f = fopen(p, "rb");
  if (!f) {
    return;
  }
  char magic[8];
  uint32_t h[8];  // n_vocab n_merges max_len bos eos pad add_bos n_special
  if (fread(magic, 1, 8, f) == 8 && memcmp(magic, "Q35TOK01", 8) == 0 && fread(h, 4, 8, f) == 8 &&
      fseek(f, 4L * (long)h[7], SEEK_CUR) == 0) {
    llb_stop_add(h[4]);
    static const char eot[] = "<|endoftext|>", ime[] = "<|im_end|>";
    int got_eot = 0, got_ime = 0;
    char buf[sizeof eot];
    for (uint32_t i = 0; i < h[0] && !(got_eot && got_ime); i += 1) {
      uint16_t l;
      uint8_t ty;
      if (fread(&l, 2, 1, f) != 1 || fread(&ty, 1, 1, f) != 1) {
        break;
      }
      if (l == sizeof eot - 1 || l == sizeof ime - 1) {
        if (fread(buf, 1, l, f) != l) {
          break;
        }
        if (l == sizeof eot - 1 && memcmp(buf, eot, l) == 0) {
          got_eot = 1;
          llb_stop_add(i);
        } else if (l == sizeof ime - 1 && memcmp(buf, ime, l) == 0) {
          got_ime = 1;
          if (i != h[4]) {
            llb_stop_add(i);
          }
        }
      } else if (fseek(f, l, SEEK_CUR) != 0) {
        break;
      }
    }
  }
  fclose(f);
}

int llb_load(const char* model_dir, const char* cache_dir) {
  llb_err[0] = 0;
  rs_reap();
  if (llb_resident() && strcmp(llb_resident(), model_dir) != 0) {
    llb_close();  // (KEVPHONE) one engine per process: another model's session is released first
  }
  if (llb_spent) {
    return llb_refuse("llb_load");
  }
  llb_runs = 0;  // (ENGINE-REPEAT) the last run was released: this load admits the next one
  if (strcmp(llb_dir, model_dir) != 0) {
    llb_head_close();  // another model: its own head and tokenizer
  }
  snprintf(llb_dir, sizeof llb_dir, "%s", model_dir);
  snprintf(llb_cache, sizeof llb_cache, "%s", cache_dir);
  char p[1200];
  snprintf(p, sizeof p, "%s/manifest.json", model_dir);
  if (access(p, R_OK) != 0) {
    return llb_fail("no llb sidecar (manifest.json) in ", model_dir);
  }
  DIR* d = opendir(model_dir);
  if (!d) {
    return llb_fail("cannot open ", model_dir);
  }
  llb_gguf[0] = 0;
  for (struct dirent* e; (e = readdir(d)) != NULL;) {
    size_t n = strlen(e->d_name);
    if (n > 5 && strcmp(e->d_name + n - 5, ".gguf") == 0) {
      if (llb_gguf[0]) {
        closedir(d);
        return llb_fail("more than one .gguf in ", model_dir);
      }
      snprintf(llb_gguf, sizeof llb_gguf, "%s/%s", model_dir, e->d_name);
    }
  }
  closedir(d);
  if (!llb_gguf[0]) {
    return llb_fail("no .gguf in ", model_dir);
  }
  struct stat st;
  if (stat(llb_gguf, &st) != 0) {
    return llb_fail("cannot stat ", llb_gguf);
  }
  llb_gguf_bytes = (u64)st.st_size;
  mkdir(cache_dir, 0755);
  snprintf(p, sizeof p, "%s/run", cache_dir);
  mkdir(p, 0755);
  snprintf(llb_exe, sizeof llb_exe, "%s/sys1mac", cache_dir);
  setenv("LLB_GGUF", llb_gguf, 1);
  llb_stop_read(model_dir);
  llb_loaded = 1;
  return 0;
}

// The compiled Metal program, in the cache: the Mac's --gpu-build, done once.
static double llb_gpu_cache(void) {
  const char* path = gpu_path();
  if (access(path, R_OK) == 0) {
    return 0;
  }
  double t0 = llb_ms();
  @autoreleasepool {
    if (!gpu_probe()) {
      llb_fail("no Metal device", NULL);
      return -1;
    }
#if TARGET_OS_SIMULATOR
    // the simulator's MTLBinaryArchive asserts ("Target device architecture is
    // nil"): compile the program to a pipeline, uncached, to show it builds here
    if (gpu_pipe(gpu_desc(), nil) == nil) {
      llb_fail("the simulator could not build the Metal pipeline", NULL);
      return -1;
    }
    fprintf(stderr, "llb: simulator: Metal program compiled in %.0f ms (no binary archive here)\n",
      llb_ms() - t0);
    return llb_ms() - t0;
#endif
    if (!gpu_make(path)) {
      llb_fail("cannot write the Metal program cache ", path);
      return -1;
    }
  }
  return llb_ms() - t0;
}

static char* llb_slurp(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    return strdup("");
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* s = malloc((size_t)n + 1);
  size_t got = fread(s, 1, (size_t)n, f);
  s[got] = 0;
  fclose(f);
  return s;
}

// a JSON error line, as dual.c's refuse(): {"error":"..."}
static char* llb_jerr(const char* msg) {
  size_t n = strlen(msg), k = 0;
  char* o = malloc(n * 6 + 16);
  k += (size_t)sprintf(o, "{\"error\":\"");
  for (const unsigned char* c = (const unsigned char*)msg; *c; c += 1) {
    if (*c == '"' || *c == '\\') {
      o[k++] = '\\';
      o[k++] = (char)*c;
    } else if (*c < 0x20) {
      k += (size_t)sprintf(o + k, "\\u%04x", *c);
    } else {
      o[k++] = (char)*c;
    }
  }
  sprintf(o + k, "\"}");
  return o;
}

// (ENGINE-ONE) Sys1.next: the next queued question as engine argv words ("ids ... --steps 1"),
// its row positions switched on in the hook (llb_hidden_positions; sys1.c set them all once), the
// rows to llb_hpath. 0 = EOF (the srv loop ends, main returns).
static int dual_next_q(const char* dir, unsigned ctx, char** cmd, const char** hidden, unsigned* n) {
  (void)dir;
  while (llb_qi < llb_qn && llb_replies[llb_q[llb_qi].r]) {
    llb_qi += 1;  // its request already failed
  }
  if (llb_qi >= llb_qn) {
    *cmd = NULL;
    return 0;
  }
  llb_dq* q = &llb_q[llb_qi];
  if ((unsigned)q->n > ctx) {
    llb_replies[q->r] = llb_jerr("request exceeds the KV capacity fixed at launch");
    return dual_next_q(dir, ctx, cmd, hidden, n);
  }
  memset(llb_hidden_positions, 0, sizeof llb_hidden_positions);
  for (int i = 0; i < q->npos; i += 1) {
    llb_hidden_positions[q->pos[i]] = 1;
  }
  unlink(llb_hpath);
  size_t cap = (size_t)q->n * 12 + 64, k = 0;
  char* c = malloc(cap);
  for (int i = 0; i < q->n; i += 1) {
    k += (size_t)snprintf(c + k, cap - k, "%d ", q->ids[i]);
  }
  if (llb_fastpf > 0) {  // (ENGINE-DECIDECHUNK) the fast tier (the srv loop's run.prompt)
    k += (size_t)snprintf(c + k, cap - k, "--fastpf %d ", llb_fastpf);
  }
  snprintf(c + k, cap - k, "--steps 1");
  *cmd = c;
  *hidden = llb_hpath;
  *n = (unsigned)q->n;
  llb_t_fwd = llb_ms();
  return 1;
}

// (ENGINE-ONE) Sys1.done: the question's prefill ran and its rows are in llb_hpath; hand them to
// the head; after the request's last question, its reply
static void dual_done_q(void) {
  llb_dq* q = &llb_q[llb_qi];
  double t1 = llb_ms();
  llb_fwd_ms += t1 - llb_t_fwd;
  char err[512] = "";
  if (llb_rawdir) {  // (ENGINE-DECIDECHUNK) llb_rows: keep the item's rows, no head
    char keep[1300];
    snprintf(keep, sizeof keep, "%s/rows-%d.hidden", llb_rawdir, q->r);
    if (rename(llb_hpath, keep) != 0) {
      llb_replies[q->r] = llb_jerr("no rows were written");
    } else {
      llb_replies[q->r] = strdup("ok");
    }
    llb_qi += 1;
    return;
  }
  jvh_job* j = llb_jobs[q->r];
  if (jvh_job_rows_file(j, (size_t)q->k, llb_hpath, err, sizeof err) != 0) {
    llb_replies[q->r] = llb_jerr(err);
  }
  if (getenv("LLB_DECIDE_TRACE")) {  // keep the question's rows: <cache>/run/rows-r<r>-q<k>.hidden
    char keep[1300];
    snprintf(keep, sizeof keep, "%s/run/rows-r%d-q%d.hidden", llb_cache, q->r, q->k);
    rename(llb_hpath, keep);
  }
  if (!llb_replies[q->r] && (size_t)q->k + 1 == jvh_job_count(j)) {
    char* r = jvh_job_reply(j, llb_model, 0 /* exact f32 */, err, sizeof err);
    llb_replies[q->r] = r ? r : llb_jerr(err);
  }
  llb_head_ms += llb_ms() - t1;
  llb_qi += 1;
}

// (KEVPHONE) the resident session (see the top of this file). One engine thread runs main in the SYS1 srv loop; the
// caller's thread posts one job at a time (llb_generate / llb_decide hold the library, one call at a time, as
// before) and waits for it; dual_next / dual_done below run on the engine thread.
static pthread_mutex_t rs_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  rs_cv = PTHREAD_COND_INITIALIZER;
static int       rs_live;      // an engine thread exists (llb_open .. llb_close)
static int       rs_ready;     // its main reached the srv loop: the weights are resident
static int       rs_ended;     // its main returned
static int       rs_quit;      // llb_close: end the loop
static int       rs_kind;      // the job: 0 none, 1 generate, 2 decide
static int       rs_sent;      // generate: its command was handed to the loop
static int       rs_done;      // the job is over (its words taken)
static char*     rs_cmd;       // generate: the command words (the loop frees them)
static unsigned  rs_cmdn;
static int       rs_exit;
static unsigned  rs_cap;       // the loop's KV capacity (2^dT, fixed at llb_open)
static pthread_t rs_tid;
static char      rs_dir[1024];
static char      rs_op[1200], rs_ep[1200];
static char*     rs_out;       // the last job's stdout / stderr
static char*     rs_err;
static char*     rs_argv[11];
static char      rs_spans[32], rs_thrs[16], rs_cs[16];
static char      rs_rep[512];
static double    rs_load_ms, rs_open_ms, rs_compile_ms, rs_warmup_ms;
static long      rs_jobs, rs_loads;
static u64       rs_span_bytes;

// the engine thread, between jobs or at the loop's end: the job's words from the session's files, which are then
// emptied (so a reader of err.txt, Bend's logits poller, starts every job on an empty file)
static void rs_take(void) {
  fflush(llb_fout);
  fflush(llb_ferr);
  free(rs_out);
  free(rs_err);
  rs_out = llb_slurp(rs_op);
  rs_err = llb_slurp(rs_ep);
  if (ftruncate(fileno(llb_fout), 0) == 0) {
    fseek(llb_fout, 0, SEEK_SET);
  }
  if (ftruncate(fileno(llb_ferr), 0) == 0) {
    fseek(llb_ferr, 0, SEEK_SET);
  }
}

static void rs_end_job(void) {  // under rs_mu
  rs_take();
  rs_done = 1;
  pthread_cond_broadcast(&rs_cv);
}

// Sys1.next: the open session's next command. The first call is the load's end (the weights are resident); then the
// decide's questions, or the generate's one command, or a wait for the next job; EOF once llb_close asks.
int dual_next(const char* dir, unsigned ctx, char** cmd, const char** hidden, unsigned* n) {
  if (!rs_live) {
    return dual_next_q(dir, ctx, cmd, hidden, n);
  }
  pthread_mutex_lock(&rs_mu);
  if (!rs_ready) {
    rs_take();  // the load's words: "resident <bytes> bytes in <ms> ms"
    rs_cap = ctx;
    rs_ready = 1;
    pthread_cond_broadcast(&rs_cv);
  }
  for (;;) {
    if (rs_kind == 2 && !rs_done) {
      pthread_mutex_unlock(&rs_mu);
      int got = dual_next_q(dir, ctx, cmd, hidden, n);
      pthread_mutex_lock(&rs_mu);
      if (got) {
        pthread_mutex_unlock(&rs_mu);
        return 1;
      }
      rs_end_job();
    } else if (rs_kind == 1 && !rs_sent) {
      rs_sent = 1;
      *cmd = rs_cmd;
      rs_cmd = NULL;
      *hidden = NULL;  // the hook off: a generate
      *n = rs_cmdn;
      pthread_mutex_unlock(&rs_mu);
      return 1;
    } else if (rs_quit) {
      pthread_mutex_unlock(&rs_mu);
      *cmd = NULL;
      return 0;
    } else {
      pthread_cond_wait(&rs_cv, &rs_mu);
    }
  }
}

// Sys1.done: a generate's run is over (its job ends here); a decide question goes to the head as before
void dual_done(void) {
  if (rs_live) {
    pthread_mutex_lock(&rs_mu);
    if (rs_kind == 1) {
      rs_end_job();
      pthread_mutex_unlock(&rs_mu);
      return;
    }
    pthread_mutex_unlock(&rs_mu);
  }
  dual_done_q();
}

static void* rs_main(void* arg) {
  (void)arg;
  int code = llb_run_main(10, rs_argv);  // main, then the release (bend_close, gguf_close), on this thread
  pthread_mutex_lock(&rs_mu);
  rs_exit = code;
  rs_ended = 1;
  if (!rs_ready || (rs_kind && !rs_done)) {
    rs_take();  // why it ended
    rs_done = rs_kind != 0;
  }
  pthread_cond_broadcast(&rs_cv);
  pthread_mutex_unlock(&rs_mu);
  return NULL;
}

const char* llb_resident(void) { return rs_live && !rs_ended ? rs_dir : NULL; }


void llb_close(void);
static int llb_generate_rs(const uint32_t* ids, int n_ids, int n_new, uint32_t* out, llb_stats* st);

// a session whose main ended by itself (it never does, short of an engine error) is closed before anything else runs
static void rs_reap(void) {
  if (rs_live && rs_ended) {
    llb_close();
  }
}

const char* llb_resident_report(void) {
  const char* m = strrchr(rs_dir, '/');
  snprintf(rs_rep, sizeof rs_rep, "resident=%s model=%s cap=%u loads=%ld jobs=%ld load_ms=%.0f open_ms=%.0f "
    "compile_ms=%.0f warmup_ms=%.0f span_mb=%llu footprint_mb=%ld", llb_resident() ? "yes" : "no",
    rs_dir[0] ? (m ? m + 1 : rs_dir) : "-", rs_cap, rs_loads, rs_jobs, rs_load_ms, rs_open_ms, rs_compile_ms,
    rs_warmup_ms, (unsigned long long)(rs_span_bytes >> 20),
    llb_footprint_mb());
  return rs_rep;
}

void llb_close(void) {
  if (!rs_live) {
    return;
  }
  pthread_mutex_lock(&rs_mu);
  rs_quit = 1;
  pthread_cond_broadcast(&rs_cv);
  pthread_mutex_unlock(&rs_mu);
  pthread_join(rs_tid, NULL);
  fclose(llb_fout);
  fclose(llb_ferr);
  llb_fout = llb_ferr = NULL;
  free(rs_cmd);
  rs_cmd = NULL;
  rs_live = rs_ready = rs_ended = rs_quit = rs_kind = rs_sent = rs_done = 0;
  rs_cap = 0;
  // the released session's model stays named in the report (resident=no)
}

int llb_open(const char* model_dir, const char* cache_dir, int ctx, int flags, llb_stats* st) {
  llb_stats z = {0};
  if (llb_resident() && strcmp(rs_dir, model_dir) == 0) {
    z.span_bytes = rs_span_bytes;  // already resident: nothing to load
    if (st) {
      *st = z;
    }
    return 0;
  }
  llb_close();
  if (llb_load(model_dir, cache_dir) != 0) {
    return -1;
  }
  if (llb_spent) {
    return llb_refuse("llb_open");
  }
  double t0 = llb_ms();
  z.gguf_bytes = llb_gguf_bytes;
  z.compile_ms = llb_gpu_cache();
  if (z.compile_ms < 0) {
    return -1;
  }
  z.span_bytes = llb_span();
  if (gpu_dev == nil && !gpu_probe()) {
    return llb_fail("no Metal device", NULL);
  }
  u64 most = (u64)[gpu_dev maxBufferLength];
  if (z.span_bytes > most && !getenv("LLB_SPAN_NOCHECK")) {
    snprintf(llb_err, sizeof llb_err, "GPU span %llu MB > this device's maxBufferLength %llu MB "
      "(the engine needs one MTLBuffer for weights + state)",
      (unsigned long long)(z.span_bytes >> 20), (unsigned long long)(most >> 20));
    return -1;
  }
  long ncpu = (long)[[NSProcessInfo processInfo] activeProcessorCount];
  const char* te = getenv("LLB_THREADS");
  long thr = te && atol(te) > 0 ? atol(te) : (ncpu < 8 ? ncpu : 8);
  // the srv loop's launch argv "0 --steps C": its KV capacity 2^dT = C, a power of two in 64 .. 2^27
  // (METAL-ANYSHAPE: the engine sizes the KV and its attention planes from C in their own Metal buffer and
  // refuses, naming the bytes, when they exceed the device's largest buffer; the span below no longer holds
  // a per-layer copy of the KV for the fused step)
  int cap = 64;
  while (cap < ctx && cap < (1 << 27)) {
    cap *= 2;
  }
  snprintf(rs_spans, sizeof rs_spans, "%lluMB", (unsigned long long)(z.span_bytes >> 20));
  snprintf(rs_thrs, sizeof rs_thrs, "%ld", thr);
  snprintf(rs_cs, sizeof rs_cs, "%d", cap);
  char* argv[] = { llb_exe, "--gpu", rs_spans, "--threads", rs_thrs, "--", llb_dir, "0", "--steps", rs_cs, NULL };
  memcpy(rs_argv, argv, sizeof argv);
  snprintf(rs_op, sizeof rs_op, "%s/run/out.txt", llb_cache);
  snprintf(rs_ep, sizeof rs_ep, "%s/run/err.txt", llb_cache);
  llb_fout = fopen(rs_op, "w");
  llb_ferr = fopen(rs_ep, "w");
  if (!llb_fout || !llb_ferr) {
    if (llb_fout) fclose(llb_fout);
    if (llb_ferr) fclose(llb_ferr);
    llb_fout = llb_ferr = NULL;
    return llb_fail("cannot open the session's files in ", llb_cache);
  }
  setvbuf(llb_ferr, NULL, _IONBF, 0);  // the logits lines as they happen (Bend's poller)
  fprintf(stderr, "llb: resident: span %llu MB, threads %ld, cap %d, engine stderr -> %s\n",
    (unsigned long long)(z.span_bytes >> 20), thr, cap, rs_ep);
  snprintf(rs_dir, sizeof rs_dir, "%s", model_dir);
  rs_live = 1;
  rs_ready = rs_ended = rs_quit = rs_kind = rs_sent = rs_done = 0;
  setenv("LLB_SYS1", "1", 1);
  pthread_attr_t at;
  pthread_attr_init(&at);
  pthread_attr_setstacksize(&at, 16u << 20);  // the engine's main is deep: a main-thread-sized stack, as Bend gives it
  int pe = pthread_create(&rs_tid, &at, rs_main, NULL);
  pthread_attr_destroy(&at);
  if (pe != 0) {
    unsetenv("LLB_SYS1");
    fclose(llb_fout);
    fclose(llb_ferr);
    llb_fout = llb_ferr = NULL;
    rs_live = 0;
    return llb_fail("cannot start the engine thread", NULL);
  }
  pthread_mutex_lock(&rs_mu);
  while (!rs_ready && !rs_ended) {
    pthread_cond_wait(&rs_cv, &rs_mu);
  }
  int ok = rs_ready && !rs_ended;
  pthread_mutex_unlock(&rs_mu);
  unsetenv("LLB_SYS1");
  free(llb_errlog);
  llb_errlog = strdup(rs_err ? rs_err : "");
  fputs(llb_errlog, stderr);
  if (!ok) {
    z.exit_code = rs_exit;
    llb_close();
    if (st) {
      *st = z;
    }
    return llb_fail("the engine ended before its weights were resident; see llb_log()", NULL);
  }
  for (char* l = llb_errlog; *l;) {
    unsigned long long a = 0, b = 0;
    if (sscanf(l, "resident %llu bytes in %llu ms", &a, &b) == 2) {
      z.load_ms = (double)b;
    }
    char* nl = strchr(l, '\n');
    l = nl ? nl + 1 : l + strlen(l);
  }
  rs_loads += 1;
  rs_span_bytes = z.span_bytes;
  // the first prefill after a load sets up what the fast chunk path keeps (~0.15 s Mac, more on the phone): a
  // throw-away 16-id chat (the fast tier, two new ids) pays it here, inside the open, so the first request runs at the
  // session's own speed. Nothing carries over: every job starts at pos 0 on fresh states. LLB_OPEN_NOWARMUP (a request
  // is waiting on this open: its own first job pays that set-up, which is less than the warm-up) or
  // LLB_OPEN_WARMUP=0: none.
  char* loadlog = strdup(llb_errlog);
  const char* wu = getenv("LLB_OPEN_WARMUP");
  rs_warmup_ms = 0;
  if (!(flags & LLB_OPEN_NOWARMUP) && !(wu && strcmp(wu, "0") == 0)) {
    double tw = llb_ms();
    uint32_t wids[16], wout[2];
    for (int i = 0; i < 16; i += 1) {
      wids[i] = 198;  // "\n"
    }
    rs_jobs -= 1;  // not a request
    if (llb_generate_rs(wids, 16, 2, wout, NULL) < 0) {
      free(loadlog);
      return -1;  // the session ended under it (closed there)
    }
    rs_warmup_ms = llb_ms() - tw;
  }
  z.total_ms = llb_ms() - t0;
  rs_load_ms = z.load_ms;
  rs_open_ms = z.total_ms;
  rs_compile_ms = z.compile_ms;
  free(llb_errlog);
  llb_errlog = loadlog;  // the load's words, not the warm-up's
  if (st) {
    *st = z;
  }
  return 0;
}

// a job on the open session; 0 once it is over, -1 when the session ended under it (then closed here)
static int rs_job(int kind, char* cmd, unsigned n) {
  pthread_mutex_lock(&rs_mu);
  rs_kind = kind;
  rs_cmd = cmd;
  rs_cmdn = n;
  rs_sent = rs_done = 0;
  pthread_cond_broadcast(&rs_cv);
  while (!rs_done && !rs_ended) {
    pthread_cond_wait(&rs_cv, &rs_mu);
  }
  int ended = rs_ended;
  rs_kind = 0;
  rs_done = 0;
  pthread_mutex_unlock(&rs_mu);
  rs_jobs += 1;
  free(llb_errlog);
  llb_errlog = strdup(rs_err ? rs_err : "");
  if (ended) {
    int code = rs_exit;
    llb_close();
    snprintf(llb_err, sizeof llb_err, "the resident engine ended under a job (exit %d); see llb_log()", code);
    return -1;
  }
  return 0;
}

// a chat on the open session: the same command a one-shot run's argv carries (ids, its default prefill tier, --steps,
// the stop ids), so the same ids come out; no load, no release
static int llb_generate_rs(const uint32_t* ids, int n_ids, int n_new, uint32_t* out, llb_stats* st) {
  llb_stats z = {0};
  double t0 = llb_ms();
  z.gguf_bytes = llb_gguf_bytes;
  z.span_bytes = rs_span_bytes;
  // the one-shot run's default tier (m.dfk, the 4B): the fast chunk (K 512) at nt >= 13 when its own dT (2^dT >=
  // nt + n - 1) is >= 5, else the exact path (the srv loop's default, fk 0)
  int dT = 0;
  while (dT < 17 && (1u << dT) < (unsigned)(n_ids + n_new - 1)) {
    dT += 1;
  }
  int fast = n_ids >= 13 && dT >= 5;
  const char* es = getenv("LLB_EARLYSTOP");
  int nstop = es && strcmp(es, "0") == 0 ? 0 : llb_nstop;
  size_t cap = (size_t)n_ids * 12 + 96, k = 0;
  char* c = malloc(cap);
  for (int i = 0; i < n_ids; i += 1) {
    k += (size_t)snprintf(c + k, cap - k, "%u ", ids[i]);
  }
  if (fast) {
    k += (size_t)snprintf(c + k, cap - k, "--fastpf 512 ");
  }
  k += (size_t)snprintf(c + k, cap - k, "--steps %d", n_new);
  for (int i = 0; i < nstop; i += 1) {
    k += (size_t)snprintf(c + k, cap - k, " --stop %u", llb_stop[i]);
  }
  if (rs_job(1, c, (unsigned)n_ids) != 0) {
    if (st) {
      *st = z;
    }
    return -1;
  }
  int got = 0;
  for (char* s = rs_out; s && *s && got < n_new;) {
    char* end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s) {
      break;
    }
    out[got++] = (uint32_t)v;
    s = end;
    while (*s == '\n' || *s == ' ') s += 1;
  }
  for (char* l = llb_errlog; *l;) {
    unsigned long long a = 0, b = 0, cc = 0;
    if (sscanf(l, "resident %llu bytes in %llu ms", &a, &b) == 2) {
      z.load_ms = (double)b;  // never on a resident job: the proof that nothing was loaded
    } else if (sscanf(l, "chunkf %llu %llu %llu", &a, &b, &cc) == 3) {
      z.prefill_ms += (double)cc;
    } else if (sscanf(l, "step %llu %llu", &a, &b) == 2) {
      if ((long long)a < (long long)n_ids - 1) {
        z.prefill_ms += (double)b;
      } else {
        z.step_ms_sum += (double)b;
        z.steps += 1;
      }
    }
    char* nl = strchr(l, '\n');
    l = nl ? nl + 1 : l + strlen(l);
  }
  z.total_ms = llb_ms() - t0;
  if (st) {
    *st = z;
  }
  return got;
}

// a decide on the open session: its questions through the loop, as a one-shot run's (the head, the hook)
static int llb_srv_rs(llb_dstats* zp) {
  llb_dstats z = *zp;
  const char* fe = getenv("LLB_DECIDE_FASTPF");
  llb_fastpf = fe ? atoi(fe) : 512;
  z.fastpf = llb_fastpf;
  z.gguf_bytes = llb_gguf_bytes;
  z.span_bytes = rs_span_bytes;
  if (llb_qn > 0) {
    if (rs_job(2, NULL, 0) != 0) {
      return -1;
    }
    for (char* l = llb_errlog; *l;) {
      unsigned long long a = 0, b = 0, c = 0;
      if (sscanf(l, "resident %llu bytes in %llu ms", &a, &b) == 2) {
        z.load_ms = (double)b;
      } else if (sscanf(l, "step %llu %llu", &a, &b) == 2) {
        z.step_ms_sum += (double)b;
        z.steps += 1;
      } else if (sscanf(l, "chunkf %llu %llu %llu", &a, &b, &c) == 3) {
        z.chunk_ms_sum += (double)c;
        z.chunks += 1;
        z.chunk_tokens += (int)b;
      }
      char* nl = strchr(l, '\n');
      l = nl ? nl + 1 : l + strlen(l);
    }
    unlink(llb_hpath);
  }
  *zp = z;
  return 0;
}

int llb_generate(const uint32_t* ids, int n_ids, int n_new, uint32_t* out, llb_stats* st) {
  llb_stats z = {0};
  llb_err[0] = 0;
  if (!llb_loaded) {
    return llb_fail("llb_generate before llb_load", NULL);
  }
  if (llb_runs > 0 || llb_spent) {
    // (ENGINE-REPEAT) each run sets up and releases the engine's statics (heap, pool threads, GPU span)
    return llb_refuse("llb_generate");
  }
  if (n_ids < 1 || n_new < 1) {
    return llb_fail("need at least one prompt id and one new id", NULL);
  }
  rs_reap();
  if (llb_resident() && strcmp(rs_dir, llb_dir) == 0) {
    if ((unsigned)(n_ids + n_new) <= rs_cap) {
      return llb_generate_rs(ids, n_ids, n_new, out, st);
    }
    llb_close();  // (KEVPHONE) longer than the session's KV: this one runs as a one-shot (the next open loads again)
  }
  double t0 = llb_ms();
  z.gguf_bytes = llb_gguf_bytes;
  z.compile_ms = llb_gpu_cache();
  if (z.compile_ms < 0) {
    return -1;
  }
  z.span_bytes = llb_span();
  // the engine's whole span is ONE no-copy MTLBuffer: refuse here, not by its _exit
  if (gpu_dev == nil && !gpu_probe()) {
    return llb_fail("no Metal device", NULL);
  }
  u64 most = (u64)[gpu_dev maxBufferLength];
  if (z.span_bytes > most && !getenv("LLB_SPAN_NOCHECK")) {  // LLB_SPAN_NOCHECK: try anyway
    if (st) {
      *st = z;
    }
    snprintf(llb_err, sizeof llb_err, "GPU span %llu MB > this device's maxBufferLength %llu MB "
      "(the engine needs one MTLBuffer for weights + state)",
      (unsigned long long)(z.span_bytes >> 20), (unsigned long long)(most >> 20));
    return -1;
  }
  long ncpu = (long)[[NSProcessInfo processInfo] activeProcessorCount];
  const char* te = getenv("LLB_THREADS");
  long thr = te && atol(te) > 0 ? atol(te) : (ncpu < 8 ? ncpu : 8);

  // argv as the Mac CLI: sys1mac --gpu <MB>MB --threads T -- <dir> ids... --steps n
  int argc = 0;
  char** argv = calloc((size_t)n_ids + 16, sizeof(char*));
  char spans[32], thrs[16], ns[16];
  snprintf(spans, sizeof spans, "%lluMB", (unsigned long long)(z.span_bytes >> 20));
  snprintf(thrs, sizeof thrs, "%ld", thr);
  snprintf(ns, sizeof ns, "%d", n_new);
  argv[argc++] = llb_exe;
  argv[argc++] = "--gpu";
  argv[argc++] = spans;
  argv[argc++] = "--threads";
  argv[argc++] = thrs;
  argv[argc++] = "--";
  argv[argc++] = llb_dir;
  for (int i = 0; i < n_ids; i += 1) {
    char b[16];
    snprintf(b, sizeof b, "%u", ids[i]);
    argv[argc++] = strdup(b);
  }
  argv[argc++] = "--steps";
  argv[argc++] = ns;
  // (ENGINE-EARLYSTOP) n_new is a maximum: the run ends at the first stop id (it is the last id returned).
  // LLB_EARLYSTOP=0: exactly n_new ids, as before.
  const char* es = getenv("LLB_EARLYSTOP");
  int nstop = es && strcmp(es, "0") == 0 ? 0 : llb_nstop;
  char stops[3][16];
  for (int i = 0; i < nstop; i += 1) {
    snprintf(stops[i], sizeof stops[i], "%u", llb_stop[i]);
    argv[argc++] = "--stop";
    argv[argc++] = stops[i];
  }

  char op[1200], ep[1200];
  snprintf(op, sizeof op, "%s/run/out.txt", llb_cache);
  snprintf(ep, sizeof ep, "%s/run/err.txt", llb_cache);
  // a fatal engine error is its _exit(1), which skips the restore below: say where
  // its words go (the scripts collect this file)
  fprintf(stderr, "llb: engine stderr -> %s\n", ep);
  fflush(stdout);
  fflush(stderr);
  int o1 = dup(1), o2 = dup(2);
  int f1 = open(op, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  int f2 = open(ep, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  dup2(f1, 1);
  dup2(f2, 2);
  close(f1);
  close(f2);
  fprintf(stderr, "llb: span %llu MB, threads %ld, %d ids, %d new, stop ids %s %s %s\n",
    (unsigned long long)(z.span_bytes >> 20), thr, n_ids, n_new, nstop ? stops[0] : "none", nstop > 1 ? stops[1] : "",
    nstop > 2 ? stops[2] : "");
  llb_runs += 1;
  z.exit_code = llb_run_main(argc, argv);
  fflush(stdout);
  fflush(stderr);
  dup2(o1, 1);
  dup2(o2, 2);
  close(o1);
  close(o2);
  for (int i = 7; i < 7 + n_ids; i += 1) {
    free(argv[i]);
  }
  free(argv);

  free(llb_errlog);
  llb_errlog = llb_slurp(ep);
  fputs(llb_errlog, stderr);  // the engine's own diagnostics, to the console
  char* o = llb_slurp(op);
  int got = 0;
  for (char* s = o; *s && got < n_new;) {
    char* end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s) {
      break;
    }
    out[got++] = (uint32_t)v;
    s = end;
    while (*s == '\n' || *s == ' ') s += 1;
  }
  free(o);
  for (char* l = llb_errlog; l && *l;) {
    unsigned long long a = 0, b = 0, c = 0;
    if (sscanf(l, "resident %llu bytes in %llu ms", &a, &b) == 2) {
      z.load_ms = (double)b;
    } else if (sscanf(l, "chunkf %llu %llu %llu", &a, &b, &c) == 3) {
      z.prefill_ms += (double)c;
    } else if (sscanf(l, "step %llu %llu", &a, &b) == 2) {
      if ((long long)a < (long long)n_ids - 1) {
        z.prefill_ms += (double)b;  // exact tier: prompt positions are steps too
      } else {
        z.step_ms_sum += (double)b;
        z.steps += 1;
      }
    }
    char* nl = strchr(l, '\n');
    l = nl ? nl + 1 : NULL;
  }
  z.total_ms = llb_ms() - t0;
  if (st) {
    *st = z;
  }
  if (z.exit_code != 0) {
    llb_fail("engine exit code != 0; see llb_log()", NULL);
    return -1;
  }
  return got;
}

// (ENGINE-ONE) decide: n /v1/systemone bodies -> n reply lines (llama-server's layout), one engine
// run. Each question is rendered by the head, tokenized by the engine's tokenizer (the sidecar's
// tokenizer.bin, parse_special), prefilled from pos 0 on fresh states (the srv loop, --steps 1), and
// its rows at the option markers + the pointer go to the head as they are produced.
static int  llb_srv(int maxT, int n, llb_dstats* zp);
static void llb_qfree(void);

int llb_decide(const char* const* bodies, int n, char** replies, llb_dstats* ds) {
  llb_dstats z = {0};
  llb_err[0] = 0;
  double t0 = llb_ms();
  if (!llb_loaded) {
    return llb_fail("llb_decide before llb_load", NULL);
  }
  if (llb_runs > 0 || llb_spent) {
    return llb_refuse("llb_decide");
  }
  if (n < 1) {
    return llb_fail("need at least one request", NULL);
  }
  char err[512] = "";
  if (!llb_head) {
    llb_head = jvh_session_open(llb_gguf, err, sizeof err);
    if (!llb_head) {
      return llb_fail("head: ", err);
    }
  }
  if (!llb_tok_on) {
    if (q35_tok_load(&llb_tok, llb_dir) != 0) {
      return llb_fail("sidecar requires tokenizer.bin (llb-tok extract) in ", llb_dir);
    }
    llb_tok_on = 1;
  }
  z.head_open_ms = llb_ms() - t0;
  const char* slash = strrchr(llb_gguf, '/');
  llb_model = slash ? slash + 1 : llb_gguf;
  snprintf(llb_hpath, sizeof llb_hpath, "%s/run/rows.hidden", llb_cache);

  // parse + render + tokenize + layout every question first: a refusal costs no engine time
  llb_replies = replies;
  llb_jobs = calloc((size_t)n, sizeof *llb_jobs);
  int qcap = 0, maxT = 0;
  llb_qn = llb_qi = 0;
  llb_fwd_ms = llb_head_ms = 0;
  for (int r = 0; r < n; r += 1) {
    replies[r] = NULL;
    jvh_job* j = jvh_job_new(llb_head, bodies[r], strlen(bodies[r]), err, sizeof err);
    if (!j) {
      replies[r] = llb_jerr(err);
      continue;
    }
    llb_jobs[r] = j;
    for (size_t k = 0; k < jvh_job_count(j) && !replies[r]; k += 1) {
      int idcap = 16384, *ids = malloc(sizeof *ids * (size_t)idcap);
      int T = q35_tok_encode(&llb_tok, jvh_job_prompt(j, k), 1, ids, idcap);
      int npos = (int)jvh_job_options(j, k) + 1;
      uint32_t* pos = malloc(sizeof *pos * (size_t)npos);
      if (T < 1) {
        replies[r] = llb_jerr("prompt must tokenize to 1..16384 tokens");
      } else if (jvh_job_layout(j, k, ids, (size_t)T, JVH_MARKER_ID_KEV, pos, err, sizeof err) < 0) {
        replies[r] = llb_jerr(err);
      }
      if (replies[r]) {
        free(ids);
        free(pos);
        break;
      }
      if (llb_qn == qcap) {
        qcap = qcap ? 2 * qcap : 8;
        llb_q = realloc(llb_q, sizeof *llb_q * (size_t)qcap);
      }
      llb_q[llb_qn++] = (llb_dq){ r, (int)k, ids, T, pos, npos };
      if (getenv("LLB_DECIDE_TRACE")) {  // J0: the engine tokenizer's ids and the layout, per question
        fprintf(stderr, "llb: decide r%d q%zu %s T=%d pos", r, k, jvh_job_id(j, k), T);
        for (int i = 0; i < npos; i += 1) {
          fprintf(stderr, " %u", pos[i]);
        }
        fprintf(stderr, " ids");
        for (int i = 0; i < T; i += 1) {
          fprintf(stderr, " %d", ids[i]);
        }
        fprintf(stderr, "\n");
      }
      maxT = T > maxT ? T : maxT;
      z.questions += 1;
      z.input_tokens += T;
    }
  }

  int rs = 0;
  rs_reap();
  if (llb_resident() && strcmp(rs_dir, llb_dir) == 0) {
    if ((unsigned)maxT <= rs_cap) {
      rs = 1;
    } else {
      llb_close();  // (KEVPHONE) a question longer than the session's KV: a one-shot run
    }
  }
  if ((rs ? llb_srv_rs(&z) : llb_srv(maxT, n, &z)) < 0) {
    for (int r = 0; r < n; r += 1) {
      free(replies[r]);
      replies[r] = NULL;
      jvh_job_free(llb_jobs[r]);
    }
    llb_qfree();
    return -1;
  }
  z.forward_ms = llb_fwd_ms;
  z.head_ms = llb_head_ms;
  int answered = 0;
  for (int r = 0; r < n; r += 1) {
    if (!replies[r]) {
      replies[r] = llb_jerr("the engine ended before this request was answered");
    } else if (strncmp(replies[r], "{\"error\"", 8) != 0) {
      answered += 1;
    }
    jvh_job_free(llb_jobs[r]);
  }
  llb_qfree();
  z.total_ms = llb_ms() - t0;
  if (ds) {
    *ds = z;
  }
  if (z.exit_code != 0) {
    llb_fail("engine exit code != 0; see llb_log()", NULL);
    return -1;
  }
  return answered;
}

// (ENGINE-DECIDECHUNK) the decide queue's engine run (one srv loop over llb_q), split out of
// llb_decide unchanged so llb_rows shares it; the fast tier's K from LLB_DECIDE_FASTPF (default 512)
static int llb_srv(int maxT, int n, llb_dstats* zp) {
  llb_dstats z = *zp;
  const char* fe = getenv("LLB_DECIDE_FASTPF");
  llb_fastpf = fe ? atoi(fe) : 512;
  z.fastpf = llb_fastpf;
  if (llb_qn > 0) {
    z.gguf_bytes = llb_gguf_bytes;
    z.compile_ms = llb_gpu_cache();
    if (z.compile_ms < 0) {
      return -1;
    }
    z.span_bytes = llb_span();
    if (gpu_dev == nil && !gpu_probe()) {
      return llb_fail("no Metal device", NULL);
    }
    u64 most = (u64)[gpu_dev maxBufferLength];
    if (z.span_bytes > most && !getenv("LLB_SPAN_NOCHECK")) {
      snprintf(llb_err, sizeof llb_err, "GPU span %llu MB > this device's maxBufferLength %llu MB "
        "(the engine needs one MTLBuffer for weights + state)",
        (unsigned long long)(z.span_bytes >> 20), (unsigned long long)(most >> 20));
      return -1;
    }
    long ncpu = (long)[[NSProcessInfo processInfo] activeProcessorCount];
    const char* te = getenv("LLB_THREADS");
    long thr = te && atol(te) > 0 ? atol(te) : (ncpu < 8 ? ncpu : 8);
    // the srv loop's launch argv "0 --steps C": the KV capacity 2^dT >= C >= the longest prompt
    int ctx = 64;
    while (ctx < maxT) {
      ctx *= 2;
    }
    char spans[32], thrs[16], cs[16];
    snprintf(spans, sizeof spans, "%lluMB", (unsigned long long)(z.span_bytes >> 20));
    snprintf(thrs, sizeof thrs, "%ld", thr);
    snprintf(cs, sizeof cs, "%d", ctx);
    char* argv[] = { llb_exe, "--gpu", spans, "--threads", thrs, "--", llb_dir, "0", "--steps", cs, NULL };
    int argc = 10;

    char op[1200], ep[1200];
    snprintf(op, sizeof op, "%s/run/out.txt", llb_cache);
    snprintf(ep, sizeof ep, "%s/run/err.txt", llb_cache);
    fprintf(stderr, "llb: engine stderr -> %s\n", ep);
    fflush(stdout);
    fflush(stderr);
    int o1 = dup(1), o2 = dup(2);
    int f1 = open(op, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int f2 = open(ep, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    dup2(f1, 1);
    dup2(f2, 2);
    close(f1);
    close(f2);
    fprintf(stderr, "llb: decide: span %llu MB, threads %ld, %d requests, %d questions, ctx %d\n",
      (unsigned long long)(z.span_bytes >> 20), thr, n, llb_qn, ctx);
    llb_runs += 1;
    setenv("LLB_SYS1", "1", 1);
    z.exit_code = llb_run_main(argc, argv);
    unsetenv("LLB_SYS1");
    fflush(stdout);
    fflush(stderr);
    dup2(o1, 1);
    dup2(o2, 2);
    close(o1);
    close(o2);
    free(llb_errlog);
    llb_errlog = llb_slurp(ep);
    fputs(llb_errlog, stderr);
    for (char* l = llb_errlog; l && *l;) {
      unsigned long long a = 0, b = 0, c = 0;
      if (sscanf(l, "resident %llu bytes in %llu ms", &a, &b) == 2) {
        z.load_ms = (double)b;
      } else if (sscanf(l, "step %llu %llu", &a, &b) == 2) {
        z.step_ms_sum += (double)b;
        z.steps += 1;
      } else if (sscanf(l, "chunkf %llu %llu %llu", &a, &b, &c) == 3) {  // (ENGINE-DECIDECHUNK)
        z.chunk_ms_sum += (double)c;
        z.chunks += 1;
        z.chunk_tokens += (int)b;
      }
      char* nl = strchr(l, '\n');
      l = nl ? nl + 1 : NULL;
    }
    unlink(llb_hpath);
  }
  *zp = z;
  return 0;
}

static void llb_qfree(void) {
  for (int i = 0; i < llb_qn; i += 1) {
    free(llb_q[i].ids);
    free(llb_q[i].pos);
  }
  free(llb_q);
  free(llb_jobs);
  llb_q = NULL;
  llb_jobs = NULL;
  llb_replies = NULL;
  llb_qn = llb_qi = 0;
}

// (ENGINE-DECIDECHUNK) the rows-parity harness: n items (ids, the hook positions) through the same
// srv loop and hook as llb_decide, fast or exact by LLB_DECIDE_FASTPF, no head (any pack); item i's
// rows (the hook's [pos, width] + row format) to <outdir>/rows-<i>.hidden. Returns items written.
int llb_rows(const uint32_t* const* ids, const int* n_ids, const uint32_t* const* pos, const int* npos,
  int n, const char* outdir, llb_dstats* ds) {
  llb_dstats z = {0};
  llb_err[0] = 0;
  double t0 = llb_ms();
  if (!llb_loaded) {
    return llb_fail("llb_rows before llb_load", NULL);
  }
  if (llb_runs > 0 || llb_spent) {
    return llb_refuse("llb_rows");
  }
  snprintf(llb_hpath, sizeof llb_hpath, "%s/run/rows.hidden", llb_cache);
  char** replies = calloc((size_t)n, sizeof *replies);
  llb_replies = replies;
  llb_rawdir = outdir;
  llb_q = calloc((size_t)n, sizeof *llb_q);
  llb_qn = llb_qi = 0;
  llb_fwd_ms = llb_head_ms = 0;
  int maxT = 0;
  for (int r = 0; r < n; r += 1) {
    llb_dq* q = &llb_q[llb_qn++];
    q->r = r;
    q->k = 0;
    q->n = n_ids[r];
    q->ids = malloc(sizeof *q->ids * (size_t)n_ids[r]);
    for (int i = 0; i < n_ids[r]; i += 1) {
      q->ids[i] = (int)ids[r][i];
    }
    q->npos = npos[r];
    q->pos = malloc(sizeof *q->pos * (size_t)npos[r]);
    memcpy(q->pos, pos[r], sizeof *q->pos * (size_t)npos[r]);
    maxT = n_ids[r] > maxT ? n_ids[r] : maxT;
    z.questions += 1;
    z.input_tokens += n_ids[r];
  }
  if (llb_srv(maxT, n, &z) < 0) {
    for (int r = 0; r < n; r += 1) {
      free(replies[r]);
    }
    free(replies);
    llb_qfree();
    llb_rawdir = NULL;
    return -1;
  }
  z.forward_ms = llb_fwd_ms;
  int got = 0;
  for (int r = 0; r < n; r += 1) {
    got += replies[r] && !strcmp(replies[r], "ok");
    free(replies[r]);
  }
  free(replies);
  llb_qfree();
  llb_rawdir = NULL;
  z.total_ms = llb_ms() - t0;
  if (ds) {
    *ds = z;
  }
  if (z.exit_code != 0) {
    llb_fail("engine exit code != 0; see llb_log()", NULL);
    return -1;
  }
  return got;
}

void llb_free(void) {
  llb_close();  // (KEVPHONE) the resident session, if any: main returns, its release runs
  llb_loaded = 0;
  if (!llb_spent) {
    llb_runs = 0;  // (ENGINE-REPEAT) the engine was released after its run: the next llb_load is admitted
  }
  llb_head_close();
  llb_dir[0] = 0;
  unsetenv("LLB_GGUF");
  free(llb_errlog);
  llb_errlog = NULL;
}

// (PHONECAM) the camera's verdict (llb_engine.h). The emitted C's session image area (llb_img_cap, llb_img_seq) and
// readout (llb_ro_*) come from patches/metal-camera.patch; the tower from vision/host/llbv.c. Built with LLB_VISION=1
#if defined(LLB_VISION) && LLB_VISION
#include "llbv.h"
static int llb_vis_d;
int llb_vision_area(int cap) {
  llb_err[0] = 0;
  if (cap < 0 || cap > 4096) {
    return llb_fail("llb_vision_area: cap 0 .. 4096 rows", NULL);
  }
  if (llb_resident() && llb_img_cap != (u32)cap) {
    return llb_fail("llb_vision_area: set it before llb_open (the session's image area is reserved at its first pass)",
      NULL);
  }
  llb_img_cap = (u32)cap;
  return 0;
}
int llb_vision_open(const char* mmproj, int d, int cap) {
  llb_err[0] = 0;
  if (cap < 1 || cap > 4096 || d < 256 || d % 256 != 0) {
    return llb_fail("llb_vision_open: cap 1 .. 4096 rows and a width that is a multiple of 256", NULL);
  }
  if (llb_resident() && llb_img_cap != (u32)cap) {
    return llb_fail("llb_vision_open: call it before llb_open (the session's image area is reserved at its first pass)",
      NULL);
  }
  char err[256] = {0};
  if (llbv_open(mmproj, (uint32_t)d, err) != 0) {
    return llb_fail("llb_vision_open: ", err);
  }
  llb_img_cap = (u32)cap;
  llb_vis_d = d;
  return 0;
}
void llb_vision_close(void) {
  llbv_close();
}
static int llb_rows_take(const uint8_t* rgb, int w, int h, uint32_t* n, uint32_t* nx, uint32_t* ny, llb_vstats* z,
    unsigned long long* key);
static void llb_px_arm(u32 want0, u32 want1, unsigned long long key);
static void llb_px_disarm(llb_vstats* z);
int llb_classify_rgb(const uint8_t* rgb, int w, int h, const uint32_t* pre, int n_pre, const uint32_t* post,
    int n_post, uint32_t id_a, uint32_t id_b, llb_vstats* st) {
  llb_vstats z = {0};
  double t0 = llb_ms();
  llb_err[0] = 0;
  rs_reap();
  if (llb_vis_d == 0 || llb_img_cap == 0) {
    return llb_fail("llb_classify_rgb before llb_vision_open", NULL);
  }
  if (!llb_resident() || strcmp(rs_dir, llb_dir) != 0) {
    return llb_fail("llb_classify_rgb needs the resident session (llb_open)", NULL);
  }
  if (id_a >= 248320 || id_b >= 248320 || n_pre < 1 || n_post < 1) {
    return llb_fail("llb_classify_rgb: readout ids in the vocab, a prompt before and after the image", NULL);
  }
  uint32_t n = 0, nx = 0, ny = 0;
  unsigned long long key = 0;
  // (IMGFAST) the frame's rows: kept ones when this exact canvas was encoded before (the tap on a frame the live gate
  // already judged; the same rows the tower would give again), else the tower's, which are then kept, so Describe on
  // this frame (llb_generate_rgb on the same canvas) needs no tower
  if (llb_rows_take(rgb, w, h, &n, &nx, &ny, &z, &key) != 0) {
    return -1;
  }
  int n_ids = n_pre + (int)n + n_post;
  if (n > llb_img_cap || (unsigned)(n_ids + 1) > rs_cap) {
    llbv_forget();
    return llb_fail("llb_classify_rgb: the image's rows exceed the session's image area or KV capacity", NULL);
  }
  uint32_t* ids = malloc((size_t)n_ids * sizeof *ids);
  memcpy(ids, pre, (size_t)n_pre * sizeof *ids);
  for (uint32_t i = 0; i < n; i += 1) {
    ids[n_pre + i] = LLB_IMG_PAD;
  }
  memcpy(ids + n_pre + n, post, (size_t)n_post * sizeof *ids);
  z.n_ids = n_ids;
  // the readout armed for this job only (set here, read after the job: the job's hand-off orders both)
  llb_ro_init = true;
  llb_ro_ids[0] = id_a, llb_ro_ids[1] = id_b, llb_ro_pos = (u32)(n_ids - 1), llb_ro_got = 0, llb_ro_on = true;
  uint32_t out[2] = {0};
  llb_stats gs = {0};
  // (IMGFAST) the camera's prefix (the system turn up to <|vision_start|>) kept once, restored on every later fire
  llb_px_arm((u32)n_pre, 0, key);
  double t1 = llb_ms();
  int got = llb_generate_rs(ids, n_ids, 1, out, &gs);
  z.llm_ms = llb_ms() - t1;
  llb_px_disarm(&z);
  llb_ro_on = false;
  free(ids);
  llbv_forget();  // a job that never reached its first chunk leaves the rows held
  if (got < 0) {
    return -1;  // llb_error set by the job
  }
  if (got < 1 || llb_ro_got != 1) {
    snprintf(llb_err, sizeof llb_err, "llb_classify_rgb: the readout did not fire (ids %d, readouts %u); see llb_log()",
      got, llb_ro_got);
    return -1;
  }
  z.first_id = out[0];
  z.logit_a = llb_ro_v[0], z.logit_b = llb_ro_v[1];
  memcpy(&z.bits_a, &llb_ro_v[0], 4);
  memcpy(&z.bits_b, &llb_ro_v[1], 4);
  z.total_ms = llb_ms() - t0;
  if (st) {
    *st = z;
  }
  return 0;
}

// (IMGPERF) the last LLB_ROWS_KEEP photos' tower rows, by their exact canvas bytes (memcmp, no hash): a look at the
// same photo again (a follow-up question, "Ask about this photo") takes them instead of the tower's encode, and needs
// no tower at all (it may have closed). The tower is deterministic, so the rows are the ones it would give again.
// ~2 MB a photo at 448 (196 rows x 2560 floats) + its 588 KB canvas.
#define LLB_ROWS_KEEP 4
typedef struct {
  uint8_t* rgb;
  size_t nb;
  int w, h;
  float* rows;
  uint32_t d, n, nx, ny;
  unsigned long long used;
  unsigned long long serial;  // (IMGFAST) this entry's rows' identity: the prefix snapshots' image key
  int looks;                  // (IMGFAST) llb_generate_rgb jobs on these rows so far
} llb_rows_kept;
static llb_rows_kept llb_rk[LLB_ROWS_KEEP];
static unsigned long long llb_rk_clock;
static unsigned long long llb_rk_serial;  // (IMGFAST) never reused: a forgotten entry's key never matches again

// (IMGFAST) LLB_ROWS_KEEP=0: nothing kept, nothing found (the A/B's before arm: a tower per call, as b0ffd9af's)
static int llb_rk_off(void) {
  static int off = -1;
  if (off < 0) {
    const char* e = getenv("LLB_ROWS_KEEP");
    off = e != NULL && strcmp(e, "0") == 0;
  }
  return off;
}

static int llb_rk_find(const uint8_t* rgb, int w, int h, uint32_t d) {
  if (llb_rk_off()) {
    return -1;
  }
  size_t nb = (size_t)w * (size_t)h * 3;
  for (int i = 0; i < LLB_ROWS_KEEP; i += 1) {
    llb_rows_kept* k = &llb_rk[i];
    if (k->rows && k->w == w && k->h == h && k->d == d && k->nb == nb && memcmp(k->rgb, rgb, nb) == 0) {
      return i;
    }
  }
  return -1;
}

static void llb_rk_put(const uint8_t* rgb, int w, int h, const float* rows, uint32_t d, uint32_t n, uint32_t nx,
    uint32_t ny) {
  if (llb_rk_off()) {
    return;
  }
  int at = 0;
  for (int i = 0; i < LLB_ROWS_KEEP; i += 1) {
    if (!llb_rk[i].rows) {
      at = i;
      break;
    }
    if (llb_rk[i].used < llb_rk[at].used) {
      at = i;
    }
  }
  llb_rows_kept* k = &llb_rk[at];
  free(k->rgb);
  free(k->rows);
  memset(k, 0, sizeof *k);
  size_t nb = (size_t)w * (size_t)h * 3, rb = (size_t)n * d * sizeof(float);
  k->rgb = malloc(nb);
  k->rows = malloc(rb);
  if (!k->rgb || !k->rows) {
    free(k->rgb);
    free(k->rows);
    memset(k, 0, sizeof *k);
    return;
  }
  memcpy(k->rgb, rgb, nb);
  memcpy(k->rows, rows, rb);
  k->nb = nb, k->w = w, k->h = h, k->d = d, k->n = n, k->nx = nx, k->ny = ny, k->used = ++llb_rk_clock;
  k->serial = ++llb_rk_serial;
}

// (IMGFAST) the next job's prefix snapshots (the emitted C's llb_px_*, host-prefix.patch): its wanted prefix ends and
// its image key, armed for that job only; afterwards the restored prefix and the host ms it cost go into the stats
static void llb_px_arm(u32 want0, u32 want1, unsigned long long key) {
  llb_px_job = true;
  llb_px_want[0] = want0, llb_px_want[1] = want1;
  llb_px_img = key;
  llb_px_hit = 0;
  llb_px_ms = 0;
}
static void llb_px_disarm(llb_vstats* z) {
  llb_px_job = false;
  llb_px_want[0] = llb_px_want[1] = 0;
  z->prefix_ids = (int)llb_px_hit;
  z->prefix_ms = llb_px_ms;
}

void llb_prefix_forget(void) {
  llb_px_drop();
}

// (IMGFAST) the canvas's rows held for the next job (llbv_hold): the kept copy when this exact canvas was encoded
// before (no tower: prep/tower ms 0, rows_kept 1), else the resident tower's encode, kept for the next job on it.
// 0, or -1 with llb_error set (nothing held).
static int llb_rows_take(const uint8_t* rgb, int w, int h, uint32_t* n, uint32_t* nx, uint32_t* ny, llb_vstats* z,
    unsigned long long* key) {
  uint32_t d = (uint32_t)llb_vis_d;
  *key = 0;
  if (!rgb || w < 1 || h < 1) {
    return llb_fail("vision: an image", NULL);
  }
  int kept = llb_rk_find(rgb, w, h, d);
  if (kept >= 0) {
    llb_rows_kept* k = &llb_rk[kept];
    size_t rb = (size_t)k->n * k->d * sizeof(float);
    float* copy = malloc(rb);
    if (!copy) {
      return llb_fail("vision: out of memory for the kept rows", NULL);
    }
    memcpy(copy, k->rows, rb);
    llbv_hold(copy, k->d, k->n, k->nx, k->ny);
    k->used = ++llb_rk_clock;
    *n = k->n, *nx = k->nx, *ny = k->ny;
    *key = k->serial;
    z->rows_kept = 1;
  } else {
    if (!llbv_is_open()) {
      return llb_fail("vision: the tower is closed and this canvas's rows are not kept (llb_vision_open)", NULL);
    }
    char err[256] = {0};
    double vm[3] = {0};
    if (llbv_prepare_rgb(rgb, w, h, d, n, nx, ny, vm, err) != 0) {
      return llb_fail("vision: ", err);
    }
    z->prep_ms = vm[0], z->tower_ms = vm[1], z->tower_gpu_ms = vm[2];
    uint32_t hd = 0, hn = 0, hx = 0, hy = 0;
    const float* held = llbv_peek(&hd, &hn, &hx, &hy);
    if (held) {
      llb_rk_put(rgb, w, h, held, hd, hn, hx, hy);
      int at = llb_rk_find(rgb, w, h, d);
      *key = at >= 0 ? llb_rk[at].serial : 0;
    }
  }
  z->n_img = (int)*n, z->nx = (int)*nx, z->ny = (int)*ny;
  return 0;
}

int llb_vision_has(const uint8_t* rgb, int w, int h) {
  return rgb && w > 0 && h > 0 && llb_vis_d > 0 && llb_rk_find(rgb, w, h, (uint32_t)llb_vis_d) >= 0;
}

void llb_vision_forget(void) {
  for (int i = 0; i < LLB_ROWS_KEEP; i += 1) {
    free(llb_rk[i].rgb);
    free(llb_rk[i].rows);
    memset(&llb_rk[i], 0, sizeof llb_rk[i]);
  }
}

int llb_vision_keep(const uint8_t* rgb, int w, int h, llb_vstats* st) {
  llb_vstats z = {0};
  double t0 = llb_ms();
  llb_err[0] = 0;
  if (llb_vis_d == 0) {
    return llb_fail("llb_vision_keep before llb_vision_open", NULL);
  }
  if (!rgb || w < 1 || h < 1) {
    return llb_fail("llb_vision_keep: an image", NULL);
  }
  int kept = llb_rk_find(rgb, w, h, (uint32_t)llb_vis_d);
  if (kept >= 0) {
    llb_rk[kept].used = ++llb_rk_clock;
    z.rows_kept = 1;
    z.n_img = (int)llb_rk[kept].n, z.nx = (int)llb_rk[kept].nx, z.ny = (int)llb_rk[kept].ny;
  } else {
    if (!llbv_is_open()) {
      return llb_fail("llb_vision_keep: the tower is closed (llb_vision_open)", NULL);
    }
    char err[256] = {0};
    double vm[3] = {0};
    uint32_t n = 0, nx = 0, ny = 0;
    if (llbv_prepare_rgb(rgb, w, h, (uint32_t)llb_vis_d, &n, &nx, &ny, vm, err) != 0) {
      return llb_fail("llb_vision_keep: ", err);
    }
    z.prep_ms = vm[0], z.tower_ms = vm[1], z.tower_gpu_ms = vm[2];
    z.n_img = (int)n, z.nx = (int)nx, z.ny = (int)ny;
    uint32_t hd = 0, hn = 0, hx = 0, hy = 0;
    const float* held = llbv_peek(&hd, &hn, &hx, &hy);
    if (held) {
      llb_rk_put(rgb, w, h, held, hd, hn, hx, hy);
    }
    llbv_forget();
  }
  z.total_ms = llb_ms() - t0;
  if (st) {
    *st = z;
  }
  return 0;
}

int llb_generate_rgb(const uint8_t* rgb, int w, int h, const uint32_t* pre, int n_pre, const uint32_t* post,
    int n_post, int n_new, uint32_t* out, llb_vstats* st) {
  llb_vstats z = {0};
  double t0 = llb_ms();
  llb_err[0] = 0;
  rs_reap();
  if (llb_img_cap == 0 || llb_vis_d == 0) {
    return llb_fail("llb_generate_rgb before llb_vision_open", NULL);
  }
  if (!llb_resident() || strcmp(rs_dir, llb_dir) != 0) {
    return llb_fail("llb_generate_rgb needs the resident session (llb_open)", NULL);
  }
  if (!rgb || w < 1 || h < 1 || n_pre < 1 || n_post < 1 || n_new < 1 || !out) {
    return llb_fail("llb_generate_rgb: an image, a prompt before and after it, n_new >= 1 and out", NULL);
  }
  uint32_t n = 0, nx = 0, ny = 0;
  unsigned long long key = 0;
  if (llb_rows_take(rgb, w, h, &n, &nx, &ny, &z, &key) != 0) {
    return -1;
  }
  int n_ids = n_pre + (int)n + n_post;
  if (n > llb_img_cap || (unsigned)(n_ids + n_new) > rs_cap) {
    llbv_forget();
    return llb_fail("llb_generate_rgb: the image's rows exceed the session's image area, or the prompt + n_new the KV "
      "capacity", NULL);
  }
  uint32_t* ids = malloc((size_t)n_ids * sizeof *ids);
  if (!ids) {
    llbv_forget();
    return llb_fail("llb_generate_rgb: out of memory", NULL);
  }
  memcpy(ids, pre, (size_t)n_pre * sizeof *ids);
  for (uint32_t i = 0; i < n; i += 1) {
    ids[n_pre + i] = LLB_IMG_PAD;
  }
  memcpy(ids + n_pre + n, post, (size_t)n_post * sizeof *ids);
  z.n_ids = n_ids;
  llb_ro_on = false;  // no readout: a generate
  llb_stats gs = {0};
  // (IMGFAST) the look's prefix, restored (kept once a session). The prefix + this image's rows only from the canvas's
  // second look on: keeping it splits the job's chunk (Mac: +15 ms, more than the restored text saves), which pays only
  // when the canvas is looked at again (then that look's prefill is its question alone: Mac 183 -> 58 ms)
  int looks = 0;
  for (int i = 0; key != 0 && i < LLB_ROWS_KEEP; i += 1) {
    if (llb_rk[i].rows && llb_rk[i].serial == key) {
      looks = llb_rk[i].looks++;
    }
  }
  llb_px_arm((u32)n_pre, looks >= 1 ? (u32)(n_pre + (int)n) : 0, key);
  double t1 = llb_ms();
  int got = llb_generate_rs(ids, n_ids, n_new, out, &gs);
  z.llm_ms = llb_ms() - t1;
  llb_px_disarm(&z);
  free(ids);
  llbv_forget();  // a job that never reached its first chunk leaves the rows held
  if (got < 0) {
    return -1;  // llb_error set by the job
  }
  z.first_id = got > 0 ? out[0] : 0;
  z.prefill_ms = gs.prefill_ms, z.step_ms_sum = gs.step_ms_sum, z.steps = gs.steps;
  z.total_ms = llb_ms() - t0;
  if (st) {
    *st = z;
  }
  return got;
}
#else
int llb_generate_rgb(const uint8_t* rgb, int w, int h, const uint32_t* pre, int n_pre, const uint32_t* post,
    int n_post, int n_new, uint32_t* out, llb_vstats* st) {
  (void)rgb, (void)w, (void)h, (void)pre, (void)n_pre, (void)post, (void)n_post, (void)n_new, (void)out, (void)st;
  return llb_fail("this engine build has no vision (LLB_VISION)", NULL);
}
int llb_vision_has(const uint8_t* rgb, int w, int h) {
  (void)rgb, (void)w, (void)h;
  return 0;
}
int llb_vision_keep(const uint8_t* rgb, int w, int h, llb_vstats* st) {
  (void)rgb, (void)w, (void)h, (void)st;
  return llb_fail("this engine build has no vision (LLB_VISION)", NULL);
}
void llb_vision_forget(void) {}
void llb_prefix_forget(void) {}
int llb_vision_area(int cap) {
  (void)cap;
  return llb_fail("this engine build has no vision (LLB_VISION)", NULL);
}
int llb_vision_open(const char* mmproj, int d, int cap) {
  (void)mmproj, (void)d, (void)cap;
  return llb_fail("this engine build has no vision (LLB_VISION)", NULL);
}
void llb_vision_close(void) {}
int llb_classify_rgb(const uint8_t* rgb, int w, int h, const uint32_t* pre, int n_pre, const uint32_t* post,
    int n_post, uint32_t id_a, uint32_t id_b, llb_vstats* st) {
  (void)rgb, (void)w, (void)h, (void)pre, (void)n_pre, (void)post, (void)n_post, (void)id_a, (void)id_b, (void)st;
  return llb_fail("this engine build has no vision (LLB_VISION)", NULL);
}
#endif
