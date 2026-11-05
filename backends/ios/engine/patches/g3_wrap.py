#!/usr/bin/env python3
# (G3) wrapper + header edits: text prompts on the prefix snapshots (llb_generate_px), the snapshots saved to and
# loaded from a file (llb_state_save / llb_state_load: the hot state across launches), packmap= in the report.
# usage: g3_wrap.py <llb_engine.m> <llb_engine.h>  (in place; exits on a missing anchor)
import sys

mp, hp = sys.argv[1], sys.argv[2]
m, h = open(mp).read(), open(hp).read()


def sub(s, old, new):
    if s.count(old) != 1:
        sys.exit(f"g3_wrap: anchor found {s.count(old)}x: {old[:70]!r}")
    return s.replace(old, new)


API = r'''void llb_prefix_forget(void) {
  llb_px_drop();
}

// (G3) a text prompt on the prefix snapshots: its first n_pre ids (the system turn, a conversation so far) kept as a
// snapshot the first time and restored on every later prompt that starts with them (and on one after llb_state_load)
int llb_generate_px(const uint32_t* ids, int n_ids, int n_pre, int n_new, uint32_t* out, llb_stats* st,
    int* prefix_ids) {
  llb_err[0] = 0;
  rs_reap();
  if (!llb_resident() || strcmp(rs_dir, llb_dir) != 0) {
    return llb_fail("llb_generate_px needs the resident session (llb_open)", NULL);
  }
  if (n_pre < 1 || n_pre >= n_ids || n_pre > (int)LLB_PX_MAXP) {
    return llb_fail("llb_generate_px: n_pre in 1 .. min(n_ids - 1, 1024)", NULL);
  }
  llb_px_arm((u32)n_pre, 0, 0);
  int got = llb_generate_rs(ids, n_ids, n_new, out, st);
  llb_vstats z = {0};
  llb_px_disarm(&z);
  if (prefix_ids) {
    *prefix_ids = z.prefix_ids;
  }
  return got;
}

// (G3) the hot state file: "LLBPXST1", a header naming what the state is valid for (the emitted C, the model's GGUF
// bytes, the model dir's name, the session's KV capacity), then the text snapshots (an image's are not saved: its rows
// are not on disk), each {p, nl, dT, r[64], nb, ids[p], state[nb]}. Written to <path>.tmp, then renamed.
typedef struct {
  char     magic[8];
  uint32_t ver, n, cap, pad;
  uint64_t gguf_bytes;
  char     tag[24], model[232];
} LlbStHdr;

static void llb_st_hdr(LlbStHdr* hd) {
  memset(hd, 0, sizeof *hd);
  memcpy(hd->magic, "LLBPXST1", 8);
  hd->ver = 1, hd->cap = rs_cap, hd->gguf_bytes = llb_gguf_bytes;
  snprintf(hd->tag, sizeof hd->tag, "%s", LLB_SRC_TAG);
  const char* b = strrchr(rs_dir, '/');
  snprintf(hd->model, sizeof hd->model, "%s", b ? b + 1 : rs_dir);
}

// the session idle (resident, no job), else -1. Under rs_mu on 0.
static int llb_st_idle(const char* who) {
  rs_reap();
  if (!llb_resident()) {
    return llb_fail(who, NULL);
  }
  pthread_mutex_lock(&rs_mu);
  if (!rs_ready || rs_kind != 0) {
    pthread_mutex_unlock(&rs_mu);
    return llb_fail(who, NULL);
  }
  return 0;
}

int llb_state_save(const char* path) {
  llb_err[0] = 0;
  if (llb_st_idle("llb_state_save: needs the resident session, idle (no job running)") != 0) {
    return -1;
  }
  char tmp[1100];
  snprintf(tmp, sizeof tmp, "%s.tmp", path);
  FILE* f = fopen(tmp, "wb");
  if (!f) {
    pthread_mutex_unlock(&rs_mu);
    return llb_fail("llb_state_save: cannot write the file", path);
  }
  LlbStHdr hd;
  llb_st_hdr(&hd);
  for (u32 k = 0; k < LLB_PX_SLOTS; k += 1) {
    hd.n += llb_px[k].st != NULL && llb_px[k].img == 0;
  }
  int ok = fwrite(&hd, sizeof hd, 1, f) == 1;
  for (u32 k = 0; k < LLB_PX_SLOTS && ok; k += 1) {
    const LlbPx* x = &llb_px[k];
    if (x->st == NULL || x->img != 0) {
      continue;
    }
    u32 h3[3] = {x->p, x->nl, x->dT};
    ok = fwrite(h3, sizeof h3, 1, f) == 1 && fwrite(x->r, sizeof x->r, 1, f) == 1
      && fwrite(&x->nb, sizeof x->nb, 1, f) == 1 && fwrite(x->ids, sizeof(u32), x->p, f) == x->p
      && fwrite(x->st, 1, x->nb, f) == x->nb;
  }
  pthread_mutex_unlock(&rs_mu);
  ok = fclose(f) == 0 && ok;
  if (!ok || rename(tmp, path) != 0) {
    unlink(tmp);
    return llb_fail("llb_state_save: write failed", path);
  }
  return (int)hd.n;
}

int llb_state_load(const char* path) {
  llb_err[0] = 0;
  FILE* f = fopen(path, "rb");
  if (!f) {
    return llb_fail("llb_state_load: no such file", path);
  }
  if (llb_st_idle("llb_state_load: needs the resident session, idle (no job running)") != 0) {
    fclose(f);
    return -1;
  }
  LlbStHdr hd, want;
  llb_st_hdr(&want);
  const char* bad = NULL;
  if (fread(&hd, sizeof hd, 1, f) != 1 || memcmp(hd.magic, want.magic, 8) != 0 || hd.ver != 1) {
    bad = "llb_state_load: not a state file";
  } else if (strcmp(hd.tag, want.tag) != 0) {
    bad = "llb_state_load: saved by another engine build";
  } else if (hd.gguf_bytes != want.gguf_bytes || strcmp(hd.model, want.model) != 0) {
    bad = "llb_state_load: saved for another model";
  } else if (hd.cap != want.cap) {
    bad = "llb_state_load: saved at another KV capacity";
  }
  int got = 0;
  for (u32 i = 0; bad == NULL && i < hd.n; i += 1) {
    u32 h3[3];
    u32 r[64];
    u64 nb = 0;
    if (fread(h3, sizeof h3, 1, f) != 1 || fread(r, sizeof r, 1, f) != 1 || fread(&nb, sizeof nb, 1, f) != 1
      || h3[0] < 1 || h3[0] > LLB_PX_MAXP || h3[1] > 64 || nb > (1ull << 31)) {
      bad = "llb_state_load: a damaged snapshot";
      break;
    }
    // the least recently used slot (a free one first)
    LlbPx* x = &llb_px[0];
    for (u32 k = 1; k < LLB_PX_SLOTS; k += 1) {
      if (x->st != NULL && (llb_px[k].st == NULL || llb_px[k].used < x->used)) {
        x = &llb_px[k];
      }
    }
    u8* st = malloc(nb);
    u32* ids = calloc(LLB_PX_MAXP, sizeof(u32));
    if (!st || !ids || fread(ids, sizeof(u32), h3[0], f) != h3[0] || fread(st, 1, nb, f) != nb) {
      free(st), free(ids);
      bad = "llb_state_load: a damaged snapshot";
      break;
    }
    free(x->st);
    memset(x, 0, sizeof *x);
    x->p = h3[0], x->nl = h3[1], x->dT = h3[2], x->st = st, x->nb = nb, x->img = 0;
    memcpy(x->r, r, sizeof r);
    memcpy(x->ids, ids, (size_t)h3[0] * sizeof(u32));
    free(ids);
    x->used = ++llb_px_clock;
    got += 1;
  }
  pthread_mutex_unlock(&rs_mu);
  fclose(f);
  if (bad) {
    return llb_fail(bad, path);
  }
  return got;
}
'''
m = sub(m, '''void llb_prefix_forget(void) {
  llb_px_drop();
}
''', API)

# stubs in the no-vision build
m = sub(m, '''void llb_prefix_forget(void) {}
''', '''void llb_prefix_forget(void) {}
int llb_generate_px(const uint32_t* ids, int n_ids, int n_pre, int n_new, uint32_t* out, llb_stats* st,
    int* prefix_ids) {
  (void)ids, (void)n_ids, (void)n_pre, (void)n_new, (void)out, (void)st, (void)prefix_ids;
  return llb_fail("this engine build has no prefix snapshots (LLB_VISION)", NULL);
}
int llb_state_save(const char* path) {
  return llb_fail("this engine build has no prefix snapshots (LLB_VISION)", path);
}
int llb_state_load(const char* path) {
  return llb_fail("this engine build has no prefix snapshots (LLB_VISION)", path);
}
''')

m = sub(m, '#include "sys1mac.c"\n', '#include "sys1mac.c"\n#ifndef LLB_SRC_TAG\n#define LLB_SRC_TAG "untagged"\n#endif\n'
  '// (G3) LLB_PACK_MAP: the pack file\'s load counters (mapped / copied / written MB)\n'
  'const char* llb_packmap_report(void) {\n  static char b[160];\n  g3_pm_report(b, sizeof b);\n  return b;\n}\n')

# the report: packmap= appended (the line's other fields unchanged)
open(mp, "w").write(m)

h = sub(h, "void        llb_prefix_forget(void);\n", '''void        llb_prefix_forget(void);
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
''')
open(hp, "w").write(h)
print("g3_wrap: ok")
