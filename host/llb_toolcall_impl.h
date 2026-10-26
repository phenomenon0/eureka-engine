// host/llb_toolcall_impl.h — definitions for host/llb_toolcall.h (inlined by the engine,
// compiled standalone via host/llb_toolcall.c). Template line references are to
// qwen3.5-9b-hf/chat_template.jinja; tests: tools/toolcall/.
#ifndef LLB_TOOLCALL_IMPL_H
#define LLB_TOOLCALL_IMPL_H

#include "llb_toolcall.h"
#include "llb_json_impl.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- small helpers

static void llb_tc_err(char* err, int errlen, const char* fmt, ...) {
  if (!err || errlen <= 0 || err[0]) return;  // first error wins
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err, (size_t)errlen, fmt, ap);
  va_end(ap);
}

// first / last index of needle in s[0..n), -1 if absent (memmem is not C11)
static long llb_tc_find(const char* s, size_t n, const char* nd) {
  size_t m = strlen(nd);
  for (size_t i = 0; i + m <= n; i += 1) {
    if (!memcmp(s + i, nd, m)) return (long)i;
  }
  return -1;
}

static long llb_tc_rfind(const char* s, size_t n, const char* nd) {
  size_t m = strlen(nd);
  for (size_t i = n >= m ? n - m + 1 : 0; i-- > 0;) {
    if (!memcmp(s + i, nd, m)) return (long)i;
  }
  return -1;
}

static int llb_tc_starts(const char* s, size_t n, const char* p) {
  size_t m = strlen(p);
  return n >= m && !memcmp(s, p, m);
}

// Python str.isspace() code points (what str.strip() / jinja |trim remove)
static int llb_tc_isspace(unsigned cp) {
  return (cp >= 9 && cp <= 13) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

// decode one UTF-8 code point at s (n bytes available); invalid -> U+FFFD, length 1
static unsigned llb_tc_cp(const unsigned char* s, size_t n, size_t* len) {
  unsigned c = s[0];
  *len = 1;
  if (c < 0x80) return c;
  int k = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
  if (k == 1 || (size_t)k > n) return 0xFFFD;
  unsigned v = c & (0x7Fu >> k);
  for (int i = 1; i < k; i += 1) {
    if ((s[i] & 0xC0) != 0x80) return 0xFFFD;
    v = v << 6 | (s[i] & 63u);
  }
  *len = (size_t)k;
  return v;
}

// Python str.strip(): [*a, *z) of s[0..n)
static void llb_tc_trim(const char* s, size_t n, size_t* a, size_t* z) {
  const unsigned char* u = (const unsigned char*)s;
  size_t i = 0, e = n, l;
  while (i < e && llb_tc_isspace(llb_tc_cp(u + i, e - i, &l))) i += l;
  while (e > i) {
    size_t k = e - 1;
    while (k > i && (u[k] & 0xC0) == 0x80 && e - k < 4) k -= 1;
    unsigned cp = llb_tc_cp(u + k, e - k, &l);
    if (l != e - k || !llb_tc_isspace(cp)) break;
    e = k;
  }
  *a = i;
  *z = e;
}

static const char* llb_tc_role(const LlbJson* m) {
  const LlbJson* r = llb_json_get(m, "role");
  return r && r->t == LLB_J_STR ? r->s : "";
}

// ---------------------------------------------------------------- Python-faithful output

// repr(float) (json = 0) or json.dumps(float) (json = 1): shortest round-trip digits,
// fixed notation for decimal exponents -4..15, else d.ddde+XX
static void llb_tc_pyfloat(LlbBuf* b, double v, int json) {
  char t[48], dig[24];
  if (isnan(v)) { llb_buf_puts(b, json ? "NaN" : "nan"); return; }
  if (isinf(v)) { llb_buf_puts(b, v < 0 ? (json ? "-Infinity" : "-inf") : (json ? "Infinity" : "inf")); return; }
  for (int p = 1; p <= 17; p += 1) {
    snprintf(t, sizeof t, "%.*e", p - 1, v);
    if (strtod(t, NULL) == v) break;
  }
  const char* q = t;
  int nd = 0, neg = 0;
  if (*q == '-') { neg = 1; q += 1; }
  for (; *q && *q != 'e'; q += 1) {
    if (*q != '.' && nd < 20) dig[nd++] = *q;
  }
  int e = atoi(q + 1);
  while (nd > 1 && dig[nd - 1] == '0') nd -= 1;
  if (neg) llb_buf_puts(b, "-");
  if (e >= -4 && e < 16) {
    if (e >= 0) {
      if (nd <= e + 1) {
        llb_buf_put(b, dig, (size_t)nd);
        for (int i = nd; i < e + 1; i += 1) llb_buf_puts(b, "0");
        llb_buf_puts(b, ".0");
      } else {
        llb_buf_put(b, dig, (size_t)e + 1);
        llb_buf_puts(b, ".");
        llb_buf_put(b, dig + e + 1, (size_t)(nd - e - 1));
      }
    } else {
      llb_buf_puts(b, "0.");
      for (int i = 0; i < -e - 1; i += 1) llb_buf_puts(b, "0");
      llb_buf_put(b, dig, (size_t)nd);
    }
  } else {
    llb_buf_put(b, dig, 1);
    if (nd > 1) {
      llb_buf_puts(b, ".");
      llb_buf_put(b, dig + 1, (size_t)nd - 1);
    }
    snprintf(t, sizeof t, "e%c%02d", e < 0 ? '-' : '+', e < 0 ? -e : e);
    llb_buf_puts(b, t);
  }
}

// a JSON number as Python sees it after json.loads: int lexeme verbatim (-0 -> 0), float repr
static void llb_tc_pynum(LlbBuf* b, const LlbJson* j, int json) {
  if (!strpbrk(j->s, ".eE")) {
    if (!strcmp(j->s, "-0")) llb_buf_puts(b, "0");
    else llb_buf_put(b, j->s, j->n);
    return;
  }
  llb_tc_pyfloat(b, strtod(j->s, NULL), json);
}

// does key i of o repeat an earlier key? (Python dict keeps the first position)
static int llb_tc_dupe(const LlbJson* o, size_t i) {
  for (size_t k = 0; k < i; k += 1) {
    if (!strcmp(o->keys[k], o->keys[i])) return 1;
  }
  return 0;
}

// the value for key i of o as a Python dict sees it (last duplicate wins)
static const LlbJson* llb_tc_last(const LlbJson* o, size_t i) {
  for (size_t k = o->n; k-- > i + 1;) {
    if (!strcmp(o->keys[k], o->keys[i])) return o->kids[k];
  }
  return o->kids[i];
}

// json.dumps(x, ensure_ascii=False) of json.loads(text): the template's tojson
static void llb_tc_tojson(LlbBuf* b, const LlbJson* j) {
  switch (j->t) {
    case LLB_J_NULL: llb_buf_puts(b, "null"); break;
    case LLB_J_BOOL: llb_buf_puts(b, j->b ? "true" : "false"); break;
    case LLB_J_NUM: llb_tc_pynum(b, j, 1); break;
    case LLB_J_STR: llb_json_quote(b, j->s, j->n); break;
    case LLB_J_ARR:
      llb_buf_puts(b, "[");
      for (size_t i = 0; i < j->n; i += 1) {
        if (i) llb_buf_puts(b, ", ");
        llb_tc_tojson(b, j->kids[i]);
      }
      llb_buf_puts(b, "]");
      break;
    case LLB_J_OBJ: {
      int first = 1;
      llb_buf_puts(b, "{");
      for (size_t i = 0; i < j->n; i += 1) {
        if (llb_tc_dupe(j, i)) continue;
        if (!first) llb_buf_puts(b, ", ");
        first = 0;
        llb_json_quote(b, j->keys[i], strlen(j->keys[i]));
        llb_buf_puts(b, ": ");
        llb_tc_tojson(b, llb_tc_last(j, i));
      }
      llb_buf_puts(b, "}");
      break;
    }
  }
}

// Python str() of a scalar (jinja `| string` and `{{ x }}`); -1 for containers
static int llb_tc_pystr(LlbBuf* b, const LlbJson* j) {
  switch (j->t) {
    case LLB_J_NULL: llb_buf_puts(b, "None"); return 0;
    case LLB_J_BOOL: llb_buf_puts(b, j->b ? "True" : "False"); return 0;
    case LLB_J_NUM: llb_tc_pynum(b, j, 0); return 0;
    case LLB_J_STR: llb_buf_put(b, j->s, j->n); return 0;
    default: return -1;
  }
}

// ---------------------------------------------------------------- prompt side

// render_content(content) (template lines 3-41) into b; -1 + err on raise / unsupported
static int llb_tc_content(LlbBuf* b, const LlbJson* c, int is_sys, char* err, int errlen) {
  if (!c || c->t == LLB_J_NULL) return 0;
  if (c->t == LLB_J_STR) {
    llb_buf_put(b, c->s, c->n);
    return 0;
  }
  if (c->t != LLB_J_ARR) {
    llb_tc_err(err, errlen, "Unexpected content type.");
    return -1;
  }
  for (size_t i = 0; i < c->n; i += 1) {
    const LlbJson* it = c->kids[i];
    if (it->t != LLB_J_OBJ) {
      llb_tc_err(err, errlen, "unsupported: non-object content item");
      return -1;
    }
    const LlbJson* ty = llb_json_get(it, "type");
    int tyis_img = ty && ty->t == LLB_J_STR && !strcmp(ty->s, "image");
    int tyis_vid = ty && ty->t == LLB_J_STR && !strcmp(ty->s, "video");
    if (llb_json_get(it, "image") || llb_json_get(it, "image_url") || tyis_img) {
      llb_tc_err(err, errlen, is_sys ? "System message cannot contain images." : "unsupported: image content");
      return -1;
    }
    if (llb_json_get(it, "video") || tyis_vid) {
      llb_tc_err(err, errlen, is_sys ? "System message cannot contain videos." : "unsupported: video content");
      return -1;
    }
    const LlbJson* tx = llb_json_get(it, "text");
    if (!tx) {
      llb_tc_err(err, errlen, "Unexpected item type in content.");
      return -1;
    }
    if (llb_tc_pystr(b, tx)) {
      llb_tc_err(err, errlen, "unsupported: non-scalar text in content item");
      return -1;
    }
  }
  return 0;
}

// one assistant tool call (template lines 102-125)
static int llb_tc_call(LlbBuf* o, const LlbJson* call, int first, int has_content, char* err, int errlen) {
  if (call->t != LLB_J_OBJ) {
    llb_tc_err(err, errlen, "unsupported: tool call is not an object");
    return -1;
  }
  const LlbJson* f = llb_json_get(call, "function");
  if (f) call = f;
  const LlbJson* nm = call->t == LLB_J_OBJ ? llb_json_get(call, "name") : NULL;
  if (!nm || nm->t != LLB_J_STR) {
    llb_tc_err(err, errlen, "tool call without a string name");
    return -1;
  }
  llb_buf_puts(o, !first ? "\n<tool_call>\n<function=" : has_content ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=");
  llb_buf_put(o, nm->s, nm->n);
  llb_buf_puts(o, ">\n");
  const LlbJson* a = llb_json_get(call, "arguments");
  LlbJson* parsed = NULL;
  if (a && a->t == LLB_J_STR) {
    size_t x, y;
    llb_tc_trim(a->s, a->n, &x, &y);
    if (x == y) a = NULL;  // "" -> no arguments
    else {
      char e2[128];
      parsed = llb_json_parse(a->s, a->n, e2, sizeof e2);
      if (!parsed) {
        llb_tc_err(err, errlen, "tool call arguments string is not JSON (%s)", e2);
        return -1;
      }
      a = parsed;
    }
  }
  if (a) {
    if (a->t != LLB_J_OBJ) {
      llb_tc_err(err, errlen, "tool call arguments is not a mapping");
      llb_json_free(parsed);
      return -1;
    }
    for (size_t i = 0; i < a->n; i += 1) {
      if (llb_tc_dupe(a, i)) continue;
      const LlbJson* v = llb_tc_last(a, i);
      llb_buf_puts(o, "<parameter=");
      llb_buf_puts(o, a->keys[i]);
      llb_buf_puts(o, ">\n");
      if (v->t == LLB_J_OBJ || v->t == LLB_J_ARR) llb_tc_tojson(o, v);
      else llb_tc_pystr(o, v);
      llb_buf_puts(o, "\n</parameter>\n");
    }
  }
  llb_json_free(parsed);
  llb_buf_puts(o, "</function>\n</tool_call>");
  return 0;
}

static char* llb_tc_render_doc(const LlbJson* req, int add_gen, int enable_thinking, char* err, int errlen) {
  if (req->t != LLB_J_OBJ) {
    llb_tc_err(err, errlen, "request is not a JSON object");
    return NULL;
  }
  const LlbJson* msgs = llb_json_get(req, "messages");
  if (!msgs || msgs->t == LLB_J_NULL || (msgs->t == LLB_J_ARR && msgs->n == 0)) {
    llb_tc_err(err, errlen, "No messages provided.");
    return NULL;
  }
  if (msgs->t != LLB_J_ARR) {
    llb_tc_err(err, errlen, "messages is not an array");
    return NULL;
  }
  size_t nm = msgs->n;
  for (size_t i = 0; i < nm; i += 1) {
    if (msgs->kids[i]->t != LLB_J_OBJ) {
      llb_tc_err(err, errlen, "unsupported: message %zu is not an object", i);
      return NULL;
    }
  }
  const LlbJson* tools = llb_json_get(req, "tools");
  if (tools && tools->t != LLB_J_ARR && tools->t != LLB_J_OBJ && tools->t != LLB_J_NULL) {
    llb_tc_err(err, errlen, "unsupported: tools is not an array");
    return NULL;
  }
  int has_tools = tools && tools->t == LLB_J_ARR && tools->n > 0;

  // render_content(m.content) for every message, kept with its trim window
  LlbBuf* cb = calloc(nm, sizeof *cb);
  size_t* ca = calloc(nm, sizeof *ca);
  size_t* cz = calloc(nm, sizeof *cz);
  LlbBuf o = {0};
  if (!cb || !ca || !cz) abort();
  llb_buf_put(&o, "", 0);
  for (size_t i = 0; i < nm; i += 1) {
    llb_buf_put(&cb[i], "", 0);
    int sys0 = i == 0 && !strcmp(llb_tc_role(msgs->kids[i]), "system");
    if (llb_tc_content(&cb[i], llb_json_get(msgs->kids[i], "content"), sys0, err, errlen)) goto fail;
    llb_tc_trim(cb[i].p, cb[i].n, &ca[i], &cz[i]);
  }
#define LLB_TC_C(i) (cb[i].p + ca[i])
#define LLB_TC_N(i) (cz[i] - ca[i])

  // system block (lines 45-68)
  int sys0 = !strcmp(llb_tc_role(msgs->kids[0]), "system");
  if (has_tools) {
    llb_buf_puts(&o, "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (size_t i = 0; i < tools->n; i += 1) {
      llb_buf_puts(&o, "\n");
      llb_tc_tojson(&o, tools->kids[i]);
    }
    llb_buf_puts(&o, "\n</tools>");
    llb_buf_puts(&o,
        "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
        "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
        "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
        "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
        "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
        "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
    if (sys0 && LLB_TC_N(0)) {
      llb_buf_puts(&o, "\n\n");
      llb_buf_put(&o, LLB_TC_C(0), LLB_TC_N(0));
    }
    llb_buf_puts(&o, "<|im_end|>\n");
  } else if (sys0) {
    llb_buf_puts(&o, "<|im_start|>system\n");
    llb_buf_put(&o, LLB_TC_C(0), LLB_TC_N(0));
    llb_buf_puts(&o, "<|im_end|>\n");
  }

  // last_query_index: the last user message that is not a pure <tool_response> (69-78)
  long lqi = -1;
  for (size_t k = nm; k-- > 0;) {
    if (strcmp(llb_tc_role(msgs->kids[k]), "user")) continue;
    const char* s = LLB_TC_C(k);
    size_t n = LLB_TC_N(k);
    if (!(llb_tc_starts(s, n, "<tool_response>") && n >= 16 && !memcmp(s + n - 16, "</tool_response>", 16))) {
      lqi = (long)k;
      break;
    }
  }
  if (lqi < 0) {
    llb_tc_err(err, errlen, "No user query found in messages.");
    goto fail;
  }

  for (size_t i = 0; i < nm; i += 1) {
    const LlbJson* m = msgs->kids[i];
    const char* role = llb_tc_role(m);
    const char* cs = LLB_TC_C(i);
    size_t cn = LLB_TC_N(i);
    if (!strcmp(role, "system")) {
      if (i != 0) {
        llb_tc_err(err, errlen, "System message must be at the beginning.");
        goto fail;
      }
    } else if (!strcmp(role, "user")) {
      llb_buf_puts(&o, "<|im_start|>user\n");
      llb_buf_put(&o, cs, cn);
      llb_buf_puts(&o, "<|im_end|>\n");
    } else if (!strcmp(role, "assistant")) {
      // reasoning: reasoning_content if a string, else split the content at </think> (86-97)
      const char* rs = "";
      size_t rn = 0, x, y;
      const LlbJson* rc = llb_json_get(m, "reasoning_content");
      if (rc && rc->t == LLB_J_STR) {
        rs = rc->s;
        rn = rc->n;
      } else {
        long k0 = llb_tc_find(cs, cn, "</think>");
        if (k0 >= 0) {
          rs = cs;
          rn = (size_t)k0;
          while (rn && rs[rn - 1] == '\n') rn -= 1;
          long t = llb_tc_rfind(rs, rn, "<think>");
          if (t >= 0) { rs += t + 7; rn -= (size_t)t + 7; }
          while (rn && rs[0] == '\n') { rs += 1; rn -= 1; }
          long k1 = llb_tc_rfind(cs, cn, "</think>");
          cs += k1 + 8;
          cn -= (size_t)k1 + 8;
          while (cn && cs[0] == '\n') { cs += 1; cn -= 1; }
        }
      }
      llb_tc_trim(rs, rn, &x, &y);
      if ((long)i > lqi) {
        llb_buf_puts(&o, "<|im_start|>assistant\n<think>\n");
        llb_buf_put(&o, rs + x, y - x);
        llb_buf_puts(&o, "\n</think>\n\n");
      } else {
        llb_buf_puts(&o, "<|im_start|>assistant\n");
      }
      llb_buf_put(&o, cs, cn);
      const LlbJson* tc = llb_json_get(m, "tool_calls");
      if (tc && tc->t == LLB_J_STR && tc->n) {
        llb_tc_err(err, errlen, "unsupported: tool_calls is a string");
        goto fail;
      }
      if (tc && tc->t == LLB_J_ARR) {
        llb_tc_trim(cs, cn, &x, &y);
        for (size_t k = 0; k < tc->n; k += 1) {
          if (llb_tc_call(&o, tc->kids[k], k == 0, y > x, err, errlen)) goto fail;
        }
      }
      llb_buf_puts(&o, "<|im_end|>\n");
    } else if (!strcmp(role, "tool")) {
      // consecutive tool messages share one user turn (127-139)
      const LlbJson* pv = i > 0 ? msgs->kids[i - 1] : NULL;
      if (pv && pv->n > 0 && strcmp(llb_tc_role(pv), "tool")) llb_buf_puts(&o, "<|im_start|>user");
      llb_buf_puts(&o, "\n<tool_response>\n");
      llb_buf_put(&o, cs, cn);
      llb_buf_puts(&o, "\n</tool_response>");
      if (i + 1 == nm || strcmp(llb_tc_role(msgs->kids[i + 1]), "tool")) llb_buf_puts(&o, "<|im_end|>\n");
    } else {
      llb_tc_err(err, errlen, "Unexpected message role.");
      goto fail;
    }
  }
  if (add_gen) llb_buf_puts(&o, enable_thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n");
#undef LLB_TC_C
#undef LLB_TC_N
  for (size_t i = 0; i < nm; i += 1) free(cb[i].p);
  free(cb); free(ca); free(cz);
  return o.p;
fail:
  for (size_t i = 0; i < nm; i += 1) free(cb[i].p);
  free(cb); free(ca); free(cz);
  free(o.p);
  return NULL;
}

char* llb_tc_render(const char* request_json, int add_generation_prompt, int enable_thinking, char* err, int errlen) {
  if (err && errlen > 0) err[0] = 0;
  if (!request_json) {
    llb_tc_err(err, errlen, "no request");
    return NULL;
  }
  LlbJson* req = llb_json_parse(request_json, strlen(request_json), err, errlen);
  if (!req) return NULL;
  char* r = llb_tc_render_doc(req, add_generation_prompt, enable_thinking, err, errlen);
  llb_json_free(req);
  return r;
}

// ---------------------------------------------------------------- response side

enum { LLB_TCP_CONTENT, LLB_TCP_REASON, LLB_TCP_CALL, LLB_TCP_AFTER, LLB_TCP_DONE };

typedef struct {
  LlbTcEvType type;
  char* text;
  size_t len;
  char* name;
  char* args;
  int index;
} LlbTcpEv;

struct LlbTcParser {
  LlbJson* tools;   // owned parse of tools_json (array, or the request holding one)
  int st;           // LLB_TCP_*
  int strip_nl;     // drop the "\n" run that follows </think>
  LlbBuf buf;       // bytes received but not yet turned into events
  size_t scan;      // CALL: bytes of buf already searched for </tool_call>
  int ncalls;
  LlbTcpEv* q;
  size_t qn, qhead, qcap;
  LlbTcpEv cur;     // the event last handed out by llb_tcp_next (freed on the next call)
};

static void llb_tcp_ev_free(LlbTcpEv* e) {
  free(e->text);
  free(e->name);
  free(e->args);
  memset(e, 0, sizeof *e);
}

static void llb_tcp_push(LlbTcParser* P, LlbTcEvType t, const char* s, size_t n, char* name, char* args) {
  if (t != LLB_TCE_TOOL_CALL && n == 0) return;
  if (P->qn == P->qcap) {
    P->qcap = P->qcap ? P->qcap * 2 : 16;
    P->q = realloc(P->q, P->qcap * sizeof *P->q);
    if (!P->q) abort();
  }
  LlbTcpEv* e = &P->q[P->qn++];
  memset(e, 0, sizeof *e);
  e->type = t;
  e->text = malloc(n + 1);
  if (!e->text) abort();
  memcpy(e->text, s, n);
  e->text[n] = 0;
  e->len = n;
  e->name = name;
  e->args = args;
  if (t == LLB_TCE_TOOL_CALL) e->index = P->ncalls++;
}

static void llb_tcp_consume(LlbTcParser* P, size_t k) {
  memmove(P->buf.p, P->buf.p + k, P->buf.n - k);
  P->buf.n -= k;
  P->buf.p[P->buf.n] = 0;
}

// bytes at the end of s[0..n) to hold back: the longest proper prefix of tag ending the
// buffer, extended so the released part does not end inside a UTF-8 sequence
static size_t llb_tcp_hold(const char* s, size_t n, const char* tag) {
  size_t m = strlen(tag), keep = 0;
  for (size_t k = m - 1; k >= 1; k -= 1) {
    if (k <= n && !memcmp(s + n - k, tag, k)) { keep = k; break; }
  }
  size_t r = n - keep;  // released length; back off an incomplete trailing code point
  for (size_t j = r, back = 0; j-- > 0 && back < 3; back += 1) {
    unsigned char c = (unsigned char)s[j];
    if ((c & 0xC0) == 0x80) continue;
    size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    if (j + need > r) r = j;
    break;
  }
  return n - r;
}

static LlbJson* llb_tcp_jstr(const char* s, size_t n) {
  LlbJson* j = llb_jnew(LLB_J_STR);
  j->s = malloc(n + 1);
  if (!j->s) abort();
  memcpy(j->s, s, n);
  j->s[n] = 0;
  j->n = n;
  return j;
}

static LlbJson* llb_tcp_jnum(const char* lex) {
  LlbJson* j = llb_tcp_jstr(lex, strlen(lex));
  j->t = LLB_J_NUM;
  return j;
}

// the schema "properties" object of tool `name`, or NULL
static const LlbJson* llb_tcp_props(const LlbJson* tools, const char* name) {
  if (tools && tools->t == LLB_J_OBJ) tools = llb_json_get(tools, "tools");
  if (!tools || tools->t != LLB_J_ARR) return NULL;
  for (size_t i = 0; i < tools->n; i += 1) {
    const LlbJson* f = llb_json_get(tools->kids[i], "function");
    if (!f) f = tools->kids[i];
    const LlbJson* nm = llb_json_get(f, "name");
    if (!nm || nm->t != LLB_J_STR || strcmp(nm->s, name)) continue;
    const LlbJson* pr = llb_json_get(llb_json_get(f, "parameters"), "properties");
    return pr && pr->t == LLB_J_OBJ ? pr : NULL;
  }
  return NULL;
}

static int llb_tcp_ieq(const char* s, size_t n, const char* w) {
  size_t m = strlen(w);
  if (n != m) return 0;
  for (size_t i = 0; i < n; i += 1) {
    char c = s[i] >= 'A' && s[i] <= 'Z' ? (char)(s[i] + 32) : s[i];
    if (c != w[i]) return 0;
  }
  return 1;
}

// a parameter value typed by its schema (see the header's table)
static LlbJson* llb_tcp_convert(const LlbJson* props, const char* pname, const char* v, size_t n) {
  const LlbJson* ps = props ? llb_json_get(props, pname) : NULL;
  if (!ps) return llb_tcp_jstr(v, n);
  const LlbJson* ty = llb_json_get(ps, "type");
  int is_null = llb_tcp_ieq(v, n, "null") || (n == 4 && !memcmp(v, "None", 4));
  if (ty && ty->t == LLB_J_ARR) {
    // ["string", "null"]: a nullable type takes "null"/"None" as null even when string-like
    const LlbJson* pick = NULL;
    for (size_t i = 0; i < ty->n; i += 1) {
      if (ty->kids[i]->t != LLB_J_STR) continue;
      if (!strcmp(ty->kids[i]->s, "null")) { if (is_null) return llb_jnew(LLB_J_NULL); }
      else if (!pick) pick = ty->kids[i];
    }
    ty = pick;
  }
  char t[32] = "string";
  if (ty && ty->t == LLB_J_STR) {
    size_t a, z, k = 0;
    llb_tc_trim(ty->s, ty->n, &a, &z);
    for (; a < z && k + 1 < sizeof t; a += 1) t[k++] = ty->s[a] >= 'A' && ty->s[a] <= 'Z' ? (char)(ty->s[a] + 32) : ty->s[a];
    t[k] = 0;
  }
  if (!strcmp(t, "string") || !strcmp(t, "str") || !strcmp(t, "text") || !strcmp(t, "varchar") || !strcmp(t, "char") || !strcmp(t, "enum"))
    return llb_tcp_jstr(v, n);
  if (is_null) return llb_jnew(LLB_J_NULL);
  size_t a, z;
  llb_tc_trim(v, n, &a, &z);
  const char* s = v + a;
  size_t sn = z - a;
  if (!strncmp(t, "int", 3) || !strncmp(t, "uint", 4) || !strncmp(t, "long", 4) || !strncmp(t, "short", 5) || !strncmp(t, "unsigned", 8)) {
    size_t i = 0;
    int neg = 0;
    if (i < sn && (s[i] == '+' || s[i] == '-')) neg = s[i++] == '-';
    size_t d0 = i;
    while (i < sn && s[i] >= '0' && s[i] <= '9') i += 1;
    if (i == sn && i > d0) {
      while (d0 + 1 < sn && s[d0] == '0') d0 += 1;
      LlbBuf b = {0};
      if (neg && !(sn - d0 == 1 && s[d0] == '0')) llb_buf_puts(&b, "-");
      llb_buf_put(&b, s + d0, sn - d0);
      LlbJson* j = llb_tcp_jnum(b.p);
      free(b.p);
      return j;
    }
    return llb_tcp_jstr(v, n);
  }
  if (!strncmp(t, "num", 3) || !strncmp(t, "float", 5)) {
    // decimal float syntax only (strtod alone would take hex, inf, nan)
    size_t i = 0, nd = 0;
    if (i < sn && (s[i] == '+' || s[i] == '-')) i += 1;
    while (i < sn && s[i] >= '0' && s[i] <= '9') { i += 1; nd += 1; }
    if (i < sn && s[i] == '.') {
      i += 1;
      while (i < sn && s[i] >= '0' && s[i] <= '9') { i += 1; nd += 1; }
    }
    if (nd && i < sn && (s[i] == 'e' || s[i] == 'E')) {
      size_t e0;
      i += 1;
      if (i < sn && (s[i] == '+' || s[i] == '-')) i += 1;
      e0 = i;
      while (i < sn && s[i] >= '0' && s[i] <= '9') i += 1;
      if (i == e0) nd = 0;
    }
    if (nd && i == sn && sn < 400) {
      char tmp[400], lex[420];
      memcpy(tmp, s, sn);
      tmp[sn] = 0;
      double d = strtod(tmp, NULL);
      if (isfinite(d)) {
        if (d == floor(d)) {
          snprintf(lex, sizeof lex, "%.0f", d);
          if (!strcmp(lex, "-0")) strcpy(lex, "0");
          return llb_tcp_jnum(lex);
        }
        LlbBuf b = {0};
        llb_tc_pyfloat(&b, d, 1);
        LlbJson* j = llb_tcp_jnum(b.p);
        free(b.p);
        return j;
      }
    }
    return llb_tcp_jstr(v, n);
  }
  if (!strcmp(t, "boolean") || !strcmp(t, "bool") || !strcmp(t, "binary")) {
    if (llb_tcp_ieq(v, n, "true") || llb_tcp_ieq(v, n, "false")) {
      LlbJson* j = llb_jnew(LLB_J_BOOL);
      j->b = llb_tcp_ieq(v, n, "true");
      return j;
    }
    return llb_tcp_jstr(v, n);
  }
  LlbJson* j = llb_json_parse(v, n, NULL, 0);
  if (j) return j;
  if (strcmp(t, "object") && strcmp(t, "array") && strcmp(t, "arr") && strncmp(t, "dict", 4) && strncmp(t, "list", 4)) {
    if ((n == 4 && !memcmp(v, "True", 4)) || (n == 5 && !memcmp(v, "False", 5))) {
      j = llb_jnew(LLB_J_BOOL);
      j->b = v[0] == 'T';
      return j;
    }
  }
  return llb_tcp_jstr(v, n);
}

static int llb_tcp_isws(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

// parse the body of one <tool_call>...</tool_call>; pushes TOOL_CALL events (all or none)
static int llb_tcp_body(LlbTcParser* P, const char* s, size_t n, char* err, int errlen) {
  char** names = NULL;
  char** argv = NULL;
  size_t nf = 0, i = 0;
  int rc = -1;
  for (;;) {
    while (i < n && llb_tcp_isws(s[i])) i += 1;
    if (i == n) break;
    if (!llb_tc_starts(s + i, n - i, "<function=")) { llb_tc_err(err, errlen, "tool call: expected <function="); goto out; }
    i += 10;
    long fe = llb_tc_find(s + i, n - i, "</function>");
    size_t fend = fe < 0 ? n : i + (size_t)fe, next = fe < 0 ? n : fend + 11;
    long gt = llb_tc_find(s + i, fend - i, ">");
    if (gt <= 0) { llb_tc_err(err, errlen, "tool call: bad function name"); goto out; }
    char* name = malloc((size_t)gt + 1);
    if (!name) abort();
    memcpy(name, s + i, (size_t)gt);
    name[gt] = 0;
    const LlbJson* props = llb_tcp_props(P->tools, name);
    LlbJson* obj = llb_jnew(LLB_J_OBJ);
    size_t j = i + (size_t)gt + 1;
    for (;;) {
      while (j < fend && llb_tcp_isws(s[j])) j += 1;
      if (j == fend) break;
      if (!llb_tc_starts(s + j, fend - j, "<parameter=")) {
        llb_tc_err(err, errlen, "tool call: unexpected text in function %s", name);
        free(name);
        llb_json_free(obj);
        goto out;
      }
      j += 11;
      long pg = llb_tc_find(s + j, fend - j, ">");
      if (pg < 0) {
        llb_tc_err(err, errlen, "tool call: bad parameter name");
        free(name);
        llb_json_free(obj);
        goto out;
      }
      char* pname = malloc((size_t)pg + 1);
      if (!pname) abort();
      memcpy(pname, s + j, (size_t)pg);
      pname[pg] = 0;
      j += (size_t)pg + 1;
      long pa = llb_tc_find(s + j, fend - j, "</parameter>");
      long pb = llb_tc_find(s + j, fend - j, "<parameter=");
      size_t vend = fend, skip = 0;
      if (pa >= 0 && (pb < 0 || pa < pb)) { vend = j + (size_t)pa; skip = 12; }
      else if (pb >= 0) vend = j + (size_t)pb;
      const char* v = s + j;
      size_t vn = vend - j;
      if (vn && v[0] == '\n') { v += 1; vn -= 1; }
      if (vn && v[vn - 1] == '\n') vn -= 1;
      llb_jadd(obj, pname, llb_tcp_convert(props, pname, v, vn));
      j = vend + skip;
    }
    LlbBuf ab = {0};
    llb_buf_put(&ab, "", 0);
    llb_tc_tojson(&ab, obj);
    llb_json_free(obj);
    names = realloc(names, (nf + 1) * sizeof *names);
    argv = realloc(argv, (nf + 1) * sizeof *argv);
    if (!names || !argv) abort();
    names[nf] = name;
    argv[nf++] = ab.p;
    i = next;
  }
  if (!nf) { llb_tc_err(err, errlen, "tool call: no <function=...> block"); goto out; }
  for (size_t k = 0; k < nf; k += 1) llb_tcp_push(P, LLB_TCE_TOOL_CALL, "", 0, names[k], argv[k]);
  nf = 0;
  rc = 0;
out:
  for (size_t k = 0; k < nf; k += 1) { free(names[k]); free(argv[k]); }
  free(names);
  free(argv);
  return rc;
}

// raw call bytes back out as content after an ERROR event
static void llb_tcp_reject(LlbTcParser* P, const char* msg, const char* body, size_t n, int closed) {
  LlbBuf b = {0};
  llb_tcp_push(P, LLB_TCE_ERROR, msg, strlen(msg), NULL, NULL);
  llb_buf_puts(&b, "<tool_call>");
  llb_buf_put(&b, body, n);
  if (closed) llb_buf_puts(&b, "</tool_call>");
  llb_tcp_push(P, LLB_TCE_CONTENT, b.p, b.n, NULL, NULL);
  free(b.p);
}

static void llb_tcp_run(LlbTcParser* P, int fin) {
  for (;;) {
    char* s = P->buf.p;
    size_t n = P->buf.n;
    if (n == 0) return;
    if (P->st == LLB_TCP_REASON) {
      long k = llb_tc_find(s, n, "</think>");
      if (k >= 0) {
        llb_tcp_push(P, LLB_TCE_REASONING, s, (size_t)k, NULL, NULL);
        llb_tcp_consume(P, (size_t)k + 8);
        P->st = LLB_TCP_CONTENT;
        P->strip_nl = 1;
        continue;
      }
      size_t keep = fin ? 0 : llb_tcp_hold(s, n, "</think>");
      llb_tcp_push(P, LLB_TCE_REASONING, s, n - keep, NULL, NULL);
      llb_tcp_consume(P, n - keep);
      return;
    }
    if (P->st == LLB_TCP_CONTENT && P->strip_nl) {
      size_t i = 0;
      while (i < n && s[i] == '\n') i += 1;
      llb_tcp_consume(P, i);
      if (i == n) return;
      P->strip_nl = 0;
      continue;
    }
    if (P->st == LLB_TCP_AFTER) {
      // whitespace after a call is template glue; another call or real text ends it
      size_t i = 0;
      while (i < n && llb_tcp_isws(s[i])) i += 1;
      if (i == n) {
        if (fin) llb_tcp_consume(P, n);
        return;
      }
      if (llb_tc_starts(s + i, n - i, "<tool_call>")) {
        llb_tcp_consume(P, i);
        P->st = LLB_TCP_CONTENT;
        continue;
      }
      if (!fin && n - i < 11 && !memcmp(s + i, "<tool_call>", n - i)) return;
      P->st = LLB_TCP_CONTENT;
      continue;
    }
    if (P->st == LLB_TCP_CONTENT) {
      long k = llb_tc_find(s, n, "<tool_call>");
      if (k >= 0) {
        llb_tcp_push(P, LLB_TCE_CONTENT, s, (size_t)k, NULL, NULL);
        llb_tcp_consume(P, (size_t)k + 11);
        P->st = LLB_TCP_CALL;
        P->scan = 0;
        continue;
      }
      size_t keep = fin ? 0 : llb_tcp_hold(s, n, "<tool_call>");
      llb_tcp_push(P, LLB_TCE_CONTENT, s, n - keep, NULL, NULL);
      llb_tcp_consume(P, n - keep);
      return;
    }
    // LLB_TCP_CALL: buffer the body up to </tool_call>
    size_t from = P->scan > 11 ? P->scan - 11 : 0;
    long k = llb_tc_find(s + from, n - from, "</tool_call>");
    if (k < 0) {
      P->scan = n;
      if (fin) {
        llb_tcp_reject(P, "unterminated <tool_call>", s, n, 0);
        llb_tcp_consume(P, n);
        P->st = LLB_TCP_CONTENT;
      }
      return;
    }
    size_t end = from + (size_t)k;
    char e[160];
    e[0] = 0;
    if (llb_tcp_body(P, s, end, e, sizeof e)) {
      llb_tcp_reject(P, e, s, end, 1);
      P->st = LLB_TCP_CONTENT;
    } else {
      P->st = LLB_TCP_AFTER;
    }
    llb_tcp_consume(P, end + 12);
  }
}

LlbTcParser* llb_tcp_new(const char* tools_json) {
  LlbTcParser* P = calloc(1, sizeof *P);
  if (!P) abort();
  if (tools_json) {
    P->tools = llb_json_parse(tools_json, strlen(tools_json), NULL, 0);
    if (!P->tools) { free(P); return NULL; }
  }
  llb_buf_put(&P->buf, "", 0);
  return P;
}

void llb_tcp_set_reasoning(LlbTcParser* P, int in_reasoning) {
  if (P->st == LLB_TCP_CONTENT || P->st == LLB_TCP_REASON) P->st = in_reasoning ? LLB_TCP_REASON : LLB_TCP_CONTENT;
}

int llb_tcp_feed(LlbTcParser* P, const char* bytes, int len) {
  if (P->st == LLB_TCP_DONE) return -1;
  if (len > 0) llb_buf_put(&P->buf, bytes, (size_t)len);
  llb_tcp_run(P, 0);
  return (int)(P->qn - P->qhead);
}

int llb_tcp_finish(LlbTcParser* P) {
  if (P->st == LLB_TCP_DONE) return -1;
  llb_tcp_run(P, 1);
  P->st = LLB_TCP_DONE;
  return (int)(P->qn - P->qhead);
}

int llb_tcp_next(LlbTcParser* P, LlbTcEvent* ev) {
  llb_tcp_ev_free(&P->cur);
  memset(ev, 0, sizeof *ev);
  if (P->qhead == P->qn) {
    P->qhead = P->qn = 0;
    return 0;
  }
  P->cur = P->q[P->qhead++];
  ev->type = P->cur.type;
  ev->text = P->cur.text;
  ev->len = P->cur.len;
  ev->name = P->cur.name;
  ev->args = P->cur.args;
  ev->index = P->cur.index;
  return 1;
}

void llb_tcp_free(LlbTcParser* P) {
  if (!P) return;
  llb_tcp_ev_free(&P->cur);
  for (size_t i = P->qhead; i < P->qn; i += 1) llb_tcp_ev_free(&P->q[i]);
  free(P->q);
  free(P->buf.p);
  llb_json_free(P->tools);
  free(P);
}

#endif
