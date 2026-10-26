// tools/qwen35_template.c — see qwen35_template.h
#include <stdio.h>
#include <string.h>
#include "qwen35_template.h"

typedef struct { char *buf; int cap; int n; } Out;
static void out_bytes(Out *o, const char *s, int n) {
    if (o->n + n < o->cap) memcpy(o->buf + o->n, s, n);
    o->n += n;
}
static void out_str(Out *o, const char *s) { out_bytes(o, s, (int)strlen(s)); }

// Jinja |trim: strip ASCII whitespace both ends
static int is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static void out_trim(Out *o, const char *s) {
    int n = (int)strlen(s), a = 0, b = n;
    while (a < b && is_ws(s[a])) a++;
    while (b > a && is_ws(s[b - 1])) b--;
    out_bytes(o, s + a, b - a);
}

int q35_chat_render(const Q35Msg *msgs, int n, int add_generation_prompt, int think_mode,
                    char *out, int out_cap) {
    Out o = { out, out_cap, 0 };
    if (n <= 0) return 0;

    // last genuine user query index (the template's last_query_index)
    int last_query = -1;
    for (int i = n - 1; i >= 0; i--) {
        if (strcmp(msgs[i].role, "user") == 0) { last_query = i; break; }
    }

    for (int i = 0; i < n; i++) {
        if (strcmp(msgs[i].role, "system") == 0) {
            out_str(&o, "<|im_start|>system\n");
            out_trim(&o, msgs[i].content);
            out_str(&o, "<|im_end|>\n");
        } else if (strcmp(msgs[i].role, "user") == 0) {
            out_str(&o, "<|im_start|>user\n");
            out_trim(&o, msgs[i].content);
            out_str(&o, "<|im_end|>\n");
        } else if (strcmp(msgs[i].role, "assistant") == 0) {
            out_str(&o, "<|im_start|>assistant\n");
            const char *content = msgs[i].content;
            if (i > last_query) {
                // split reasoning out of an assistant turn that carries </think>
                const char *th = strstr(content, "</think>");
                const char *reason = "";
                if (th) {
                    const char *start = content;
                    const char *open = strstr(content, "<think>");
                    if (open && open < th) start = open + 7;
                    // trim reasoning
                    static char rbuf[1 << 20];
                    int rn = (int)(th - start);
                    int a = 0, b = rn;
                    while (a < b && is_ws(start[a])) a++;
                    while (b > a && is_ws(start[b - 1])) b--;
                    if (b - a < (int)sizeof rbuf) { memcpy(rbuf, start + a, b - a); rbuf[b - a] = 0; reason = rbuf; }
                    const char *rest = th + 8;
                    while (*rest == '\n') rest++;
                    content = rest;
                }
                out_str(&o, "<think>\n");
                out_str(&o, reason);
                out_str(&o, "\n</think>\n\n");
                out_trim(&o, content);
            } else {
                out_trim(&o, content);
            }
            out_str(&o, "<|im_end|>\n");
        }
    }

    if (add_generation_prompt) {
        out_str(&o, "<|im_start|>assistant\n");
        if (think_mode == 1) out_str(&o, "<think>\n");
        else if (think_mode == 2) out_str(&o, "<think>\n\n</think>\n\n");
    }
    if (o.cap > 0) {
        int end = o.n < o.cap - 1 ? o.n : o.cap - 1;
        if (end < 0) end = 0;
        out[end] = 0;
    }
    return o.n;
}
