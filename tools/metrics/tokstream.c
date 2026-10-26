// tokstream.c: a FILE* that keeps everything written to it (like open_memstream)
// and stamps llb_req_token() on every newline, so an engine that prints one
// token id per line gets true per-token clocks without an engine change.
// Line-buffered: the engine's "id\n" reaches the cookie as it is printed.
#define _GNU_SOURCE
#include "metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

typedef struct {
    llb_req *r;     // NULL = keep bytes only, no clocks
    char **p;
    size_t *n, cap;
} tok_cookie;

static ssize_t tok_write(void *c, const char *data, size_t len) {
    tok_cookie *k = c;
    if (*k->n + len + 1 > k->cap) {
        size_t nc = k->cap ? k->cap * 2 : 4096;
        while (nc < *k->n + len + 1) nc *= 2;
        char *np = realloc(*k->p, nc);
        if (!np) return -1;
        *k->p = np;
        k->cap = nc;
    }
    memcpy(*k->p + *k->n, data, len);
    *k->n += len;
    (*k->p)[*k->n] = 0;
    if (k->r) {
        double t = llb_now();
        for (size_t i = 0; i < len; i++)
            if (data[i] == '\n') llb_req_token_at(k->r, t);
    }
    return (ssize_t)len;
}

static int tok_close(void *c) {
    free(c);
    return 0;
}

FILE *llb_token_stream(llb_req *r, char **buf, size_t *len) {
    tok_cookie *k = calloc(1, sizeof *k);
    if (!k) return NULL;
    k->r = r;
    k->p = buf;
    k->n = len;
    *buf = calloc(1, 1);   // "" even if nothing is written (memstream contract)
    *len = 0;
    k->cap = 1;
    cookie_io_functions_t io = {.write = tok_write, .close = tok_close};
    FILE *f = fopencookie(k, "w", io);
    if (!f) {
        free(k);
        return NULL;
    }
    setvbuf(f, NULL, _IOLBF, 0);
    return f;
}

int llb_stdin_queued(FILE *f) {
    // complete request lines already read into f's buffer and not yet consumed
    // (glibc FILE internals; a lower bound: bytes still in the pipe are not seen)
    int q = 0;
    for (const char *s = f->_IO_read_ptr; s && s < f->_IO_read_end; s++) q += *s == '\n';
    return q;
}
