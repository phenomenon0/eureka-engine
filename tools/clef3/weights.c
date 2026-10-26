#define _POSIX_C_SOURCE 200809L
#include "head.h"
#include "../../host/llb_gguf.c"
#include <math.h>

static float half(const unsigned char *p) {
    _Float16 h;
    memcpy(&h, p, 2);
    return (float)h;
}

static void dequant(const unsigned char *b, float *w, int n, int type) {
    if (type == 0) memcpy(w, b, n*4);
    else if (type == 1) {
        for (int i = 0; i < n; i++) w[i] = half(b+2*i);
    } else if (type == 30) {
        for (int i = 0; i < n; i++) {
            uint16_t v;
            memcpy(&v, b+2*i, 2);
            uint32_t u=(uint32_t)v<<16;
            memcpy(w+i, &u, 4);
        }
    } else if (type == 8) {
        for (int i = 0; i < n; i += 32, b += 34) {
            float d = half(b);
            for (int j = 0; j < 32; j++) w[i+j] = d * (int8_t)b[2+j];
        }
    } else if (type == 14) {
        // GGML Q6_K: 128 low bytes, 64 high bytes, 16 signed scales, f16 d.
        for (int i = 0; i < n; i += 256, b += 210) {
            float d = half(b+208);
            for (int j = 0; j < 256; j++) {
                int group=j/128, lane=j%32, quarter=(j%128)/32;
                unsigned low=(b[group*64+lane+(quarter%2)*32]>>(quarter>=2 ? 4 : 0))&15;
                unsigned high=(b[128+group*32+lane]>>(quarter*2))&3;
                w[i+j] = (d * (int8_t)b[192+group*8+quarter*2+lane/16]) * ((int)(low|(high<<4))-32);
            }
        }
    } else clef_fail("unsupported head/lexical tensor type");
    for (int i = 0; i < n; i++) if (!isfinite(w[i])) clef_fail("nonfinite tensor");
}

// "<arch>.embedding_length" (one dot, so not e.g. "clef.vision.embedding_length"), an integer value.
static int width_key(const char *key, uint32_t type) {
    const char *dot=key ? strchr(key,'.') : NULL;
    return dot && dot==strrchr(key,'.') && !strcmp(dot,".embedding_length") && (type==4 || type==5 || type==10 || type==11);
}

void clef_head_open(Head *h, const char *path) {
    memset(h, 0, sizeof *h);
    h->fd = open(path, O_RDONLY);
    struct stat st;
    if (h->fd < 0 || fstat(h->fd, &st)) clef_fail("cannot open GGUF");
    GgRd r = {
        h->fd,0,0,malloc(1<<20),0,0
    };
    char magic[4];
    gg_rd_get(&r,magic,4);
    uint32_t ver=gg_u32(&r);
    uint64_t nt=gg_u64(&r), nkv=gg_u64(&r), align=32;
    if (memcmp(magic,"GGUF",4) || ver!=3 || nt>100000 || nkv>100000) clef_fail("invalid GGUF header");
    // The head's shape comes from the GGUF: <arch>.embedding_length (the trunk width the head reads) and the
    // tokenizer's token count (the vocab), checked against output.weight below. Absent keys are not required.
    int64_t meta_d=-1, meta_vocab=-1;
    for (uint64_t i=0; i<nkv && !r.bad; i++) {
        char *key=gg_str(&r,1);
        uint32_t type=gg_u32(&r);
        if (key && type==9 && !strcmp(key,"tokenizer.ggml.tokens")) {
            uint32_t et=gg_u32(&r);
            uint64_t n=gg_u64(&r);
            if (et!=8) r.bad=1;
            for (uint64_t j=0; j<n && !r.bad; j++) gg_str(&r,0);
            meta_vocab=(int64_t)n;
            free(key);
            continue;
        }
        uint64_t v=gg_val(&r,type);
        if (key && !strcmp(key,"general.alignment")) align=v;
        if (width_key(key,type)) meta_d=(int64_t)v;
        free(key);
    }
    for (uint64_t i=0; i<nt && !r.bad; i++) {
        char *name=gg_str(&r,1);
        uint32_t nd=gg_u32(&r);
        uint64_t ne[4]={
            1,1,1,1
        }, n=1;
        if (!name || nd<1 || nd>4) clef_fail("invalid tensor header");
        for (unsigned j=0; j<nd; j++) {
            ne[j]=gg_u64(&r);
            if (!ne[j] || ne[j]>1000000000 || n>UINT64_MAX/ne[j]) clef_fail("invalid tensor shape");
            n*=ne[j];
        } uint32_t type=gg_u32(&r);
        uint64_t off=gg_u64(&r);
        int keep=!strncmp(name,"dec.",4) || !strncmp(name,"decision.",9) || !strcmp(name,"token_types.weight") || !strcmp(name,"output.weight");
        if (!keep) {
            free(name);
            continue;
        }
        for (int j=0; j<h->nt; j++) if (!strcmp(name,h->t[j].name)) clef_fail("duplicate head tensor");
        int bb,el;
        if (!gg_type(type,&bb,&el) || n%el || nd>2 || ne[0]%el || h->nt==161) clef_fail("invalid head layout");
        if (strcmp(name,"output.weight") && type!=0 && type!=8 && type!=30) clef_fail("head requires F32, Q8_0 or BF16");
        h->t[h->nt++]=(Tensor){
            name,(int)ne[1],(int)ne[0],(int)type,off,n/el*bb,NULL
        };
    }
    if (r.bad || !align || align>4096 || h->nt!=161) clef_fail("missing/corrupt head tensors");
    uint64_t data=(r.off+r.p+align-1)/align*align;
    free(r.buf);
    for (int i=0; i<h->nt; i++) {
        Tensor *t=&h->t[i];
        if (t->offset>UINT64_MAX-data) clef_fail("tensor offset overflow");
        t->offset+=data;
        if (t->offset>(uint64_t)st.st_size || t->bytes>(uint64_t)st.st_size-t->offset) clef_fail("tensor outside GGUF");
        if (!strcmp(t->name,"output.weight")) {
            // Any width/vocab, as long as the GGUF agrees with itself; types = the ones dequant() reads.
            static char why[4608];
            const char *bad=NULL;
            if (t->type!=0 && t->type!=1 && t->type!=8 && t->type!=14 && t->type!=30) bad="type has no CPU dequant (F32/F16/BF16/Q8_0/Q6_K)";
            else if (t->cols<256 || t->cols>8192 || t->cols%256) bad="cols must be a multiple of 256 in 256..8192";
            else if (meta_d>=0 && t->cols!=meta_d) bad="cols != <arch>.embedding_length";
            else if (meta_vocab>=0 && t->rows!=meta_vocab) bad="rows != tokenizer.ggml.tokens count";
            if (bad) {
                snprintf(why,sizeof why,"unsupported output.weight %d x %d type %d (embedding_length %lld, vocab %lld): %s: %s",
                         t->rows,t->cols,t->type,(long long)meta_d,(long long)meta_vocab,bad,path);
                clef_fail(why);
            }
            h->d=t->cols;
            h->vocab=t->rows;
            h->orow=t->bytes/(uint64_t)t->rows;
            continue;
        } uint64_t n=(uint64_t)t->rows*t->cols;
        if (n>16777216) clef_fail("head tensor too large");
        unsigned char *raw=malloc(t->bytes);
        t->w=malloc(n*4);
        if (!raw || !t->w || gg_pread(h->fd,raw,t->bytes,t->offset)) clef_fail("tensor read failed");
        dequant(raw,t->w,(int)n,t->type);
        free(raw);
    }
    if (!h->d) clef_fail("output.weight missing");
}
float *clef_weight(Head *h, const char *name, int rows, int cols) {
    for (int i=0; i<h->nt; i++) if (!strcmp(name,h->t[i].name)) {
        Tensor *t=&h->t[i];
        if (t->rows!=rows || t->cols!=cols || !t->w) clef_fail(name);
        return t->w;
    }
    clef_fail(name);
    return NULL;
}

void clef_output_row(Head *h, int id, float *out) {
    for (int i=0; i<h->nt; i++) if (!strcmp(h->t[i].name,"output.weight")) {
        Tensor *t=&h->t[i];
        unsigned char raw[8192*4];
        if (id<0 || id>=t->rows || h->orow>sizeof raw || gg_pread(h->fd,raw,h->orow,t->offset+(uint64_t)id*h->orow)) clef_fail("output row read failed");
        dequant(raw,out,t->cols,t->type);
        return;
    }
    clef_fail("output.weight missing");
}

void clef_head_close(Head *h) {
    for (int i=0; i<h->nt; i++) {
        free(h->t[i].name);
        free(h->t[i].w);
    }
    close(h->fd);
}
float *clef_hidden_read(const char *path, int tokens, int d) {
    FILE *f=fopen(path,"rb");
    if (!f) clef_fail("cannot open hidden dump");
    float *h=malloc((size_t)tokens*d*4);
    if (!h) clef_fail("hidden allocation failed");
    for (int i=0; i<tokens; i++) {
        uint32_t head[2];
        if (fread(head,4,2,f)!=2 || head[0]!=(uint32_t)i || head[1]!=(uint32_t)d || fread(h+(size_t)i*d,4,d,f)!=(size_t)d) clef_fail("hidden row order/shape mismatch");
    }
    if (fgetc(f)!=EOF) clef_fail("extra hidden rows");
    fclose(f);
    for (size_t i=0; i<(size_t)tokens*d; i++) if (!isfinite(h[i])) clef_fail("nonfinite hidden row");
    return h;
}

int clef_gguf_width(const char *path) {
    int fd=open(path,O_RDONLY);
    if (fd<0) clef_fail("cannot open GGUF");
    GgRd r={
        fd,0,0,malloc(1<<20),0,0
    };
    char magic[4];
    gg_rd_get(&r,magic,4);
    uint32_t ver=gg_u32(&r);
    uint64_t nt=gg_u64(&r), nkv=gg_u64(&r);
    (void)nt;
    if (memcmp(magic,"GGUF",4) || ver!=3 || nkv>100000) clef_fail("invalid GGUF header");
    int64_t d=-1;
    for (uint64_t i=0; i<nkv && !r.bad; i++) {
        char *key=gg_str(&r,1);
        uint32_t type=gg_u32(&r);
        uint64_t v=gg_val(&r,type);
        if (width_key(key,type)) d=(int64_t)v;
        free(key);
    }
    free(r.buf);
    close(fd);
    if (r.bad || d<=0 || d>8192) {
        static char why[4352];
        snprintf(why,sizeof why,"trunk GGUF has no usable <arch>.embedding_length (%lld): %s",(long long)d,path);
        clef_fail(why);
    }
    return (int)d;
}
