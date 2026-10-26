#define _POSIX_C_SOURCE 200809L
#include "head.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv) {
    if (argc!=5) clef_fail("usage: head-test GGUF TOKENIZER_DIR REQUEST HIDDEN|synthetic");
    Q35Tokenizer tok; if (q35_tok_load(&tok,argv[2])) clef_fail("tokenizer load");
    char *raw=json_read_file(argv[3]); const char *err=NULL; if (!raw) clef_fail("request read");
    jv *req=json_parse(raw,&err); if (!req) clef_fail(err); Record *r=calloc(1,sizeof *r); clef_encode(&tok,req,r,CLEF_MAX_T);
    Head h; clef_head_open(&h,argv[1]); const int d=h.d;
    float *hidden;
    if (!strcmp(argv[4],"synthetic")) {
        hidden=malloc((size_t)r->n*d*4);
        for (size_t i=0; i<(size_t)r->n*d; i++) hidden[i]=sinf((float)(i%7001)*0.03125f)+cosf((float)(i%113)*0.125f);
    } else hidden=clef_hidden_read(argv[4],r->n,d);
    const char *dump=getenv("CLEF_HEAD_DUMP");
    if (dump) { char path[4096]; snprintf(path,sizeof path,"%s/input.f32",dump); FILE *f=fopen(path,"wb"); if (!f || fwrite(hidden,4,(size_t)r->n*d,f)!=(size_t)r->n*d || fclose(f)) clef_fail("input dump"); }
    float *logits=calloc(r->no,4); clef_head_run(&h,r,hidden,logits);
    for (int i=0; i<r->no; i++) printf("%.9g\n",logits[i]);
    clef_head_close(&h); free(logits); free(hidden); clef_record_free(r); free(r); json_free(req); free(raw); q35_tok_free(&tok);
}
