// dual.c: SYS1-DUAL request side (docs/sys1dual-registration.md). Linked into
// the engine; called by the Sys1.next / Sys1.done effects (sys1.c).
// stdin: one JSON request per line. stdout: one JSON response per line.
//   {"mode":"generate","prompt":TEXT,"n":N} -> {"prompt_ids":[...],"ids":[...],"text":"..."}
//   anything else = a clef-decide request  -> the clef-decide response
// The engine's own stdout prints (gen's ids) go to a memstream for the request;
// responses go to the real stdout. Timings: one JSON line per request on stderr.
#define _POSIX_C_SOURCE 200809L
#include "../clef3/head.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static Q35Tokenizer tok;
static Head head;
static FILE *out,*cap;
static char *capp,hidden_path[4096];
static size_t capn;
static char *text;
static jv *req;
static Record *rec;
static int mode,steps,seq,prompt_n,prompt_ids[CLEF_MAX_T];
static double t_req;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}

static void reply(const char *s) {
    fputs(s,out);
    fputc('\n',out);
    fflush(out);
}

static void refuse(const char *msg) {
    sb e={0};
    sb_add(&e,"{\"error\":");
    json_quote(&e,msg);
    sb_addc(&e,'}');
    reply(e.p);
    free(e.p);
    fprintf(stderr,"{\"seq\":%d,\"error\":\"%s\"}\n",seq,msg);
}

static void drop(void) {
    if (rec) {
        clef_record_free(rec);
        free(rec);
    }
    json_free(req);
    free(text);
    rec=NULL;
    req=NULL;
    text=NULL;
}

// copied from tools/clef3/decide.c response() (CLEF-3, commit a752e101), unchanged
static void response(const jv *req,const Record *r,const float *logits,sb *out) {
    sb_add(out,"{\"model\":");
    json_dump(out,json_get(req,"model"));
    sb_add(out,",\"answers\":{");
    const char *types[]={
        "noul","choice","score"
    };
    for (int i=0; i<r->nq; i++) {
        Question q=r->q[i];
        int n=q.count,best=0;
        double p[CLEF_MAX_O],sum=0,max=logits[q.first];
        for (int j=1; j<n; j++) if (logits[q.first+j]>max) {
            max=logits[q.first+j];
            best=j;
        }
        for (int j=0; j<n; j++) {
            p[j]=exp((double)logits[q.first+j]-max);
            sum+=p[j];
        }
        for (int j=0; j<n; j++) {
            p[j]/=sum;
            if (!isfinite(p[j])) clef_fail("nonfinite probability");
        }
        if (i) sb_addc(out,',');
        json_quote(out,q.id);
        sb_add(out,":{\"type\":");
        json_quote(out,types[q.type]);
        if (!q.type) {
            sb_fmt(out,",\"noul\":%.17g}",p[0]);
            continue;
        }
        double confidence;
        if (q.type==1) {
            sb_add(out,",\"choice\":");
            json_quote(out,r->o[q.first+best].id);
            confidence=n==1?1:fmax(0,(p[best]-1.0/n)/(1-1.0/n));
        } else {
            double score=0,dist=0,uniform=0;
            for (int j=0; j<n; j++) {
                score+=j*p[j];
                dist+=p[j]*abs(j-best);
                uniform+=fabs(j-(n-1)*.5)/n;
            } confidence=fmax(0,1-dist/uniform);
            sb_fmt(out,",\"score\":%.17g,\"legend\":{",score);
            const jv *criteria=json_get(json_get(json_get(req,"questions"),q.id),"criteria");
            for (int j=0; j<n; j++) {
                if (j) sb_addc(out,',');
                json_quote(out,r->o[q.first+j].id);
                sb_addc(out,':');
                json_dump(out,criteria->items[j]);
            }sb_addc(out,'}');
        }
        sb_fmt(out,",\"confidence\":%.17g,\"probabilities\":{",confidence);
        for (int j=0; j<n; j++) {
            if (j) sb_addc(out,',');
            json_quote(out,r->o[q.first+j].id);
            sb_fmt(out,":%.17g",p[j]);
        }sb_add(out,"}}");
    }
    sb_fmt(out,"},\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}",r->n);
}

static void init(const char *dir) {
    double t0=now();
    const char *gguf=getenv("LLB_GGUF"),*tmp=getenv("TMPDIR");
    if (!gguf || !*gguf) clef_fail("sys1 needs LLB_GGUF");
    if (!tmp || !*tmp) clef_fail("set TMPDIR to owned scratch");
    if (snprintf(hidden_path,sizeof hidden_path,"%s/sys1dual-%d.hidden",tmp,(int)getpid())>=(int)sizeof hidden_path) clef_fail("TMPDIR too long");
    if (q35_tok_load(&tok,dir)) clef_fail("sidecar requires tokenizer.bin (llb-tok extract)");
    clef_head_open(&head,gguf);
    out=stdout;
    fprintf(stderr,"sys1: FP32 clef head + tokenizer ready %.3f ms\n",(now()-t0)*1000);
}

// argv words for the request's ids + steps
static char *command(const int *ids,int n,int st) {
    sb c={0};
    for (int i=0; i<n; i++) sb_fmt(&c,"%d ",ids[i]);
    sb_fmt(&c,"--steps %d",st);
    return c.p;
}

// one request line -> 1 (cmd set) or 0 (refused, answered already)
static int parse(char *line,unsigned ctx,char **cmd,const char **hidden,unsigned *n) {
    const char *err=NULL;
    text=line;
    req=json_parse(text,&err);
    if (!req || req->type!=J_OBJ) return refuse(err?err:"request must be a JSON object"),0;
    jv *m=json_get(req,"mode");
    if (m && m->type==J_STR && !strcmp(m->str,"generate")) {
        jv *p=json_get(req,"prompt"),*k=json_get(req,"n");
        if (!p || p->type!=J_STR || !k || k->type!=J_NUM || k->num<1 || k->num>65536 || k->num!=floor(k->num))
            return refuse("generate needs prompt (string) and n (integer 1..65536)"),0;
        int cap_ids=CLEF_MAX_T,*ids=malloc(sizeof *ids*cap_ids);
        int c=q35_tok_encode(&tok,p->str,0,ids,cap_ids);
        if (c<1) return free(ids),refuse("prompt must tokenize to 1..16384 tokens"),0;
        if ((unsigned)c+(unsigned)k->num-1>ctx) return free(ids),refuse("prompt + n exceeds the KV capacity fixed at launch"),0;
        mode=2;
        steps=(int)k->num;
        prompt_n=c;
        memcpy(prompt_ids,ids,sizeof *ids*c);
        *n=(unsigned)c;
        *hidden=NULL;
        *cmd=command(ids,c,steps);
        free(ids);
        return 1;
    }
    // the clef-decide request checks (tools/clef3/decide.c main), as error lines
    jv *model=json_get(req,"model"),*qs=json_get(req,"questions");
    if (!model || model->type!=J_STR || !qs || qs->type!=J_OBJ) return refuse("model/questions required"),0;
    for (size_t i=0; i<qs->n; i++) {
        jv *ins=json_get(qs->items[i],"instructions"),*type=json_get(qs->items[i],"type"),*criteria=json_get(qs->items[i],"criteria");
        if (!ins || ins->type==J_NULL) return refuse("instructions must be provided"),0;
        if (type && type->type==J_STR && !strcmp(type->str,"score") && (!criteria || criteria->type!=J_ARR || criteria->n<2 || criteria->n>10)) return refuse("score requires 2..10 criteria"),0;
    }
    // ponytail: clef_encode still exits the process on the inputs it rejects (spans.c clef_fail); error lines need a non-exiting encode
    rec=calloc(1,sizeof *rec);
    clef_encode(&tok,req,rec,CLEF_MAX_T);
    if ((unsigned)rec->n>ctx) return refuse("request exceeds the KV capacity fixed at launch"),0;
    mode=1;
    steps=1;
    *n=(unsigned)rec->n;
    *hidden=hidden_path;
    *cmd=command(rec->ids,rec->n,1);
    return 1;
}

int dual_next(const char *dir,unsigned ctx,char **cmd,const char **hidden,unsigned *n) {
    if (!out) init(dir);
    for (;;) {
        char *line=NULL;
        size_t len=0;
        if (getline(&line,&len,stdin)<0) {
            free(line);
            unlink(hidden_path);
            return 0;
        }
        seq++;
        t_req=now();
        if (parse(line,ctx,cmd,hidden,n)) break;
        drop();
    }
    capp=NULL;
    capn=0;
    cap=open_memstream(&capp,&capn);
    if (!cap) clef_fail("open_memstream failed");
    stdout=cap;
    return 1;
}

void dual_done(void) {
    double t_ran=now();
    fflush(cap);
    stdout=out;
    // gen's prints: one id per line
    static int ids[65536];
    int k=0;
    for (char *s=capp,*e; k<65536; s=e) {
        long v=strtol(s,&e,10);
        if (e==s) break;
        ids[k++]=(int)v;
    }
    fclose(cap);
    free(capp);
    sb o={0};
    if (mode==2) {
        if (k!=steps) clef_fail("generate: id count differs from n");
        sb_add(&o,"{\"prompt_ids\":[");
        for (int i=0; i<prompt_n; i++) sb_fmt(&o,i?",%d":"%d",prompt_ids[i]);
        sb_add(&o,"],\"ids\":[");
        for (int i=0; i<k; i++) sb_fmt(&o,i?",%d":"%d",ids[i]);
        int need=q35_tok_decode(&tok,ids,k,NULL,0);
        char *t=malloc(need+1);
        q35_tok_decode(&tok,ids,k,t,need);
        t[need]=0;
        sb_add(&o,"],\"text\":");
        json_quote(&o,t);
        sb_addc(&o,'}');
        free(t);
        reply(o.p);
        fprintf(stderr,"{\"seq\":%d,\"mode\":\"generate\",\"prompt_tokens\":%d,\"n\":%d,\"forward_ms\":%.3f,\"total_ms\":%.3f}\n",
            seq,prompt_n,k,(t_ran-t_req)*1000,(now()-t_req)*1000);
    } else {
        float *hidden=clef_hidden_read(hidden_path,rec->n,head.d);
        float *logits=calloc(rec->no,4);
        double t_h=now();
        clef_head_run(&head,rec,hidden,logits);
        response(req,rec,logits,&o);
        reply(o.p);
        fprintf(stderr,"{\"seq\":%d,\"mode\":\"decide\",\"input_tokens\":%d,\"forward_ms\":%.3f,\"head_ms\":%.3f,\"total_ms\":%.3f}\n",
            seq,rec->n,(t_ran-t_req)*1000,(now()-t_h)*1000,(now()-t_req)*1000);
        free(hidden);
        free(logits);
    }
    free(o.p);
    drop();
}
