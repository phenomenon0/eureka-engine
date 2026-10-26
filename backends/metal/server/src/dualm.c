// dual.c: SYS1-DUAL request side (docs/sys1dual-registration.md). Linked into
// the engine; called by the Sys1.next / Sys1.done effects (sys1.c).
// stdin: one JSON request per line. stdout: one JSON response per line.
//   {"mode":"generate","prompt":TEXT,"n":N} -> {"prompt_ids":[...],"ids":[...],"text":"..."}
//   anything else = a clef-decide request  -> the clef-decide response
// The engine's own stdout prints (gen's ids) go to a memstream for the request;
// responses go to the real stdout. Timings: one JSON line per request on stderr.
// E-METAL-LEAD (dualm.c): the Mac request side with a chat provider's needs:
//  - headless: a GGUF without the 9B clef head (dec.*/decision.* tensors) serves generate only; decide is refused.
//  - "special":true parses control tokens (the chat template's <|im_start|>/<|im_end|>).
//  - LLB_SYS1_STREAM=1: every id the engine prints is answered at once as {"token":ID,"text":PIECE} (PIECE =
//    the id's bytes, an incomplete UTF-8 tail held back for the next event); the final line is the usual response.
//  - stop ids (<|im_end|>, <|endoftext|>) end a generate request; the stop id is not part of "ids"/"text" and
//    the final line carries "finish":"stop" (else "length").
// MAC-ENGINE-PENALTY: the server's "sampling" object -> the W1-SAMPLER spec (CUDA's dual.c sample_word map, base
//    LLB_SAMPLE or preset=greedy), set per request with llb_sampler_configure; the engine's fused-step head
//    then picks on the host (patches/apply_sampler.py). Every generated id must be a host pick (llb_sampler_picks).
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "clef3/head.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static Q35Tokenizer tok;
static Head head;
static int headless,stream_on;
#define STOP_A 248046
#define STOP_B 248044
static FILE *out,*cap;
static char hidden_path[4096];
static char *text;
static jv *req;
static Record *rec;
static int mode,steps,seq,prompt_n,prompt_ids[CLEF_MAX_T];
static double t_req;
// the engine's W1-SAMPLER (patches/apply_sampler.py): the spec, the prompt length (heads before it are prefill
// rows), the host pick count; the penalty window is the engine's own (each fused step's input token)
int llb_sampler_configure(const char *spec);
void llb_sampler_prompt(int n_prompt);
int llb_sampler_picks(void);
static int smp_req,smp_picks0;
static char smp_spec[1024];

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

// does the GGUF carry the clef head (dec.*/decision.* tensor names)? The tensor-name table sits in the first ~16 MiB.
static int gguf_has_head(const char *path) {
    FILE *f=fopen(path,"rb");
    if (!f) return 1;
    size_t cap=24u<<20,n;
    char *b=malloc(cap);
    n=fread(b,1,cap,f);
    fclose(f);
    int has=0;
    for (size_t i=0;i+18<n&&!has;i++)
        if (b[i]=='t'&&!memcmp(b+i,"token_types.weight",18)) has=1;
    free(b);
    return has;
}

static void init(const char *dir) {
    double t0=now();
    const char *gguf=getenv("LLB_GGUF"),*tmp=getenv("TMPDIR");
    if (!gguf || !*gguf) clef_fail("sys1 needs LLB_GGUF");
    if (!tmp || !*tmp) clef_fail("set TMPDIR to owned scratch");
    if (snprintf(hidden_path,sizeof hidden_path,"%s/sys1dual-%d.hidden",tmp,(int)getpid())>=(int)sizeof hidden_path) clef_fail("TMPDIR too long");
    if (q35_tok_load(&tok,dir)) clef_fail("sidecar requires tokenizer.bin (llb-tok extract)");
    headless=!gguf_has_head(gguf);
    if (!headless) clef_head_open(&head,gguf);
    stream_on=getenv("LLB_SYS1_STREAM")&&*getenv("LLB_SYS1_STREAM")=='1';
    out=stdout;
    fprintf(stderr,"sys1: %s + tokenizer ready %.3f ms\n",headless?"headless (generate only)":"FP32 clef head",(now()-t0)*1000);
}

// argv words for the request's ids + steps
static char *command(const int *ids,int n,int st) {
    sb c={0};
    for (int i=0; i<n; i++) sb_fmt(&c,"%d ",ids[i]);
    sb_fmt(&c,"--steps %d",st);
    if (mode==2) {
        sb_fmt(&c," --stop %d --stop %d",STOP_A,STOP_B);
        const char *fk=getenv("LLB_DUAL_FASTPF"); // the fast prefill tier (chunked, K=512 default) for generate; LLB_DUAL_FASTPF=0 = the exact per-token path
        int k=fk&&*fk?atoi(fk):512;
        if (k>0) sb_fmt(&c," --fastpf %d",k);
    }
    return c.p;
}

// CUDA's dual.c G15 map: OpenAI/llama.cpp sampling knobs -> host/llb_sampler.h spec keys; 0 = refused
static const char *smp_map[][2]={{"temperature","temp"},{"top_p","top_p"},{"top_k","top_k"},
    {"min_p","min_p"},{"seed","seed"},{"presence_penalty","presence"},{"frequency_penalty","frequency"},
    {"repeat_penalty","repeat_penalty"},{"repetition_penalty","repeat_penalty"},{NULL,NULL}};
static int sample_spec(void) {
    jv *smp=json_get(req,"sampling");
    const char *d=getenv("LLB_SAMPLE");
    sb c={0};
    sb_add(&c,d&&*d?d:"preset=greedy");
    int any=0;
    for (int i=0; smp&&smp->type==J_OBJ&&smp_map[i][0]; i++) {
        jv *v=json_get(smp,smp_map[i][0]);
        if (!v || v->type==J_NULL) continue;
        if (v->type!=J_NUM) return free(c.p),refuse("sampling values must be numbers"),0;
        if (!strcmp(smp_map[i][1],"seed") || !strcmp(smp_map[i][1],"top_k")) sb_fmt(&c,",%s=%lld",smp_map[i][1],(long long)v->num);
        else sb_fmt(&c,",%s=%.9g",smp_map[i][1],v->num);
        any=1;
    }
    snprintf(smp_spec,sizeof smp_spec,"%s",any||(d&&*d)?c.p:"");
    free(c.p);
    int live=llb_sampler_configure(smp_spec);  // -1 bad, 0 argmax-equivalent, 1 the host picks
    if (live<0) return refuse("bad sampling values"),0;
    smp_req=live;
    return 1;
}

// one request line -> 1 (cmd set) or 0 (refused, answered already)
static int parse(char *line,unsigned ctx,char **cmd,const char **hidden,unsigned *n) {
    const char *err=NULL;
    text=line;
    req=json_parse(text,&err);
    if (!req || req->type!=J_OBJ) return refuse(err?err:"request must be a JSON object"),0;
    jv *m=json_get(req,"mode");
    if (m && m->type==J_STR && !strcmp(m->str,"generate")) {
        jv *p=json_get(req,"prompt"),*k=json_get(req,"n"),*pi=json_get(req,"prompt_ids");
        if (pi && pi->type==J_ARR && pi->n>=1 && pi->n<=CLEF_MAX_T && k && k->type==J_NUM && k->num>=1 && k->num<=65536) {
            for (size_t i=0; i<pi->n; i++) {
                if (pi->items[i]->type!=J_NUM || pi->items[i]->num<0 || pi->items[i]->num>=248320) return refuse("prompt_ids must be integers in [0,248320)"),0;
                prompt_ids[i]=(int)pi->items[i]->num;
            }
            prompt_n=(int)pi->n;
            if ((unsigned)prompt_n+(unsigned)k->num-1>ctx) return refuse("prompt + n exceeds the KV capacity fixed at launch"),0;
            if (!sample_spec()) return 0;
            mode=2;
            steps=(int)k->num;
            *n=(unsigned)prompt_n;
            *hidden=NULL;
            *cmd=command(prompt_ids,prompt_n,steps);
            return 1;
        }
        if (!p || p->type!=J_STR || !k || k->type!=J_NUM || k->num<1 || k->num>65536 || k->num!=floor(k->num))
            return refuse("generate needs prompt (string) and n (integer 1..65536)"),0;
        int cap_ids=CLEF_MAX_T,*ids=malloc(sizeof *ids*cap_ids);
        jv *sp=json_get(req,"special");
        int c=q35_tok_encode(&tok,p->str,sp&&sp->type==J_BOOL&&sp->b,ids,cap_ids);
        if (c<1) return free(ids),refuse("prompt must tokenize to 1..16384 tokens"),0;
        if ((unsigned)c+(unsigned)k->num-1>ctx) return free(ids),refuse("prompt + n exceeds the KV capacity fixed at launch"),0;
        if (!sample_spec()) return free(ids),0;
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
    if (headless) return refuse("decide needs the clef head; this model was launched headless (generate only)"),0;
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
    llb_sampler_configure(NULL);
    smp_req=0;
    mode=1;
    steps=1;
    *n=(unsigned)rec->n;
    *hidden=hidden_path;
    *cmd=command(rec->ids,rec->n,1);
    return 1;
}

// the engine prints one id per line into `cap`; this writer collects them and (stream_on) answers each as an event
static int gids[65536],gk;
static char gline[32],gpend[64];
static int gl,gpn;
static int utf8_complete(const char *b,int n) { // longest prefix of b[0..n) that does not end inside a UTF-8 sequence
    int i=n;
    while (i>0&&(b[i-1]&0xC0)==0x80&&n-i<3) i--;
    if (i>0&&(b[i-1]&0xC0)==0xC0) {
        unsigned char c=b[i-1];
        int need=c>=0xF0?4:c>=0xE0?3:2;
        if (n-(i-1)<need) return i-1;
    }
    return n;
}
static void gid(int id) {
    if (gk<65536) gids[gk++]=id;
    if (!stream_on||id==STOP_A||id==STOP_B) return;
    char pc[512];
    int pn=q35_tok_piece(&tok,id,pc,sizeof pc);
    if (pn<0) pn=0;
    if (pn>(int)sizeof pc) pn=sizeof pc;
    char all[sizeof gpend+sizeof pc];
    memcpy(all,gpend,gpn);
    memcpy(all+gpn,pc,pn);
    int n=gpn+pn,ok=utf8_complete(all,n);
    gpn=n-ok;
    memcpy(gpend,all+ok,gpn);
    char *tx=malloc(ok+1);
    memcpy(tx,all,ok);
    tx[ok]=0;
    sb e={0};
    sb_fmt(&e,"{\"token\":%d,\"text\":",id);
    json_quote(&e,tx);
    sb_addc(&e,'}');
    reply(e.p);
    free(e.p);
    free(tx);
}
static int gwrite(void *ck,const char *p,int n) {
    (void)ck;
    for (int i=0;i<n;i++) {
        if (p[i]=='\n') {
            gline[gl]=0;
            if (gl) gid(atoi(gline));
            gl=0;
        } else if (gl<(int)sizeof gline-1) gline[gl++]=p[i];
    }
    return n;
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
    gk=gl=gpn=0;
    if (mode==2) {
        llb_sampler_prompt(prompt_n);
        smp_picks0=llb_sampler_picks();
    }
    cap=funopen(NULL,NULL,gwrite,NULL,NULL);
    if (!cap) clef_fail("funopen failed");
    setvbuf(cap,NULL,_IOLBF,0);
    stdout=cap;
    return 1;
}

void dual_done(void) {
    double t_ran=now();
    fflush(cap);
    stdout=out;
    fclose(cap);
    int *ids=gids,k=gk;
    int stopped=k&&(ids[k-1]==STOP_A||ids[k-1]==STOP_B);
    if (stopped) k--;
    sb o={0};
    if (mode==2) {
        if (k>steps||(!stopped&&k!=steps)) clef_fail("generate: id count differs from n");
        if (smp_req&&llb_sampler_picks()-smp_picks0!=gk) {
            fprintf(stderr,"sampler: %d host picks for %d ids\n",llb_sampler_picks()-smp_picks0,gk);
            clef_fail("generate: an id was not a host pick (sampler hook missed a head path)");
        }
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
        sb_fmt(&o,",\"finish\":\"%s\"}",stopped?"stop":"length");
        free(t);
        reply(o.p);
        fprintf(stderr,"{\"seq\":%d,\"mode\":\"generate\",\"prompt_tokens\":%d,\"n\":%d,\"sampler\":\"%s\",\"picks\":%d,\"forward_ms\":%.3f,\"total_ms\":%.3f}\n",
            seq,prompt_n,k,smp_req?smp_spec:"argmax",llb_sampler_picks()-smp_picks0,(t_ran-t_req)*1000,(now()-t_req)*1000);
    } else {
        float *hidden=clef_hidden_read(hidden_path,rec->n);
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
