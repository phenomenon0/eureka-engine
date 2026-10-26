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
// KEV-HEAD-METAL: a GGUF without the clef head but with the Kev head (<arch>.decision.type "kev", cls.output.{weight,bias})
// serves decide through kevhead.c: one exact pass per question (the hook on), queued here like tools/kevserve/kev.c
// (dual_next hands out the next question without reading stdin), one response line when the last is answered.
// The clef probe and path are checked first and are unchanged; neither head = headless as before.
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "clef3/head.h"
#include "kevhead.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static Q35Tokenizer tok;
static Head head;
static int headless,stream_on,kev,kev_qi;
static char kev_why[256];
static sb kev_ans;
#define STOP_A 248046
#define STOP_B 248044
static FILE *out,*cap;
static char hidden_path[4096];
static char *text;
static jv *req;
static Record *rec;
static int mode,steps,seq,prompt_n,*prompt_ids,prompt_cap;
// (METAL-ANYSHAPE) a generate prompt of any length up to the KV capacity (not CLEF_MAX_T): prompt_ids grows to fit
static int prompt_room(size_t n) {
    if (n>(size_t)prompt_cap) {
        int *q=realloc(prompt_ids,sizeof *q*n);
        if (!q) return 0;
        prompt_ids=q,prompt_cap=(int)n;
    }
    return 1;
}
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
    if (kev) {
        kev_drop();
        free(kev_ans.p);
        kev_ans=(sb){0};
        kev_qi=0;
    }
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
    else if ((kev=kev_head_probe(gguf,kev_why,sizeof kev_why))) {
        headless=0;
        kev_head_open(gguf,&tok);
    } else if (*kev_why) fprintf(stderr,"sys1: partial kev head ignored: %s\n",kev_why);
    stream_on=getenv("LLB_SYS1_STREAM")&&*getenv("LLB_SYS1_STREAM")=='1';
    out=stdout;
    fprintf(stderr,"sys1: %s + tokenizer ready %.3f ms\n",headless?"headless (generate only)":kev?"kev head":"FP32 clef head",(now()-t0)*1000);
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

// one request line -> 1 (cmd set) or 0 (refused, answered already)
static int parse(char *line,unsigned ctx,char **cmd,const char **hidden,unsigned *n) {
    const char *err=NULL;
    text=line;
    req=json_parse(text,&err);
    if (!req || req->type!=J_OBJ) return refuse(err?err:"request must be a JSON object"),0;
    jv *m=json_get(req,"mode");
    if (m && m->type==J_STR && !strcmp(m->str,"generate")) {
        jv *p=json_get(req,"prompt"),*k=json_get(req,"n"),*pi=json_get(req,"prompt_ids");
        if (pi && pi->type==J_ARR && pi->n>=1 && pi->n<=ctx && k && k->type==J_NUM && k->num>=1 && k->num<=65536) {
            if (!prompt_room(pi->n)) return refuse("out of memory for prompt_ids"),0;
            for (size_t i=0; i<pi->n; i++) {
                if (pi->items[i]->type!=J_NUM || pi->items[i]->num<0 || pi->items[i]->num>=248320) return refuse("prompt_ids must be integers in [0,248320)"),0;
                prompt_ids[i]=(int)pi->items[i]->num;
            }
            prompt_n=(int)pi->n;
            if ((unsigned)prompt_n+(unsigned)k->num-1>ctx) return refuse("prompt + n exceeds the KV capacity fixed at launch"),0;
            mode=2;
            steps=(int)k->num;
            *n=(unsigned)prompt_n;
            *hidden=NULL;
            *cmd=command(prompt_ids,prompt_n,steps);
            return 1;
        }
        if (!p || p->type!=J_STR || !k || k->type!=J_NUM || k->num<1 || k->num>65536 || k->num!=floor(k->num))
            return refuse("generate needs prompt (string) and n (integer 1..65536)"),0;
        // (METAL-ANYSHAPE) a token is at least one byte of the prompt: strlen + 1 ids always suffice
        int cap_ids=(int)strlen(p->str)+1,*ids=malloc(sizeof *ids*(size_t)cap_ids);
        jv *sp=json_get(req,"special");
        int c=ids?q35_tok_encode(&tok,p->str,sp&&sp->type==J_BOOL&&sp->b,ids,cap_ids):-1;
        if (c<1) return free(ids),refuse("prompt must tokenize to 1 or more tokens"),0;
        if ((unsigned)c+(unsigned)k->num-1>ctx) return free(ids),refuse("prompt + n exceeds the KV capacity fixed at launch"),0;
        if (!prompt_room((size_t)c)) return free(ids),refuse("out of memory for the prompt"),0;
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
    if (headless&&*kev_why) { // a Kev key/tensor without a usable Kev head: say what is missing
        char m[400];
        snprintf(m,sizeof m,"decide needs the clef or Kev head; this model's Kev head is unusable (%s), launched headless (generate only)",kev_why);
        return refuse(m),0;
    }
    if (headless) return refuse("decide needs the clef head; this model was launched headless (generate only)"),0;
    if (kev) {
        const char *e=kev_request(req,ctx);
        if (e) return refuse(e),0;
        mode=3;
        steps=1;
        kev_qi=0;
        int kn;
        const int *ids=kev_ids(0,&kn);
        *n=(unsigned)kn;
        *hidden=hidden_path;
        *cmd=command(ids,kn,1);
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
    // (CLEF-PICK) clef_encode's rejections (e.g. "unknown question type") are error lines (400), not an engine exit: with
    // err set it returns the message instead of clef_fail's exit, rec left empty (drop() frees it)
    rec=calloc(1,sizeof *rec);
    const char *fail;
    if (clef_encode(&tok,req,rec,CLEF_MAX_T,&fail)) {
        char m[400];
        snprintf(m,sizeof m,"clef: %s",fail);
        return refuse(m),0;
    }
    if ((unsigned)rec->n>ctx) return refuse("request exceeds the KV capacity fixed at launch"),0;
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
    if (mode==3&&kev_qi<kev_count()) { // the next question of the queued Kev decide: no stdin read
        int kn;
        const int *ids=kev_ids(kev_qi,&kn);
        *n=(unsigned)kn;
        *hidden=hidden_path;
        *cmd=command(ids,kn,1);
    } else for (;;) {
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
        fprintf(stderr,"{\"seq\":%d,\"mode\":\"generate\",\"prompt_tokens\":%d,\"n\":%d,\"forward_ms\":%.3f,\"total_ms\":%.3f}\n",
            seq,prompt_n,k,(t_ran-t_req)*1000,(now()-t_req)*1000);
    } else if (mode==3) {
        kev_answer(kev_qi,hidden_path,&kev_ans);
        if (++kev_qi<kev_count()) return; // the next question's pass comes from dual_next
        const jv *model=json_get(req,"model");
        sb_add(&o,"{\"model\":");
        if (model) json_dump(&o,model); else sb_add(&o,"null");
        sb_add(&o,",\"answers\":{");
        sb_add(&o,kev_ans.p);
        sb_fmt(&o,"},\"usage\":{\"input_tokens\":%d,\"output_tokens\":0}}",kev_tokens());
        reply(o.p);
        fprintf(stderr,"{\"seq\":%d,\"mode\":\"decide\",\"head\":\"kev\",\"questions\":%d,\"input_tokens\":%d,\"total_ms\":%.3f,\"raw\":%s}\n",
            seq,kev_count(),kev_tokens(),(now()-t_req)*1000,kev_raw());
        mode=0;
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
