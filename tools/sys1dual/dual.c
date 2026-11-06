// dual.c: SYS1-DUAL request side (docs/records/sys1dual-registration.md). Linked into
// the engine; called by the Sys1.next / Sys1.done effects (sys1.c).
// stdin: one JSON request per line. stdout: one JSON response per line.
//   {"mode":"generate","prompt":TEXT,"n":N} -> {"prompt_ids":[...],"ids":[...],"text":"..."}
//     G1, optional: "json_schema":TEXT or "grammar":GBNF (constrained from the first
//     generated position on, host/llb_grammar.h; needs LLB_TOKENIZER), "sample":SPEC
//     (W1-SAMPLER key=val list, sent to the engine as its --sample word)
//   anything else = a clef-decide request  -> the clef-decide response
// The engine's own stdout prints (gen's ids) go to a memstream for the request;
// responses go to the real stdout. Timings: one JSON line per request on stderr.
// W1-METRICS: every request is timed through tools/metrics (TTFT, per-token
// clocks from the id stream, cache = retained/continuation tokens); /metrics is
// served when LLB_METRICS_PORT is set; "timings":true on a request (or
// LLB_TIMINGS=1) appends a llama.cpp-style "timings" object to its response.
#define _GNU_SOURCE /* W1-SERVER: fopencookie */
#define _POSIX_C_SOURCE 200809L
#include "../clef3/head.h"
#include "../metrics/metrics.h"
int llb_grammar_configure(const char* kind, const char* text, unsigned at); /* host/llb_host.c (G1) */
#include "../glue/glue_out.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static Q35Tokenizer tok;
static Head head;
/* S3 twoheads: optional SECOND head ("head2") loaded from LLB_HEAD2_GGUF (any
 * GGUF holding a clef-joint-head tensor set; loader is path-parameterized).
 * When set, every decide request runs BOTH heads on the SAME engine pass's
 * hidden rows and the response carries "head2":{...} next to the head-0
 * fields. Env-driven only; with the var unset the behavior is byte-identical
 * to S2 (head table/registry + per-request "head" key are the follow-up). */
static Head head2;
static int head2_on;
/* BATCH-NATIVE-3: head registry. LLB_HEADS="name=gguf[,name=gguf...]" (up to
 * HREG_MAX) opens more clef heads at init; a decide's "head":"<name>" runs only
 * that head on the request's hidden rows (tail marker "head":"<name>", as S4).
 * The trunk pass never sees the head, so in a native batch each row picks its
 * own head: multi-head batch: N rows, per-row head, exact per row. Unset = no
 * change (head0/head2 keep the S3/S4 behavior). */
#define HREG_MAX 6
static Head hreg[HREG_MAX];
static char hreg_name[HREG_MAX][24];
static int hreg_n;
/* ENGINE-HEAD27: LLB_DISCIPLINE=strict (llb-serve --discipline strict) = a decide with no "head" key runs the
 * registry head named "strict" (as if it said "head":"strict"); normal/unset = head0. */
static int disc_head=-1;
static FILE *out,*cap;
static char *capp,hidden_path[4096];
static size_t capn;
static char *text;
static jv *req;
static Record *rec;
static int mode,steps,seq,prompt_n,*prompt_ids,prompt_cap;
/* tools/payload-anyshape.patch: a generate prompt is bounded by the engine's KV capacity
 * (ctx, prompt + steps fixed at launch), not CLEF_MAX_T (16384, the decide head's budget);
 * the id buffers grow to the request */
static int *ids_fit(int *p,int *cap,int n) {
    if (n<=*cap) return p;
    int *q=realloc(p,sizeof *q*(size_t)n);
    if (!q) clef_fail("out of memory for the request ids");
    *cap=n;
    return q;
}
static double t_req;

/* S2 twoheads: retained session. The last request that carried "session":S and
 * its exact response line are kept; a byte-identical repeat of it under the
 * same S is served from the record with NO engine pass (no re-prefill; the
 * engine's retained state is not touched). A request of S that differs from
 * the retained one is refused (never truncated). A request with a new/absent
 * session resets the record (state resets at pos 0 as before). */
static char rt_session[256];
static int  rt_valid,rt_mode,rt_gen_n,rt_req_n,*rt_req_ids,rt_req_cap;
static char *rt_response;
static char cur_session[256];

/* S5 twoheads W5 P1: continuation. A generate request whose token ids strictly
 * extend the retained record's ids is a continuation, not a differing request:
 * the engine receives ONLY the appended ids plus `--pos K` (K = retained token
 * count = all tokens the engine has processed for the session, prompt+gen), so
 * the retained state continues with no re-prefill. cont_pos resets to 0 (fresh
 * semantics) for every request; decide stays refused when differing (its hidden
 * rows cover the full request; not plumbed for partial runs). */
static int cont_pos;
/* W1-KVPFX: prefix cache (host llb_pfx_*, env LLB_PREFIX_CACHE=1). cached_n = the
 * request's ids the engine did not run (a continuation's K or a prefix hit's K). */
int llb_pfx_on(void);
unsigned llb_pfx_begin(const int *ids,unsigned n,unsigned start,int reuse);
static int cached_n;
/* G20: S1 banks (host llb_bank_*, env LLB_BANKS=B). "bank":j on a request runs it on the
 * engine's sequence state j (KV + GDN state + prefix cache); absent = bank 0. */
int llb_banks(void);
int llb_bank_select(int j);
/* BATCH-NATIVE (host llb_bx_*, env LLB_BATCH=N, 2..8): a {"decide_batch":[R1,..,Rk]} line (k <= N)
 * is k decide request lines. Each Ri is queued and served as if it had arrived on its own line
 * (one reply line each, in order). Before that, the ids of every Ri that encodes go to the host's
 * plan, and each decide names its ids (llb_bx_select): the host then runs the k sequences in one
 * forward a step and serves each request's steps from it. Unset: the line is a plain request. */
int llb_bx_max(void);
unsigned llb_heap_bump(void); /* BATCH-NATIVE-2: LLB_MEMPROBE */
void llb_bx_plan(const int *const *ids,const unsigned *n,unsigned k);
void llb_bx_select(const int *ids,unsigned n);
int llb_bx_next_done(void);
/* BATCH-NATIVE-2 stage 3b (LLB_BATCH_OOO=1, serve --native-cont): a queued line is taken when its
 * item is done, not in line order, and its reply follows a {"bx_item":k} line (k = its place in the
 * decide_batch); refused (unplanned) lines go first */
static int bx_ooo=-1,*bxq_plan,*bxq_len;
static char *bxq_took;
static char **bxq;
static int bxq_n,bxq_h;

/* W1-SERVER: token streaming for the HTTP front-end (tools/sys1dual/serve.c).
 * LLB_SYS1_STREAM=1: during a generate request every id the engine prints is
 * also answered at once as an event line {"token":ID,"text":PIECE} on stdout
 * (PIECE = the id's decoded bytes, an incomplete UTF-8 tail held back for the
 * next event; the final line carries the full text as before). The final
 * response line is unchanged, so the env unset = the old protocol exactly.
 * "special":true on a generate request tokenizes the prompt with the special
 * (control) tokens parsed (chat templates); absent = the old encode. */
static int stream_on;
static FILE *scap;                 /* the cookie stream gen prints into */
static char sline[32],spend[64];   /* partial id line / held-back UTF-8 bytes */
static int slen,spn;

/* W1-SERVER: token streaming for the HTTP front-end (tools/sys1dual/serve.c).
 * LLB_SYS1_STREAM=1: during a generate request every id the engine prints is
 * also answered at once as an event line {"token":ID,"text":PIECE} on stdout
 * (PIECE = the id's decoded bytes, an incomplete UTF-8 tail held back for the
 * next event; the final line carries the full text as before). The final
 * response line is unchanged, so the env unset = the old protocol exactly.
 * "special":true on a generate request tokenizes the prompt with the special
 * (control) tokens parsed (chat templates); absent = the old encode. */
static int stream_on;
static FILE *scap;                 /* the cookie stream gen prints into */
static char sline[32],spend[64];   /* partial id line / held-back UTF-8 bytes */
static int slen,spn;

/* S4 twoheads: strict single-session option (LLB_SESSION_STRICT=1). While a
 * retained record is live: a *new* session id and a sessionless request are
 * REFUSED (the record is never replaced/cleared); an identical repeat is
 * still served. Unset / "0" = S3 replace behavior (regression-compatible). */
static int strict_on;
/* S4 twoheads: per-request head selection (decide only, registry = head0 +
 * resident head2). "head":"head0" runs only head0; "head":"head2" runs only
 * head2 (requires LLB_HEAD2_GGUF); the selected head's response line carries
 * "head":"<name>". Key absent = S3 behavior exactly (head0; plus the spliced
 * "head2" object when the second head is resident). */
static int sel_head; /* 0=absent, 1=head0-only, 2=head2-only, 3+k=registry head k (BATCH-NATIVE-3) */

/* G4 F1 (fused trunk, connector rows): a generate request may carry
 *   "image":{"rows":PATH,"n":N}  N rows of 4096 F32 (glue_rows.py's rows.f32), spliced
 *       by the engine at the prompt's one run of exactly N <|image_pad|> ids, at text
 *       positions (host llb_vis_request / llb_img_step); the prompt is encoded with the
 *       special tokens parsed so the pads are ids, not text;
 *   "readout":[ID,...]  1..8 vocab ids: the last prompt row (final-normed) . the
 *       output.weight rows -> "readout":{"ids","logits","p"} in the response (p = softmax
 *       over the listed ids; two ids = the glue p_yes). Same hidden hook as decide.
 * Both are sessionless. Every request resets the host's image, so none persists.
 * LLB_SYS1_NOHEAD=1: no clef head is opened (a trunk GGUF without the clef joint head,
 * e.g. Qwen3.5-9B-Base); decide requests are refused; readout rows come from
 * glue_out.c. Unset (and no image/readout key) = the old behavior exactly. */
void llb_vis_request(const char *path, unsigned n);
/* G24 F2/F4: the rows' stamp. "image":{"rows":"X.f32","n"} needs glue_rows.py's X.json beside
 * it, with trunk_tensor_table_sha256 equal to the loaded GGUF's (content, not a label: the
 * sha over every tensor's bytes, cached under LLB_GLUE_CACHE/tabsha), rows_sha256 equal to
 * the file's, K = n and d = 4096. Unstamped (pre-F2) rows are refused. */
static double now(void);
int cuMemGetInfo_v2(size_t *free_b,size_t *total_b); /* libcuda (the engine links it) */
static const char *trunk_gguf;
static char trunk_tab[65];
static int trunk_tab_ok;
static const char *stamp_check(const char *rows, unsigned n) {
    static char msg[512];
    size_t L=strlen(rows);
    char sp[4096],fs[65];
    if (L<5 || L>4000 || strcmp(rows+L-4,".f32")) return "image rows must be a .f32 file with its .json stamp beside it";
    snprintf(sp,sizeof sp,"%.*s.json",(int)(L-4),rows);
    char *txt=json_read_file(sp);
    const char *err=NULL;
    jv *st=txt?json_parse(txt,&err):NULL;
    free(txt);
    if (!st || st->type!=J_OBJ) return json_free(st),"image rows need their glue_rows.py stamp (rows .json beside the .f32)";
    jv *ts=json_get(st,"trunk_tensor_table_sha256"),*rs=json_get(st,"rows_sha256"),*k=json_get(st,"K"),*d=json_get(st,"d");
    const char *why=NULL;
    if (!ts || ts->type!=J_STR || strlen(ts->str)!=64) why="image rows are not bound to a trunk (stamp lacks trunk_tensor_table_sha256)";
    else if (!k || k->type!=J_NUM || k->num!=n || !d || d->type!=J_NUM || d->num!=4096) why="image rows stamp: K/d differ from the request (n) / 4096";
    else if (!rs || rs->type!=J_STR || glue_file_sha(rows,fs) || strcmp(fs,rs->str)) why="image rows differ from their stamp (rows_sha256)";
    if (!why && !trunk_tab_ok) {
        double t0=now();
        int hit=0;
        if (glue_table_sha_cached(trunk_gguf,getenv("LLB_GLUE_CACHE"),trunk_tab,&hit)) why="cannot hash the loaded GGUF's tensor table";
        else trunk_tab_ok=1,fprintf(stderr,"sys1: trunk tensor_table_sha256 %s (%s, %.0f ms)\n",trunk_tab,hit?"cache hit":"computed",(now()-t0)*1e3);
    }
    if (!why && strcmp(ts->str,trunk_tab))
        snprintf(msg,sizeof msg,"image rows were made for another trunk (stamp tensor_table_sha256 %.12s..., loaded GGUF %.12s...)",ts->str,trunk_tab),why=msg;
    json_free(st);
    return why;
}
/* FOLD-CONTINUE MIMG: "images":[{"rows":"X.f32","nx":NX,"ny":NY},...] (1..16 images, in prompt order) = the Qwen3.5
 * tower's rows (vision/build/cudavis-cli IMG MMPROJ --out X.f32), IMRoPE positions (host llb_vis_request_list): the
 * prompt holds one run of exactly nx*ny <|image_pad|> per image, in order. Tower rows carry no glue stamp (no trunk
 * table in them); they are bound instead by tools/vision/tower_stamp.py's X.json beside X.f32:
 *   {"kind":"llb-tower-rows-v1","mmproj_sha256":S,"nx","ny","d":4096,"rows_sha256"}
 * with S equal to the engine's LLB_VIS_MMPROJ_SHA256 (the operator names the mmproj that belongs to this trunk; unset =
 * "images" refused) and rows_sha256 equal to the file's. The G4 "image" form above is unchanged. */
void llb_vis_request_list(const char *spec);
static int trunk_d; /* defined below (ENGINE-HEAD27); the stamp check compares the rows' d to it */
static const char *tower_stamp_check(const char *rows, unsigned nx, unsigned ny) {
    static char msg[512];
    size_t L=strlen(rows);
    char sp[4096],fs[65];
    const char *want=getenv("LLB_VIS_MMPROJ_SHA256");
    if (!want || strlen(want)!=64) return "images (tower rows) need the engine's LLB_VIS_MMPROJ_SHA256 (the mmproj bound to this trunk)";
    if (L<5 || L>4000 || strcmp(rows+L-4,".f32") || strchr(rows,';')) return "images rows must be a .f32 path (no ';') with its tower_stamp.py .json beside it";
    snprintf(sp,sizeof sp,"%.*s.json",(int)(L-4),rows);
    char *txt=json_read_file(sp);
    const char *err=NULL;
    jv *st=txt?json_parse(txt,&err):NULL;
    free(txt);
    if (!st || st->type!=J_OBJ) return json_free(st),"images rows need their tower_stamp.py stamp (rows .json beside the .f32)";
    jv *kd=json_get(st,"kind"),*ms=json_get(st,"mmproj_sha256"),*rs=json_get(st,"rows_sha256"),*sx=json_get(st,"nx"),*sy=json_get(st,"ny"),*d=json_get(st,"d");
    const char *why=NULL;
    if (!kd || kd->type!=J_STR || strcmp(kd->str,"llb-tower-rows-v1")) why="images rows stamp is not an llb-tower-rows-v1 stamp";
    else if (!ms || ms->type!=J_STR || strlen(ms->str)!=64) why="images rows are not bound to an mmproj (stamp lacks mmproj_sha256)";
    else if (!sx || sx->type!=J_NUM || sx->num!=nx || !sy || sy->type!=J_NUM || sy->num!=ny || !d || d->type!=J_NUM || d->num!=trunk_d) {
        if (d && d->type==J_NUM && d->num!=trunk_d && sx && sx->type==J_NUM && sx->num==nx && sy && sy->type==J_NUM && sy->num==ny)
            snprintf(msg,sizeof msg,"images rows are %.0f wide (their tower's mmproj); this trunk is %d wide",d->num,trunk_d),why=msg;
        else why="images rows stamp: nx/ny/d differ from the request / the trunk width";
    }
    else if (!rs || rs->type!=J_STR || glue_file_sha(rows,fs) || strcmp(fs,rs->str)) why="images rows differ from their stamp (rows_sha256)";
    else if (strcmp(ms->str,want))
        snprintf(msg,sizeof msg,"images rows were made with another mmproj (stamp %.12s..., engine LLB_VIS_MMPROJ_SHA256 %.12s...)",ms->str,want),why=msg;
    json_free(st);
    return why;
}
static int nohead,glue_ok,ro_n,ro_ids[8];
/* ENGINE-HEAD27: the trunk's hidden width (LLB_GGUF <arch>.embedding_length); every head must read it. */
static int trunk_d;
static int rt_head;  /* sel_head recorded with the retained record (identity) */

/* W1-METRICS: the request's timer, its mode label, whether it is already
 * folded into the aggregates, and whether its response carries "timings". */
static llb_req mreq;
static const char *mmode="invalid";
static int mdone,want_timings,timings_all;

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

/* S4 twoheads: append `,"<name>":<value>` as the last member of the response
 * object (value = raw JSON; o owns the result). Used for the head marker and
 * for the S3-compatible head2 splice (byte-identical to S3's inline splice). */
static void tail_field(sb *o,const char *name,const char *value) {
    size_t n=strlen(o->p);
    if (!n || o->p[n-1]!='}') clef_fail("response tail splice layout");
    char *tmp=strdup(o->p);
    tmp[n-1]=0;
    sb f={0};
    sb_add(&f,tmp);
    free(tmp);
    sb_add(&f,",\"");
    sb_add(&f,name);
    sb_add(&f,"\":");
    sb_add(&f,value);
    sb_addc(&f,'}');
    free(o->p);
    o->p=f.p;
    o->n=f.n;
    o->cap=f.cap;
}

/* W1-METRICS: close the request's timer (once) with its outcome */
static void mend(const char *outcome) {
    if (mdone) return;
    llb_req_end(&mreq,mmode,outcome);
    mdone=1;
}

/* W1-METRICS: reply, with a "timings" tail when requested. The retained record
 * (sess_commit) keeps the untimed line, so a repeat is byte-identical to it. */
static void reply_timed(const char *s) {
    if (!want_timings) {
        reply(s);
        return;
    }
    char t[1024];
    llb_req_timings_json(&mreq,t,sizeof t);
    sb o={0};
    sb_add(&o,s);
    tail_field(&o,"timings",t);
    reply(o.p);
    free(o.p);
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

// A head reads the trunk's hidden rows, so its width must be the trunk's (a 27B head on a 9B trunk is refused).
static void head_width(const Head *h,const char *path) {
    if (h->d==trunk_d) return;
    static char why[8448];
    snprintf(why,sizeof why,"head %s reads %d-wide rows (output.weight %d x %d) but the trunk %s is %d wide",path,h->d,h->vocab,h->d,trunk_gguf,trunk_d);
    clef_fail(why);
}

static void init(const char *dir) {
    double t0=now();
    const char *gguf=getenv("LLB_GGUF"),*tmp=getenv("TMPDIR");
    if (!gguf || !*gguf) clef_fail("sys1 needs LLB_GGUF");
    trunk_gguf=gguf;
    if (!tmp || !*tmp) clef_fail("set TMPDIR to owned scratch");
    if (snprintf(hidden_path,sizeof hidden_path,"%s/sys1dual-%d.hidden",tmp,(int)getpid())>=(int)sizeof hidden_path) clef_fail("TMPDIR too long");
    if (q35_tok_load(&tok,dir)) clef_fail("sidecar requires tokenizer.bin (llb-tok extract)");
    const char *nh=getenv("LLB_SYS1_NOHEAD");
    nohead=nh && *nh && strcmp(nh,"0");
    if (getenv("LLB_VIS_EMBD")) clef_fail("sys1: the image is per request (\"image\"), unset LLB_VIS_EMBD");
    /* glue_out (the NOHEAD readout rows) reads the width from the GGUF: output.weight, or token_embd.weight on a
       tied-embedding trunk (SENSES-MATRIX), 256k cols <= 8192. Any other width serves generate requests and refuses
       readout (tools/payload-anyshape.patch; was: refused at start) */
    trunk_d=clef_gguf_width(gguf);
    glue_ok=nohead && trunk_d%256==0 && trunk_d<=8192;
    if (glue_ok) glue_out_open(gguf);
    if (glue_ok && glue_out_width()!=trunk_d) clef_fail("sys1: NOHEAD readout rows are not the trunk width (embedding_length)");
    if (nohead) fprintf(stderr,"sys1: NOHEAD (no clef head; decide refused; readout %s, d %d)\n",glue_ok?"from output.weight":"off: glue_out reads rows of 256k cols <= 8192 only",trunk_d);
    else clef_head_open(&head,gguf);
    if (!nohead) head_width(&head,gguf);
    const char *gguf2=getenv("LLB_HEAD2_GGUF");
    if (gguf2 && *gguf2) {
        clef_head_open(&head2,gguf2);
        head_width(&head2,gguf2);
        head2_on=1;
        fprintf(stderr,"sys1: resident head2 (second head) ready %.3f ms\n",(now()-t0)*1000);
    }
    const char *hs=getenv("LLB_HEADS");
    if (hs && *hs) {
        if (nohead) clef_fail("LLB_HEADS needs the clef head (unset LLB_SYS1_NOHEAD)");
        char *spec=strdup(hs),*save=NULL;
        for (char *e=strtok_r(spec,",",&save); e; e=strtok_r(NULL,",",&save)) {
            char *eq=strchr(e,'=');
            if (!eq || eq==e || eq-e>=(long)sizeof hreg_name[0] || !eq[1]) clef_fail("LLB_HEADS: entries are name=gguf");
            *eq=0;
            if (strspn(e,"abcdefghijklmnopqrstuvwxyz0123456789_-")!=strlen(e)) clef_fail("LLB_HEADS: names are [a-z0-9_-]");
            if (!strcmp(e,"head0") || !strcmp(e,"head2")) clef_fail("LLB_HEADS: head0/head2 are reserved names");
            for (int j=0; j<hreg_n; j++) if (!strcmp(e,hreg_name[j])) clef_fail("LLB_HEADS: duplicate name");
            if (hreg_n==HREG_MAX) clef_fail("LLB_HEADS: too many heads");
            double th=now();
            clef_head_open(&hreg[hreg_n],eq+1);
            head_width(&hreg[hreg_n],eq+1);
            strcpy(hreg_name[hreg_n++],e);
            fprintf(stderr,"sys1: registry head %s = %s ready %.3f ms\n",e,eq+1,(now()-th)*1000);
        }
        free(spec);
    }
    const char *disc=getenv("LLB_DISCIPLINE");
    if (disc && *disc && strcmp(disc,"normal")) {
        if (strcmp(disc,"strict")) clef_fail("LLB_DISCIPLINE is strict or normal");
        for (int j=0; j<hreg_n; j++) if (!strcmp(hreg_name[j],"strict")) disc_head=j;
        if (disc_head<0) clef_fail("--discipline strict needs --heads strict=GGUF");
        fprintf(stderr,"sys1: discipline strict (a decide with no \"head\" runs registry head strict)\n");
    }
    const char *strict=getenv("LLB_SESSION_STRICT");
    strict_on=strict && *strict && strcmp(strict,"0");
    if (strict_on) fprintf(stderr,"sys1: STRICT single-session mode (a new session is refused while one is retained)\n");
    const char *st=getenv("LLB_SYS1_STREAM");
    stream_on=st && *st && strcmp(st,"0");
    if (stream_on) fprintf(stderr,"sys1: STREAM mode (generate ids answered as {\"token\",\"text\"} events)\n");
    out=stdout;
    const char *tm=getenv("LLB_TIMINGS");
    timings_all=tm && *tm && strcmp(tm,"0");
    const char *mp=getenv("LLB_METRICS_PORT"),*mh=getenv("LLB_METRICS_HOST");
    if (mp && *mp) {
        if (llb_metrics_listen(mh,atoi(mp))) clef_fail("LLB_METRICS_PORT: cannot listen");
        fprintf(stderr,"sys1: metrics on http://%s:%s/metrics\n",mh&&*mh?mh:"127.0.0.1",mp);
    }
    fprintf(stderr,"sys1: FP32 clef head + tokenizer ready %.3f ms\n",(now()-t0)*1000);
}

// one request line -> 1 (cmd set) or 0 (refused/retained, answered already)
// S2: session handler. id = the request's session (NULL = sessionless request),
// m = mode (1 decide, 2 generate), ids/n the request's token ids, gen_n the
// generate count (0 decide).
// S4: strict_on refuses a new session / sessionless request while a record is
// live; sel_head is part of the retained record's identity.
// Returns 1 = let the engine run it; 0 = already answered (retained serve or refuse).
static int sess_check(const char *id,int m,const int *ids,int n,int gen_n) {
    if (!id) { /* sessionless request */
        if (rt_valid && strict_on) {
            refuse("strict: a session is retained; a sessionless request is refused (unset LLB_SESSION_STRICT to replace)");
            return 0;
        }
        return 1; /* S1/S2 legacy: runs; sess_commit clears the record */
    }
    if (rt_valid && !strcmp(id,rt_session)) {
        int same=rt_mode==m && rt_req_n==n && rt_head==sel_head;
        for (int i=0; same && i<n; i++) if (ids[i]!=rt_req_ids[i]) same=0;
        if (same && m==2) same=rt_gen_n==gen_n;
        if (same) {
            double t=now();
            llb_req_start(&mreq,n,n);
            mend("retained");
            reply_timed(rt_response);
            fprintf(stderr,"{\"seq\":%d,\"mode\":\"%s\",\"retained\":true,\"input_tokens\":%d,\"forward_ms\":0.0,\"total_ms\":%.3f}\n",
                seq,m==2?"generate":"decide",n,(now()-t)*1000);
            return 0;
        }
        /* S5 twoheads W5 P1: strict extension of a retained GENERATE record
         * = continuation (run the appended ids at --pos K; no re-prefill). */
        if (m==2 && rt_mode==2 && rt_head==sel_head && n>rt_req_n
            && !memcmp(ids,rt_req_ids,sizeof *ids*rt_req_n)) {
            /* K = pass_pos+1 (host guard): a generate record's LAST sampled id
             * is printed but never stepped into the KV, so the retained token
             * count minus one is the true engine position (decide records
             * process all their tokens: no -1). S5 V2 measured. */
            cont_pos=rt_req_n+(rt_gen_n?rt_gen_n-1:0);
            rt_req_ids=ids_fit(rt_req_ids,&rt_req_cap,n);
            memcpy(rt_req_ids,ids,sizeof *ids*n);
            rt_req_n=n;
            rt_gen_n=gen_n;
            strcpy(cur_session,id);
            fprintf(stderr,"sys1: continuation session %s pos %d tokens %d\n",id,cont_pos,n);
            return 1;
        }
        refuse("session retained: a differing request is refused (identical repeats are served; a new session resets)");
        return 0;
    }
    if (rt_valid && strict_on) {
        refuse("strict: a session is retained; a new session is refused (identical repeats are served; unset LLB_SESSION_STRICT to replace)");
        return 0;
    }
    memset(rt_session,0,sizeof rt_session);
    rt_req_ids=ids_fit(rt_req_ids,&rt_req_cap,n);
    memcpy(rt_req_ids,ids,sizeof *ids*n);
    rt_req_n=n;
    rt_mode=m;
    rt_gen_n=gen_n;
    strcpy(cur_session,id);
    return 1;
}

// S2: after a request ran, keep its record and the exact response line
static void sess_commit(const char *response) {
    if (cur_session[0]) {
        free(rt_response);
        rt_response=strdup(response);
        strcpy(rt_session,cur_session);
        rt_head=sel_head;
        rt_valid=1;
    } else {
        rt_valid=0;
        rt_session[0]=0;
    }
}

// argv words for the request's ids + steps + start pos (S5: the --pos word is
// on EVERY request; K=0 fresh keeps the old path byte-identical, K>0 is a
// continuation start pos consumed by cfg --pos (P1), sys1.c latch as belt)
static char *command(const int *ids,int n,int st,int pos) {
    sb c={0};
    for (int i=0; i<n; i++) sb_fmt(&c,"%d ",ids[i]);
    sb_fmt(&c,"--steps %d --pos %d",st,pos);
    return c.p;
}

/* G15: OpenAI/llama.cpp sampling knobs -> host/llb_sampler.h spec keys. Stop strings stay with
 * the front (llb_stop_*, W1-SAMPLER); logprobs are not returned over this protocol yet. */
static const char *smp_map[][2]={{"temperature","temp"},{"top_p","top_p"},{"top_k","top_k"},
    {"min_p","min_p"},{"seed","seed"},{"presence_penalty","presence"},{"frequency_penalty","frequency"},
    {"repeat_penalty","repeat_penalty"},{"repetition_penalty","repeat_penalty"},{NULL,NULL}};
static int sample_word(jv *smp,char **cmd) {
    sb c={0};
    const char *d=getenv("LLB_SAMPLE");
    sb_fmt(&c,"%s --sample %s",*cmd,d&&*d?d:"preset=greedy");
    int any=0;
    for (int i=0; smp_map[i][0]; i++) {
        jv *v=json_get(smp,smp_map[i][0]);
        if (!v || v->type==J_NULL) continue;
        if (v->type!=J_NUM) return free(c.p),refuse("sampling values must be numbers"),0;
        if (!strcmp(smp_map[i][1],"seed") || !strcmp(smp_map[i][1],"top_k")) sb_fmt(&c,",%s=%lld",smp_map[i][1],(long long)v->num);
        else sb_fmt(&c,",%s=%.9g",smp_map[i][1],v->num);
        any=1;
    }
    if (!any) return free(c.p),1;  /* only stop/logprobs/...: the default sampler, old command line */
    free(*cmd);
    *cmd=c.p;
    return 1;
}

// one request line -> 1 (cmd set) or 0 (refused, answered already)
static int parse(char *line,unsigned ctx,char **cmd,const char **hidden,unsigned *n) {
    const char *err=NULL;
    text=line;
    req=json_parse(text,&err);
    cont_pos=0;
    llb_grammar_configure(NULL,NULL,0);   /* G1: off unless this generate request sets it */
    ro_n=0;
    llb_vis_request(NULL,0);
    if (!req || req->type!=J_OBJ) return refuse(err?err:"request must be a JSON object"),0;
    jv *tj=json_get(req,"timings");
    want_timings=timings_all || (tj && tj->type==J_BOOL && tj->b);
    jv *bk=json_get(req,"bank");
    if (bk && (bk->type!=J_NUM || bk->num!=floor(bk->num) || bk->num<0 || bk->num>=llb_banks()))
        return refuse("bank must be an integer 0..LLB_BANKS-1"),0;
    /* the retained session record is one per process: not per bank yet */
    if (llb_banks()>1 && json_get(req,"session"))
        return refuse("session is not supported with LLB_BANKS (one retained record per process); the prefix cache reuses each bank's state"),0;
    llb_bank_select(bk?(int)bk->num:0);
    jv *m=json_get(req,"mode");
    mmode=m && m->type==J_STR && !strcmp(m->str,"generate")?"generate":"decide";
    if (m && m->type==J_STR && !strcmp(m->str,"generate")) {
        if (llb_bx_max()>1) llb_bx_select(NULL,0); /* BATCH-NATIVE */
        jv *p=json_get(req,"prompt"),*k=json_get(req,"n");
        if (!p || p->type!=J_STR || !k || k->type!=J_NUM || k->num<1 || k->num>65536 || k->num!=floor(k->num))
            return refuse("generate needs prompt (string) and n (integer 1..65536)"),0;
        /* G4 F1: image / readout validation (refused before any engine work) */
        jv *img=json_get(req,"image"),*ro=json_get(req,"readout");
        jv *ir=img?json_get(img,"rows"):NULL,*in=img?json_get(img,"n"):NULL;
        if (img && (img->type!=J_OBJ || !ir || ir->type!=J_STR || !in || in->type!=J_NUM || in->num<1 || in->num>4096 || in->num!=floor(in->num)))
            return refuse("image needs rows (path string) and n (integer 1..4096)"),0;
        if (img && trunk_d!=4096) return refuse("image rows are 4096 wide (the 9B tower); this trunk is not"),0;
        if (ro) {
            if (ro->type!=J_ARR || ro->n<1 || ro->n>8) return refuse("readout must be an array of 1..8 token ids"),0;
            if (nohead && !glue_ok) return refuse("readout needs a trunk of 256k cols <= 8192 under LLB_SYS1_NOHEAD (glue_out)"),0;
            for (size_t i=0; i<ro->n; i++) {
                jv *v=ro->items[i];
                if (v->type!=J_NUM || v->num<0 || v->num>=248320 || v->num!=floor(v->num)) return refuse("readout ids must be integers 0..248319"),0;
            }
        }
        if ((img || ro) && json_get(req,"session")) return refuse("image/readout requests are sessionless"),0;
        /* MIMG: "images" (tower rows, IMRoPE), validated before any engine work */
        jv *ims=json_get(req,"images");
        unsigned im_k=0,im_nx[16],im_ny[16];
        const char *im_spec=NULL;
        if (ims) {
            if (img) return refuse("image and images are exclusive (image = G4 connector rows, images = tower rows)"),0;
            if (ims->type!=J_ARR || ims->n<1 || ims->n>16) return refuse("images must be an array of 1..16 {rows,nx,ny}"),0;
            if (json_get(req,"session")) return refuse("images requests are sessionless"),0;
            const char *pf=getenv("LLB_PREFILL");
            if (getenv("LLB_SPEC") || getenv("LLB_MTP") || (pf && !strcmp(pf,"fast")))
                return refuse("images run on the exact tier only (this engine has LLB_SPEC, LLB_MTP or LLB_PREFILL=fast)"),0;
            sb w={0};
            unsigned tot=0;
            for (size_t i=0; i<ims->n; i++) {
                jv *e=ims->items[i],*er=e->type==J_OBJ?json_get(e,"rows"):NULL,*ex=e->type==J_OBJ?json_get(e,"nx"):NULL,*ey=e->type==J_OBJ?json_get(e,"ny"):NULL;
                if (!er || er->type!=J_STR || !ex || ex->type!=J_NUM || ex->num<1 || ex->num>4096 || ex->num!=floor(ex->num)
                    || !ey || ey->type!=J_NUM || ey->num<1 || ey->num>4096 || ey->num!=floor(ey->num) || ex->num*ey->num>16384)
                    return free(w.p),refuse("each images entry needs rows (path string), nx and ny (integers, nx*ny <= 16384)"),0;
                im_nx[i]=(unsigned)ex->num, im_ny[i]=(unsigned)ey->num, tot+=im_nx[i]*im_ny[i];
                FILE *f=fopen(er->str,"rb");
                long sz=-1;
                if (f && !fseek(f,0,SEEK_END)) sz=ftell(f);
                if (f) fclose(f);
                if (sz!=(long)im_nx[i]*im_ny[i]*trunk_d*4) return free(w.p),refuse("images rows must be a file of nx*ny x d F32 (d = the trunk width)"),0;
                const char *why=tower_stamp_check(er->str,im_nx[i],im_ny[i]);
                if (why) return free(w.p),refuse(why),0;
                sb_fmt(&w,"%s%s:%u:%u",i?";":"",er->str,im_nx[i],im_ny[i]);
            }
            if (tot>16384) return free(w.p),refuse("images total more than 16384 rows"),0;
            static char im_buf[16384];
            if (!w.p || strlen(w.p)>=sizeof im_buf) return free(w.p),refuse("images rows paths are too long"),0;
            strcpy(im_buf,w.p), free(w.p);
            im_k=(unsigned)ims->n, im_spec=im_buf;
        }
        if (img) {
            FILE *f=fopen(ir->str,"rb");
            long sz=-1;
            if (f && !fseek(f,0,SEEK_END)) sz=ftell(f);
            if (f) fclose(f);
            if (sz!=(long)in->num*4096*4) return refuse("image rows must be a file of n x 4096 F32"),0;
            const char *why=stamp_check(ir->str,(unsigned)in->num);
            if (why) return refuse(why),0;
        }
        int cap_ids=ctx>0 && ctx<(1u<<30)?(int)ctx:CLEF_MAX_T,*ids=malloc(sizeof *ids*(size_t)cap_ids);
        if (!ids) return refuse("out of memory for the prompt ids"),0;
        jv *spc=json_get(req,"special");
        int c=q35_tok_encode(&tok,p->str,img || ims || (spc && spc->type==J_BOOL && spc->b),ids,cap_ids);
        /* C1A-LAND: q35_tok_encode returns the full count (and copies nothing) when it exceeds cap_ids; c>cap_ids
           used to fall through, overrun ids/prompt_ids and send garbage ids -> "sys1: bad request command", rc 2 */
        if (c<1 || c>cap_ids) return free(ids),refuse("prompt must tokenize to 1 .. the KV capacity fixed at launch (--steps)"),0;
        if ((unsigned)c+(unsigned)k->num-1>ctx) return free(ids),refuse("prompt + n exceeds the KV capacity fixed at launch"),0;
        if (json_get(req,"head")) return free(ids),refuse("head applies to decide requests only"),0;
        if (img) {
            int first=-1,np=0;
            for (int i=0; i<c; i++) if (ids[i]==248056) { if (first<0) first=i; np++; }
            if (np!=(int)in->num || ids[first+np-1]!=248056 || (np>1 && ids[first+1]!=248056))
                return free(ids),refuse("the prompt must hold one run of exactly n <|image_pad|>"),0;
            for (int i=first; i<first+np; i++) if (ids[i]!=248056) return free(ids),refuse("the prompt must hold one run of exactly n <|image_pad|>"),0;
        }
        if (ims) { /* MIMG: exactly one run of nx*ny pads per image, in order */
            unsigned r=0;
            for (int i=0; i<c; ) {
                if (ids[i]!=248056) { i++; continue; }
                int j=i;
                while (j<c && ids[j]==248056) j++;
                if (r>=im_k || (unsigned)(j-i)!=im_nx[r]*im_ny[r])
                    return free(ids),refuse("the prompt must hold one run of exactly nx*ny <|image_pad|> per images entry, in order"),0;
                r++, i=j;
            }
            if (r!=im_k) return free(ids),refuse("the prompt must hold one run of exactly nx*ny <|image_pad|> per images entry, in order"),0;
        }
        sel_head=0;
        cur_session[0]=0;
        const char *sid=NULL;
        jv *se=json_get(req,"session");
        if (se) {
            if (se->type!=J_STR || strlen(se->str)>=sizeof cur_session) return free(ids),refuse("session must be a short string"),0;
            sid=se->str;
        }
        jv *gs=json_get(req,"json_schema"),*gg=json_get(req,"grammar"),*sm=json_get(req,"sample");
        if ((gs && gs->type!=J_STR) || (gg && gg->type!=J_STR) || (gs && gg))
            return free(ids),refuse("json_schema / grammar: at most one, as a string"),0;
        if (sm && (sm->type!=J_STR || !*sm->str || strpbrk(sm->str," \t\r\n") || strlen(sm->str)>1000))
            return free(ids),refuse("sample must be a key=val,... string without spaces"),0;
        /* G1: the first generated position is c (positions 0..c-1 are the prompt, a
           continuation included); no field = the grammar off for this request */
        if (llb_grammar_configure(gs?"json_schema":gg?"gbnf":NULL,gs?gs->str:gg?gg->str:NULL,(unsigned)c)!=0)
            return free(ids),refuse("bad json_schema / grammar (see stderr)"),0;
        if (!sess_check(sid,2,ids,c,(int)k->num)) return free(ids),0;
        mode=2;
        steps=(int)k->num;
        prompt_n=c;
        prompt_ids=ids_fit(prompt_ids,&prompt_cap,c);
        memcpy(prompt_ids,ids,sizeof *ids*c);
        *n=(unsigned)c;
        *hidden=NULL;
        if (img) llb_vis_request(ir->str,(unsigned)in->num);
        else if (ims) llb_vis_request_list(im_spec);
        if (ro) {
            ro_n=(int)ro->n;
            for (int i=0; i<ro_n; i++) ro_ids[i]=(int)ro->items[i]->num;
            *hidden=hidden_path;
        }
        /* S5: a continuation runs only the appended ids at --pos cont_pos.
         * W1-KVPFX: else a fresh request sharing a chunk-aligned prefix with the
         * resident ids runs only its suffix at --pos K (host restores the state). */
        cached_n=cont_pos;
        if (llb_pfx_on()) {
            cached_n=(int)llb_pfx_begin(ids,(unsigned)c,(unsigned)cont_pos,cont_pos==0 && !img && !ims && !ro); /* merge-train S9: no reuse for G4 image/readout requests (rows at placeholder ids; readout reads the last prompt row) */
            if (!cont_pos && cached_n) fprintf(stderr,"sys1: prefix hit pos %d tokens %d\n",cached_n,c-cached_n);
        }
        *cmd=command(ids+cached_n,c-cached_n,steps,cached_n);
        if (sm) {
            sb w={0};
            sb_fmt(&w,"%s --sample %s",*cmd,sm->str);
            free(*cmd);
            *cmd=w.p;
        }
        else { /* G15: the server's "sampling" object -> the W1-SAMPLER --sample word (G16's "sample" string wins) */
            jv *smp=json_get(req,"sampling");
            if (smp && smp->type==J_OBJ && !sample_word(smp,cmd)) return free(ids),0;
        }
        free(ids);
        return 1;
    }
    if (nohead) return refuse("decide needs the clef head (LLB_SYS1_NOHEAD is set)"),0;
    // the clef-decide request checks (tools/clef3/decide.c main), as error lines
    jv *model=json_get(req,"model"),*qs=json_get(req,"questions");
    if (!model || model->type!=J_STR || !qs || qs->type!=J_OBJ) return refuse("model/questions required"),0;
    for (size_t i=0; i<qs->n; i++) {
        jv *ins=json_get(qs->items[i],"instructions"),*type=json_get(qs->items[i],"type"),*criteria=json_get(qs->items[i],"criteria");
        if (!ins || ins->type==J_NULL) return refuse("instructions must be provided"),0;
        if (type && type->type==J_STR && !strcmp(type->str,"score") && (!criteria || criteria->type!=J_ARR || criteria->n<2 || criteria->n>10)) return refuse("score requires 2..10 criteria"),0;
    }
    // S4 twoheads: per-request head selection (registry: head0 always, head2
    // when LLB_HEAD2_GGUF loaded it). Validated before the encode; absent key
    // keeps the S3 behavior untouched.
    jv *hd=json_get(req,"head");
    int hk=-1; /* BATCH-NATIVE-3: registry index of the named head */
    if (hd && hd->type==J_STR) for (int j=0; j<hreg_n; j++) if (!strcmp(hd->str,hreg_name[j])) hk=j;
    if (!hd) hk=disc_head;
    if (hd && hk<0) {
        if (hd->type!=J_STR || (strcmp(hd->str,"head0") && strcmp(hd->str,"head2")))
            return refuse(hreg_n?"head must be \"head0\", \"head2\" or a name in LLB_HEADS":"head must be \"head0\" or \"head2\""),0;
        if (!strcmp(hd->str,"head2") && !head2_on) return refuse("head2 is not resident (set LLB_HEAD2_GGUF)"),0;
    }
    sel_head=hk>=0?3+hk:hd?(hd->str[4]=='2'?2:1):0;
    // SERVE-BATCH: the inputs clef_encode rejects (spans.c clef_fail) are refused as error lines, the engine stays up
    rec=calloc(1,sizeof *rec);
    jmp_buf trap;
    if (setjmp(trap)) {
        clef_trap=NULL;
        return refuse(clef_trap_msg),0; /* drop() frees the partial record */
    }
    clef_trap=&trap;
    clef_encode(&tok,req,rec,CLEF_MAX_T);
    clef_trap=NULL;
    if ((unsigned)rec->n>ctx) return refuse("request exceeds the KV capacity fixed at launch"),0;
    if (llb_bx_max()>1) llb_bx_select(rec->ids,(unsigned)rec->n); /* BATCH-NATIVE */
    cur_session[0]=0;
    const char *sid=NULL;
    jv *se=json_get(req,"session");
    if (se) {
        if (se->type!=J_STR || strlen(se->str)>=sizeof cur_session) return refuse("session must be a short string"),0;
        sid=se->str;
    }
    if (!sess_check(sid,1,rec->ids,rec->n,0)) return 0;
    mode=1;
    steps=1;
    *n=(unsigned)rec->n;
    *hidden=hidden_path;
    if (llb_pfx_on()) llb_pfx_begin(rec->ids,(unsigned)rec->n,0,0); /* W1-KVPFX: decide runs from 0 */
    *cmd=command(rec->ids,rec->n,1,0);
    return 1;
}

/* W1-SERVER: the length of the longest prefix of b that ends on a complete
 * UTF-8 sequence (a lead byte whose continuation bytes are all present). */
static int utf8_whole(const char *b,int n) {
    int i=n;
    while (i>0 && i>n-4 && ((unsigned char)b[i-1]&0xC0)==0x80) i--;
    if (i==0) return 0;
    unsigned char c=(unsigned char)b[i-1];
    int need=c<0x80?1:(c&0xE0)==0xC0?2:(c&0xF0)==0xE0?3:(c&0xF8)==0xF0?4:1;
    return n-(i-1)>=need?n:i-1;
}

static void stream_id(int id) {
    char piece[4096];
    int g=q35_tok_decode(&tok,&id,1,piece,sizeof piece);
    if (g>(int)sizeof piece) g=sizeof piece;
    char buf[sizeof spend+sizeof piece+1];
    memcpy(buf,spend,spn);
    memcpy(buf+spn,piece,g);
    int n=spn+g,w=utf8_whole(buf,n);
    if (n-w>(int)sizeof spend) w=n; /* never valid UTF-8 anyway: let it through */
    spn=n-w;
    memcpy(spend,buf+w,spn);
    buf[w]=0;
    sb e={0};
    sb_fmt(&e,"{\"token\":%d,\"text\":",id);
    json_quote(&e,buf);
    sb_addc(&e,'}');
    reply(e.p);
    free(e.p);
}

/* the cookie write: keep the memstream capture byte-identical, answer each
 * complete id line as an event */
static ssize_t stream_write(void *c,const char *b,size_t n) {
    (void)c;
    if (fwrite(b,1,n,cap)!=n) return -1;
    for (size_t i=0; i<n; i++) {
        if (b[i]=='\n') {
            sline[slen]=0;
            char *e;
            long v=strtol(sline,&e,10);
            if (e!=sline) stream_id((int)v);
            slen=0;
        } else if (slen<(int)sizeof sline-1) sline[slen++]=b[i];
    }
    return (ssize_t)n;
}

/* BATCH-NATIVE: 1 = line was a decide_batch (its items queued, the plan handed over) */
static int bx_split(const char *line,unsigned ctx) {
    if (llb_bx_max()<2 || strncmp(line,"{\"decide_batch\":",16)) return 0;
    const char *err=NULL;
    jv *v=json_parse(line,&err),*a=v?json_get(v,"decide_batch"):NULL;
    if (!a || a->type!=J_ARR || a->n<1 || a->n>256) return json_free(v),0; /* BATCH-NATIVE-2: up to 256 items, N rows */
    bxq=realloc(bxq,sizeof *bxq*a->n);
    bxq_plan=realloc(bxq_plan,sizeof *bxq_plan*a->n);
    bxq_took=realloc(bxq_took,a->n);
    bxq_len=realloc(bxq_len,sizeof *bxq_len*a->n);
    if (!bxq || !bxq_plan || !bxq_took || !bxq_len) clef_fail("out of memory");
    const int *ids[256];
    unsigned nn[256],k=0;
    Record *rs[256];
    for (size_t i=0; i<a->n; i++) {
        sb o={0};
        json_dump(&o,a->items[i]);
        bxq[i]=o.p,bxq_plan[i]=-1,bxq_took[i]=0;
        jv *it=a->items[i],*m=it->type==J_OBJ?json_get(it,"mode"):NULL;
        if (it->type!=J_OBJ || (m && m->type==J_STR && !strcmp(m->str,"generate")) || nohead) continue;
        Record *r=calloc(1,sizeof *r);
        jmp_buf trap;
        if (setjmp(trap)) { clef_trap=NULL; clef_record_free(r); free(r); continue; }
        clef_trap=&trap;
        clef_encode(&tok,it,r,CLEF_MAX_T);
        clef_trap=NULL;
        if ((unsigned)r->n>ctx || r->n<2) { clef_record_free(r); free(r); continue; }
        bxq_plan[i]=(int)k,bxq_len[i]=r->n;
        rs[k]=r,ids[k]=r->ids,nn[k]=(unsigned)r->n,k++;
    }
    bxq_n=(int)a->n,bxq_h=0;
    llb_bx_plan(ids,nn,k);
    for (unsigned j=0; j<k; j++) clef_record_free(rs[j]),free(rs[j]);
    json_free(v);
    return 1;
}

int dual_next(const char *dir,unsigned ctx,char **cmd,const char **hidden,unsigned *n) {
    if (!out) init(dir);
    for (;;) {
        char *line=NULL;
        size_t len=0;
        if (bxq_h<bxq_n && bx_ooo<0) bx_ooo=getenv("LLB_BATCH_OOO")&&atoi(getenv("LLB_BATCH_OOO"))==1;
        if (bxq_h<bxq_n && bx_ooo) { /* stage 3b: the next line is the next done item's */
            int at=-1;
            for (int i=0; at<0 && i<bxq_n; i++) if (!bxq_took[i] && bxq_plan[i]<0) at=i;
            if (at<0) {
                int j=llb_bx_next_done();
                for (int i=0; at<0 && j>=0 && i<bxq_n; i++) if (!bxq_took[i] && bxq_plan[i]==j) at=i;
                /* the plan's first request: the shortest of the items that start together (rows 0..N-1) */
                for (int i=0; j<0 && i<bxq_n; i++)
                    if (!bxq_took[i] && bxq_plan[i]>=0 && bxq_plan[i]<llb_bx_max() && (at<0 || bxq_len[i]<bxq_len[at])) at=i;
            }
            for (int i=0; at<0 && i<bxq_n; i++) if (!bxq_took[i]) at=i;
            bxq_took[at]=1,bxq_h++;
            line=bxq[at];
            fprintf(out,"{\"bx_item\":%d}\n",at);
            fflush(out);
        } else if (bxq_h<bxq_n) line=bxq[bxq_h++]; /* BATCH-NATIVE: a queued decide_batch item */
        else if (getline(&line,&len,stdin)<0) {
            free(line);
            unlink(hidden_path);
            return 0;
        } else if (bx_split(line,ctx)) {
            free(line);
            continue;
        } else if (llb_bx_max()>1) llb_bx_plan(NULL,NULL,0); /* a plain line: the last batch's plan is spent */
        seq++;
        t_req=now();
        llb_req_begin(&mreq);
        mmode="invalid";
        mdone=0;
        want_timings=0;
        int go=parse(line,ctx,cmd,hidden,n);
        llb_metrics_queue_depth(llb_stdin_queued(stdin));
        if (go) break;
        mend("refused"); /* no-op after a retained serve (already ended) */
        drop();
    }
    if (mode==2) llb_req_start(&mreq,prompt_n,cached_n); /* integ: KVPFX hits count as cache (== cont_pos when off) */
    else llb_req_start(&mreq,rec->n,0);
    /* W1-METRICS: the id stream stamps one token per printed line (generate) */
    cap=llb_token_stream(mode==2?&mreq:NULL,&capp,&capn);
    if (!cap) clef_fail("token stream failed");
    stdout=cap;
    if (stream_on && mode==2) {
        slen=spn=0;
        scap=fopencookie(NULL,"w",(cookie_io_functions_t){.write=stream_write});
        if (!scap) clef_fail("fopencookie failed");
        setvbuf(scap,NULL,_IONBF,0);
        stdout=scap;
    }
    return 1;
}

void dual_done(void) {
    double t_ran=now();
    if (scap) {
        fflush(scap);
        fclose(scap);
        scap=NULL;
    }
    fflush(cap);
    stdout=out;
    // gen's prints: one id per line (up to the request's steps, any count since
    // tools/payload-anyshape.patch; was a 65536-id buffer)
    static int *ids,ids_cap;
    ids=ids_fit(ids,&ids_cap,(steps>1?steps:1)+1);
    int k=0;
    for (char *s=capp,*e; k<ids_cap; s=e) {
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
        if (llb_pfx_on()) sb_fmt(&o,",\"cached_tokens\":%d",cached_n); /* W1-KVPFX */
        if (ro_n) { /* G4 F1 readout: last prompt row . output.weight rows (float64 dot, as readout.py) */
            float *h=clef_hidden_read(hidden_path,prompt_n,trunk_d),row[8192];
            const float *last=h+(size_t)(prompt_n-1)*trunk_d;
            double lg[8],mx=-INFINITY,z=0;
            for (int j=0; j<ro_n; j++) {
                if (nohead) glue_out_row(ro_ids[j],row); else clef_output_row(&head,ro_ids[j],row);
                lg[j]=0;
                for (int i=0; i<trunk_d; i++) lg[j]+=(double)row[i]*last[i];
                if (lg[j]>mx) mx=lg[j];
            }
            for (int j=0; j<ro_n; j++) z+=exp(lg[j]-mx);
            sb_add(&o,",\"readout\":{\"ids\":[");
            for (int j=0; j<ro_n; j++) sb_fmt(&o,j?",%d":"%d",ro_ids[j]);
            sb_add(&o,"],\"logits\":[");
            for (int j=0; j<ro_n; j++) sb_fmt(&o,j?",%.9g":"%.9g",lg[j]);
            sb_add(&o,"],\"p\":[");
            for (int j=0; j<ro_n; j++) sb_fmt(&o,j?",%.9g":"%.9g",exp(lg[j]-mx)/z);
            sb_add(&o,"]}");
            free(h);
            unlink(hidden_path);
        }
        sb_addc(&o,'}');
        free(t);
        mend("ok");
        reply_timed(o.p);
        if (llb_pfx_on())
            fprintf(stderr,"{\"seq\":%d,\"mode\":\"generate\",\"prompt_tokens\":%d,\"cached_tokens\":%d,\"n\":%d,\"forward_ms\":%.3f,\"total_ms\":%.3f}\n",
                seq,prompt_n,cached_n,k,(t_ran-t_req)*1000,(now()-t_req)*1000);
        else
        fprintf(stderr,"{\"seq\":%d,\"mode\":\"generate\",\"prompt_tokens\":%d,\"n\":%d,\"forward_ms\":%.3f,\"total_ms\":%.3f}\n",
            seq,prompt_n,k,(t_ran-t_req)*1000,(now()-t_req)*1000);
    } else {
        float *hidden=clef_hidden_read(hidden_path,rec->n,trunk_d);
        float *logits=calloc(rec->no,4);
        double t_head=0,t_head2=0;
        sb o2={0};
        if (sel_head>=3) {
            /* BATCH-NATIVE-3: a registry head only (timed as head2_ms: "the selected non-head0 head") */
            double th=now();
            clef_head_run(&hreg[sel_head-3],rec,hidden,logits);
            t_head2=(now()-th)*1000;
            response(req,rec,logits,&o);
            sb nm={0};
            sb_fmt(&nm,"\"%s\"",hreg_name[sel_head-3]);
            tail_field(&o,"head",nm.p);
            free(nm.p);
        } else if (sel_head==2) {
            /* S4 twoheads: head2-only selection — head0 does NOT run; the
             * response line carries the "head":"head2" tail marker. */
            double th=now();
            clef_head_run(&head2,rec,hidden,logits);
            t_head2=(now()-th)*1000;
            response(req,rec,logits,&o);
            tail_field(&o,"head","\"head2\"");
        } else {
            double t_h=now();
            clef_head_run(&head,rec,hidden,logits);
            t_head=(now()-t_h)*1000;
            response(req,rec,logits,&o);
            if (sel_head==1) tail_field(&o,"head","\"head0\"");
            if (head2_on && sel_head==0) {
                // S3: second head, same rows, same pass. Output spliced below.
                float *logits2=calloc(rec->no,4);
                double th=now();
                clef_head_run(&head2,rec,hidden,logits2);
                t_head2=(now()-th)*1000;
                response(req,rec,logits2,&o2);
                free(logits2);
                if (!o2.p) clef_fail("head2 splice layout");
                tail_field(&o,"head2",o2.p); /* o owns the spliced line (reply + sess_commit) */
            }
        }
        mend("ok");
        reply_timed(o.p);
        fprintf(stderr,"{\"seq\":%d,\"mode\":\"decide\",\"input_tokens\":%d,\"forward_ms\":%.3f,\"head_ms\":%.3f,\"head2_ms\":%.3f,\"total_ms\":%.3f}\n",
            seq,rec->n,(t_ran-t_req)*1000,t_head,t_head2,(now()-t_req)*1000);
        free(o2.p);
        free(hidden);
        free(logits);
    }
    sess_commit(o.p);
    free(o.p);
    drop();
    /* BATCH-NATIVE-2: LLB_MEMPROBE=1 prints the device's used MiB after every request
       (the per-decide leak receipt); unset, nothing runs */
    if (getenv("LLB_MEMPROBE")) {
        size_t fr=0,tot=0;
        if (cuMemGetInfo_v2(&fr,&tot)==0)
            fprintf(stderr,"{\"memprobe\":%d,\"dev_used_mib\":%.3f,\"heap_bump_kib\":%u}\n",seq,(double)(tot-fr)/1048576.0,llb_heap_bump());
    }
}
