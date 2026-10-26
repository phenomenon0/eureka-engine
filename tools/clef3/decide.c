#define _POSIX_C_SOURCE 200809L
#include "head.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static char hidden_path[4096],engine_log[4096],run_dir[4096];
static void cleanup(void) {
    if (*hidden_path) unlink(hidden_path);
    if (*engine_log) unlink(engine_log);
    if (*run_dir) rmdir(run_dir);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}

static void backbone(const char *gguf,const char *side,const Record *r) {
    const char *tmp=getenv("TMPDIR");
    if (!tmp || !*tmp) clef_fail("set TMPDIR to owned scratch");
    if (snprintf(run_dir,sizeof run_dir,"%s/clef-XXXXXX",tmp)>=(int)sizeof run_dir || !mkdtemp(run_dir)) clef_fail("temporary directory failed");
    atexit(cleanup);
    snprintf(hidden_path,sizeof hidden_path,"%s/hidden",run_dir);
    snprintf(engine_log,sizeof engine_log,"%s/engine.log",run_dir);
    char engine[4096];
    const char *override=getenv("CLEF_ENGINE");
    if (override) {
        if (snprintf(engine,sizeof engine,"%s",override)>=(int)sizeof engine) clef_fail("engine path too long");
    } else {
        ssize_t n=readlink("/proc/self/exe",engine,sizeof engine-1);
        if (n<0) clef_fail("resolve engine path");
        engine[n]=0;
        char *slash=strrchr(engine,'/');
        if (!slash) clef_fail("resolve executable directory");
        if ((size_t)(slash+1-engine)+sizeof "llb-engine">sizeof engine) clef_fail("engine path too long");
        strcpy(slash+1,"llb-engine");
    }
    char **args=calloc(r->n+9,sizeof *args);
    args[0]=engine;
    args[1]="--threads";
    args[2]="16";
    args[3]="--";
    args[4]=(char*)side;
    for (int i=0; i<r->n; i++) {
        args[i+5]=malloc(16);
        snprintf(args[i+5],16,"%d",r->ids[i]);
    } args[r->n+5]="--steps";
    args[r->n+6]="1";
    pid_t child=fork();
    if (child<0) clef_fail("fork failed");
    if (!child) {
        const char *clear[]={
            "LLB_PREFILL","LLB_SPEC","LLB_MTP","LLB_DUMP","LLB_DUMP_POS","LLB_DUMP_HIDDEN_POS","LLB_Q4_DP4A","LLB_Q4GLUE","LLB_WARP_GRID"
        };
        for (size_t i=0; i<sizeof clear/sizeof *clear; i++) unsetenv(clear[i]);
        char count[24];
        snprintf(count,sizeof count,"%d",r->n);
        setenv("LLB_GGUF",gguf,1);
        setenv("LLB_DUMP_HIDDEN",hidden_path,1);
        setenv("LLB_DUMP_HIDDEN_TOKENS",count,1);
        int out=open("/dev/null",O_WRONLY),err=open(engine_log,O_WRONLY|O_CREAT|O_TRUNC,0600);
        if (out<0 || err<0 || dup2(out,STDOUT_FILENO)<0 || dup2(err,STDERR_FILENO)<0) _exit(126);
        close(out);
        close(err);
        execv(engine,args);
        _exit(127);
    }
    int status;
    while (waitpid(child,&status,0)<0) if (errno!=EINTR) clef_fail("wait engine failed");
    for (int i=0; i<r->n; i++) free(args[i+5]);
    free(args);
    if (!WIFEXITED(status) || WEXITSTATUS(status)) {
        char *log=json_read_file(engine_log);
        if (log) {
            fputs(log,stderr);
            free(log);
        }
        clef_fail("backbone failed");
    }
}

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

int main(int argc,char **argv) {
    const char *gguf=NULL,*side=NULL,*request=NULL;
    for (int i=1; i<argc; i+=2) {
        if (i+1==argc) clef_fail("missing argument");
        if (!strcmp(argv[i],"--gguf")) gguf=argv[i+1];
        else if (!strcmp(argv[i],"--sidecar")) side=argv[i+1];
        else if (!strcmp(argv[i],"--request")) request=argv[i+1];
        else clef_fail("unknown argument");
    }
    if (!gguf || !side || !request) clef_fail("usage: clef-decide --gguf GGUF --sidecar DIR --request REQUEST.json");
    double start=now();
    char *text=json_read_file(request);
    if (!text) clef_fail("request read failed");
    const char *err=NULL;
    jv *req=json_parse(text,&err);
    if (!req) clef_fail(err);
    jv *model=json_get(req,"model"),*qs=json_get(req,"questions");
    if (!model || model->type!=J_STR || !qs || qs->type!=J_OBJ) clef_fail("model/questions required");
    for (size_t i=0; i<qs->n; i++) {
        jv *ins=json_get(qs->items[i],"instructions"),*type=json_get(qs->items[i],"type"),*criteria=json_get(qs->items[i],"criteria");
        if (!ins || ins->type==J_NULL) clef_fail("instructions must be provided");
        if (type && type->type==J_STR && !strcmp(type->str,"score") && (!criteria || criteria->type!=J_ARR || criteria->n<2 || criteria->n>10)) clef_fail("score requires 2..10 criteria");
    } Q35Tokenizer tok;
    if (q35_tok_load(&tok,side)) clef_fail("sidecar requires tokenizer.bin (llb-tok extract)");
    Record *r=calloc(1,sizeof *r);
    clef_encode(&tok,req,r,CLEF_MAX_T);
    q35_tok_free(&tok);
    double encoded=now();
    backbone(gguf,side,r);
    double ran=now();
    float *hidden=clef_hidden_read(hidden_path,r->n,clef_gguf_width(gguf));
    Head h;
    clef_head_open(&h,gguf);
    double loaded=now();
    float *logits=calloc(r->no,4);
    clef_head_run(&h,r,hidden,logits);
    double decided=now();
    sb out={
        0
    };
    response(req,r,logits,&out);
    puts(out.p);
    fprintf(stderr,"{\"render_ms\":%.3f,\"backbone_ms\":%.3f,\"head_load_ms\":%.3f,\"head_ms\":%.3f,\"total_ms\":%.3f}\n",(encoded-start)*1000,(ran-encoded)*1000,(loaded-ran)*1000,(decided-loaded)*1000,(decided-start)*1000);
    clef_head_close(&h);
    free(hidden);
    free(logits);
    clef_record_free(r);
    free(r);
    json_free(req);
    free(text);
    free(out.p);
    return 0;
}
