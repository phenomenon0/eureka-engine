#define _POSIX_C_SOURCE 200809L
#include "head.h"
#include <openblas/cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int n,d;
    float *v;
} Mat;
static Mat mat(int n,int d) {
    Mat a={
        n,d,calloc((size_t)n*d,4)
    };
    if (!a.v) clef_fail("head allocation failed");
    return a;
}

static void drop(Mat a) {
    free(a.v);
}

static void checkpoint(const char *name,Mat a) {
    const char *dir=getenv("CLEF_HEAD_DUMP");
    if (!dir || !*dir) return;
    char path[4096];
    if (snprintf(path,sizeof path,"%s/%s.f32",dir,name)>=(int)sizeof path) clef_fail("checkpoint path too long");
    FILE *f=fopen(path,"wb");
    uint32_t shape[2]={
        (uint32_t)a.n,(uint32_t)a.d
    };
    if (!f || fwrite(shape,4,2,f)!=2 || fwrite(a.v,4,(size_t)a.n*a.d,f)!=(size_t)a.n*a.d || fclose(f)) clef_fail("checkpoint write failed");
}

static Mat linear(Head *h,const char *name,Mat x,int d,int bias) {
    char key[128];
    snprintf(key,sizeof key,"%s.weight",name);
    float *w=clef_weight(h,key,d,x.d);
    Mat y=mat(x.n,d);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,x.n,d,x.d,1,x.v,x.d,w,x.d,0,y.v,d);
    if (bias) {
        snprintf(key,sizeof key,"%s.bias",name);
        float *b=clef_weight(h,key,1,d);
        for (int i=0; i<x.n; i++) for (int j=0; j<d; j++) y.v[i*d+j]+=b[j];
    }
    return y;
}

static Mat norm(Head *h,const char *name,Mat x) {
    char key[128];
    snprintf(key,sizeof key,"%s.weight",name);
    float *w=clef_weight(h,key,1,x.d);
    snprintf(key,sizeof key,"%s.bias",name);
    float *b=clef_weight(h,key,1,x.d);
    Mat y=mat(x.n,x.d);
    for (int i=0; i<x.n; i++) {
        float mean=0,var=0;
        const float *p=x.v+(size_t)i*x.d;
        for (int j=0; j<x.d; j++) mean+=p[j];
        mean/=x.d;
        for (int j=0; j<x.d; j++) {
            float a=p[j]-mean;
            var+=a*a;
        }
        float inv=1/sqrtf(var/x.d+1e-5f);
        for (int j=0; j<x.d; j++) y.v[(size_t)i*x.d+j]=((p[j]-mean)*inv)*w[j]+b[j];
    }
    return y;
}

static void add(Mat x,Mat y) {
    if (x.n!=y.n || x.d!=y.d) clef_fail("add shape mismatch");
    for (size_t i=0; i<(size_t)x.n*x.d; i++) x.v[i]+=y.v[i];
}

static void gelu(Mat x) {
    for (size_t i=0; i<(size_t)x.n*x.d; i++) {
        float a=x.v[i];
        x.v[i]=(0.5f*a)*(1+erff(a*0.7071067811865475f));
    }
}

static void softmax(float *x,int n) {
    float high=x[0],sum=0;
    for (int i=1; i<n; i++) high=fmaxf(high,x[i]);
    for (int i=0; i<n; i++) {
        x[i]=expf(x[i]-high);
        sum+=x[i];
    }
    for (int i=0; i<n; i++) x[i]/=sum;
}

static Mat attention(Head *h,const char *prefix,Mat query,Mat memory) {
    char key[128];
    snprintf(key,sizeof key,"%s_q",prefix);
    Mat q=linear(h,key,query,1024,1);
    snprintf(key,sizeof key,"%s_k",prefix);
    Mat k=linear(h,key,memory,1024,1);
    snprintf(key,sizeof key,"%s_v",prefix);
    Mat v=linear(h,key,memory,1024,1);
    Mat out=mat(q.n,1024),scores=mat(q.n,k.n);
    for (int head=0; head<16; head++) {
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,q.n,k.n,64,0.125f,q.v+head*64,1024,k.v+head*64,1024,0,scores.v,k.n);
        for (int i=0; i<q.n; i++) softmax(scores.v+(size_t)i*k.n,k.n);
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,q.n,64,k.n,1,scores.v,k.n,v.v+head*64,1024,0,out.v+head*64,1024);
    }
    snprintf(key,sizeof key,"%s_o",prefix);
    Mat result=linear(h,key,out,1024,1);
    drop(q);
    drop(k);
    drop(v);
    drop(out);
    drop(scores);
    return result;
}

static void block(Head *h,int layer,Mat x,Mat memory) {
    char key[128];
    Mat q,m,y;
    if (layer>=2) {
        snprintf(key,sizeof key,"dec.blk.%d.attn_norm",layer);
        q=norm(h,key,x);
        snprintf(key,sizeof key,"dec.blk.%d.attn",layer);
        y=attention(h,key,q,q);
        add(x,y);
        drop(q);
        drop(y);
    }
    snprintf(key,sizeof key,"dec.blk.%d.cross_attn_norm",layer);
    q=norm(h,key,x);
    m=memory;
    if (layer<2) {
        snprintf(key,sizeof key,"dec.blk.%d.cross_attn_norm_kv",layer);
        m=norm(h,key,memory);
    }
    snprintf(key,sizeof key,"dec.blk.%d.cross_attn",layer);
    y=attention(h,key,q,m);
    add(x,y);
    drop(q);
    drop(y);
    if (layer<2) drop(m);
    snprintf(key,sizeof key,"dec.blk.%d.ffn_norm",layer);
    q=norm(h,key,x);
    snprintf(key,sizeof key,"dec.blk.%d.ffn_up",layer);
    y=linear(h,key,q,4096,1);
    gelu(y);
    snprintf(key,sizeof key,"dec.blk.%d.ffn_down",layer);
    m=linear(h,key,y,1024,1);
    add(x,m);
    drop(q);
    drop(y);
    drop(m);
    snprintf(key,sizeof key,"block%d",layer);
    checkpoint(key,x);
}

static void mean_span(Mat src,Span s,float *dest) {
    if (s.start<0 || s.end>src.n || s.start>=s.end) clef_fail("invalid pooling span");
    for (int i=s.start; i<s.end; i++) for (int j=0; j<src.d; j++) dest[j]+=src.v[(size_t)i*src.d+j];
    for (int j=0; j<src.d; j++) dest[j]/=s.end-s.start;
}

static float cosine(const float *x,const float *y,int n,float eps) {
    float xy=0,xx=0,yy=0;
    for (int j=0; j<n; j++) {
        xy+=x[j]*y[j];
        xx+=x[j]*x[j];
        yy+=y[j]*y[j];
    }
    return xy/(fmaxf(sqrtf(xx),eps)*fmaxf(sqrtf(yy),eps));
}

void clef_head_run(Head *h,const Record *r,const float *hidden,float *logits) {
    const int d=h->d; // the trunk width the head reads (output.weight cols, clef_head_open)
    Mat raw={
        r->n,d,(float*)hidden
    }, n=norm(h,"decision.hidden_norm",raw);
    checkpoint("hidden_norm",n);
    Mat mem=linear(h,"decision.proj_memory",n,1024,0);
    checkpoint("memory",mem);
    Mat q=mat(r->nq,d), ctx=mat(r->no,d), lex=mat(r->no,d);
    for (int i=0; i<r->nq; i++) mean_span(n,r->q[i].span,q.v+(size_t)i*d);
    // Cache only token rows referenced by the option spans, never the whole vocabulary.
    float **cache=calloc(h->vocab,sizeof *cache);
    if (!cache) clef_fail("lexical cache allocation failed");
    for (int i=0; i<r->no; i++) {
        Span s=r->o[i].span;
        mean_span(n,s,ctx.v+(size_t)i*d);
        for (int t=s.start; t<s.end; t++) {
            int id=r->ids[t];
            if (id<0 || id>=h->vocab) clef_fail("invalid token id");
            if (!cache[id]) {
                cache[id]=malloc((size_t)d*4);
                if (!cache[id]) clef_fail("lexical row allocation failed");
                clef_output_row(h,id,cache[id]);
            }
            for (int j=0; j<d; j++) lex.v[(size_t)i*d+j]+=cache[id][j];
        }
        for (int j=0; j<d; j++) lex.v[(size_t)i*d+j]/=s.end-s.start;
    }
    for (int i=0; i<h->vocab; i++) free(cache[i]);
    free(cache);
    checkpoint("question_pool",q);
    checkpoint("context_pool",ctx);
    checkpoint("lexical_pool",lex);
    Mat pc=linear(h,"decision.proj_option_context",ctx,1024,0), pl=linear(h,"decision.proj_option_lexical",lex,1024,0);
    Mat pq=linear(h,"decision.proj_option_question",q,1024,0);
    checkpoint("proj_context",pc);
    checkpoint("proj_lexical",pl);
    checkpoint("proj_option_question",pq);
    add(pc,pl);
    for (int i=0; i<r->no; i++) for (int j=0; j<1024; j++) pc.v[(size_t)i*1024+j]+=pq.v[(size_t)r->o[i].parent*1024+j];
    checkpoint("option_seed",pc);
    drop(pl);
    drop(pq);
    drop(ctx);
    block(h,0,pc,mem);
    block(h,1,pc,mem);
    Mat base=linear(h,"decision.proj_question",q,1024,0), summaries=mat(r->nq,1024);
    checkpoint("proj_question",base);
    for (int i=0; i<r->nq; i++) {
        Question field=r->q[i];
        float *weights=calloc(field.count,4);
        for (int o=0; o<field.count; o++) {
            for (int j=0; j<1024; j++) weights[o]+=pc.v[(size_t)(field.first+o)*1024+j]*base.v[(size_t)i*1024+j];
            weights[o]/=32;
        } softmax(weights,field.count);
        for (int o=0; o<field.count; o++) for (int j=0; j<1024; j++) summaries.v[(size_t)i*1024+j]+=weights[o]*pc.v[(size_t)(field.first+o)*1024+j];
        free(weights);
    } checkpoint("summaries",summaries);
    Mat sn=norm(h,"decision.option_summary_norm",summaries);
    Mat global={
        1,d,n.v+(size_t)(n.n-1)*d
    }, pg=linear(h,"decision.proj_global",global,1024,0);
    checkpoint("proj_global",pg);
    checkpoint("summary_norm",sn);
    add(base,sn);
    float *types=clef_weight(h,"token_types.weight",3,1024);
    for (int i=0; i<r->nq; i++) for (int j=0; j<1024; j++) {
        base.v[(size_t)i*1024+j]+=pg.v[j];
        base.v[(size_t)i*1024+j]+=types[r->q[i].type*1024+j];
    }
    checkpoint("field_seed",base);
    drop(sn);
    drop(pg);
    drop(summaries);
    for (int layer=2; layer<6; layer++) block(h,layer,base,mem);
    Mat fields=norm(h,"decision.field_norm",base), options=norm(h,"decision.option_norm",pc);
    checkpoint("fields",fields);
    checkpoint("options",options);
    Mat prior=mat(r->no,1), cos=mat(r->no,1), features=mat(r->no,4096);
    float *scales=clef_weight(h,"decision.scales",1,3);
    float *anchor=malloc((size_t)d*4);
    if (!anchor) clef_fail("head allocation failed");
    for (int i=0; i<r->no; i++) {
        int parent=r->o[i].parent;
        const float *f=fields.v+(size_t)parent*1024,*o=options.v+(size_t)i*1024;
        for (int j=0; j<d; j++) anchor[j]=q.v[(size_t)parent*d+j]+global.v[j];
        prior.v[i]=scales[0]*cosine(lex.v+(size_t)i*d,anchor,d,1e-12f);
        cos.v[i]=cosine(f,o,1024,1e-8f);
        float *feat=features.v+(size_t)i*4096;
        for (int j=0; j<1024; j++) {
            feat[j]=f[j];
            feat[1024+j]=o[j];
            feat[2048+j]=f[j]*o[j];
            feat[3072+j]=fabsf(f[j]-o[j]);
        }
    } free(anchor);
    checkpoint("prior",prior);
    checkpoint("cosine",cos);
    checkpoint("features",features);
    Mat score=linear(h,"decision.scorer",features,1024,1);
    gelu(score);
    checkpoint("scorer",score);
    Mat residual=linear(h,"decision.scorer_out",score,1,1);
    checkpoint("residual",residual);
    for (int i=0; i<r->no; i++) {
        logits[i]=prior.v[i]+scales[2]*(scales[1]*cos.v[i]+residual.v[i]);
        if (!isfinite(logits[i])) clef_fail("nonfinite head logit");
    } checkpoint("logits",(Mat){r->no,1,logits});
    Mat probs=mat(r->no,1);
    memcpy(probs.v,logits,r->no*4);
    for (int i=0; i<r->nq; i++) softmax(probs.v+r->q[i].first,r->q[i].count);
    checkpoint("probabilities",probs);
    drop(n);
    drop(mem);
    drop(q);
    drop(lex);
    drop(pc);
    drop(base);
    drop(fields);
    drop(options);
    drop(prior);
    drop(cos);
    drop(features);
    drop(score);
    drop(residual);
    drop(probs);
}
