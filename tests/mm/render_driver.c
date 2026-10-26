// render_driver.c: G5 parity harness. Compiles serve.c itself (main renamed) and prints,
// per request file, the exact prompt the server would send plus its media manifest:
//   render_driver REQ.json...  ->  one JSON line per file: {"file","status","prompt"|"error","media":[...]}
#define main serve_main
#include "../../tools/sys1dual/serve.c"
#undef main
int main(int argc, char **argv) {
    mm_caps_default(&mmc);
    mmc.images = 1; // render as if the engine took images (the gate is tested elsewhere)
    for (int i = 1; i < argc; i++) {
        char *s = json_read_file(argv[i]);
        const char *err = NULL;
        jv *body = s ? json_parse(s, &err) : NULL;
        Req r = {0};
        sb o = {0};
        sb_add(&o, "{\"file\":");
        json_quote(&o, argv[i]);
        if (!body || (err = parse_req(body, 1, &r))) {
            sb_fmt(&o, ",\"status\":%d,\"error\":", r.status ? r.status : 400);
            json_quote(&o, err ? err : "bad json");
        } else {
            sb_add(&o, ",\"status\":200,\"prompt\":");
            json_quote(&o, r.prompt);
            sb_add(&o, ",\"media\":[");
            for (size_t k = 0; k < r.media.n; k++)
                sb_fmt(&o, "%s{\"mime\":\"%s\",\"bytes\":%zu,\"w\":%u,\"h\":%u,\"sha256\":\"%s\"}", k ? "," : "",
                    r.media.v[k].mime, r.media.v[k].n, r.media.v[k].w, r.media.v[k].h, r.media.v[k].sha256);
            sb_addc(&o, ']');
        }
        sb_addc(&o, '}');
        puts(o.p);
        free(o.p), free(r.prompt), media_done(&r), json_free(body), free(s);
    }
    return 0;
}
