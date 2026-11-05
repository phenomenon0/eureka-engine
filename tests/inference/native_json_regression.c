#include "json.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    const char *bad[] = {"1e999", "-", "1.", "01", "1e+", "1+2", "[\"a\nb\"]", NULL};
    for (int i = 0; bad[i]; i++) { const char *err = NULL; jv *v = json_parse(bad[i], &err); assert(!v && err); }
    const char *err = NULL; jv *v = json_parse("{\"temperature\":0.75,\"x\":[1,true]}", &err); assert(v);
    sb out = {0}; json_dump(&out, v); assert(!strcmp(out.p, "{\"temperature\":0.75,\"x\":[1,true]}")); free(out.p); json_free(v);
    char deep[300]; memset(deep, '[', 140); deep[140] = '0'; memset(deep + 141, ']', 140); deep[281] = 0;
    v = json_parse(deep, &err); assert(!v && err);
    return 0;
}
