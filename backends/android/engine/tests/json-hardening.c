#include "json.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    const char *err = NULL;
    const char *invalid[] = {"1e999", "1e", "01", "1+2", "\"bad\ntext\"", NULL};
    for (int i = 0; invalid[i]; i++) {
        jv *value = json_parse(invalid[i], &err);
        assert(!value && err);
    }
    jv *value = json_parse("{\"n\":1.25e-2}", &err);
    assert(value); json_free(value);
    puts("Android native JSON hardening passed");
    return 0;
}
