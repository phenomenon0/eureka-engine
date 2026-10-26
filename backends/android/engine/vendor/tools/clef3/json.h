// minimal ordered JSON (enough for /v1/systemone requests and responses). no dependencies.
#pragma once
#include <stddef.h>
typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;
typedef struct jv {
    jtype type;
    int b;              // J_BOOL
    double num;         // J_NUM
    char *raw;          // J_NUM: the literal text as written (kev renders numbers with dump())
    char *str;          // J_STR (utf-8, NUL terminated; \u0000 not supported)
    size_t n;           // J_ARR / J_OBJ: item count
    struct jv **items;  // J_ARR / J_OBJ values
    char **keys;        // J_OBJ keys, insertion order
} jv;
jv *json_parse(const char *s, const char **err);
jv *json_get(const jv *o, const char *key);   // NULL if absent
void json_free(jv *v);
char *json_read_file(const char *path);       // malloc'd, NULL on error
// serialization (sb = growable string)
typedef struct { char *p; size_t n, cap; } sb;
void sb_add(sb *s, const char *t);
void sb_addn(sb *s, const char *t, size_t n);
void sb_addc(sb *s, char c);
void sb_fmt(sb *s, const char *fmt, ...);
void json_quote(sb *s, const char *str);
void json_dump(sb *s, const jv *v);           // compact, nlohmann-style separators (',' and ':' no spaces)
