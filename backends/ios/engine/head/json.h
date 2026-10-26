// head/json.h — small JSON value + parser + serializer for the Jev (Kev) decision head.
//
// Two properties matter for parity with llama.cpp's server (nlohmann ordered_json):
//   1. objects keep INSERTION ORDER (Kev's answer depends on option order, SPEC §2);
//      a duplicate key replaces the value in place, at the position of its first occurrence;
//   2. numbers keep their kind (int / uint / double) and print as nlohmann's dump() does:
//      integers in decimal, doubles as the shortest round-trip digits in nlohmann's layout
//      (decimal notation for exponents in (-4, 15], else d.ddde±XX), "1.0" not "1".
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef enum { JN_NULL, JN_BOOL, JN_INT, JN_UINT, JN_DBL, JN_STR, JN_ARR, JN_OBJ } jtype;

typedef struct jv {
  jtype t;
  int b;               // JN_BOOL
  int64_t i;           // JN_INT
  uint64_t u;          // JN_UINT
  double d;            // JN_DBL
  char *s;             // JN_STR (NUL-terminated, may contain NUL: use slen)
  size_t slen;
  size_t n;            // JN_ARR / JN_OBJ: number of items
  char **keys;         // JN_OBJ: keys[n] (NUL-terminated)
  struct jv **items;   // JN_ARR / JN_OBJ: items[n]
} jv;

jv *jv_parse(const char *text, size_t len, char *err, size_t errn);  // NULL on error
void jv_free(jv *v);
const jv *jv_get(const jv *obj, const char *key);                   // NULL if absent or not an object

// growable string buffer
typedef struct { char *p; size_t n, cap; } sbuf;
void sb_putn(sbuf *b, const char *s, size_t n);
void sb_puts(sbuf *b, const char *s);
void sb_putc(sbuf *b, char c);
void sb_printf(sbuf *b, const char *fmt, ...);
char *sb_take(sbuf *b);  // NUL-terminated, caller frees

void jv_dump(sbuf *b, const jv *v);                  // nlohmann dump(): compact, ensure_ascii=false
void jv_dump_str(sbuf *b, const char *s, size_t n);  // a JSON string literal
void jv_dump_double(sbuf *b, double d);              // nlohmann double layout
