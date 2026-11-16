/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md). */
#include_next <string.h>
#ifndef LLBW_STRING_H
#define LLBW_STRING_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
char *strndup(const char *s, size_t n);
size_t strnlen(const char *s, size_t n);
void *memmem(const void *h, size_t hn, const void *n, size_t nn);
char *stpcpy(char *d, const char *s);
int strerror_r(int e, char *buf, size_t n);
#ifdef __cplusplus
}
#endif
#define strdup _strdup
#define strtok_r strtok_s
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif
