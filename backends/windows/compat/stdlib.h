/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md). */
#include_next <stdlib.h>
#include "llbw_posix.h"
#ifndef LLBW_STDLIB_H
#define LLBW_STDLIB_H
#ifdef __cplusplus
extern "C" {
#endif
/* over _putenv_s: getenv in this process and the environment of its children both see it */
int setenv(const char *name, const char *value, int overwrite);
int unsetenv(const char *name);
char *realpath(const char *path, char *resolved);
int posix_memalign(void **p, size_t align, size_t n);
#ifdef __cplusplus
}
#endif
#endif
