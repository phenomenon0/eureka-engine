/* glibc234.h: pre-included (-include) so the binaries need GLIBC <= 2.34. glibc >= 2.38 redirects
 * strtol/strtoul/sscanf/fscanf to __isoc23_* (GLIBC_2.38) under _GNU_SOURCE or C23; the only change
 * is C23's "0b" binary prefix, which no caller here parses. Pull in features.h the way the sources
 * would (_GNU_SOURCE) and switch the redirect off; the source files' own text is not touched. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <features.h>
#undef __GLIBC_USE_C23_STRTOL
#define __GLIBC_USE_C23_STRTOL 0
#undef __GLIBC_USE_C2X_STRTOL
#define __GLIBC_USE_C2X_STRTOL 0
