/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md). */
#include_next <time.h>
#include "llbw_posix.h"
#ifndef LLBW_TIME_H
#define LLBW_TIME_H
#ifdef __cplusplus
extern "C" {
#endif
struct tm *gmtime_r(const time_t *t, struct tm *out);
struct tm *localtime_r(const time_t *t, struct tm *out);
#ifdef __cplusplus
}
#endif
#endif
