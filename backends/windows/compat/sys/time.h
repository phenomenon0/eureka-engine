/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md). */
#pragma once
#include <time.h>
#include "../llbw_posix.h"
struct timezone { int tz_minuteswest, tz_dsttime; };
int gettimeofday(struct timeval *tv, void *tz);
