/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md). */
/* the UCRT's sys/types.h first, then the POSIX types it leaves out (llbw_posix.h) */
#include_next <sys/types.h>
#include "../llbw_posix.h"
