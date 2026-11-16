/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md).
 * The UCRT's signal.h, then sigaction and the signal numbers it lacks; signal() is
 * llbw_signal, which accepts the numbers the UCRT does not know (its own signal()
 * reports those as an invalid parameter). */
#include_next <signal.h>
#include "llbw_posix.h"
#ifndef LLBW_SIGNAL_H
#define LLBW_SIGNAL_H
#define signal llbw_signal
#endif
