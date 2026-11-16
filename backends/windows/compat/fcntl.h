/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md). */
/* XPLAT-WIN: open() is llbw_open, which always adds _O_BINARY; O_CLOEXEC is
 * _O_NOINHERIT; O_NONBLOCK is accepted and dropped for files. */
#include_next <fcntl.h>
#include "llbw_posix.h"
#ifndef LLBW_FCNTL_H
#define LLBW_FCNTL_H
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#define O_WRONLY _O_WRONLY
#define O_RDWR _O_RDWR
#define O_APPEND _O_APPEND
#define O_CREAT _O_CREAT
#define O_TRUNC _O_TRUNC
#define O_EXCL _O_EXCL
#define O_TEXT _O_TEXT
#define O_BINARY _O_BINARY
#endif
#define O_CLOEXEC _O_NOINHERIT
#define O_NONBLOCK 0x10000000
#define O_NOCTTY 0
#define O_SYNC 0
#define O_DSYNC 0
#define O_NOFOLLOW 0
#define O_DIRECTORY 0
#define O_NOATIME 0
#define open llbw_open
#endif
