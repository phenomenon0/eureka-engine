/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md). */
/* XPLAT-WIN: struct stat / stat() / fstat() are the shim's llbw_stat: the UCRT's 64-bit
 * _stat64 underneath (its plain struct stat holds st_size in a 32-bit long, and the GGUFs
 * pass 2 GiB), with POSIX's st_atim / st_mtim / st_ctim timespecs (tools/glue/glue_out.c
 * keys a cache on st_mtim). st_ino is 0 and st_uid / st_gid are 0, as the UCRT reports. */
#include_next <sys/types.h>
#include_next <sys/stat.h>
#include "../llbw_posix.h"
#ifndef LLBW_SYS_STAT_H
#define LLBW_SYS_STAT_H
#include <time.h>
struct llbw_stat {
  unsigned int st_dev, st_ino;
  unsigned short st_mode;
  short st_nlink;
  short st_uid, st_gid;
  unsigned int st_rdev;
  long long st_size;
  struct timespec st_atim, st_mtim, st_ctim;
};
#ifdef __cplusplus
extern "C" {
#endif
int llbw_stat(const char *path, struct llbw_stat *st);
int llbw_fstat(int fd, struct llbw_stat *st);
#ifdef __cplusplus
}
#endif
#define stat llbw_stat
#define fstat llbw_fstat
#define lstat llbw_stat
#define st_atime st_atim.tv_sec
#define st_mtime st_mtim.tv_sec
#define st_ctime st_ctim.tv_sec
#define mkdir llbw_mkdir
#ifndef S_IFMT
#define S_IFMT _S_IFMT
#define S_IFDIR _S_IFDIR
#define S_IFCHR _S_IFCHR
#define S_IFREG _S_IFREG
#endif
#ifndef S_IFIFO
#define S_IFIFO _S_IFIFO
#endif
#define S_IFLNK 0
#define S_IFSOCK 0
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#define S_ISCHR(m) (((m) & S_IFMT) == S_IFCHR)
#define S_ISFIFO(m) (((m) & S_IFMT) == S_IFIFO)
#define S_ISLNK(m) 0
#define S_ISSOCK(m) 0
#define S_IRUSR 0400
#define S_IWUSR 0200
#define S_IXUSR 0100
#define S_IRWXU 0700
#define S_IRGRP 040
#define S_IWGRP 020
#define S_IXGRP 010
#define S_IRWXG 070
#define S_IROTH 04
#define S_IWOTH 02
#define S_IXOTH 01
#define S_IRWXO 07
#endif
