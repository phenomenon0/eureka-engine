/* XPLAT-WIN: part of the Windows POSIX shim (backends/windows/compat/README.md).
 * The UCRT's stdio.h, then: fopencookie, an assignable stdout, getline/getdelim, and
 * the write side of stdio routed through llbw_* so a cookie stream works anywhere a
 * FILE * is written (llbw_posix.h, "stdio extensions", says why). */
#include_next <stdio.h>
#include "llbw_posix.h"
#ifndef LLBW_STDIO_H
#define LLBW_STDIO_H
#include <stdarg.h>
#ifdef __cplusplus
extern "C" {
#endif
FILE *fopencookie(void *cookie, const char *mode, cookie_io_functions_t io);
FILE **llbw_stdout_ref(void);
int llbw_fprintf(FILE *f, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
int llbw_vfprintf(FILE *f, const char *fmt, va_list ap);
int llbw_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
int llbw_vprintf(const char *fmt, va_list ap);
int llbw_fputs(const char *s, FILE *f);
int llbw_puts(const char *s);
int llbw_fputc(int c, FILE *f);
int llbw_putchar(int c);
size_t llbw_fwrite(const void *p, size_t sz, size_t n, FILE *f);
size_t llbw_fread(void *p, size_t sz, size_t n, FILE *f);
int llbw_fflush(FILE *f);
int llbw_fclose(FILE *f);
int llbw_setvbuf(FILE *f, char *buf, int mode, size_t n);
void llbw_setbuf(FILE *f, char *buf);
int llbw_ferror(FILE *f);
int llbw_fileno(FILE *f);
ssize_t getline(char **line, size_t *cap, FILE *f);
ssize_t getdelim(char **line, size_t *cap, int delim, FILE *f);
FILE *open_memstream(char **buf, size_t *len);
#ifdef __cplusplus
}
#endif
#undef stdout
#define stdout (*llbw_stdout_ref())
#define fprintf llbw_fprintf
#define vfprintf llbw_vfprintf
#define printf llbw_printf
#define vprintf llbw_vprintf
#define fputs llbw_fputs
#define puts llbw_puts
#define fputc llbw_fputc
#undef putc
#define putc llbw_fputc
#undef putchar
#define putchar llbw_putchar
#define fwrite llbw_fwrite
/* fread: a large fread goes to ReadFile on the caller's buffer, which must be committed first */
#define fread llbw_fread
#define fflush llbw_fflush
#define fclose llbw_fclose
#define setvbuf llbw_setvbuf
#define setbuf llbw_setbuf
#define ferror llbw_ferror
#define fileno llbw_fileno
#define fdopen _fdopen
#define fseeko _fseeki64
#define ftello _ftelli64
/* XPLAT-WIN: tools/metrics/tokstream.c llb_stdin_queued reads glibc's FILE buffer pointers to
 * count request lines already buffered (a metric, documented there as a lower bound). The
 * UCRT's FILE is opaque (one _Placeholder pointer), so both names are that one field: the
 * loop runs zero times and the queue depth reads 0, the bound's floor. */
#define _IO_read_ptr _Placeholder
#define _IO_read_end _Placeholder
#define popen _popen
#define pclose _pclose
#endif
