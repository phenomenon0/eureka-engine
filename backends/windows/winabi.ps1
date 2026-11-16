# backends/windows/winabi.ps1 - make an emitted Bend C file (llb-engine.c, sys1mac.c, any
# bun main.ts output) safe for the Win64 calling convention (XPLAT-WIN).
#
#   powershell -ExecutionPolicy Bypass -File backends\windows\winabi.ps1 -In llb-engine.c -Out llb-engine.win.c
#
# Why. The runtime's work loop is a chain of segment functions (WL_FN) entered by
# musttail, each taking `Env e` by value. Env is two pointers (16 bytes). The System V
# x86-64 ABI (Linux, macOS) passes it in two registers. The Win64 ABI passes any struct
# over 8 bytes by a hidden pointer to a copy in the caller's frame, and a musttail call
# frees that frame. The callee then reads a dead copy: on this machine an emitted program
# died at once ("bend: memory fault", "bend: solo delivery lost") with Env.mem == NULL.
# Marking the segments sysv_abi (clang honours it on Windows x64) puts Env back in
# registers, and the same program ran correctly on 1 and 16 threads.
#
# What it changes: exactly two lines of the runtime preamble, each of which must occur
# once, or this script fails and writes nothing:
#   #define WL_FN      static PRESERVE(preserve_none) __attribute__((noinline)) Reply
#   typedef Reply (PRESERVE(preserve_none) *WlFn)(WL_SIG);
# preserve_none is itself a calling convention and cannot be combined with sysv_abi, so the
# segments lose it on Windows (a register-allocation speed attribute, not a semantic one).
# Nothing else in the file changes. The Bend compiler and its patch stack are not touched;
# this runs on its output, as tools/remap_cc.sh does on Linux.
param(
  [Parameter(Mandatory = $true)] [string] $In,
  [Parameter(Mandatory = $true)] [string] $Out
)
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'

$text = [IO.File]::ReadAllText((Resolve-Path $In).Path)
$nl = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
$edits = @(
  @{ old = '#define WL_FN      static PRESERVE(preserve_none) __attribute__((noinline)) Reply'
     new = '#define WL_FN      static __attribute__((sysv_abi, noinline)) Reply /* XPLAT-WIN: backends/windows/winabi.ps1 */' },
  @{ old = 'typedef Reply (PRESERVE(preserve_none) *WlFn)(WL_SIG);'
     new = 'typedef Reply (__attribute__((sysv_abi)) *WlFn)(WL_SIG); /* XPLAT-WIN: backends/windows/winabi.ps1 */' }
)
$lines = [regex]::Split($text, "`r?`n")
foreach ($e in $edits) {
  $hits = @(0..($lines.Count - 1) | Where-Object { $lines[$_] -ceq $e.old })
  if ($hits.Count -ne 1) {
    Write-Error ("winabi: expected exactly one line '" + $e.old + "' in $In, found " + $hits.Count + "; the runtime preamble changed: re-check the Win64 Env/musttail issue before building")
    exit 1
  }
  $lines[$hits[0]] = $e.new
}
[IO.File]::WriteAllText($Out, ($lines -join $nl), (New-Object Text.UTF8Encoding($false)))
Write-Host "winabi: $In -> $Out (2 lines: WL_FN and WlFn are sysv_abi)"
exit 0
