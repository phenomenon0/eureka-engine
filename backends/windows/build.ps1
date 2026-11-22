# backends/windows/build.ps1 - build the engine on native Windows (x64).
#
#   build\win\cpu\qwen35-cli.exe   CPU twin   (clang or MSVC cl)
#   build\win\cpu\gguf2pack.exe    GGUF -> pack / sidecar converter
#   build\win\cpu\llb-tok.exe      tokenizer.bin extractor (clang, POSIX shim)
#   build\win\cpu\engine_cpu.exe   the desktop's CPU engine, driven by llb-serve (clang, POSIX shim)
#   build\win\cpu\engine\cpu\      the desktop contract dir (+ llb-serve.exe with -LlbServe)
#   build\qwen35-cli-gpu.exe       CUDA (--gpu), if nvcc is on PATH; arch from nvidia-smi
#
# The CPU part was built and run on Windows 11 (clang 23, x86_64-pc-windows-msvc) on
# 2026-11-16; WINDOWS-TEST-STATUS.md has the numbers. The CUDA part has not run yet.
# See backends/windows/BUILD.md.
#
# Usage (from the repo root; PowerShell 7 or Windows PowerShell 5.1; clang is found on PATH or
# in Program Files\LLVM; nvcc needs "Developer PowerShell for VS 2022" so cl.exe is on PATH):
#   powershell -ExecutionPolicy Bypass -File backends\windows\build.ps1 [-Cc auto|clang|cl] [-Arch sm_XX] [-NoCuda] [-LlbServe PATH]
# Every run rebuilds every target (safe to re-run). Prints each command it runs.

param(
  [ValidateSet('auto', 'clang', 'cl')] [string] $Cc = 'auto',
  [string] $Arch = '',
  [switch] $NoCuda,
  [switch] $NoCpu,
  [string] $CudaPath = '',
  [string] $LlbServe = ''
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Build = Join-Path $Root 'build'
$Tools = Join-Path $Root 'tools'
Set-Location $Root
if (-not (Test-Path $Build)) { New-Item -ItemType Directory -Path $Build | Out-Null }

$results = New-Object System.Collections.ArrayList

function Have([string] $exe) { return [bool](Get-Command $exe -ErrorAction SilentlyContinue) }

# Print the command line, run it, return its exit code. Its output goes to the console
# (Out-Host), not into the return value; stderr from a native tool must not throw.
function Run([string] $exe, [string[]] $argv) {
  $ErrorActionPreference = 'Continue'
  Write-Host ('> ' + $exe + ' ' + ($argv -join ' '))
  & $exe @argv | Out-Host
  return $LASTEXITCODE
}

function Record([string] $name, [int] $rc, [string] $what) {
  $label = $name
  if ($what -ne '') { $label = "$name ($what)" }
  if ($rc -eq 0) { [void]$results.Add(@{ name = $name; ok = $true; text = $label }) }
  else { [void]$results.Add(@{ name = $name; ok = $false; text = "${label}: exit $rc" }) }
}

# ---- CPU: clang (LLVM for Windows) preferred, else MSVC cl ----------------
# Everything CPU goes to build\win\cpu (XPLAT-WIN: one output dir per lane, so the CUDA and
# llb-serve builds never overwrite these). Numerical contract: no FMA contraction, no
# fast-math (Linux: -ffp-contract=off -fno-fast-math). clang: the same flags. cl: /fp:precise,
# and no /fp:contract (VS 2022 does not contract under /fp:precise unless /fp:contract is
# given); no /arch:AVX2 (lets VS 2019 contract).
$CpuOut = Join-Path $Build 'win\cpu'
$ObjOut = Join-Path $CpuOut 'obj'
$Compat = Join-Path $Root 'backends\windows\compat'
$WinSrc = Join-Path $Root 'backends\windows'
foreach ($d in @($CpuOut, $ObjOut)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
# clang need not be on PATH: LLVM's installer leaves it in Program Files by default
$ClangExe = 'clang'
if (-not (Have 'clang')) {
  $pf = Join-Path $env:ProgramFiles 'LLVM\bin\clang.exe'
  if (Test-Path $pf) { $ClangExe = $pf }
}
if ($Cc -eq 'auto') {
  if ($ClangExe -ne 'clang' -or (Have 'clang')) { $Cc = 'clang' } elseif (Have 'cl') { $Cc = 'cl' } else { $Cc = 'none' }
}
Write-Host "CPU compiler: $Cc"

# -NoCpu: leave build\win\cpu as it is (the CUDA lane still copies its gguf2pack.exe and
# llb-tok.exe). The CPU lane below is unchanged inside this block.
if ($NoCpu) { Write-Host 'CPU: skipped (-NoCpu)' } else {
$cpuSrc = @(
  @{ name = 'qwen35-cli.exe'; src = (Join-Path $Tools 'qwen35_engine.c'); fp = $true; what = 'CPU' },
  @{ name = 'gguf2pack.exe';  src = (Join-Path $Tools 'gguf2pack.c');     fp = $false; what = '' }
)
foreach ($t in $cpuSrc) {
  $out = Join-Path $CpuOut $t.name
  if (Test-Path $out) { Remove-Item $out -Force }
  if ($Cc -eq 'clang') {
    $a = @('-O2', '-std=c11', '-D_CRT_SECURE_NO_WARNINGS')
    if ($t.fp) { $a += @('-ffp-contract=off', '-fno-fast-math') }
    $a += @($t.src, '-o', $out)
    $rc = Run $ClangExe $a
  } elseif ($Cc -eq 'cl') {
    $obj = Join-Path $ObjOut ''
    $a = @('/nologo', '/O2', '/std:c11', '/D_CRT_SECURE_NO_WARNINGS')
    if ($t.fp) { $a += @('/fp:precise') }
    $a += @($t.src, "/Fe:$out", "/Fo:$obj")
    $rc = Run 'cl' $a
  } else {
    Write-Host "no CPU compiler: neither clang nor cl on PATH"
    $rc = 127
  }
  if ($rc -eq 0 -and -not (Test-Path $out)) { $rc = 1 }
  Record $t.name $rc $t.what
}

# ---- the desktop CPU engine: engine_cpu.exe + llb-tok.exe over the POSIX shim -------------
# engine_cpu = tools/qwen35_engine.c's forward + tools/sys1dual/dual.c's request side, the
# stdin/stdout JSON-line protocol llb-serve drives (backends/windows/BUILD.md, "engine_cpu").
# The POSIX sources compile unchanged against backends\windows\compat (a -I dir); the shim's
# own implementation (compat\llbw_posix.c) is built without it. clang only: the sources use
# GNU C (#include_next in the shim, __attribute__, _Float16, __builtin_ctzll).
$engineNames = @('llbw_posix.o (POSIX shim)', 'llb-tok.exe', 'engine_cpu.exe')
if ($Cc -ne 'clang') {
  foreach ($n in $engineNames) { Record $n 127 'needs clang (LLVM for Windows)' }
} else {
  $nonstd = '-D_CRT_DECLARE_NONSTDC_NAMES=0'  # the shim declares the POSIX names itself (compat\README.md)
  $req = @('-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math', '-D_POSIX_C_SOURCE=200809L',
           '-D_CRT_SECURE_NO_WARNINGS', $nonstd, '-Wall', '-Wextra', '-Wno-unused-function', '-Wno-unused-parameter',
           '-I', $Compat)
  $shimObj = Join-Path $ObjOut 'llbw_posix.o'
  if (Test-Path $shimObj) { Remove-Item $shimObj -Force }
  $rc = Run $ClangExe @('-std=c11', '-O2', $nonstd, '-Wall', '-c', (Join-Path $Compat 'llbw_posix.c'), '-o', $shimObj)
  if ($rc -eq 0 -and -not (Test-Path $shimObj)) { $rc = 1 }
  Record 'llbw_posix.o (POSIX shim)' $rc ''
  # compiler-rt's builtins: _Float16 conversions (tools/clef3, tools/glue) call __extendhfsf2
  $rtDir = (& $ClangExe --print-resource-dir | Select-Object -First 1).ToString().Trim()
  $rtLib = Join-Path $rtDir 'lib\windows\clang_rt.builtins-x86_64.lib'
  $sysLibs = @($rtLib, '-lws2_32', '-lwinmm')

  # the shim's self-test (compat\test_shim.c): the reservation, first-touch commit, pread into reserved
  # pages, threads, select/poll on a pipe, sockets, the stdout cookie; run in obj\ (it writes temp files)
  $out = Join-Path $CpuOut 'test_shim.exe'
  if (Test-Path $out) { Remove-Item $out -Force }
  $rc = 1
  if (Test-Path $shimObj) {
    $rc = Run $ClangExe (@('-std=c11', '-O2', '-D_CRT_SECURE_NO_WARNINGS', $nonstd, '-D_POSIX_C_SOURCE=200809L', '-I', $Compat,
                          (Join-Path $Compat 'test_shim.c'), $shimObj) + $sysLibs + @('-o', $out))
    if ($rc -eq 0) {
      Push-Location $ObjOut
      try { $rc = Run $out @() } finally { Pop-Location }
    }
  }
  Record 'test_shim.exe' $rc 'shim self-test, run'

  # llb-tok: tools/llb/llb-tok.c (the desktop runs `llb-tok extract GGUF DIR` for tokenizer.bin)
  $out = Join-Path $CpuOut 'llb-tok.exe'
  if (Test-Path $out) { Remove-Item $out -Force }
  $rc = 1
  if (Test-Path $shimObj) {
    $rc = Run $ClangExe (@('-std=c11', '-O2', '-D_CRT_SECURE_NO_WARNINGS', $nonstd, '-I', $Compat,
                          (Join-Path $Tools 'llb\llb-tok.c'), $shimObj) + $sysLibs + @('-o', $out))
  }
  if ($rc -eq 0 -and -not (Test-Path $out)) { $rc = 1 }
  Record 'llb-tok.exe' $rc ''

  # engine_cpu: the request objects tools/sys1dual/build.sh links (minus qwen35_tokenizer.c, which
  # qwen35_engine.c includes), the Bend host's sampler, the GGUF loader, the driver, the shim
  $objs = @()
  $units = @(
    (Join-Path $Tools 'sys1dual\dual.c'), (Join-Path $Tools 'metrics\metrics.c'),
    (Join-Path $Tools 'metrics\tokstream.c'), (Join-Path $Tools 'glue\glue_out.c'),
    (Join-Path $Tools 'clef3\head.c'), (Join-Path $Tools 'clef3\weights.c'),
    (Join-Path $Tools 'clef3\spans.c'), (Join-Path $Tools 'clef3\json.c'),
    (Join-Path $Root 'host\llb_sampler.c'), (Join-Path $WinSrc 'cblas_sgemm.c'),
    (Join-Path $WinSrc 'engine_cpu_gguf.c'), (Join-Path $WinSrc 'engine_cpu.c')
  )
  $rc = 0
  if (-not (Test-Path $shimObj)) { $rc = 1 }
  foreach ($u in $units) {
    if ($rc -ne 0) { break }
    $o = Join-Path $ObjOut ([IO.Path]::GetFileNameWithoutExtension($u) + '.o')
    if (Test-Path $o) { Remove-Item $o -Force }
    $rc = Run $ClangExe ($req + @('-c', $u, '-o', $o))
    if ($rc -eq 0 -and -not (Test-Path $o)) { $rc = 1 }
    $objs += $o
  }
  $out = Join-Path $CpuOut 'engine_cpu.exe'
  if (Test-Path $out) { Remove-Item $out -Force }
  if ($rc -eq 0) {
    # an 8 MiB main-thread stack, glibc's default (Windows' is 1 MiB)
    $rc = Run $ClangExe (@('-O2', '-o', $out) + $objs + @($shimObj) + $sysLibs + @('-Wl,/STACK:8388608'))
  }
  if ($rc -eq 0 -and -not (Test-Path $out)) { $rc = 1 }
  Record 'engine_cpu.exe' $rc 'tools/qwen35_engine.c + dual.c'

  # Inkling 2 Mini native CPU shape engine (optional when cmake + inkling2/ are present)
  $InkSrc = Join-Path $Root 'inkling2\engine\src\inkling2'
  if ((Test-Path (Join-Path $InkSrc 'CMakeLists.txt')) -and (Have 'cmake')) {
    $inkCpuBuild = Join-Path $CpuOut 'inkling-build'
    $rcInk = Run 'cmake' @('-S', $InkSrc, '-B', $inkCpuBuild, '-DCMAKE_BUILD_TYPE=Release', '-DSTRATA_ENABLE_CUDA=OFF')
    if ($rcInk -eq 0) { $rcInk = Run 'cmake' @('--build', $inkCpuBuild, '--config', 'Release', '--target', 'strata-inkling', '--parallel') }
    $inkBin = @(Join-Path $inkCpuBuild 'Release\strata-inkling.exe'; Join-Path $inkCpuBuild 'strata-inkling.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($rcInk -eq 0 -and $inkBin) { Copy-Item $inkBin (Join-Path $CpuOut 'engine_inkling_cpu.exe') -Force }
    Record 'engine_inkling_cpu.exe' $(if ($inkBin) { $rcInk } else { 1 }) 'Inkling 2 Mini CPU'
  }

  # the desktop contract's cpu\ dir (bend-app desktop/engine/build.sh stages EUREKA_WINDOWS_ENGINE\cpu):
  # engine_cpu.exe, gguf2pack.exe, llb-tok.exe from here; llb-serve.exe from the llb-serve build
  # (-LlbServe PATH), else it is named as missing
  $stage = Join-Path $CpuOut 'engine\cpu'
  if (-not (Test-Path $stage)) { New-Item -ItemType Directory -Path $stage | Out-Null }
  foreach ($n in @('engine_cpu.exe', 'gguf2pack.exe', 'llb-tok.exe', 'engine_inkling_cpu.exe')) {
    $p = Join-Path $CpuOut $n
    if (Test-Path $p) { Copy-Item $p (Join-Path $stage $n) -Force }
  }
  if ($LlbServe -ne '') {
    if (Test-Path $LlbServe) { Copy-Item $LlbServe (Join-Path $stage 'llb-serve.exe') -Force }
    else { Write-Host "-LlbServe ${LlbServe}: no such file" }
  }
  $have = @(Get-ChildItem $stage -Filter '*.exe' | ForEach-Object { $_.Name })
  $lack = @(@('llb-serve.exe', 'engine_cpu.exe', 'gguf2pack.exe', 'llb-tok.exe') | Where-Object { $have -notcontains $_ })
  if ($lack.Count -gt 0) { Write-Host ("contract dir ${stage}: missing " + ($lack -join ', ')) }
  else { Write-Host "contract dir ${stage}: complete (EUREKA_WINDOWS_ENGINE=$(Join-Path $CpuOut 'engine'))" }
}
}  # -NoCpu

# ---- CUDA lane: build\win\cuda ---------------------------------------------------------------
# XPLAT-WIN: everything CUDA goes to build\win\cuda (its own obj\, its own shim object), so this
# lane never overwrites the CPU lane's binaries. Two targets:
#   qwen35-cli-gpu.exe  the C twin with the device GEMV (--gpu / --selftest), nvcc + cl
#   engine_gpu.exe      the desktop CUDA engine: llb-engine.c emitted from engine_qwen35_gpu.bend
#                       by the patched compiler (tools/build_gpu.sh, as ./build.sh does on Linux),
#                       made Win64-safe by winabi.ps1, compiled by clang with -DBEND_CUDA=1 against
#                       the POSIX shim, linked with cuda.lib (nvcuda.dll, the driver) and nvrtc.lib
# and the contract dir build\win\cuda\engine\cuda\ (engine_gpu.exe + gguf2pack.exe + llb-tok.exe
# from the CPU lane + llb-serve.exe with -LlbServe + the NVRTC DLLs, which the driver does not
# install; backends/cuda/WINDOWS.md says why they may ship).
$CudaOut = Join-Path $Build 'win\cuda'
$CudaObj = Join-Path $CudaOut 'obj'

# The newest CUDA toolkit: -CudaPath, else CUDA_PATH, else Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*
function Find-Cuda {
  $cands = @()
  if ($CudaPath -ne '') { $cands += $CudaPath }
  if ($env:CUDA_PATH) { $cands += $env:CUDA_PATH }
  $pf = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
  if (Test-Path $pf) {
    $cands += @(Get-ChildItem $pf -Directory -Filter 'v*' |
      Sort-Object { [version]($_.Name.TrimStart('v') + '.0') } -Descending | ForEach-Object { $_.FullName })
  }
  foreach ($c in $cands) { if (Test-Path (Join-Path $c 'bin\nvcc.exe')) { return (Resolve-Path $c).Path } }
  return ''
}

# A toolkit DLL by pattern: CUDA 13 keeps them in bin\x64, CUDA 12 in bin
function Find-CudaDll([string] $cuda, [string] $pattern) {
  foreach ($d in @('bin\x64', 'bin')) {
    $p = Join-Path $cuda $d
    if (Test-Path $p) {
      $f = @(Get-ChildItem $p -Filter $pattern | Sort-Object Name -Descending)
      if ($f.Count -gt 0) { return $f[0].FullName }
    }
  }
  return ''
}

# nvcc's host compiler is cl.exe, which needs the VS developer environment. Outside "Developer
# PowerShell for VS 2022" import it from vcvars64.bat (found with vswhere) into this process.
function Import-VsEnv {
  if (Have 'cl') { return $true }
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) { return $false }
  $vs = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
  if (-not $vs) { return $false }
  $bat = Join-Path $vs.ToString().Trim() 'VC\Auxiliary\Build\vcvars64.bat'
  if (-not (Test-Path $bat)) { return $false }
  Write-Host "> vcvars64.bat ($bat) for nvcc's cl.exe"
  $lines = & cmd.exe /d /c "`"$bat`" >nul 2>&1 && set"
  foreach ($l in $lines) {
    if ($l -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
  }
  return (Have 'cl')
}

# Git for Windows' bash (tools/build_gpu.sh needs bash + GNU patch). Not plain "bash": on a
# machine with WSL, C:\Windows\System32\bash.exe is WSL's.
function Find-GitBash {
  $cands = @()
  if (Have 'git') {
    $ep = (& git --exec-path | Select-Object -First 1)
    if ($ep) { $cands += (Join-Path $ep.ToString().Trim() '..\..\..\bin\bash.exe') }
  }
  $cands += (Join-Path $env:ProgramFiles 'Git\bin\bash.exe')
  foreach ($c in $cands) { if (Test-Path $c) { return (Resolve-Path $c).Path } }
  return ''
}

# Copy the git-tracked files under $paths from the work tree to $dst with LF line ends (git's
# rule: a file with a NUL in its first 8000 bytes is binary and copied as is). XPLAT-WIN: the
# checkout may be core.autocrlf=true (CRLF), and the patch stack (--fuzz=0) and the emitted hash
# need the LF text Linux and macOS see. Work-tree content, not HEAD's: like ./build.sh, a local
# edit to host/ or a .bend file is built.
function Copy-Lf([string] $dst, [string[]] $paths) {
  $files = @(& git -C $Root ls-files -- $paths)
  if ($LASTEXITCODE -ne 0 -or $files.Count -eq 0) { return $false }
  $lf = [byte]10; $cr = [byte]13
  foreach ($f in $files) {
    $src = Join-Path $Root $f
    if (-not (Test-Path $src)) { continue }
    $out = Join-Path $dst $f
    $dir = Split-Path $out -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $b = [IO.File]::ReadAllBytes($src)
    $n = [Math]::Min($b.Length, 8000)
    $bin = $false
    for ($i = 0; $i -lt $n; $i++) { if ($b[$i] -eq 0) { $bin = $true; break } }
    if (-not $bin -and [Array]::IndexOf($b, $cr) -ge 0) {
      $ms = New-Object IO.MemoryStream($b.Length)
      for ($i = 0; $i -lt $b.Length; $i++) {
        if ($b[$i] -eq $cr -and $i + 1 -lt $b.Length -and $b[$i + 1] -eq $lf) { continue }
        $ms.WriteByte($b[$i])
      }
      $b = $ms.ToArray()
    }
    [IO.File]::WriteAllBytes($out, $b)
  }
  return $true
}

if ($NoCuda) {
  Write-Host 'CUDA: skipped (-NoCuda)'
} else {
  foreach ($d in @($CudaOut, $CudaObj)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null } }
  $Cuda = Find-Cuda
  if ($Cuda -eq '') {
    Write-Host 'CUDA: no toolkit (nvcc.exe) under -CudaPath, CUDA_PATH or Program Files\NVIDIA GPU Computing Toolkit; install it or pass -NoCuda'
    Record 'qwen35-cli-gpu.exe' 127 'CUDA toolkit not found'
    Record 'engine_gpu.exe' 127 'CUDA toolkit not found'
  } else {
    Write-Host "CUDA toolkit: $Cuda"
    $Nvcc = Join-Path $Cuda 'bin\nvcc.exe'
    # Linux tools/build_qwen35_cli.sh hardcodes sm_86; here it comes from nvidia-smi compute_cap.
    # (engine_gpu.exe needs no arch: NVRTC compiles for the card it finds, at run time.)
    if ($Arch -eq '') {
      $cap = ''
      if (Have 'nvidia-smi') {
        Write-Host '> nvidia-smi --query-gpu=compute_cap --format=csv,noheader'
        $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1)
        if ($null -eq $cap) { $cap = '' }
        $cap = $cap.ToString().Trim()
      }
      if ($cap -match '^(\d+)\.(\d+)$') { $Arch = 'sm_' + $Matches[1] + $Matches[2] }
      else { Write-Host "could not read compute_cap from nvidia-smi (got '$cap'); pass -Arch sm_XX" }
    }

    # -- qwen35-cli-gpu.exe: nvcc, host side through cl ----------------------------------
    $out = Join-Path $CudaOut 'qwen35-cli-gpu.exe'
    if (Test-Path $out) { Remove-Item $out -Force }
    if ($Arch -eq '') {
      Record 'qwen35-cli-gpu.exe' 1 'CUDA, arch unknown'
    } elseif (-not (Import-VsEnv)) {
      Record 'qwen35-cli-gpu.exe' 127 'nvcc needs cl.exe: install VS 2022 "Desktop development with C++"'
    } else {
      Write-Host "CUDA arch: $Arch"
      [void](Run $Nvcc @('--version'))
      # -fmad=false: the device side of the contract; host side through cl /fp:precise
      $a = @('-O3', '-fmad=false', '-std=c++14', "-arch=$Arch", '-Xcompiler', '/fp:precise',
             '-D_CRT_SECURE_NO_WARNINGS', '-o', $out, (Join-Path $Tools 'qwen35_engine.cu'))
      $rc = Run $Nvcc $a
      if ($rc -eq 0 -and -not (Test-Path $out)) { $rc = 1 }
      Record 'qwen35-cli-gpu.exe' $rc "CUDA $Arch"
    }

    # -- engine_gpu.exe ---------------------------------------------------------------------
    # 1. LF copies: the pinned compiler (git archive of pins/bend-240ecef4's bend2, tree checked as
    #    ./build.sh checks it) and this tree's .bend files, kernels, host and tools.
    # 2. tools/build_gpu.sh applies the patch stack (--fuzz=0) to a compiler copy. Its own emit then
    #    fails (bend2 looks for effs/ in the working directory, see BUILD.md), so it is run again with
    #    the copy's bend2 as the working directory: that run re-checks the whole stack and emits.
    # 3. winabi.ps1 (the Win64 Env/musttail fix), then clang -DBEND_CUDA=1 against the shim.
    $rc = 0
    $why = ''
    $Bash = Find-GitBash
    $Src = Join-Path $CudaOut 'src'
    $emitC = Join-Path $CudaOut 'llb-engine.c'
    $winC = Join-Path $CudaOut 'llb-engine.win.c'
    if ($Cc -ne 'clang') { $rc = 127; $why = 'needs clang (LLVM for Windows)' }
    elseif (-not (Have 'bun')) { $rc = 127; $why = 'needs bun (the Bend compiler runs on it)' }
    elseif ($Bash -eq '') { $rc = 127; $why = 'needs Git for Windows (bash + patch)' }
    if ($rc -eq 0) {
      foreach ($p in @($Src, $emitC, $winC)) { if (Test-Path $p) { Remove-Item $p -Recurse -Force } }
      New-Item -ItemType Directory -Path $Src | Out-Null
      $pin = ''
      foreach ($ref in @('pins/bend-240ecef4', 'origin/pins/bend-240ecef4')) {
        $t = (& git -C $Root rev-parse -q --verify "${ref}:bend2" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $t -eq 'aada4cbf6e04d0ade6a7e421ed9fa945a9504177') { $pin = $ref; break }
      }
      if ($pin -eq '') { $rc = 1; $why = 'no pins/bend-240ecef4 with bend2 tree aada4cbf (git fetch origin pins/bend-240ecef4:pins/bend-240ecef4)' }
    }
    if ($rc -eq 0) {
      $pinDir = Join-Path $Src 'bend-pinned'
      New-Item -ItemType Directory -Path $pinDir | Out-Null
      $tar = Join-Path $Src 'pin.tar'
      $rc = Run 'git' @('-C', $Root, '-c', 'core.autocrlf=false', 'archive', '-o', $tar, $pin, 'bend2')
      if ($rc -eq 0) { $rc = Run 'tar' @('-xf', $tar, '-C', $pinDir) }
      if (Test-Path $tar) { Remove-Item $tar -Force }
      if ($rc -ne 0) { $why = 'exporting the pinned compiler failed' }
    }
    $tree = Join-Path $Src 'tree'
    if ($rc -eq 0) {
      Write-Host "> LF copy of *.bend kernels host tools -> $tree"
      if (-not (Copy-Lf $tree @('*.bend', 'kernels', 'host', 'tools'))) { $rc = 1; $why = 'copying the engine tree failed' }
    }
    if ($rc -eq 0) {
      $fwd = { param($p) $p.Replace('\', '/') }
      $joint = Join-Path $Src 'bend-joint'
      $env:BEND_ROOT = & $fwd $pinDir
      $env:BEND_BUILD = & $fwd $joint
      $env:LLB_HOST_DIR = & $fwd (Join-Path $tree 'host')
      $gpuSh = & $fwd (Join-Path $tree 'tools\build_gpu.sh')
      $entry = & $fwd (Join-Path $tree 'engine_qwen35_gpu.bend')
      $outC = & $fwd $emitC
      Push-Location $Src
      try { [void](Run $Bash @($gpuSh, '--version')) } finally { Pop-Location }
      if (-not (Test-Path (Join-Path $joint 'bend2\main.ts'))) { $rc = 1; $why = 'the patch stack did not apply (tools/build_gpu.sh)' }
      else {
        Push-Location (Join-Path $joint 'bend2')
        try { $rc = Run $Bash @($gpuSh, $entry, '-o', $outC) } finally { Pop-Location }
        if ($rc -eq 0 -and -not (Test-Path $emitC)) { $rc = 1 }
        if ($rc -ne 0) { $why = 'emitting llb-engine.c failed' }
      }
      foreach ($v in @('BEND_ROOT', 'BEND_BUILD', 'LLB_HOST_DIR')) { Remove-Item "env:$v" -ErrorAction SilentlyContinue }
    }
    if ($rc -eq 0) {
      Write-Host ('llb-engine.c sha256 ' + (Get-FileHash $emitC -Algorithm SHA256).Hash.ToLower())
      $rc = Run 'powershell' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $WinSrc 'winabi.ps1'), '-In', $emitC, '-Out', $winC)
      if ($rc -ne 0) { $why = 'winabi.ps1 refused the emitted runtime' }
    }
    if ($rc -eq 0) {
      $rc = Run 'powershell' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $WinSrc 'wincuda.ps1'), '-In', $winC, '-Out', $winC)
      if ($rc -ne 0) { $why = 'wincuda.ps1 refused the emitted runtime' }
    }
    # the request objects ./build.sh links (same sources and flags) + the shim + the plain-C cblas
    $objs = @()
    if ($rc -eq 0) {
      $nonstd = '-D_CRT_DECLARE_NONSTDC_NAMES=0'
      $shimObj = Join-Path $CudaObj 'llbw_posix.o'
      $rc = Run $ClangExe @('-std=c11', '-O2', $nonstd, '-Wall', '-c', (Join-Path $Compat 'llbw_posix.c'), '-o', $shimObj)
      $objs += $shimObj
      $units = @('sys1dual\dual.c', 'metrics\metrics.c', 'metrics\tokstream.c', 'glue\glue_out.c', 'clef3\head.c',
                 'clef3\weights.c', 'clef3\spans.c', 'clef3\json.c', 'qwen35_tokenizer.c')
      foreach ($u in $units) {
        if ($rc -ne 0) { break }
        $o = Join-Path $CudaObj ([IO.Path]::GetFileNameWithoutExtension($u) + '.o')
        $rc = Run $ClangExe @('-D_POSIX_C_SOURCE=200809L', '-std=c11', '-O2', '-ffp-contract=off', '-Wall', '-Wextra',
                              '-Wno-unused-function', '-D_CRT_SECURE_NO_WARNINGS', $nonstd, '-I', $Compat, '-c', (Join-Path $Tools $u), '-o', $o)
        $objs += $o
      }
      if ($rc -eq 0) {
        $o = Join-Path $CudaObj 'cblas_sgemm.o'
        $rc = Run $ClangExe @('-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math', '-D_CRT_SECURE_NO_WARNINGS', '-c', (Join-Path $WinSrc 'cblas_sgemm.c'), '-o', $o)
        $objs += $o
      }
      if ($rc -ne 0) { $why = 'a request object failed to compile' }
    }
    $out = Join-Path $CudaOut 'engine_gpu.exe'
    if (Test-Path $out) { Remove-Item $out -Force }
    if ($rc -eq 0) {
      # tools/remap_cc.sh's flags, less its Linux-only glibc234.h / rpath; -include llbw_cuda.h adds
      # the Windows CUDA pieces (backends/windows/compat/llbw_cuda.h)
      $rtDir = (& $ClangExe --print-resource-dir | Select-Object -First 1).ToString().Trim()
      $rtLib = Join-Path $rtDir 'lib\windows\clang_rt.builtins-x86_64.lib'
      $cl = Join-Path $Cuda 'lib\x64'
      $o = Join-Path $CudaObj 'llb-engine.o'
      $rc = Run $ClangExe @('-DBEND_CUDA=1', '-I', (Join-Path $Cuda 'include'), '-std=c11', '-O3', '-ffp-contract=off',
                            '-D_CRT_SECURE_NO_WARNINGS', '-D_CRT_DECLARE_NONSTDC_NAMES=0', '-I', $Compat,
                            '-Wno-deprecated-declarations', '-c', $winC, '-o', $o)
      if ($rc -eq 0) {
        $rc = Run $ClangExe (@('-O3', '-o', $out, $o) + $objs + @($rtLib, (Join-Path $cl 'cuda.lib'), (Join-Path $cl 'nvrtc.lib'),
                              '-lws2_32', '-lwinmm', '-Wl,/STACK:8388608'))
      }
      if ($rc -eq 0 -and -not (Test-Path $out)) { $rc = 1 }
      if ($rc -ne 0) { $why = 'compiling llb-engine.win.c failed' }
    }
    # NVRTC: engine_gpu compiles its device program at run time. nvcuda.dll comes with the driver;
    # nvrtc64_*.dll and nvrtc-builtins64_*.dll do not, so they go beside the exe.
    $dlls = @()
    if ($rc -eq 0) {
      foreach ($pat in @('nvrtc64_*.dll', 'nvrtc-builtins64_*.dll')) {
        $d = Find-CudaDll $Cuda $pat
        if ($d -eq '') { $rc = 1; $why = "no $pat in the toolkit" } else { Copy-Item $d $CudaOut -Force; $dlls += (Split-Path $d -Leaf) }
      }
    }
    Record 'engine_gpu.exe' $rc $(if ($rc -eq 0) { 'Bend CUDA engine, NVRTC: ' + ($dlls -join ', ') } else { $why })

    # Inkling 2 Mini native CUDA shape engine (optional when cmake + inkling2/ + VS are present)
    $InkSrc = Join-Path $Root 'inkling2\engine\src\inkling2'
    if ((Test-Path (Join-Path $InkSrc 'CMakeLists.txt')) -and (Have 'cmake') -and (Import-VsEnv)) {
      $inkCudaBuild = Join-Path $CudaOut 'inkling-build'
      $rcInk = Run 'cmake' @('-S', $InkSrc, '-B', $inkCudaBuild, '-DCMAKE_BUILD_TYPE=Release', '-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_INK_CUDA_ARCHS=75;80;86;89;120')
      if ($rcInk -eq 0) { $rcInk = Run 'cmake' @('--build', $inkCudaBuild, '--config', 'Release', '--target', 'strata-inkling', '--parallel') }
      $inkBin = @(Join-Path $inkCudaBuild 'Release\strata-inkling.exe'; Join-Path $inkCudaBuild 'strata-inkling.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1
      if ($rcInk -eq 0 -and $inkBin) { Copy-Item $inkBin (Join-Path $CudaOut 'engine_inkling_gpu.exe') -Force }
      Record 'engine_inkling_gpu.exe' $(if ($inkBin) { $rcInk } else { 1 }) 'Inkling 2 Mini CUDA'
    }

    # the desktop contract's cuda\ dir (bend-app desktop/engine/build.sh stages EUREKA_WINDOWS_ENGINE\cuda)
    $stage = Join-Path $CudaOut 'engine\cuda'
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    New-Item -ItemType Directory -Path $stage | Out-Null
    foreach ($n in @('engine_gpu.exe', 'engine_inkling_gpu.exe') + $dlls) {
      $p = Join-Path $CudaOut $n
      if (Test-Path $p) { Copy-Item $p (Join-Path $stage $n) -Force }
    }
    foreach ($n in @('gguf2pack.exe', 'llb-tok.exe')) {
      $p = Join-Path $CpuOut $n
      if (Test-Path $p) { Copy-Item $p (Join-Path $CudaOut $n) -Force; Copy-Item $p (Join-Path $stage $n) -Force }
    }
    if ($LlbServe -ne '' -and (Test-Path $LlbServe)) { Copy-Item $LlbServe (Join-Path $stage 'llb-serve.exe') -Force }
    $have = @(Get-ChildItem $stage -Filter '*.exe' | ForEach-Object { $_.Name })
    $lack = @(@('llb-serve.exe', 'engine_gpu.exe', 'gguf2pack.exe', 'llb-tok.exe') | Where-Object { $have -notcontains $_ })
    if ($lack.Count -gt 0) { Write-Host ("contract dir ${stage}: missing " + ($lack -join ', ')) }
    else { Write-Host "contract dir ${stage}: complete (EUREKA_WINDOWS_ENGINE=$(Join-Path $CudaOut 'engine'))" }
  }
}

# ---- summary ----------------------------------------------------------------
$order = @('qwen35-cli.exe', 'qwen35-cli-gpu.exe', 'gguf2pack.exe', 'llbw_posix.o (POSIX shim)', 'test_shim.exe', 'llb-tok.exe', 'engine_cpu.exe', 'engine_gpu.exe')
$sorted = @($results | Sort-Object { [array]::IndexOf($order, $_.name) })
$ok = @($sorted | Where-Object { $_.ok } | ForEach-Object { $_.text })
$bad = @($sorted | Where-Object { -not $_.ok } | ForEach-Object { $_.text })
Write-Host ''
if ($ok.Count -gt 0) { Write-Host ('built: ' + ($ok -join ' | ')) } else { Write-Host 'built: nothing' }
if ($bad.Count -gt 0) { Write-Host ('FAILED: ' + ($bad -join ' | ')); exit 1 }
exit 0
