# backends/windows/build.ps1 - build the engine CLI on native Windows (x64).
#
#   build\qwen35-cli.exe      CPU twin   (clang or MSVC cl)
#   build\gguf2pack.exe       GGUF -> pack converter
#   build\qwen35-cli-gpu.exe  CUDA (--gpu), if nvcc is on PATH; arch from nvidia-smi
#
# UNTESTED ON WINDOWS: written on macOS, never run, never parsed by PowerShell.
# See backends/windows/BUILD.md and WINDOWS-TEST-STATUS.md.
#
# Usage (from the repo root, in "Developer PowerShell for VS 2022" so cl.exe is on PATH):
#   powershell -ExecutionPolicy Bypass -File backends\windows\build.ps1 [-Cc auto|clang|cl] [-Arch sm_XX] [-NoCuda]
# Every run rebuilds all three into build\ (safe to re-run). Prints each command it runs.

param(
  [ValidateSet('auto', 'clang', 'cl')] [string] $Cc = 'auto',
  [string] $Arch = '',
  [switch] $NoCuda
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
# Numerical contract: no FMA contraction, no fast-math (Linux: -ffp-contract=off -fno-fast-math).
# clang: the same flags. cl: /fp:precise, and no /fp:contract (VS 2022 does not contract under
# /fp:precise unless /fp:contract is given); no /arch:AVX2 (lets VS 2019 contract).
if ($Cc -eq 'auto') {
  if (Have 'clang') { $Cc = 'clang' } elseif (Have 'cl') { $Cc = 'cl' } else { $Cc = 'none' }
}
Write-Host "CPU compiler: $Cc"

$cpuSrc = @(
  @{ name = 'qwen35-cli.exe'; src = (Join-Path $Tools 'qwen35_engine.c'); fp = $true; what = 'CPU' },
  @{ name = 'gguf2pack.exe';  src = (Join-Path $Tools 'gguf2pack.c');     fp = $false; what = '' }
)
foreach ($t in $cpuSrc) {
  $out = Join-Path $Build $t.name
  if (Test-Path $out) { Remove-Item $out -Force }
  if ($Cc -eq 'clang') {
    $a = @('-O2', '-std=c11', '-D_CRT_SECURE_NO_WARNINGS')
    if ($t.fp) { $a += @('-ffp-contract=off', '-fno-fast-math') }
    $a += @($t.src, '-o', $out)
    $rc = Run 'clang' $a
  } elseif ($Cc -eq 'cl') {
    $obj = Join-Path $Build ''
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

# ---- CUDA: nvcc, arch detected from the card --------------------------------
# Linux tools/build_qwen35_cli.sh hardcodes sm_86; here it comes from nvidia-smi compute_cap.
if ($NoCuda) {
  Write-Host 'CUDA: skipped (-NoCuda)'
} elseif (-not (Have 'nvcc')) {
  Write-Host 'CUDA: nvcc not on PATH; install the CUDA Toolkit or pass -NoCuda'
  Record 'qwen35-cli-gpu.exe' 127 'CUDA, nvcc not found'
} else {
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
  if ($Arch -eq '') {
    Record 'qwen35-cli-gpu.exe' 1 'CUDA, arch unknown'
  } else {
    Write-Host "CUDA arch: $Arch"
    [void](Run 'nvcc' @('--version'))
    $out = Join-Path $Build 'qwen35-cli-gpu.exe'
    if (Test-Path $out) { Remove-Item $out -Force }
    # -fmad=false: the device side of the contract; host side through cl /fp:precise
    $a = @('-O3', '-fmad=false', '-std=c++14', "-arch=$Arch", '-Xcompiler', '/fp:precise',
           '-D_CRT_SECURE_NO_WARNINGS', '-o', $out, (Join-Path $Tools 'qwen35_engine.cu'))
    $rc = Run 'nvcc' $a
    if ($rc -eq 0 -and -not (Test-Path $out)) { $rc = 1 }
    Record 'qwen35-cli-gpu.exe' $rc "CUDA $Arch"
  }
}

# ---- summary ----------------------------------------------------------------
$order = @('qwen35-cli.exe', 'qwen35-cli-gpu.exe', 'gguf2pack.exe')
$sorted = @($results | Sort-Object { [array]::IndexOf($order, $_.name) })
$ok = @($sorted | Where-Object { $_.ok } | ForEach-Object { $_.text })
$bad = @($sorted | Where-Object { -not $_.ok } | ForEach-Object { $_.text })
Write-Host ''
if ($ok.Count -gt 0) { Write-Host ('built: ' + ($ok -join ' | ')) } else { Write-Host 'built: nothing' }
if ($bad.Count -gt 0) { Write-Host ('FAILED: ' + ($bad -join ' | ')); exit 1 }
exit 0
