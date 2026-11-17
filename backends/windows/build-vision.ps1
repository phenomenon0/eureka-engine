# backends/windows/build-vision.ps1 - the CUDA vision tower for Windows: vision/cudavis-cli.c + vitc.cu -> cudavis-cli.exe
# (image -> Qwen3.5 tower rows, the engine's "images" request form). XPLAT-WIN.
#
#   powershell -ExecutionPolicy Bypass -File backends\windows\build-vision.ps1 [-CudaPath D:\cuda\v13.3] [-Out build\win\vision]
#
# Built with -DVITC_NO_CUBLAS: the tower's one GEMM is vitc.cu's own WMMA kernel, so no cuBLAS DLLs (~500 MB) ship.
# Needs the CUDA toolkit with nvcc, cl.exe from VS 2022 (vcvars64.bat, found with vswhere)
# for nvcc, and LLVM clang for the C host files (built against backends/windows/compat, as engine_gpu is).
# The exe links cudart statically and needs only the driver (nvcuda.dll).
param(
  [string] $CudaPath = '',
  [string] $Out = '',
  [string] $Arch = ''
)
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Vis = Join-Path $Root 'vision'
$Compat = Join-Path $Root 'backends\windows\compat'
if (-not $Out) { $Out = Join-Path $Root 'build\win\vision' }
$Obj = Join-Path $Out 'obj'
New-Item -ItemType Directory -Force $Obj | Out-Null
if (-not $CudaPath) { $CudaPath = $env:CUDA_PATH }
if (-not $CudaPath -or -not (Test-Path (Join-Path $CudaPath 'bin\nvcc.exe'))) { throw "no CUDA toolkit with nvcc (give -CudaPath)" }
$Clang = 'clang'
$pf = Join-Path $env:ProgramFiles 'LLVM\bin\clang.exe'
if (Test-Path $pf) { $Clang = $pf }
if (-not $Arch) { $cc = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim(); $Arch = 'sm_' + $cc.Replace('.', '') }

function Run([string] $exe, [string[]] $a) {
  Write-Host ("> " + $exe + " " + ($a -join ' '))
  & $exe @a
  if ($LASTEXITCODE -ne 0) { throw "$exe exited $LASTEXITCODE" }
}

$cf = @('-D_POSIX_C_SOURCE=200809L', '-std=c11', '-O3', '-ffp-contract=off', '-D_CRT_SECURE_NO_WARNINGS',
        '-D_CRT_DECLARE_NONSTDC_NAMES=0', '-Wno-deprecated-declarations', '-I', $Compat, '-I', $Vis, '-I', (Join-Path $Root 'host'))
Run $Clang (@('-std=c11', '-O2', '-D_CRT_DECLARE_NONSTDC_NAMES=0', '-c', (Join-Path $Compat 'llbw_posix.c'), '-o', (Join-Path $Obj 'llbw_posix.o')))
foreach ($u in 'vpre', 'vmm', 'cudavis_cli') {
  Run $Clang ($cf + @('-w', '-c', (Join-Path $Vis "$u.c"), '-o', (Join-Path $Obj "$u.o")))
}
# nvcc's host compiler is cl.exe: import vcvars64 into this process when cl is not on PATH (after the clang compiles:
# vcvars' INCLUDE gives clang the SDK's winsock headers, which the shim replaces)
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  $bat = Join-Path $vs.ToString().Trim() 'VC\Auxiliary\Build\vcvars64.bat'
  cmd /c "`"$bat`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] } }
}
$nv = Join-Path $CudaPath 'bin\nvcc.exe'
Run $nv @('-O3', '-fmad=false', "-arch=$Arch", '-std=c++17', '-DVITC_NO_CUBLAS', '-I', $Vis, '-I', (Join-Path $Root 'host'), '-c', (Join-Path $Vis 'vitc.cu'), '-o', (Join-Path $Obj 'vitc.obj'))
$exe = Join-Path $Out 'cudavis-cli.exe'
$rt = Get-ChildItem (Join-Path $env:ProgramFiles 'LLVM\lib\clang') -Recurse -Filter 'clang_rt.builtins-x86_64.lib' | Select-Object -First 1
Run $Clang @('-O3', '-o', $exe, (Join-Path $Obj 'cudavis_cli.o'), (Join-Path $Obj 'vpre.o'), (Join-Path $Obj 'vmm.o'), (Join-Path $Obj 'vitc.obj'),
  (Join-Path $Obj 'llbw_posix.o'), $rt.FullName, (Join-Path $CudaPath 'lib\x64\cudart_static.lib'),
  '-lws2_32', '-lwinmm', '-ladvapi32', '-Wl,/STACK:8388608')
Write-Host "built: $exe"
