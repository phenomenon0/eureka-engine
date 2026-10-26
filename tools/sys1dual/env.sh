#!/usr/bin/env bash
# SYS1-DUAL scratch: everything the build and server write lives under $W.
export W=${SYS1_WORK:-$PWD/build/sys1dual}
export TMPDIR=$W/tmp TMP=$W/tmp TEMP=$W/tmp
export XDG_CACHE_HOME=$W/cache CUDA_CACHE_PATH=$W/cache/cuda
export BUN_INSTALL_CACHE_DIR=$W/cache/bun PYTHONDONTWRITEBYTECODE=1
export CCACHE_DISABLE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export SYS1_GGUF=${SYS1_GGUF:-} SYS1_SIDECAR=${SYS1_SIDECAR:-$W/sidecar}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$BUN_INSTALL_CACHE_DIR"
# Turbo default (RETROHARVEST 2026-10-09): Clef fast prefill ON unless CLEF_CANON=1 (or LLB_PREFILL already set). Gate: retroharvest/clefgate (360 decisions, 99.17% top-1, 15.4x).
[ "${CLEF_CANON:-0}" = 1 ] && unset LLB_PREFILL || export LLB_PREFILL="${LLB_PREFILL-fast}"
# SHIP45 (owner 10-16): the 9B clef-flash fast tier needs the Q8_0 shadow of its F32 ssm_alpha/beta
# (LLB_PF_SSMQ8=1, docs/reference/decide-configuration.md); without it fast falls back to exact (S6-GUARD). Inert when
# LLB_PREFILL is unset; LLB_PF_SSMQ8=0 opts out.
[ "${CLEF_CANON:-0}" = 1 ] && unset LLB_PF_SSMQ8 || export LLB_PF_SSMQ8="${LLB_PF_SSMQ8-1}"
# QUANTS-AUDIT (PREPARED 2026-10-16, owner-gated, NOT merged): mode-9 decode on the 9B clef-flash. KEPT +11.66 %
# [+10.76, +12.58] (BEST-AR-VERDICTS Q1 r2), KL half PASS (KL ratio 1.0001x, drift 0.00115 vs 0.696), decide battery G4:
# PASS. Not bit-identical to exact for generation. CLEF_CANON=1 unsets it
# (and ref.sh), LLB_Q4_DP4A=0 opts out.
[ "${CLEF_CANON:-0}" = 1 ] && unset LLB_Q4_DP4A || export LLB_Q4_DP4A="${LLB_Q4_DP4A-9}"
