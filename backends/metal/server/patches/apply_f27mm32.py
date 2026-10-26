#!/usr/bin/env python3
"""apply_f27mm32.py IN OUT (CHUNKF-F32): the emitted C -> the fast chunk's batched GEMM gets an F32 arm, and the CHUNKF arm's
format gate takes F32 ssm_beta / ssm_alpha (linear slots 7, 8) as the exact PASS arm already does (SYS1-MAC). clef 9B carries
those two as F32 in every linear layer; without this its default (--fastpf) tier exits on the first chunk ("warp hybrid:
unsupported operands", bang fid 295). The arm is f27_mm<32>: same 128-thread tile, 16 floats a thread a K step read as 4 float4
from the F32 row (A.rb = K * 4), so the accumulation is the Q8_0 arm's with no dequant. A model with no F32 slots 7 / 8 never
selects it (f27_mms picks the pipeline by format). Idempotent: an already-patched IN is written out unchanged."""
import sys
src, out = sys.argv[1:3]
S = open(src).read()

H = []
def hunk(old, new):
    H.append((old, new))

# 1. the template arm, before f27_mm's final (Q6_K) else
# (the f35_mm template has the same K-quant chain but no QT == 0 arm: the Q8_0 arm's tail makes the anchor f27_mm's)
Q8T = '''        v[i] = (float)as_type<char4>(qw[i / 4])[i % 4] * d;
      }
    } else if (QT == 4) {
      f27_deq4(wr + (k >> 8) * 144, (short)((k & 255) + 16 * il0), v);
'''
hunk(Q8T + '''    } else if (QT == 5) {
      f27_deq5(wr + (k >> 8) * 176, (short)((k & 255) + 16 * il0), v);
    } else {
      f27_deq6(wr + (k >> 8) * 210, (short)((k & 255) + 16 * il0), v);
    }''', Q8T + '''    } else if (QT == 5) {
      f27_deq5(wr + (k >> 8) * 176, (short)((k & 255) + 16 * il0), v);
    } else if (QT == 32) {
      // (CHUNKF-F32) an F32 row: A.rb = K * 4, 16 columns from k + 16 * il0
      device const float4* fp = (device const float4*)(wr + (u64)(k + 16 * il0) * 4);
      const float4 w0 = fp[0], w1 = fp[1], w2 = fp[2], w3 = fp[3];
      v[0] = w0.x; v[1] = w0.y; v[2] = w0.z; v[3] = w0.w;
      v[4] = w1.x; v[5] = w1.y; v[6] = w1.z; v[7] = w1.w;
      v[8] = w2.x; v[9] = w2.y; v[10] = w2.z; v[11] = w2.w;
      v[12] = w3.x; v[13] = w3.y; v[14] = w3.z; v[15] = w3.w;
    } else {
      f27_deq6(wr + (k >> 8) * 210, (short)((k & 255) + 16 * il0), v);
    }''')

# 2. the kernel, beside its siblings (inside BEND_SRC: the file is its own #embed)
mmk5 = '''// (lane m31) Q5_K
kernel void llb_f27_mmk5(device u8* C [[buffer(0)]], device u8* S [[buffer(1)]],
  constant FMArgs& A [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]],
  ushort ti [[thread_index_in_threadgroup]],
  ushort sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float shm[3072];
  f27_mm<5>(C, S, A, tg, ti, sg, shm);
}
'''
hunk(mmk5, mmk5 + '''
// (CHUNKF-F32) F32 (clef 9B's ssm_beta / ssm_alpha)
kernel void llb_f27_mm32(device u8* C [[buffer(0)]], device u8* S [[buffer(1)]],
  constant FMArgs& A [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]],
  ushort ti [[thread_index_in_threadgroup]],
  ushort sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float shm[3072];
  f27_mm<32>(C, S, A, tg, ti, sg, shm);
}
''')

# 3. the pipeline slot
hunk('static id<MTLComputePipelineState> gpu_f27_mm[4], gpu_f27_gdl,',
     'static id<MTLComputePipelineState> gpu_f27_mm[5], gpu_f27_gdl,')
hunk('''  gpu_f27_mm[3] = warp_pipe("llb_f27_mmk5");
''', '''  gpu_f27_mm[3] = warp_pipe("llb_f27_mmk5");
  gpu_f27_mm[4] = warp_pipe("llb_f27_mm32");  // (CHUNKF-F32)
''')
hunk('''    || gpu_f27_mm[3].maxTotalThreadsPerThreadgroup < 128
''', '''    || gpu_f27_mm[3].maxTotalThreadsPerThreadgroup < 128
    || gpu_f27_mm[4].maxTotalThreadsPerThreadgroup < 128
''')

# 4. the dispatch: F32 = pipeline 4, no superblock K constraint
hunk('''  } else if (fm == LLB_FM_Q5_K) {
    f = 3;
  } else {
    f = 4;
  }
  bool ok = f < 4 && ns >= 1 && ns <= 4 && K % 32 == 0 && (x0 & 15) == 0
    && rb == LLB_FM_RB(fm, K) && (f == 0 || K % 256 == 0);''', '''  } else if (fm == LLB_FM_Q5_K) {
    f = 3;
  } else if (fm == LLB_FM_F32) {
    f = 4;  // (CHUNKF-F32)
  } else {
    f = 5;
  }
  bool ok = f < 5 && ns >= 1 && ns <= 4 && K % 32 == 0 && (x0 & 15) == 0
    && rb == LLB_FM_RB(fm, K) && (f == 0 || f == 4 || K % 256 == 0);''')

# 5. the CHUNKF arm's format gate: the PASS arm's SYS1-MAC clause
hunk('''          ok = ok && (!(has >> f & 1) || (gv >> f & 1
            ? fm == LLB_FM_Q8_0 || LLB_FM_KQ(fm) : fm == LLB_FM_F32));''', '''          // (CHUNKF-F32) Clef's F32 ssm_beta / ssm_alpha (linear slots 7, 8):
          // f27_mms's F32 arm (llb_f27_mm32) serves them
          ok = ok && (!(has >> f & 1) || (gv >> f & 1
            ? fm == LLB_FM_Q8_0 || LLB_FM_KQ(fm)
              || (!full && (f == 7 || f == 8) && fm == LLB_FM_F32)
            : fm == LLB_FM_F32));''')

# 6. bend_close: the fifth pipeline back to nil with the others
hunk('''  for (u32 k = 0; k < 4; k += 1) {
    gpu_kx[k] = nil;
    gpu_f27_mm[k] = nil;
  }
''', '''  for (u32 k = 0; k < 4; k += 1) {
    gpu_kx[k] = nil;
    gpu_f27_mm[k] = nil;
  }
  gpu_f27_mm[4] = nil;  // (CHUNKF-F32)
''')

if 'llb_f27_mm32' in S:
    for old, new in H:
        assert S.count(new) == 1, ('partially patched', S.count(new), new[:80])
    print('apply_f27mm32: already applied, unchanged')
else:
    for old, new in H:
        assert S.count(old) == 1, (S.count(old), old[:80])
        S = S.replace(old, new)
open(out, 'w').write(S)
