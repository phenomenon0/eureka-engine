# edit_catchup.py COMP.TS : the reports/cuda-catchup edits to the compiler's comp.ts (after every
# earlier patch of tools/build_gpu.sh); mkpatch.sh diffs the result into tools/payload-catchup.patch
import sys
p = sys.argv[1]; s = open(p).read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:100], s.count(old))
    s = s.replace(old, new)

# 1. pass_mail: fl word 47 bit 28 is the decode run (count in bits 0..27): s = 0, o->bsn bit 31 | n
rep('''  u32  w47 = (u32)H[a + 47], s = w47 >> 30 & 1 ? 0 : w47, k = (u32)H[a + 48], dT = c & 31, pos = c >> 5;''',
'''  // bit 28 (reports/cuda-catchup): gen's decode run, the greedy steps left in bits 0..27; s = 0
  u32  w47 = (u32)H[a + 47], s = w47 >> 28 & 5 ? 0 : w47, k = (u32)H[a + 48], dT = c & 31, pos = c >> 5;''')
rep('''  o->bsn = w47 >> 30 & 1 ? (w47 >> 29 & 1 ? 2 + (w47 & 0xffffu) : 1) : 0;
  M->sh = sh;''',
'''  o->bsn = w47 >> 30 & 1 ? (w47 >> 29 & 1 ? 2 + (w47 & 0xffffu) : 1)
    : w47 >> 28 & 1 ? 0x80000000u | (w47 & 0x0fffffffu) : 0;
  M->sh = sh;''')

# 2. llb_embt: the embedding row of the token the device picked (y word 1, an exact F32)
rep('''static CUfunction gpu_embed, gpu_embedk, gpu_embed6;''',
'''static CUfunction gpu_embed, gpu_embedk, gpu_embed6;
static CUfunction gpu_embt;  // tools/payload-catchup.patch: the decode run's embedding''')
rep('''    || cuModuleGetFunction(&gpu_embed6, gpu_lib, "llb_embed6") != CUDA_SUCCESS''',
'''    || cuModuleGetFunction(&gpu_embed6, gpu_lib, "llb_embed6") != CUDA_SUCCESS
    || cuModuleGetFunction(&gpu_embt, gpu_lib, "llb_embt") != CUDA_SUCCESS''')
i = s.index('extern "C" __global__ void llb_embed6(')
j = s.index('\n}\n', i) + 3
rep(s[i:j], s[i:j] + '''
// reports/cuda-catchup: the decode run's embedding. The token is t[1], the device's pick of the
// step before (llb_amax's word 1, an exact F32 id), so the host queues the step before it knows
// the id; the row is base + id * rb, fm its format, and each element is llb_embed / llb_embedk /
// llb_embed6's (the same f32 operations on the same bytes)
extern "C" __global__ void llb_embt(const u8* __restrict__ base, u32 rb, u32 fm,
  const float* __restrict__ t, u32 c, float* __restrict__ y) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c) {
    return;
  }
  const u8* row = base + (u64)(u32)t[1] * rb;
  if (fm == LLB_FM_Q4_K) {
    const u8* b = row + (i >> 8) * 144;
    const u8* sc = b + 4;
    u32 g = (i >> 6) & 3, hi = (i >> 5) & 1, l = i & 31, j = 2 * g + hi;
    u32 sd = j < 4 ? sc[j] & 63 : (sc[j + 4] & 15) | (sc[j - 4] >> 6) << 4;
    u32 sm = j < 4 ? sc[j + 4] & 63 : (sc[j + 4] >> 4) | (sc[j] >> 6) << 4;
    u32 q = b[16 + 32 * g + l];
    float d1 = __fmul_rn(llb_f16(b[0] | b[1] << 8), (float)sd);
    float m1 = __fmul_rn(llb_f16(b[2] | b[3] << 8), (float)sm);
    y[i] = __fsub_rn(__fmul_rn(d1, (float)(hi ? q >> 4 : q & 15)), m1);
  } else if (fm == LLB_FM_Q6_K) {
    const u8* B = row + (u64)(i >> 8) * 210;
    u32 r = i & 255, n = r >> 7, rr = r & 127, quad = rr >> 5, l = rr & 31;
    u32 ql = B[n * 64 + (quad & 1) * 32 + l], qh = B[128 + n * 32 + l];
    u32 nsh = quad >= 2 ? 4u : 0u, qsh = 2 * quad;
    int scv = (int)(signed char)B[192 + n * 8 + quad * 2 + (l >> 4)];
    float ds = __fmul_rn(llb_f16(B[208] | B[209] << 8), (float)scv);
    y[i] = __fmul_rn(ds, (float)(int)(((ql >> nsh) & 15) | (((qh >> qsh) & 3) << 4)) - 32.0f);
  } else {
    y[i] = __fmul_rn((float)(signed char)row[i],
      llb_f16(*(const unsigned short*)(row + c + 2 * (i >> 5))));
  }
}
''')

# 3. kp_pair: a partial last quad (rows of whole superblocks, cg % 64 == 0, not whole quads): the
# lanes of a superblock past the row take zero weights and multiply the real x, as the tile body
# (warp_dq4k / warp_dq6k: w = 0 past cg) does, so the leaves, the quad's tree and the carry are the
# tile body's; the quads past it stay pvs's analytic pads. The tile body's rows (cg % 256 != 0,
# d 9..11) move to the pair body; the residual store (KpRes) then takes them too
rep('''  const u32 l = threadIdx.x & 31, w = threadIdx.x >> 5;
  const u32 npr = cg >> 8;
  for (u32 p = npr + w; p < NP; p += blockDim.x >> 5) {''', '''  const u32 l = threadIdx.x & 31, w = threadIdx.x >> 5;
  const u32 npr = (cg + 255) >> 8, nsb = cg >> 6;
  for (u32 p = npr + w; p < NP; p += blockDim.x >> 5) {''')
rep('''        typename F::Raw rw[R];
        #pragma unroll
        for (u32 j = 0; j < R; j++) rw[j] = F::template ld<LP>(row[j] + (u64)(4 * p + sbk) * F::SB, l);
        float v[R];
        #pragma unroll
        for (u32 j = 0; j < R; j++) {
          float wl[16], wh[16];
          F::dec(rw[j], l, wl, wh);
          v[j] = kp_tlo<F::KIN>(kp_span(wl, xl), kp_span(wh, xh));
        }''', '''        typename F::Raw rw[R];
        const bool live = 4 * p + sbk < nsb;
        if (live) {
          #pragma unroll
          for (u32 j = 0; j < R; j++) rw[j] = F::template ld<LP>(row[j] + (u64)(4 * p + sbk) * F::SB, l);
        }
        float v[R];
        #pragma unroll
        for (u32 j = 0; j < R; j++) {
          float wl[16], wh[16];
          if (live) {
            F::dec(rw[j], l, wl, wh);
          } else {
            #pragma unroll
            for (u32 i = 0; i < 16; i++) wl[i] = wh[i] = 0.f;
          }
          v[j] = kp_tlo<F::KIN>(kp_span(wl, xl), kp_span(wh, xh));
        }''')
rep('''  if (cg % 256 == 0 && d >= 8 && d <= 13) {
    kp_warp<4, 1>(W, rb, cg, x, y, n, d, blockIdx.x, gridDim.x, nullptr);
  } else {
    warp_dq4k(W, rb, cg, x, y, n, d);
  }''', '''  if (cg % 64 == 0 && d >= 8 && d <= 13) {
    kp_warp<4, 1>(W, rb, cg, x, y, n, d, blockIdx.x, gridDim.x, nullptr);
  } else {
    warp_dq4k(W, rb, cg, x, y, n, d);
  }''')
rep('''  if (cg % 256 == 0 && d >= 8 && d <= 13) {
    kp_warp<6, 0>(W, rb, cg, x, y, n, d, blockIdx.x, gridDim.x, nullptr);
  } else {
    warp_dq6k(W, rb, cg, x, y, n, d);
  }''', '''  if (cg % 64 == 0 && d >= 8 && d <= 13) {
    kp_warp<6, 0>(W, rb, cg, x, y, n, d, blockIdx.x, gridDim.x, nullptr);
  } else {
    warp_dq6k(W, rb, cg, x, y, n, d);
  }''')


# 4. llb_gemv_anyk: llb_gemv_any's K-quant rows (Q4_K, Q6_K, Q5_K) on moe_dotn (reports/cuda-moe-speed:
# moe_dot's bits, which are llb_gemv_any's leaf and order: lane l sums groups l, l + 32, .. then the
# xor butterfly), two rows a warp sharing x, loads first; llb_gany takes it for those formats
rep('''// one block of 1024: lg the ne router logits, sg the shared expert's gate logit, pv ne floats of''',
'''extern "C" __global__ void __launch_bounds__(128) llb_gemv_anyk(
  const u8* __restrict__ W, u32 rb, u32 fm, u32 cg, const float* __restrict__ x,
  float* __restrict__ y, u32 n) {
  const u32 l = threadIdx.x & 31, nw = gridDim.x * (blockDim.x >> 5);
  for (u32 r0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 2; r0 < n; r0 += nw * 2) {
    const u8* rw[2] = { W + (u64)r0 * rb, W + (u64)(r0 + 1 < n ? r0 + 1 : r0) * rb };
    const float* xs[1] = { x };
    float v[2];
    if (fm == 3) moe_dotn<3, 2, true, 2>(rw, xs, cg, v);
    else if (fm == 4) moe_dotn<4, 2, true, 2>(rw, xs, cg, v);
    else moe_dotn<9, 2, true, 2>(rw, xs, cg, v);
    if (l == 0) y[r0] = v[0];
    if (l == 1 && r0 + 1 < n) y[r0 + 1] = v[1];
  }
}

// one block of 1024: lg the ne router logits, sg the shared expert's gate logit, pv ne floats of''')
rep('''static CUfunction gpu_embt;  // tools/payload-catchup.patch: the decode run's embedding''',
'''static CUfunction gpu_embt;  // tools/payload-catchup.patch: the decode run's embedding
static CUfunction gpu_ganyk;  // tools/payload-catchup.patch: llb_gemv_any's K-quant rows on moe_dotn''')
rep('''    || cuModuleGetFunction(&gpu_embt, gpu_lib, "llb_embt") != CUDA_SUCCESS''',
'''    || cuModuleGetFunction(&gpu_embt, gpu_lib, "llb_embt") != CUDA_SUCCESS
    || cuModuleGetFunction(&gpu_ganyk, gpu_lib, "llb_gemv_anyk") != CUDA_SUCCESS''')
open(p, 'w').write(s)
