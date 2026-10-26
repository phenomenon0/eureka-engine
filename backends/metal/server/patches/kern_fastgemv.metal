// ---- E-METAL-LEAD fast decode GEMV (the tolerance tier; host switch LLB_FASTGEMV=1). The summation structure is
// llama.cpp's kernel_mul_mv_q{4,5,6}_K_f32 (MIT, ggml-metal): one simdgroup walks the blocks of NR0 rows, each lane a
// slice of every block, partials joined by simd_sum, FMA accumulation. NOT the fixed-tree contract (bd2/rd2 above stay
// the bit-exact CANON bodies and the default). ABI = llb_warp_q8's (C weights / att x / WarpArgs / Y); 2 simdgroups a
// threadgroup, Q4_K / Q6_K NR0 2 (4 rows a group), Q5_K NR0 1 (2 rows). y is the f32 bits at Y + A.y, like bd2.
template <short NR0, short NSG>
kernel void llb_fq4k(device const u8* C [[buffer(0)]], device const float* att [[buffer(1)]],
  constant WarpArgs& A [[buffer(2)]], device u8* Y [[buffer(3)]],
  uint gid [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  const ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  const short ix = tiisg / 8, it = tiisg % 8, iq = it / 4, ir = it % 4;
  const int nb = (int)(A.cg >> 6);
  const uint first_row = (gid * NSG + sgitg) * NR0;
  device const u8* W = C + A.w;
  device u32* y = (device u32*)(Y + A.y);
  float yl[16], yh[16];
  float sumf[NR0];
  for (short r = 0; r < NR0; r++) sumf[r] = 0.0f;
  device const float* y4 = att + A.slot * 2048 + ix * 256 + 64 * iq + 8 * ir;
  ushort sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;
  for (int ib = ix; ib < nb; ib += 4) {
    float4 sumy = float4(0.0f);
    for (short i = 0; i < 8; ++i) {
      yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
      yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
      yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
      yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
    }
    for (short row = 0; row < NR0; row++) {
      device const u8* blk = W + (ulong)min(first_row + (uint)row, A.n - 1) * A.rq + (ulong)ib * 144;
      device const half* dh = (device const half*)blk;
      device const ushort* sc = (device const ushort*)(blk + 4) + iq;
      device const ushort* q1 = (device const ushort*)(blk + 16) + 16 * iq + 4 * ir;
      device const ushort* q2 = q1 + 32;
      sc16[0] = sc[0] & kmask1;
      sc16[1] = sc[2] & kmask1;
      sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
      sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
      float4 acc1 = float4(0.0f), acc2 = float4(0.0f);
      for (short i = 0; i < 4; ++i) {
        const ushort a = q1[i], b = q2[i];
        acc1[0] = fma(yl[2 * i + 0], (float)(a & 0x000F), acc1[0]);
        acc1[1] = fma(yl[2 * i + 1], (float)(a & 0x0F00), acc1[1]);
        acc1[2] = fma(yl[2 * i + 8], (float)(a & 0x00F0), acc1[2]);
        acc1[3] = fma(yl[2 * i + 9], (float)(a & 0xF000), acc1[3]);
        acc2[0] = fma(yh[2 * i + 0], (float)(b & 0x000F), acc2[0]);
        acc2[1] = fma(yh[2 * i + 1], (float)(b & 0x0F00), acc2[1]);
        acc2[2] = fma(yh[2 * i + 8], (float)(b & 0x00F0), acc2[2]);
        acc2[3] = fma(yh[2 * i + 9], (float)(b & 0xF000), acc2[3]);
      }
      sumf[row] += (float)dh[0] * ((acc1[0] + (1.0f / 256.0f) * acc1[1]) * sc8[0]
                                 + (acc1[2] + (1.0f / 256.0f) * acc1[3]) * sc8[1] * (1.0f / 16.0f)
                                 + (acc2[0] + (1.0f / 256.0f) * acc2[1]) * sc8[4]
                                 + (acc2[2] + (1.0f / 256.0f) * acc2[3]) * sc8[5] * (1.0f / 16.0f))
                 - (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
    }
    y4 += 4 * 256;
  }
  for (short row = 0; row < NR0 && first_row + (uint)row < A.n; ++row) {
    const float t = simd_sum(sumf[row]);
    if (tiisg == 0) y[first_row + row] = as_type<u32>(t);
  }
}

template <short NR0, short NSG>
kernel void llb_fq5k(device const u8* C [[buffer(0)]], device const float* att [[buffer(1)]],
  constant WarpArgs& A [[buffer(2)]], device u8* Y [[buffer(3)]],
  uint gid [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  const ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
  const int nb = (int)(A.cg >> 6);
  const uint first_row = (gid * NSG + sgitg) * NR0;
  device const u8* W = C + A.w;
  device u32* y = (device u32*)(Y + A.y);
  float sumf[NR0];
  for (short r = 0; r < NR0; r++) sumf[r] = 0.0f;
  float yl[16], yh[16];
  const short tid = tiisg / 4, ix = tiisg % 4, iq = tid / 4, ir = tid % 4;
  const short l0 = 8 * ir, q_offset = 32 * iq + l0, y_offset = 64 * iq + l0;
  const uchar hm1 = 1u << (2 * iq), hm2 = hm1 << 1, hm3 = hm1 << 4, hm4 = hm2 << 4;
  ushort sc16[4];
  thread const uchar* sc8 = (thread const uchar*)sc16;
  device const float* y1 = att + A.slot * 2048 + ix * 256 + y_offset;
  for (int i = ix; i < nb; i += 4) {
    device const float* y2 = y1 + 128;
    float4 sumy = float4(0.0f);
    for (short l = 0; l < 8; ++l) {
      yl[l + 0] = y1[l + 0];  sumy[0] += yl[l + 0];
      yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
      yh[l + 0] = y2[l + 0];  sumy[2] += yh[l + 0];
      yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
    }
    for (short row = 0; row < NR0; ++row) {
      device const u8* blk = W + (ulong)min(first_row + (uint)row, A.n - 1) * A.rq + (ulong)i * 176;
      device const half* dh = (device const half*)blk;
      device const ushort* a = (device const ushort*)(blk + 4) + iq;
      device const uchar* qh = blk + 16 + l0;
      device const uchar* q1 = blk + 48 + q_offset;
      device const uchar* q2 = q1 + 64;
      sc16[0] = a[0] & kmask1;
      sc16[1] = a[2] & kmask1;
      sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
      sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
      float4 acc1 = float4(0.0f), acc2 = float4(0.0f);
      for (short l = 0; l < 8; ++l) {
        const uchar h = qh[l], u = q1[l], v = q2[l];
        acc1[0] = fma(yl[l + 0], (float)(u & 0x0F), acc1[0]);
        acc1[1] = fma(yl[l + 8], (float)(u & 0xF0), acc1[1]);
        acc1[2] = fma(yh[l + 0], (float)(v & 0x0F), acc1[2]);
        acc1[3] = fma(yh[l + 8], (float)(v & 0xF0), acc1[3]);
        acc2[0] += (h & hm1) ? yl[l + 0] : 0.0f;
        acc2[1] += (h & hm2) ? yl[l + 8] : 0.0f;
        acc2[2] += (h & hm3) ? yh[l + 0] : 0.0f;
        acc2[3] += (h & hm4) ? yh[l + 8] : 0.0f;
      }
      sumf[row] += (float)dh[0] * (sc8[0] * (acc1[0] + 16.0f * acc2[0])
                                 + sc8[1] * (acc1[1] / 16.0f + 16.0f * acc2[1])
                                 + sc8[4] * (acc1[2] + 16.0f * acc2[2])
                                 + sc8[5] * (acc1[3] / 16.0f + 16.0f * acc2[3]))
                 - (float)dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
    }
    y1 += 4 * 256;
  }
  for (short row = 0; row < NR0 && first_row + (uint)row < A.n; ++row) {
    const float t = simd_sum(sumf[row]);
    if (tiisg == 0) y[first_row + row] = as_type<u32>(t);
  }
}

template <short NR0, short NSG>
kernel void llb_fq6k(device const u8* C [[buffer(0)]], device const float* att [[buffer(1)]],
  constant WarpArgs& A [[buffer(2)]], device u8* Y [[buffer(3)]],
  uint gid [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
  ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  const int nb = (int)(A.cg >> 6);
  const uint first_row = (gid * NSG + sgitg) * NR0;
  device const u8* W = C + A.w;
  device u32* y = (device u32*)(Y + A.y);
  float sumf[NR0];
  for (short r = 0; r < NR0; r++) sumf[r] = 0.0f;
  float yl[16];
  const short tid = tiisg / 2, ix = tiisg % 2, ip = tid / 8, il = tid % 8, l0 = 4 * il, is = 8 * ip + l0 / 16;
  const short y_offset = 128 * ip + l0, q_offset_l = 64 * ip + l0, q_offset_h = 32 * ip + l0;
  for (int i = ix; i < nb; i += 2) {
    device const float* yp = att + A.slot * 2048 + i * 256 + y_offset;
    for (short l = 0; l < 4; ++l) {
      yl[4 * l + 0] = yp[l + 0];
      yl[4 * l + 1] = yp[l + 32];
      yl[4 * l + 2] = yp[l + 64];
      yl[4 * l + 3] = yp[l + 96];
    }
    for (short row = 0; row < NR0; ++row) {
      device const u8* blk = W + (ulong)min(first_row + (uint)row, A.n - 1) * A.rq + (ulong)i * 210;
      device const uchar* q1 = blk + q_offset_l;
      device const uchar* q2 = q1 + 32;
      device const uchar* qh = blk + 128 + q_offset_h;
      device const char* sc = (device const char*)(blk + 192) + is;
      const float d = (float)*(device const half*)(blk + 208);
      float4 sums = float4(0.0f);
      for (short l = 0; l < 4; ++l) {
        sums[0] = fma(yl[4 * l + 0], (float)((char)((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32), sums[0]);
        sums[1] = fma(yl[4 * l + 1], (float)((char)((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32), sums[1]);
        sums[2] = fma(yl[4 * l + 2], (float)((char)((q1[l] >> 4) | ((qh[l] & 0x30) << 0)) - 32), sums[2]);
        sums[3] = fma(yl[4 * l + 3], (float)((char)((q2[l] >> 4) | ((qh[l] & 0xC0) >> 2)) - 32), sums[3]);
      }
      sumf[row] += d * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
    }
  }
  for (short row = 0; row < NR0 && first_row + (uint)row < A.n; ++row) {
    const float t = simd_sum(sumf[row]);
    if (tiisg == 0) y[first_row + row] = as_type<u32>(t);
  }
}
#define LLB_FQ_INST(K, NR, NS) \
  template [[host_name("llb_fq" #K "k_r" #NR "_s" #NS)]] kernel void llb_fq##K##k<NR, NS>( \
    device const u8*, device const float*, constant WarpArgs&, device u8*, uint, ushort, ushort);
LLB_FQ_INST(4, 1, 1) LLB_FQ_INST(4, 1, 2) LLB_FQ_INST(4, 1, 4) LLB_FQ_INST(4, 2, 1) LLB_FQ_INST(4, 2, 2) LLB_FQ_INST(4, 2, 4) LLB_FQ_INST(4, 4, 1) LLB_FQ_INST(4, 4, 2) LLB_FQ_INST(4, 4, 4)
LLB_FQ_INST(5, 1, 1) LLB_FQ_INST(5, 1, 2) LLB_FQ_INST(5, 1, 4) LLB_FQ_INST(5, 2, 1) LLB_FQ_INST(5, 2, 2) LLB_FQ_INST(5, 2, 4) LLB_FQ_INST(5, 4, 1) LLB_FQ_INST(5, 4, 2) LLB_FQ_INST(5, 4, 4)
LLB_FQ_INST(6, 1, 1) LLB_FQ_INST(6, 1, 2) LLB_FQ_INST(6, 1, 4) LLB_FQ_INST(6, 2, 1) LLB_FQ_INST(6, 2, 2) LLB_FQ_INST(6, 2, 4) LLB_FQ_INST(6, 4, 1) LLB_FQ_INST(6, 4, 2) LLB_FQ_INST(6, 4, 4)

