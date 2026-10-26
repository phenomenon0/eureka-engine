// tools/qwen35_engine_metal.mm — Metal (Apple Silicon) backend for the
// Qwen3.5-9B hybrid decode engine.
//
// This file includes the CPU twin (tools/qwen35_engine.c) with __METAL__
// defined, so the same host flow, pack reader and CPU reference math are used
// verbatim. Only the dominant GEMV (all 9.5 GB of Q8 weights) is offloaded: the
// pack is bound as one shared MTLBuffer, and one MSL thread per row reproduces
// the CPU gemv's accumulation order exactly, with fp-contract off, so the two
// lanes are bit-for-bit.
//
// build: clang++ -O3 -std=c++17 -fobjc-arc -x objective-c++ \
//          tools/qwen35_engine_metal.mm -framework Metal -framework Foundation \
//          -o build/qwen35-cli-metal

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>

#define __METAL__ 1
#include "qwen35_engine.c"

static const char *gMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// exact replica of the engine's half_to_float (no fp16 hardware path)
static inline float hf_half(uint h) {
  uint sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, f;
  if (exp == 0) {
    if (mant == 0) f = sign << 31;
    else {
      exp = 127 - 15 + 1;
      while (!(mant & 0x400)) { mant <<= 1; exp--; }
      mant &= 0x3ff;
      f = (sign << 31) | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  return as_type<float>(f);
}

kernel void gemv_q8(device const uchar* W     [[buffer(0)]],
                    constant ulong&      off    [[buffer(1)]],
                    constant int&        rows   [[buffer(2)]],
                    constant int&        cols   [[buffer(3)]],
                    constant int&        rb     [[buffer(4)]],
                    device const float*  x      [[buffer(5)]],
                    device float*        y      [[buffer(6)]],
                    uint                 gid    [[thread_position_in_grid]]) {
  if (gid >= (uint)rows) return;
  device const uchar* row = W + off + (ulong)gid * (ulong)rb;
  device const char*  q   = (device const char*)row;
  device const uchar* d   = row + cols;
  float acc = 0.0f;
  int nb = cols >> 5;
for (int b = 0; b < nb; b++) {
    float dd = hf_half((uint)d[2 * b] | ((uint)d[2 * b + 1] << 8));
    for (int j = 0; j < 32; j++) acc += ((float)q[b * 32 + j] * dd) * x[b * 32 + j];
  }
  y[gid] = acc;
}

kernel void gemv_f32(device const float* W      [[buffer(0)]],
                     constant ulong&      off    [[buffer(1)]],
                     constant int&        rows   [[buffer(2)]],
                     constant int&        cols   [[buffer(3)]],
                     device const float*  x      [[buffer(5)]],
                     device float*        y      [[buffer(6)]],
                     uint                 gid    [[thread_position_in_grid]]) {
  if (gid >= (uint)rows) return;
  device const float* row = (device const float*)((device const char*)W + off)
                            + (ulong)gid * (ulong)cols;
  float acc = 0.0f;
for (int c = 0; c < cols; c++) acc += row[c] * x[c];
  y[gid] = acc;
}
)MSL";

static id<MTLDevice>              gDev  = nil;
static id<MTLCommandQueue>        gQ    = nil;
static id<MTLComputePipelineState> gPipeQ8 = nil, gPipeF32 = nil;
static id<MTLBuffer>              gBufPack = nil, gBufX = nil, gBufY = nil;

int metal_init(size_t psz, int maxrows, int maxcols) {
  @autoreleasepool {
    gDev = MTLCreateSystemDefaultDevice();
    if (!gDev) { fprintf(stderr, "metal: no device\n"); exit(1); }
    gQ = [gDev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:[NSString stringWithUTF8String:gMSL]
                                            options:nil error:&err];
    if (!lib) { fprintf(stderr, "metal: library: %s\n", err.localizedDescription.UTF8String); exit(1); }
    gPipeQ8  = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_q8"]  error:&err];
    gPipeF32 = [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_f32"] error:&err];
    if (!gPipeQ8 || !gPipeF32) { fprintf(stderr, "metal: pipeline: %s\n", err.localizedDescription.UTF8String); exit(1); }
    if ((size_t)psz > (size_t)gDev.maxBufferLength) {
      fprintf(stderr, "metal: pack %zu > maxBufferLength %zu\n", psz, (size_t)gDev.maxBufferLength);
      exit(1);
    }
    // The pack (mmap) is not page-length aligned, so no-copy is not offered by
    // Metal here; one shared buffer copy it is (unified memory, ~100 GB/s).
    gBufPack = [gDev newBufferWithBytes:gPack length:psz options:MTLResourceStorageModeShared];
    if (!gBufPack) { fprintf(stderr, "metal: pack buffer %zu failed\n", psz); exit(1); }
    gBufX = [gDev newBufferWithLength:(NSUInteger)maxcols * 4 options:MTLResourceStorageModeShared];
    gBufY = [gDev newBufferWithLength:(NSUInteger)maxrows * 4 options:MTLResourceStorageModeShared];
    if (!gBufX || !gBufY) { fprintf(stderr, "metal: scratch buffers failed\n"); exit(1); }
    fprintf(stderr, "metal: device %s, pack %.1f GB resident (copied), maxrows %d maxcols %d\n",
            gDev.name.UTF8String, psz / 1e9, maxrows, maxcols);
  }
  return 0;
}

void metal_gemv(int tid, const float *x, float *y) {
  @autoreleasepool {
    int rows = gT[tid].rows, cols = gT[tid].cols, rb = gT[tid].row_bytes, is_f32 = gT[tid].is_f32;
    uint64_t off = (uint64_t)gT[tid].off;
    memcpy(gBufX.contents, x, (size_t)cols * 4);
    id<MTLCommandBuffer> cb = [gQ commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:is_f32 ? gPipeF32 : gPipeQ8];
    [e setBuffer:gBufPack offset:0 atIndex:0];
    [e setBytes:&off length:sizeof(off) atIndex:1];
    [e setBytes:&rows length:4 atIndex:2];
    [e setBytes:&cols length:4 atIndex:3];
    [e setBytes:&rb length:4 atIndex:4];
    [e setBuffer:gBufX offset:0 atIndex:5];
    [e setBuffer:gBufY offset:0 atIndex:6];
    NSUInteger tg = 256;
    NSUInteger groups = ((NSUInteger)rows + tg - 1) / tg;
    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(y, gBufY.contents, (size_t)rows * 4);
  }
}
