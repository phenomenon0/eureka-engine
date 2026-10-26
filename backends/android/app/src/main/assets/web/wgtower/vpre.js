// vpre.js: the tower's image preprocessing, cudavis/mac/vision/host/vpre.c line for line (which copies mtmd-image.cpp /
// clip-impl.h): calc_size_preserved_ratio (factor 32, 8..4096 tokens), pillow bicubic (22-bit fixed point), PAD_CEIL, then
// (v/255 - 0.5)/0.5 into planar [3][ny][nx]. f32 arithmetic emulated with Math.fround where the C is float.
// Decoding: the browser's (createImageBitmap, no color conversion). The C decodes with stb_image; PNG is lossless so both
// agree; a JPEG's IDCT/upsampling can differ by a level (the gate compares against llama.cpp's own pre.f32).
const fr = Math.fround;

export async function decode_rgb(blob) {
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const c = new OffscreenCanvas(bmp.width, bmp.height), x = c.getContext("2d", { willReadFrequently: true });
  x.drawImage(bmp, 0, 0);
  const rgba = x.getImageData(0, 0, bmp.width, bmp.height, { colorSpace: "srgb" }).data, n = bmp.width * bmp.height, rgb = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) { rgb[3 * i] = rgba[4 * i]; rgb[3 * i + 1] = rgba[4 * i + 1]; rgb[3 * i + 2] = rgba[4 * i + 2]; }
  return { nx: bmp.width, ny: bmp.height, rgb };
}

export function vpre_target(width, height, { patch = 16, merge = 2, min_px = 8 * 1024, max_px = 4096 * 1024 } = {}) {
  const f = patch * merge;
  const ROUND_BY = (x) => Math.round(fr(x / f)) * f, CEIL_BY = (x) => Math.ceil(fr(x / f)) * f, FLOOR_BY = (x) => Math.floor(fr(x / f)) * f;
  let w = ROUND_BY(width), h = ROUND_BY(height);
  if (w < f) w = f;
  if (h < f) h = f;
  if (max_px > 0 && h * w > max_px) {
    const beta = fr(Math.sqrt(fr(fr(fr(height) * width) / max_px)));
    h = FLOOR_BY(fr(height / beta)); w = FLOOR_BY(fr(width / beta));
    if (h < f) h = f;
    if (w < f) w = f;
  } else if (min_px > 0 && h * w < min_px) {
    const beta = fr(Math.sqrt(fr(min_px / fr(fr(height) * width))));
    h = CEIL_BY(fr(height * beta)); w = CEIL_BY(fr(width * beta));
  }
  return [w, h];
}

const PREC = 22;
function bicubic(x) {
  if (x < 0) x = -x;
  const a = -0.5;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}
const clip8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
function weights(inn, out) {
  let filterscale = inn / out; const scale = filterscale;
  if (filterscale < 1) filterscale = 1;
  const support = 2 * filterscale, ksize = Math.ceil(support) * 2 + 1;
  const pre = new Float64Array(out * ksize), bounds = new Int32Array(out * 2), kk = new Int32Array(out * ksize);
  for (let xx = 0; xx < out; xx++) {
    const center = (xx + 0.5) * scale, ss = 1 / filterscale;
    let ww = 0, xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inn) xmax = inn;
    xmax -= xmin;
    let x;
    for (x = 0; x < xmax; x++) { const w = bicubic((x + xmin - center + 0.5) * ss); pre[xx * ksize + x] = w; ww += w; }
    for (x = 0; x < xmax; x++) if (ww !== 0) pre[xx * ksize + x] /= ww;
    for (; x < ksize; x++) pre[xx * ksize + x] = 0;
    bounds[2 * xx] = xmin; bounds[2 * xx + 1] = xmax;
  }
  const fxp = 2 ** PREC;
  for (let i = 0; i < out * ksize; i++) kk[i] = Math.trunc(pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5));
  return { ksize, bounds, kk };
}
function horiz(src, inx, iny, onx, { ksize, bounds, kk }) {
  const out = new Uint8Array(onx * iny * 3);
  for (let yy = 0; yy < iny; yy++) for (let xx = 0; xx < onx; xx++) {
    const xmin = bounds[2 * xx], cnt = bounds[2 * xx + 1], k = xx * ksize;
    let s0 = 1 << (PREC - 1), s1 = s0, s2 = s0, p = (yy * inx + xmin) * 3;
    for (let x = 0; x < cnt; x++, p += 3) { s0 += src[p] * kk[k + x]; s1 += src[p + 1] * kk[k + x]; s2 += src[p + 2] * kk[k + x]; }
    const o = (yy * onx + xx) * 3;
    out[o] = clip8(s0 >> PREC); out[o + 1] = clip8(s1 >> PREC); out[o + 2] = clip8(s2 >> PREC);
  }
  return out;
}
function vert(src, inx, ony, { ksize, bounds, kk }) {
  const re = inx * 3, out = new Uint8Array(re * ony), acc = new Int32Array(re);
  for (let yy = 0; yy < ony; yy++) {
    const ymin = bounds[2 * yy], cnt = bounds[2 * yy + 1], k = yy * ksize;
    acc.fill(1 << (PREC - 1));
    for (let y = 0; y < cnt; y++) { const row = (ymin + y) * re, w = kk[k + y]; for (let i = 0; i < re; i++) acc[i] += src[row + i] * w; }
    for (let i = 0; i < re; i++) out[yy * re + i] = clip8(acc[i] >> PREC);
  }
  return out;
}
function pillow(src, tw, th) {
  const nh = tw !== src.nx, nv = th !== src.ny;
  if (nh && nv) return vert(horiz(src.rgb, src.nx, src.ny, tw, weights(src.nx, tw)), tw, th, weights(src.ny, th));
  if (nh) return horiz(src.rgb, src.nx, src.ny, tw, weights(src.nx, tw));
  if (nv) return vert(src.rgb, src.nx, th, weights(src.ny, th));
  return src.rgb.slice();
}
export function vpre_resize(src, tw, th) {
  if (src.nx === tw && src.ny === th) return { nx: tw, ny: th, rgb: src.rgb.slice() };
  const sw = fr(tw / src.nx), sh = fr(th / src.ny), scale = sw < sh ? sw : sh;
  const nw = Math.min(Math.ceil(fr(src.nx * scale)), tw), nh = Math.min(Math.ceil(fr(src.ny * scale)), th);
  const r = pillow(src, nw, nh), rgb = new Uint8Array(tw * th * 3), ox = (tw - nw) >> 1, oy = (th - nh) >> 1;
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
    const dx = x + ox, dy = y + oy;
    if (dx < 0 || dy < 0 || dx >= tw || dy >= th) continue;
    rgb.set(r.subarray((y * nw + x) * 3, (y * nw + x) * 3 + 3), (dy * tw + dx) * 3);
  }
  return { nx: tw, ny: th, rgb };
}
export function vpre_normalize(img) {
  const n = img.nx * img.ny, planar = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) planar[c * n + i] = fr(fr(fr(img.rgb[3 * i + c] / 255) - 0.5) / 0.5);
  return planar;
}
export async function vpre_image(blob) {
  const src = await decode_rgb(blob), [tw, th] = vpre_target(src.nx, src.ny);
  return { planar: vpre_normalize(vpre_resize(src, tw, th)), nx: tw, ny: th, src: [src.nx, src.ny] };
}
