# WebGPU

Browser tab. Tested in a browser on an M3 Ultra, and on an RTX 3090 through
Vulkan/ANGLE. Preview tier.

## Source

`public/trunk/` and `public/emit/` in `github.com/phenomenon0/bend-app`.

## Build and run

```sh
npm install
npm run dev
```

## Result

clef-flash Q4_K_S (9B): 35.8 tok/s decode on the M3 Ultra, 16.4 tok/s on the
RTX 3090.

## Gaps

The graded engine is a hand-written WGSL port. It is a test bed, not the
reference implementation.
