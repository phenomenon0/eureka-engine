# Figure credits

The README images (`docs/img/header.svg` and `docs/img/fig-five-verbs.svg`) are drawn in the style of **iso-glow** by April Zhu, which builds on
**iso-figure** by Tolga Cohce (github.com/MrBongoC/ai-iso-skill). The projection kernel,
the box and face-plane helpers, the dark palette, and the glow filters in
`make_figures.mjs` follow those projects. Both are MIT-licensed.

Regenerate the figures with:

```sh
node docs/img/src/make_figures.mjs
```

Set `OUT=dir` to write somewhere else, and `STILL=off` to write unlit still frames for review.

## License (iso-glow and iso-figure)

```text
MIT License

Copyright (c) 2026 April Zhu (iso-glow: lighting, palettes, gallery shell, examples)
Copyright (c) 2026 Tolga Cohce (iso-figure, on which iso-glow is based)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
