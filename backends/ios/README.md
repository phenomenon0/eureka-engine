# iOS

iPhone, Metal. Tested on an iPhone 18 Pro Max (12 GB).

## Source

`ios/` in `github.com/phenomenon0/bend-app`. The app embeds the Metal engine
built by [`../metal`](../metal/README.md) (emitted C `c13b9e28`).

## Build

```sh
cd ios
xcodegen generate
open Bend.xcodeproj
```

## Result

Kev-4B Q4_K_M: 20.6 tok/s end to end.

## Gaps

Receipts are from a debug build.
