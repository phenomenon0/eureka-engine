# Clef-flash oracle goldens

24 successful text-only POST /v1/systemone exchanges from local llama.cpp
2ca15f5, Clef-flash Q4_K_S, on the RTX 3090. Each request and response JSON
file contains the exact transmitted/received body bytes; the meta file records
HTTP status, raw headers and HTTP wall time. SHA256SUMS covers all 72 files.
These are oracle outputs, not human accuracy labels or proof of PyTorch parity.

Case 16 is absent: a request with omitted instructions returns HTTP 400 by
design. Case 25 is the valid counterpart with explicit instructions.

Case 24 has 2,791 state-only tokens measured using the server /tokenize route,
and 3,020 complete input tokens. This is measured token length, not a
character-count estimate. The state includes the decisive update at the end.
