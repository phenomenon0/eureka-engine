# serve-harden — llb-serve connection/robustness fixes (2026-10-19)

Branch `serve-harden`, based on canonical `73f8d735`. One file: `tools/sys1dual/serve.c` (+97/−7).

## Fixes
1. **`--max-conns N` (default 256, 0 = unlimited).** In the accept loop, above the cap the server writes a 503
   `unavailable_error` and closes — **no thread, no fd held**. `inflight` is the atomic counter.
2. **`--send-timeout S` (default 30).** `SO_SNDTIMEO` is set on every accepted socket, so a client that stops reading
   cannot park a thread (and, through the engine pipe, a slot) forever. `send_all` already returns −1 on timeout.
3. **Disconnect drain.** The generation loop observes `c->broken` → `c->cancelled`; it stops streaming and accumulating
   but keeps draining engine lines to `fin`, so the slot is released normally. (There is **no engine cancel protocol**;
   the engine still runs to completion — documented, not hidden.)
4. **Engine watchdog.** `--engine-timeout S` (default 180). A watchdog thread checks every 500 ms; a **busy** slot whose
   engine has produced no line for S seconds is `SIGKILL`ed and marked dead, and the waiting request gets 503 via EOF.
   `t_last` is stamped on slot acquire and on every engine line.
5. **Strict `Content-Length`.** Duplicate headers, non-numeric values, or trailing garbage → **400**; > `--max-body-mb` → 413.
   (Was `strtol(cl, NULL, 10)` on the first header only — request-smuggling shaped behind a proxy.)
6. **Unchecked `json_get(v,"text")->str`** in the token loop is now type-checked (crash path on engine oddity).

## Tests (fake engine, no GPU/model)
`/tmp/serve-fix/test_harden.py`, server `--max-conns 2 --send-timeout 5`:
- valid completion → **200**, body contains the scripted text — PASS
- `Content-Length: 5, 5` → **400** — PASS
- duplicate `Content-Length` headers → **400** — PASS
- two idle keep-alive connections + a third → **503** at the cap — PASS

Watchdog (`stuck.sh` engine, `--engine-timeout 2`): client received **503 after 2.5 s**, log line
`watchdog: engine[0] silent 2s (limit 2s) — killing pid …` — PASS.

Build: clang 21, `-O2 -Wall -Wextra`, rc 0, no warnings. SO_SNDTIMEO is set at accept; it is not yet stress-tested with a
deliberately non-reading client (noted as the one unproven item).

## Not changed
No engine code, no API shape, no defaults that alter successful requests. `--max-conns`/`--send-timeout`/`--engine-timeout`
are new flags with the stated defaults.

## Round 2 — the rest of GLM's structural items
- **#8 fd inheritance fixed.** `close_extra_fds()` runs in every forked child (engine, glue, tts, ear): only stdio survives,
  so the listen fd and other slots' pipe ends are no longer inherited.
- **#9 shutdown race fixed.** `fclose(slots[i].in)` only runs when `inflight == 0`; otherwise the pipes are left to the kills.
- **#10 second crash path fixed.** `r->paths[i]` is guarded in `engine_line` (NULL → empty string), and the OOM path in
  `media_stage` already returns an error.
- **#5 partial.** `conn_main`'s header buffer is checked (OOM → close + free, no NULL deref). The full malloc sweep across
  every call site is **not** done.
- **#7 open.** Strict FIFO + one 4096-item batch ticket can still monopolize a slot. That is an admission/fairness design
  change, not a one-line fix; left open deliberately.
- Re-tested after round 2: 200 / 400 / 400 / 503 cap / watchdog 503 at 2.4 s — all PASS.
