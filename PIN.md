# Bend compiler pin 240ecef4 — content record

llama-bend `build.sh` pins `BEND_COMMIT=240ecef4b52a1efcc2151df689cca9a2a769aac6`
("foldshape: host Array.get skips the wrap mask in range (blk_get)", docs/REPRO-inventory.md) and uses only
`git archive 240ecef4 bend2`. That commit object is lost: the shared Bend checkout (~/Documents/Project/bend)
is gone, and `240ecef4` is "not our ref" on phenomenon0/bend and bendlang/bend; no local repo, Mac repo or
bundle holds it (checked 2026-10-14 bend-audit-digest, re-checked 2026-10-15 SI-CANON).

This commit's `bend2/` is the surviving `git archive 240ecef4 bend2` export, byte for byte:
- 139 files; content hash (find bend2 -type f | LC_ALL=C sort | xargs cat | sha256sum)
  `ff0242dd757448a37ae9c33235679b73afd1664aa3a064eb12708b3fbd3d5a31`, identical in all six build-dir copies
  (/mnt/storage/worktrees/llama-bend-{unify1/build/s1,s2,s3,integ1/build,gguf2/build/base,gguf2/build/g2}/bend-pinned, .commit = 240ecef4).
- File modes are NOT original (the copies sit on ntfs-3g, every file reads 0777); all are recorded 100644.
  Bun runs the .ts sources, so the build does not depend on modes.
- It is therefore NOT commit 240ecef4 and does not hash to it; the identity is the content hash above and the
  git tree id of `bend2/` recorded in llama-bend build.sh (BEND_TREE).

Verified (SI-CANON 2026-10-15): this bend2/ + the canonical patch stack emits the landed CUDA engine C
`7d939b8f…` (S6-GUARD) and the Metal emit `c13b9e28…` / VISION=1 `e15c60e0…`.
No upstream sync: nothing from bendlang/bend was merged in.
