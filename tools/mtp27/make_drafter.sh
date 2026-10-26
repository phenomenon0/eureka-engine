#!/usr/bin/env bash
# tools/mtp27/make_drafter.sh: the LLB_MTP drafter pack from a clean checkout (W1-SAMPLER rider).
#   make_drafter.sh OUT.mtp1 [--aux1 OUT.aux1]
# Inputs: the official Qwen3.8-27B MTP sidecar GGUF (MTP_SIDE: blk.64 = the NextN block) and the trunk
# GGUF (MTP_TRUNK: output.weight rows of the draft vocab), plus the in-repo draft vocabulary
# tools/mtp27/draft_ids_40960.json (40,960 ids, the mtp27 lane's vocab.py count; extracted from the
# production pack blk64-wip.bin). Output = the production pack, sha256-pinned below; --aux1 also
# writes the named AUX1 container (tools/auxpack, S6-b) — mtp_load reads either.
# Needs python3 + the gguf package (pip install gguf numpy). CPU/IO only, ~440 MB written.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:?usage: make_drafter.sh OUT.mtp1 [--aux1 OUT.aux1]}
export MTP_SIDE=${MTP_SIDE:?set MTP_SIDE to the NextN sidecar GGUF}
export MTP_TRUNK=${MTP_TRUNK:?set MTP_TRUNK to the 27B Q4_K_M GGUF}
for f in "$MTP_SIDE" "$MTP_TRUNK"; do [ -f "$f" ] || { echo "make_drafter: missing $f (set MTP_SIDE / MTP_TRUNK)" >&2; exit 1; }; done
ids=$here/draft_ids_40960.json
echo "e3a69c2d2b688c4e4d01968dadf688f42d3fa18b08fdc36d9b8ed7bdb7801600  -" > /tmp/.mtp_ids_sha.$$
python3 -c "import json,struct,sys; a=json.load(open('$ids')); sys.stdout.buffer.write(struct.pack('<%dI'%len(a),*a))" \
  | sha256sum -c --quiet /tmp/.mtp_ids_sha.$$ || { echo "make_drafter: draft_ids_40960.json does not match its pin" >&2; exit 1; }
rm -f /tmp/.mtp_ids_sha.$$
python3 "$here/mtp_pack.py" "$ids" "$out"
want=c00130c43aa4bdcff3c8bc628f2682b93e77aa31357dbf73eb51ccb063e3c6cd   # blk64-wip.bin (mtp27/mtp28 lanes)
got=$(sha256sum "$out" | cut -d' ' -f1)
if [ "$got" = "$want" ]; then echo "make_drafter: $out == the production pack ($want)"
else echo "make_drafter: WARNING $out sha $got != pinned $want (different sidecar/trunk bytes?)" >&2; fi
if [ "${2:-}" = --aux1 ]; then python3 "$here/../auxpack/auxpack.py" convert "$out" "${3:?--aux1 OUT}"; fi
