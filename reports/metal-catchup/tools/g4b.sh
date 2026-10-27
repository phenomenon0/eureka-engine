#!/bin/bash
# g4b.sh: G4b, the MoE FFN float64 oracle (mx_ref.py) on the final build's LLB_MX_DUMP records: p2 (a 16-token fast
# chunk, then the decode kernels) and a 1,000-id prompt (two chunks; 240 records sampled); the no_renorm mutant (p2)
cd "${LANE:-$HOME/scratch/metal-catchup}"; M=Qwen3.5-35B-A3B-Q4_K_S; R=runs/g4b; mkdir -p $R
export PYTHONPATH=$HOME/llama.cpp/gguf-py:$PWD/pystub
for c in "p2 runs/p2.ids 8" "pp1000 runs/pp1000.ids 2" "mut runs/p2.ids 8"; do set -- $c
  rm -f $R/$1.dump; mt=$([ $1 = mut ] && echo no_renorm || echo "")
  LLB_MX_MUT=$mt LLB_MX_DUMP=$R/$1.dump ./run.sh final $M 35b @$2 --steps $3 > $R/$1.out 2> $R/$1.err
  python3 - $R/$1.dump $R/$1.sub $1 <<'PY'
import sys
R = (3 + 2 * 2048 + 17) * 4
b = open(sys.argv[1], "rb").read(); n = len(b) // R
sel = list(range(n)) if sys.argv[3] != "pp1000" else list(range(0, n, max(1, n // 240)))[:240]
open(sys.argv[2], "wb").write(b"".join(b[i * R:(i + 1) * R] for i in sel))
open(sys.argv[2] + ".idx", "w").write(" ".join(map(str, sel)) + "\n" + str(n) + "\n")
PY
  python3 mx_ref.py models/$M.gguf $R/$1.sub > $R/$1.oracle.jsonl
  tail -1 $R/$1.oracle.jsonl
done
echo done > $R/DONE
