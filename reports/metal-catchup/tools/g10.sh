#!/bin/bash
# g10.sh: G10, the MoE fast prefill's quality: two prompts cut from the needle haystack (chat template tail kept), 2,000
# and 4,096 ids, + 64 greedy tokens; tf.py against llama.cpp CPU; the floor = llama.cpp Metal on the same prompts; and
# --exactpf dispatches no chunk (its stderr has no chunkf line)
cd "${LANE:-$HOME/scratch/metal-catchup}"; M=Qwen3.5-35B-A3B-Q4_K_S; R=runs/g10; mkdir -p $R
for p in pf2k pf4k; do
  python3 tf.py final $M 35b runs/$p.ids 64 > $R/final-$p.json
  python3 tf.py llamagpu $M x runs/$p.ids 64 > $R/floor-$p.json
done
./run.sh final $M 35b @runs/pf2k.ids --steps 4 --exactpf > $R/x.out 2> $R/x.err
echo "exactpf chunkf lines: $(grep -c '^chunkf' $R/x.err); step lines: $(grep -c '^step' $R/x.err)" > $R/exactpf.txt
echo done > $R/DONE
