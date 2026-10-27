#!/bin/bash
# g3.sh: G3, the 35B-A3B through the final build vs llama.cpp CPU (teacher-forced, tf.py): three prompts x 64; the floor
# (llama.cpp Metal) on the same prompts; the controls: llama.cpp CPU against itself (perfect), the no_renorm mutant
cd "${LANE:-$HOME/scratch/metal-catchup}"; M=Qwen3.5-35B-A3B-Q4_K_S; S=35b; R=runs/g3; mkdir -p $R
for p in p1 p2 p3; do
  python3 tf.py final $M $S runs/$p.ids 64 > $R/final-$p.json
  python3 tf.py llamagpu $M x runs/$p.ids 64 > $R/floor-$p.json
done
python3 tf.py llamacpu $M x runs/p1.ids 64 > $R/perfect-p1.json
LLB_MX_MUT=no_renorm python3 tf.py final $M $S runs/p2.ids 64 > $R/mutant-p2.json
echo done > $R/DONE
