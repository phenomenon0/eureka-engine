#!/bin/bash
# g4c.sh MODEL...: per synthetic model: its Metal sidecar; G4c (the final build vs llama.cpp Metal, tf.py TFREF=gpu, two
# prompts x 32); G8 / G9 (g89.sh, p2 x 32)
cd "${LANE:-$HOME/scratch/metal-catchup}"; R=runs/g4c; mkdir -p $R
for M in "$@"; do S=$M; mkdir -p side/$S
  ./bin/llb-tok extract models/$M.gguf side/$S > /dev/null && ./bin/gguf2pack models/$M.gguf --sidecar-only side/$S --target metal > /dev/null
  for p in p1 p2; do TFREF=gpu python3 tf.py final $M $S runs/$p.ids 32 > $R/$M-final-vsgpu-$p.json; done
  ./g89.sh $M $S runs/p2.ids 32 >> $R/g89.txt
done
echo done > $R/DONE
