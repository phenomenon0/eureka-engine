#!/usr/bin/env bash
# gmode.sh TAG : G-MODE scoring (host / stream / forced-small auto vs device), wall_ms excluded
S=$HOME/worktrees/.scratch-cuda-moe; tag=$1; o=$S/receipts/gate/$tag
for m in Chost Cstream Csmall; do ok=1
  for i in 1 2 3; do for k in greedy tf; do
    cmp -s <(grep -v wall_ms $o/C-$k-p$i.txt.stdout) <(grep -v wall_ms $o/$m-$k-p$i.txt.stdout) || ok=0
    cmp -s <(grep '^logits' $o/C-$k-p$i.txt.stderr) <(grep '^logits' $o/$m-$k-p$i.txt.stderr) || ok=0
    grep -q '^rc=0' $o/$m-$k-p$i.txt || ok=0
  done; done
  echo "G-MODE $tag $m vs device: $([ $ok = 1 ] && echo 'byte-identical (stdout ids + logits, 3 prompts, greedy + TF)' || echo DIFFERS) [$(grep -h 'moe:' $o/$m-greedy-p1.txt | sed 's/.*-> //')]"
done > $o/GMODE.txt
cat $o/GMODE.txt
