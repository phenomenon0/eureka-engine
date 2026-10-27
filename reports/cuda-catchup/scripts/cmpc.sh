#!/usr/bin/env bash
# cmpc.sh TAG A B : byte-compare arm A vs arm B (stdout ids sans wall_ms + every logits line), greedy + TF, 3 prompts
S=$HOME/worktrees/.scratch-cuda-catchup; o=$S/receipts/${GD:-gate}/$1; n=0; ok=0
for k in greedy tf; do for i in 1 2 3; do a=$o/$2-$k-p$i.txt b=$o/$3-$k-p$i.txt; n=$((n+1))
  if grep -q '^rc=0' $a && grep -q '^rc=0' $b && [ -s $b.stdout ] && cmp -s <(grep -v wall_ms $a.stdout) <(grep -v wall_ms $b.stdout) && cmp -s <(grep '^logits' $a.stderr) <(grep '^logits' $b.stderr); then ok=$((ok+1)); else echo "  DIFF $1 $2 vs $3 $k p$i"; fi
done; done
echo "$1: $2 vs $3: $ok/$n identical"
