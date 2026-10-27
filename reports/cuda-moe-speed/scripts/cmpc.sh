#!/usr/bin/env bash
# cmpc.sh TAG A B : stdout ids (wall_ms excluded) + every logits line of arm A vs arm B, greedy + TF, 3 prompts -> n identical of 6
S=$HOME/worktrees/.scratch-cuda-moe-speed; o=$S/receipts/gate/$1; n=0
for i in 1 2 3; do for k in greedy tf; do
  grep -q '^rc=0' $o/$2-$k-p$i.txt && grep -q '^rc=0' $o/$3-$k-p$i.txt \
  && cmp -s <(grep -v wall_ms $o/$2-$k-p$i.txt.stdout) <(grep -v wall_ms $o/$3-$k-p$i.txt.stdout) \
  && cmp -s <(grep '^logits' $o/$2-$k-p$i.txt.stderr) <(grep '^logits' $o/$3-$k-p$i.txt.stderr) \
  && [ "$(grep -c '^logits' $o/$3-$k-p$i.txt.stderr)" -gt 0 ] && n=$((n+1))
done; done; echo "$1 $2 vs $3: $n/6 identical"
