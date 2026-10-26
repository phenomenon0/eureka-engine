#!/usr/bin/env bash
# pull.sh LLB W: llb pull without big downloads. A file:// fixture (6000 bytes) via LLB_CONF for resume / size checks,
# HF HEAD only for 9b and 9b-q8 (the >5 GB question is answered "n" on a pty with expect, and refused without a tty).
set -u
L=$1 W=$2; rm -rf $W; mkdir -p $W/src $W/conf $W/models; bad=0
ok() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }
head -c 6000 /dev/urandom > $W/src/tiny.gguf; head -c 2000 /dev/zero > $W/models/tiny.gguf.part
cp $(dirname $L)/defaults.conf $W/conf/
{ cat $(dirname $L)/models.conf
  echo "tiny    qwen35 32 15 6000 - - 0 0 tiny.gguf file://$W/src"
  echo "tinybad qwen35 32 15 6001 - - 0 0 tiny.gguf file://$W/src"; } > $W/conf/models.conf
export LLB_CONF=$W/conf LLB_MODELS=$W/models LLB_HOME=$W/h
$L pull tiny 2> $W/e; r=$?
[ $r = 0 ] && cmp -s -n 2000 $W/models/tiny.gguf /dev/zero && cmp -s -i 2000 $W/models/tiny.gguf $W/src/tiny.gguf && grep -q 'resuming at 2000 bytes' $W/e; ok $? "resume keeps the .part bytes and appends the rest"
$L pull tiny 2> $W/e; [ $? = 0 ] && grep -q 'already here' $W/e; ok $? "second pull: already here"
$L pull tinybad 2> $W/e; [ $? != 0 ] && [ $(wc -l < $W/e) = 1 ] && grep -q 'is 6000 bytes but tinybad expects 6001 bytes' $W/e && grep -q 'tiny=6000, tinybad=6001' $W/e && grep -q 'LLB_MODELS' $W/e; ok $? "wrong-size destination: refused, names both aliases and sizes"
LLB_MODELS=$W/models2 $L pull tinybad 2> $W/e; [ $? != 0 ] && grep -q 'expected 6001' $W/e; ok $? "server size mismatch still refused"
$L pull 27b 2> $W/e; [ $? != 0 ] && [ $(wc -l < $W/e) = 1 ]; ok $? "27b without a source: one line"
s=$(curl -sIL https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf | tr -d '\r' | awk 'tolower($1) == "content-length:" { n = $2 } END { print n }')
[ "$s" = 5680522464 ]; ok $? "HF HEAD size of 9b = models.conf ($s)"
$L pull 9b-q8 < /dev/null 2> $W/e; [ $? != 0 ] && grep -q 'pass -y' $W/e; ok $? "9b-q8 (8.9 GB) without a tty: refused, asks for -y"
printf 'spawn {*}$argv\nexpect "y/N\\\\] " { send "n\\r" }\nexpect eof\ncatch wait r; exit [lindex $r 3]\n' > $W/ask.exp
expect $W/ask.exp $L pull 9b-q8 > $W/e 2>&1; r=$?; [ $r != 0 ] && grep -q 'not downloaded' $W/e && ! [ -e $W/models/Qwen3.5-9B-Q8_0.gguf.part ]; ok $? "9b-q8 on a tty, answer n: not downloaded, nothing written"
echo "pull $((7 - bad))/7 PASS"
