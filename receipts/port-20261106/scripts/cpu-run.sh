(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libgomp1 patch) >/dev/null 2>&1
cp -r /src /w && cd /w && bash backends/android/engine/build.sh host /tmp/cpu >/dev/null 2>&1; echo "build exit=$?"
clang -std=c11 -O2 -o /tmp/gguf2pack tools/gguf2pack.c && clang -O2 -o /tmp/llb-tok tools/llb/llb-tok.c
echo '$ gguf2pack $MODELS/Qwen3.5-0.8B-Q4_K_M.gguf PACK'; /tmp/gguf2pack /m/Qwen3.5-0.8B-Q4_K_M.gguf /tmp/pack 2>&1 | tail -1; echo "exit=$?"
/tmp/llb-tok extract /m/Qwen3.5-0.8B-Q4_K_M.gguf /tmp/pack >/dev/null
ids=$(/tmp/llb-tok enc /tmp/pack "The capital of France is")
/tmp/cpu/kev-engine-host 2>&1 | head -3
echo "\$ kev-engine-host PACK $ids --steps 8"; OMP_NUM_THREADS=8 /tmp/cpu/kev-engine-host /tmp/pack $ids --steps 8 > /tmp/o 2>/tmp/e; echo "exit=$?"; tail -3 /tmp/e
out=$(grep -oE '^[0-9]+$' /tmp/o | tr '\n' ' '); echo "ids: $out"; echo "text: $(/tmp/llb-tok dec /tmp/pack $out)"
