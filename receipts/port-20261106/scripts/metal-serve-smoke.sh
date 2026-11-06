# Metal served-path smoke: llb-serve (backends/metal/server/src) + sys1mac (./build.sh LLB_BACKEND=metal) on Qwen3.5-0.8B Q4_K_M
B=/tmp/eepub-build/mac-serve E=/tmp/eepub-build/metal-port/sys1mac P=18951
export LLB_GGUF=$MODELS/Qwen3.5-0.8B-Q4_K_M.gguf LLB_DUAL_FASTPF=0
echo "\$ LLB_GGUF=$LLB_GGUF LLB_DUAL_FASTPF=0 $B/llb-serve --port $P --api-key smoke --engine $E -- --gpu 4GB --threads 8 -- /tmp/eepub-run/side08 0 --steps 2048"
$B/llb-serve --port $P --api-key smoke --engine $E -- --gpu 4GB --threads 8 -- /tmp/eepub-run/side08 0 --steps 2048 > /tmp/eepub-run/serve.log 2>&1 &
pid=$!
for i in $(seq 1 120); do c=$(curl -s -o /dev/null -w '%{http_code}' localhost:$P/health); [ "$c" = 200 ] && break; kill -0 $pid 2>/dev/null || break; sleep 0.5; done
echo "health=$c after ${i} polls"
curl -s -m 120 localhost:$P/v1/chat/completions -H 'Authorization: Bearer smoke' -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'; echo
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "--- server log tail"; tail -15 /tmp/eepub-run/serve.log
