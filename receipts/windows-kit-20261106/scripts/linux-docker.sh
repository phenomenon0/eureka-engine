# linux-docker.sh: run in eepub-linux:1 (receipts/port-20261106/scripts/Dockerfile.*) with -v REPO:/src:ro -v MODELS:/m:ro
# -v K:/k:ro, where K/base-src = `git archive dd824b9 tools host`. --network none. The image has clang only; gcc -> clang.
mkdir -p /tmp/bin && ln -sf "$(command -v clang)" /tmp/bin/gcc && export PATH=/tmp/bin:$PATH
echo "# Linux $(uname -m), $(ldd --version | head -1), $(clang --version | head -1); no gcc in the image: gcc -> clang"
echo '$ bash tools/build_qwen35_cli.sh'; cp -r /src /w && cd /w && rm -rf build && bash tools/build_qwen35_cli.sh 2>&1 | sed 's/ [A-Z][a-z][a-z] [ 0-9][0-9] [0-9:]* / /'; echo "exit=${PIPESTATUS[0]}"
echo '$ posix-ab.sh $MODELS/Qwen3.5-0.8B-Q4_K_M.gguf /tmp/ab <dd824b9 tools+host>'
bash /w/receipts/windows-kit-20261106/scripts/posix-ab.sh /m/Qwen3.5-0.8B-Q4_K_M.gguf /tmp/ab /k/base-src
