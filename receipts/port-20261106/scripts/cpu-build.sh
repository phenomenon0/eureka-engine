(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libgomp1 patch) >/dev/null 2>&1
gcc --version | head -1; uname -m
cp -r /src /w && cd /w
echo '$ bash backends/android/engine/build.sh host /tmp/cpu'
bash backends/android/engine/build.sh host /tmp/cpu; echo "exit=$?"
ls -la /tmp/cpu/kev-engine-host 2>&1
