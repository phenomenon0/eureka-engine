# tests/mm/run.sh (this branch's, with MM_COCO) on Linux, minus template parity (needs transformers + MM_TEMPLATE);
# unit step allowed to fail so e2e runs (2 unit checks pin the real COCO file's bytes; MM_COCO here is a 640x480 stand-in)
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libclang-rt-18-dev curl) >/dev/null 2>&1
pip3 install -q --break-system-packages pillow >/dev/null 2>&1
cp -r /src /w && cd /w && cp /t/run.sh tests/mm/run.sh
sed -e '/== parity/,+1d' -e 's#> "$O/unit.log" 2>&1;#> "$O/unit.log" 2>\&1 || true;#' tests/mm/run.sh > tests/mm/.mmrun.sh
MM_COCO=/t/stand-in-640x480.jpg bash tests/mm/.mmrun.sh /tmp/mm; echo "exit=$?"
echo "unit: $(tail -1 /tmp/mm/unit.log)"; grep '^NOT' /tmp/mm/unit.log
echo "e2e: PASS $(grep -c '^PASS' /tmp/mm/e2e.log) FAIL $(grep -c '^FAIL' /tmp/mm/e2e.log); $(tail -1 /tmp/mm/e2e.log)"; grep '^FAIL' /tmp/mm/e2e.log
