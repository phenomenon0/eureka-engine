# tests/mm/run.sh as shipped, minus the template-parity step (needs transformers + MM_TEMPLATE),
# with MM_COCO a 640x480 stand-in for the COCO 142472 image (not on this machine),
# and the unit step allowed to fail so the HTTP e2e step still runs (2 unit checks pin the real COCO file's bytes)
sed -e '/== parity/,+1d' \
    -e 's#> "$O/unit.log" 2>&1;#> "$O/unit.log" 2>\&1 || true;#' tests/mm/run.sh > tests/mm/.mmrun.sh
rm -rf "$1"; MM_COCO=/tmp/eepub-tools/stand-in-640x480.jpg PATH=/tmp/eepub-venv/bin:$PATH CC=/tmp/eepub-tools/cc-compat PY=/tmp/eepub-venv/bin/python3 bash tests/mm/.mmrun.sh "$1"; rc=$?
rm -f tests/mm/.mmrun.sh; echo "exit=$rc"
echo "unit: $(tail -1 "$1/unit.log")"; grep -v '^ok' "$1/unit.log" | grep '^NOT'
echo "e2e:  $(tail -1 "$1/e2e.log")"
exit $rc
