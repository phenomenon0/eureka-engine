#!/usr/bin/env bash
# check-isa.sh [--strict] TARBALL_DIR: Q1b by default, original Q1 via --strict.
# Scan executable sections of EVERY ELF, including bundled vendor libraries.
# A static failure is not proof an extension is executed on an older CPU.
set -euo pipefail
strict=0
if [ "${1-}" = --strict ]; then strict=1; shift; fi
[ $# = 1 ] && [ -d "$1" ] || { echo 'usage: check-isa.sh [--strict] TARBALL_DIR' >&2; exit 2; }
fail=0 count=0
while IFS= read -r -d '' f; do
  if file -b "$f" | grep -q '^ELF'; then
    count=$((count + 1))
    if ! file -b "$f" | grep -q 'x86-64'; then echo "FAIL non-x86-64 ELF: $f"; fail=1; continue; fi
    vendor=0
    case ${f##*/} in libnvrtc.so.*|libnvrtc-builtins.so.*) vendor=1 ;; esac
    # raw encodings catch VEX/EVEX even for scalar BMI instructions and unknown disassembly.
    if ! objdump -d -w -M intel "$f" | awk -v file="$f" -v strict="$strict" -v vendor="$vendor" '
      /^[[:space:]]*[0-9a-f]+:/ {
        n=split($0,a,"\t"); if(n<3) next;
        bytes=a[2]; ins=a[3]; sub(/^[[:space:]]+/,"",bytes); sub(/^[[:space:]]+/,"",ins);
        split(ins,m," "); op=m[1]; bad=0;
        if(bytes ~ /^(c4|c5|62) / || ins ~ /[xyz]mm(1[6-9]|2[0-9]|3[01])/ || ins ~ /[yz]mm[0-9]/) bad=1;
        # SSE3/SSSE3/SSE4, POPCNT, BMI, AES/PCLMUL, random, XSAVE, and later scalar extensions.
        if(op ~ /^(addsubp[sd]|haddp[sd]|hsubp[sd]|lddqu|movddup|movs[hl]dup|pabs[bwd]|palignr|phadd.*|phsub.*|pmaddubsw|pmulhrsw|pshufb|psign.*|blend.*|dpp[sd]|extractps|insertps|movntdqa|mpsadbw|packusdw|pblend.*|pcmpeqq|pcmpgtq|pextr[bdq]|phminposuw|pinsr[bqd]|pmaxs[bd]|pmaxu[dw]|pmins[bd]|pminu[dw]|pmovsx.*|pmovzx.*|pmuldq|pmulld|ptest|round.*|pcmpestr.*|pcmpistr.*|crc32|popcnt|lzcnt|tzcnt|andn|bextr|blsi|blsmsk|blsr|bzhi|mulx|pdep|pext|rorx|sarx|shlx|shrx|aes.*|pclmul.*|rdrand|rdseed|xsave.*|xrstor.*|xgetbv|xsetbv|movbe|adcx|adox|sha1.*|sha256.*|gf2p8.*|clflushopt|clwb|clzero|rdpid|serialize|enqcmd.*|movdiri|movdir64b|umonitor|umwait|tpause|wr[fg]sbase|rd[fg]sbase|endbr32|endbr64)$/) bad=1;
        if(op ~ /^v/ && op !~ /^(verr|verw)$/) bad=1;
        if(ins ~ /\(bad\)/) bad=1;
        class="";
        if(bytes ~ /^f3 0f 1e fa/ || (bytes ~ /^f[23] / && ins ~ /^(pause|rep(z|nz)?[[:space:]]+nop)/)) class="COMPAT-HINT";
        if(bytes ~ /^f3 (4[0-9a-f] )?0f bc /) class="COMPAT-ENCODING";
        if(class!="") {
          compatible[class]++; print (vendor ? "THIRD-PARTY " : "OWN ") class " " file ": " $0;
          if(strict) total++;
        } else if(bad) {
          extensions++; total++;
          if(extensions<=12) print (vendor ? "THIRD-PARTY " : "OWN ") "EXTENSION " file ": " $0;
        }
      }
      END {
        print (vendor ? "THIRD-PARTY" : "OWN") " SCAN " file ": COMPAT-HINT=" compatible["COMPAT-HINT"]+0 " COMPAT-ENCODING=" compatible["COMPAT-ENCODING"]+0 " extension/undecodable=" extensions+0 " strict-flagged=" total+0;
        exit total>0 && (strict || !vendor)
      }
    '; then fail=1; fi
  fi
done < <(find "$1" -type f -print0)
while IFS= read -r -d '' f; do
  if grep -nE -- '(^|[[:space:]"=])-(march|mavx)[^[:space:]]*' "$f"; then echo "FAIL forbidden build flag: $f"; fail=1; fi
done < <(find "$1" -type f \( -name '*.sh' -o -name '*flags*' -o -name '*build*.txt' \) -print0)
[ "$count" -gt 0 ] || { echo 'FAIL no ELF files'; fail=1; }
echo "ELF_COUNT=$count RESULT=$([ "$fail" = 0 ] && echo PASS || echo FAIL)"
exit "$fail"
