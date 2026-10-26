#!/usr/bin/env bash
# publish-github.sh VERSION DIR [--publish]
#
# Builds the GitHub release that release/install.sh downloads from:
#   https://github.com/phenomenon0/eureka-engine/releases/latest/download/{install.sh,manifest.json,<tarballs>}
#
# DIR must contain:
#   manifest.json  schema-1 manifest (template: release/dist-patch/site/manifest.json). Platforms with
#                  "status": "available" must have "url" = a tarball file name in DIR and its "sha256".
#   GATES.json     {"<platform>": {"verdict": "PASS", "receipt": "<path>"}, ...} for every available platform.
#   the tarballs   one per available platform.
#
# The script refuses to publish a platform whose gate verdict is not PASS, whose tarball is missing, or
# whose sha256 does not match. Without --publish it only prints what it would upload.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ $# -ge 2 ] || { echo "usage: $0 VERSION DIR [--publish]" >&2; exit 2; }
version=$1 dir=$2 publish=${3-}
[ -f "$dir/manifest.json" ] || { echo "missing $dir/manifest.json" >&2; exit 1; }
[ -f "$dir/GATES.json" ] || { echo "missing $dir/GATES.json" >&2; exit 1; }

out=$(mktemp -d -p "$dir" .publish-XXXX); trap 'rm -rf "$out"' EXIT
python3 -I - "$dir" "$version" "$out" <<'PY'
import hashlib, json, os, sys
d, version, out = sys.argv[1:4]
m = json.load(open(os.path.join(d, 'manifest.json')))
gates = json.load(open(os.path.join(d, 'GATES.json')))
files, bad = [], []
for plat, p in m['platforms'].items():
    if p.get('status') != 'available':
        print(f'skip      {plat}: status {p.get("status")}')
        continue
    g = gates.get(plat, {})
    if g.get('verdict') != 'PASS':
        bad.append(f'{plat}: gate verdict is {g.get("verdict")!r}, not PASS')
        continue
    f = os.path.join(d, p['url'])
    if '/' in p['url'] or not os.path.isfile(f):
        bad.append(f'{plat}: url must be a file name in {d}, got {p["url"]!r}')
        continue
    h = hashlib.sha256(open(f, 'rb').read()).hexdigest()
    if h != p['sha256']:
        bad.append(f'{plat}: sha256 {h} does not match manifest {p["sha256"]}')
        continue
    if p.get('version') != version:
        bad.append(f'{plat}: manifest version {p.get("version")!r} is not {version!r}')
        continue
    files.append(f)
    print(f'publish   {plat}: {p["url"]} ({os.path.getsize(f)} bytes, gate receipt {g.get("receipt")})')
if bad:
    print('refused:', *bad, sep='\n  ', file=sys.stderr); sys.exit(1)
if not files:
    print('refused: no platform is available', file=sys.stderr); sys.exit(1)
# install.sh parses this exact layout: 2-space indent, one field per line.
json.dump(m, open(os.path.join(out, 'manifest.json'), 'w'), indent=2); open(os.path.join(out, 'manifest.json'), 'a').write('\n')
open(os.path.join(out, 'files'), 'w').write('\n'.join(files) + '\n')
PY
cp release/install.sh "$out/install.sh"
mapfile -t tarballs < "$out/files"
echo "assets    : install.sh manifest.json ${tarballs[*]##*/}"
if [ "$publish" != --publish ]; then echo "(dry run: pass --publish to create release v$version)"; exit 0; fi
gh release create "v$version" --repo phenomenon0/eureka-engine --title "llb $version" \
  --notes "Install: curl -fsSL https://github.com/phenomenon0/eureka-engine/releases/latest/download/install.sh | sh" \
  "$out/install.sh" "$out/manifest.json" "${tarballs[@]}"
