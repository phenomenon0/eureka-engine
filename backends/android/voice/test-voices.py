#!/usr/bin/env python3
"""KOKORODROID device tests (no JUnit in the offline gradle cache, so these drive the installed debug app through
voice/page.py). The app must be running on the phone. Each device call takes /tmp/pixel6a.lock.
  voice/test-voices.py                                   # house voice + G2P parity
  voice/test-voices.py --packs DIR --host <lan-ip>       # + shop: resume, sha gate, install, remove
DIR holds hexgrad .pt packs (>= 2); --host is this machine's LAN address as the phone sees it."""

import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ADB = [os.environ.get("ADB", "adb")] + (["-s", os.environ["SERIAL"]] if os.environ.get("SERIAL") else [])
LOCK = ["flock", "/tmp/pixel6a.lock"]
fails = []


def page(expr):
    r = subprocess.run(
        LOCK + [os.path.join(HERE, "page.py"), expr], capture_output=True, text=True
    )
    if r.returncode:
        sys.exit(f"page.py failed: {r.stderr or r.stdout}")
    return json.loads(r.stdout)


def post(path, body):
    return page(
        f'await (await fetch({json.dumps(path)}, {{method: "POST", headers: {{"X-Api-Key": window.__BEND_NATIVE__.key}}, body: {json.dumps(json.dumps(body))}}})).json()'
    )


def check(name, ok, detail=""):
    print(
        ("ok   " if ok else "FAIL ")
        + name
        + (f"  ({detail})" if detail and not ok else "")
    )
    if not ok:
        fails.append(name)


def house_voice():
    # Owner pick (binding): bella is the bundle + playback default, and the picker leads bella, heart, michael.
    v = page('await (await fetch("/voices")).json()')
    ids = [x["id"] for x in v["voices"]]
    check("default voice is af_bella", v["default"] == "kokoro:af_bella", v["default"])
    check(
        "picker order starts bella, heart, michael",
        ids[:3] == ["kokoro:af_bella", "kokoro:af_heart", "kokoro:am_michael"],
        ids,
    )


def g2p_parity():
    # iPhone and Android must read a reply the same way: the Kotlin G2P has to match iOS KokoroG2P byte for byte.
    for line in open(os.path.join(HERE, "g2p-golden-ios.tsv"), encoding="utf-8"):
        lang, text, want = line.rstrip("\n").split("\t")
        got = post("/voices/g2p", {"text": text, "british": lang == "gb"})["phonemes"]
        check(f"g2p {lang} {text[:40]}", got == want, f"got {got!r}")


def shop(packs, host):
    # A pack is only installed when its bytes match the catalog's sha256; an interrupted download resumes (Range)
    # instead of starting over. One honest entry, one whose catalog sha256 is wrong.
    files = sorted(f for f in os.listdir(packs) if f.endswith(".pt"))[:2]
    root = tempfile.mkdtemp(prefix="voiceshop-test-")
    d = os.path.join(root, "v0", "voices")
    os.makedirs(os.path.join(d, "packs"))
    voices = []
    for i, f in enumerate(files):
        shutil.copy(os.path.join(packs, f), os.path.join(d, "packs", f))
        data = open(os.path.join(packs, f), "rb").read()
        sha = hashlib.sha256(data).hexdigest()
        voices.append(
            {
                "id": "test_" + f[:-3],
                "engine": "kokoro",
                "name": "Test " + f[:-3],
                "lang": "en-us",
                "gender": "female",
                "license": "Apache-2.0",
                "pack_url": "packs/" + f,
                "bytes": len(data),
                "file": "test_" + f,
                "sha256": sha if i == 0 else "0" * 64,
            }
        )
    json.dump(
        {"version": 1, "voices": voices}, open(os.path.join(d, "catalog.json"), "w")
    )
    good, bad = voices[0]["id"], voices[1]["id"]
    since = int(time.time() * 1000) - 5000  # engine.log lines are epoch ms; only this run counts
    srv = subprocess.Popen(
        [
            sys.executable,
            os.path.join(HERE, "..", "tools", "serve-models.py"),
            root,
            "--port",
            "8797",
            "--drop-at",
            "200000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1)
        s = post(
            "/voices/shop/source", {"url": f"http://{host}:8797/v0/voices/catalog.json"}
        )
        check(
            "shop lists the test catalog",
            s["listing"] == "ready" and {good, bad} <= {v["id"] for v in s["voices"]},
            s.get("why"),
        )
        for id in (good, bad):
            post("/voices/shop/download", {"id": id})
        for _ in range(60):
            rows = {
                v["id"]: v
                for v in page('await (await fetch("/voices/shop")).json()')["voices"]
            }
            if all(rows[i]["state"] != "downloading" for i in (good, bad)):
                break
            time.sleep(1)
        check("honest pack installs", rows[good]["state"] == "installed", rows[good])
        check(
            "wrong-sha pack is refused",
            rows[bad]["state"] == "failed" and "checksum" in (rows[bad]["error"] or ""),
            rows[bad],
        )
        mine = [
            v["id"] for v in page('await (await fetch("/voices")).json()')["voices"]
        ]
        check("installed pack is in the picker", "kokoro:" + good in mine, mine)
        check("refused pack is not in the picker", "kokoro:" + bad not in mine, mine)
        log = (
            subprocess.run(
                LOCK
                + ADB
                + [
                    "shell",  # one string: adb shell re-splits its argv
                    f"grep -a 'file {voices[0]['file']} ok' /sdcard/Android/data/dev.bend.app/files/engine.log",
                ],
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
        log = [l for l in log if int(l.split()[0]) > since]
        check(
            "dropped download resumed from 200000",
            bool(log) and "resumed_from=200000" in log[-1],
            log[-1:] or "no log line",
        )
        r = post("/voices/preview", {"id": "kokoro:" + good, "play": False})
        check("installed pack speaks", r.get("audio_s", 0) > 1, r)
    finally:
        for id in (good, bad):
            post("/voices/shop/remove", {"id": id})
        post("/voices/shop/source", {"url": ""})
        srv.terminate()
        shutil.rmtree(root)
    mine = [v["id"] for v in page('await (await fetch("/voices")).json()')["voices"]]
    check("remove takes it out of the picker", "kokoro:" + good not in mine, mine)


a = argparse.ArgumentParser()
a.add_argument("--packs")
a.add_argument("--host")
args = a.parse_args()
house_voice()
g2p_parity()
if args.packs:
    shop(args.packs, args.host)
print(f"{len(fails)} failed" if fails else "all ok")
sys.exit(1 if fails else 0)
