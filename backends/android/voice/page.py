#!/usr/bin/env python3
"""Evaluate JS in the app's WebView (debug build: setWebContentsDebuggingEnabled) — how the receipts drive the routes
with the page's own per-launch key. Run under the device lock:
  flock /tmp/pixel6a.lock voice/page.py 'await (await fetch("/voices")).json()'
  flock /tmp/pixel6a.lock voice/page.py --nav /voices.html
The expression runs as an async function body's return value; its JSON is printed."""

import json, os, subprocess, sys, urllib.request
import websocket

ADB = [os.environ.get("ADB", "adb")] + (["-s", os.environ["SERIAL"]] if os.environ.get("SERIAL") else [])
PORT = 9339
args = sys.argv[1:]
if args[:1] == ["--serial"]:
    ADB[1:] = ["-s", args[1]]
    args = args[2:]
pid = subprocess.run(
    ADB + ["shell", "pidof", "dev.bend.app"], capture_output=True, text=True
).stdout.strip()
if not pid:
    sys.exit("dev.bend.app is not running")
subprocess.run(
    ADB + ["forward", f"tcp:{PORT}", f"localabstract:webview_devtools_remote_{pid}"],
    check=True,
    capture_output=True,
)
try:
    page = next(
        t
        for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
        if t["type"] == "page"
    )
    ws = websocket.create_connection(
        page["webSocketDebuggerUrl"], timeout=180, suppress_origin=True
    )
    if args[:1] == ["--nav"]:
        expr = f"location.href = {json.dumps(args[1])}; 'ok'"
    else:
        expr = f"(async () => JSON.stringify(await (async () => {{ return {args[0]} }})()))()"
    ws.send(
        json.dumps(
            {
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": expr,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            }
        )
    )
    while True:
        m = json.loads(ws.recv())
        if m.get("id") == 1:
            break
    r = m["result"]
    if "exceptionDetails" in r:
        sys.exit(json.dumps(r["exceptionDetails"], indent=1))
    v = r["result"].get("value")
    try:
        print(json.dumps(json.loads(v), indent=1, ensure_ascii=False))
    except Exception:
        print(v)
finally:
    subprocess.run(ADB + ["forward", "--remove", f"tcp:{PORT}"], capture_output=True)
