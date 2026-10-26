#!/usr/bin/env python3
"""G5 template parity: llb-serve's C chat renderer (serve.c + mm.c, via render_driver) vs the
model's own chat_template.jinja compiled by transformers' chat-template machinery.
  parity.py --driver render_driver --template flash-chat_template.jinja --fix FIXDIR --out OUTDIR
Writes OUTDIR/cases/*.json (the requests), OUTDIR/parity.json; exit 1 on any mismatch."""
import argparse, base64, hashlib, json, os, subprocess, sys

ap = argparse.ArgumentParser()
for k in ("driver", "template", "fix", "out"): ap.add_argument("--" + k, required=True)
a = ap.parse_args()
from transformers.utils.chat_template_utils import _compile_jinja_template

def uri(f, mime):
    return f"data:{mime};base64," + base64.b64encode(open(os.path.join(a.fix, f), "rb").read()).decode()

PNG, JPG, WEBP = uri("a.png", "image/png"), uri("coco142472.jpg", "image/jpeg"), uri("a-lossy.webp", "image/webp")
img = lambda u, **kw: {"type": "image_url", "image_url": {"url": u, **kw}}
txt = lambda t: {"type": "text", "text": t}
C = {
    "text-string": {"messages": [{"role": "user", "content": "Hello there"}]},
    "text-parts": {"messages": [{"role": "user", "content": [txt("Hello"), txt(" there ")]}]},
    "system-user": {"messages": [{"role": "system", "content": "  Be terse.\n"}, {"role": "user", "content": "hi"}]},
    "img1-coco": {"messages": [{"role": "user", "content": [img(JPG), txt("Is there a car in the image?")]}]},
    "img1-after-text": {"messages": [{"role": "user", "content": [txt("Describe: "), img(PNG, detail="high")]}]},
    "img1-only": {"messages": [{"role": "user", "content": [img(PNG)]}]},
    "img2": {"messages": [{"role": "user", "content": [txt("Compare "), img(PNG), txt(" and "), img(WEBP), txt(".")]}]},
    "img2-adjacent": {"messages": [{"role": "user", "content": [img(JPG), img(PNG), txt("Which is red?")]}]},
    "img-type-image": {"messages": [{"role": "user", "content": [{"type": "image", "image": PNG}, txt("what?")]}]},
    "img-multiturn": {"messages": [
        {"role": "system", "content": "You see images."},
        {"role": "user", "content": [img(PNG), txt("Colour?")]},
        {"role": "assistant", "content": "<think>\nIt is red.\n</think>\n\nRed."},
        {"role": "user", "content": [txt("And this one? "), img(WEBP)]}]},
    "img-thinking-off": {"messages": [{"role": "user", "content": [img(JPG), txt(" car? ")]}],
                         "chat_template_kwargs": {"enable_thinking": False}},
    "img-no-genprompt": {"messages": [{"role": "user", "content": [img(PNG)]}], "add_generation_prompt": False},
    "assistant-parts-after-last-user": {"messages": [
        {"role": "user", "content": [img(PNG), txt("go")]},
        {"role": "assistant", "content": [txt("Sure: red")], "reasoning_content": "look at pixels"}],
        "add_generation_prompt": False},
    "whitespace-trim": {"messages": [{"role": "user", "content": [txt("\n\n  "), img(PNG), txt("  \n")]}]},
    "unicode": {"messages": [{"role": "user", "content": [txt("café 🚗 "), img(PNG), txt(" ¿coche?")]}]},
}
E = {  # the template raises -> the server must 400
    "err-system-image": {"messages": [{"role": "system", "content": [img(PNG)]}, {"role": "user", "content": "x"}]},
    "err-unknown-part": {"messages": [{"role": "user", "content": [{"type": "input_file", "file": {}}]}]},
}
os.makedirs(f"{a.out}/cases", exist_ok=True)
files = []
for name, body in {**C, **E}.items():
    p = f"{a.out}/cases/{name}.json"; json.dump(body, open(p, "w")); files.append(p)
out = subprocess.run([a.driver, *files], capture_output=True, text=True, check=True).stdout.splitlines()
got = {os.path.basename(json.loads(l)["file"])[:-5]: json.loads(l) for l in out}
tpl = _compile_jinja_template(open(a.template).read())
res, bad = [], 0
for name, body in {**C, **E}.items():
    kw = dict(messages=body["messages"], add_generation_prompt=body.get("add_generation_prompt", True), tools=None,
              **body.get("chat_template_kwargs", {}))
    try:
        ref, ref_err = tpl.render(**kw), None
    except Exception as ex:  # TemplateError from raise_exception
        ref, ref_err = None, str(ex)
    g = got[name]
    if ref is not None:
        ok = g["status"] == 200 and g["prompt"] == ref
    else:
        ok = g["status"] == 400 and ref_err.split(":")[-1].strip() in g.get("error", "")
    bad += not ok
    r = {"case": name, "match": ok, "server_status": g["status"], "jinja_error": ref_err,
         "prompt_sha256": hashlib.sha256((g.get("prompt") or "").encode()).hexdigest(),
         "jinja_sha256": hashlib.sha256((ref or "").encode()).hexdigest(), "n_images": len(g.get("media", [])),
         "server_error": g.get("error")}
    if not ok: r["server_prompt"], r["jinja_prompt"] = g.get("prompt"), ref
    res.append(r); print(("MATCH " if ok else "DIFF  ") + name, r["n_images"], g["status"], ref_err or "")
json.dump({"template_sha256": hashlib.sha256(open(a.template, "rb").read()).hexdigest(), "cases": res,
           "matched": len(res) - bad, "total": len(res)}, open(f"{a.out}/parity.json", "w"), indent=1)
print(f"{len(res) - bad}/{len(res)} byte-identical to the jinja template"); sys.exit(1 if bad else 0)
