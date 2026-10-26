#!/usr/bin/env python3
"""tools/toolcall/run_tests.py - gate for host/llb_toolcall.h (G1 structured output).

  /tmp/g1/venv/bin/python tools/toolcall/run_tests.py [--src REPO] [--build DIR] [--template JINJA]

1. render: >= 40 conversations through the HF chat template (jinja2 sandbox the way
   transformers sets it up) and through llb_tc_render; byte-identical output required
   (error cases: both sides must raise).
2. round-trip: every conversation with an assistant tool-call turn is cut after that turn,
   rendered, and the assistant body fed to the streaming parser at every chunking
   (whole, 1-byte, all 2-chunk splits, random); identical events at every chunking and
   {name, arguments} semantically equal to the original (bool != int, key order kept).
3. adversarial streams with hand-written expected events.
Everything runs on the -O2 -Werror build and again on the ASan/UBSan build.
"""
import argparse, copy, json, os, shutil, subprocess, sys

ap = argparse.ArgumentParser()
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--src", default=os.path.normpath(os.path.join(here, "..", "..")))
ap.add_argument("--build", default="/tmp/g1/tc")
ap.add_argument("--template", required=True)
ap.add_argument("--nrandom", type=int, default=40)
A = ap.parse_args()

PASS = FAIL = 0
def ok(cond, what, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"PASS {what}")
    else:
        FAIL += 1; print(f"FAIL {what}")
        if detail: print("     " + detail.replace("\n", "\n     "))

# ---------------------------------------------------------------- build
os.makedirs(A.build, exist_ok=True)
for f in ["host/llb_json.h", "host/llb_json_impl.h", "host/llb_toolcall.h", "host/llb_toolcall_impl.h",
          "host/llb_toolcall.c", "tools/toolcall/tc_test.c"]:
    shutil.copy(os.path.join(A.src, f), A.build)
print("gcc:", subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0])
W = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-I", A.build]
builds = [
    ("module", ["gcc", *W, "-O2", "-c", f"{A.build}/llb_toolcall.c", "-o", f"{A.build}/llb_toolcall.o"]),
    ("opt", ["gcc", *W, "-O2", f"{A.build}/tc_test.c", "-lm", "-o", f"{A.build}/tc_test"]),
    ("asan", ["gcc", *W, "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
              f"{A.build}/tc_test.c", "-lm", "-o", f"{A.build}/tc_test_asan"]),
]
# fallback when gcc's libasan runtime is not installed (this box: /usr/lib64/libasan.so.8 missing):
# a native clang (emsdk's upstream build) against the system clang-21 compiler-rt
SAN_CLANG = os.environ.get("SAN_CLANG", "clang")
SAN_FALLBACK = [SAN_CLANG, "--target=x86_64-redhat-linux-gnu", "-resource-dir", "/usr/lib/clang/21", "-fuse-ld=bfd", *W, "-O1", "-g",
                "-fsanitize=address,undefined", "-fno-sanitize-recover=all", f"{A.build}/tc_test.c", "-lm", "-o", f"{A.build}/tc_test_asan"]
for name, cmd in builds:
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("build", name, ":", " ".join(cmd))
    if r.returncode and name == "asan" and os.path.exists(SAN_CLANG):
        print("  gcc sanitizer build failed:", r.stderr.strip().splitlines()[-2:])
        cmd = SAN_FALLBACK
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("build asan (fallback):", " ".join(cmd))
        print("  ", subprocess.run([SAN_CLANG, "--version"], capture_output=True, text=True).stdout.splitlines()[0])
    ok(r.returncode == 0, f"build {name}", r.stderr)
    if r.returncode: sys.exit(1)
nm = subprocess.run(["nm", f"{A.build}/tc_test_asan"], capture_output=True, text=True).stdout
ok("__asan_report_load1" in nm and "__ubsan_handle" in nm, "asan binary is instrumented (asan + ubsan symbols present)")
BINS = [("opt", f"{A.build}/tc_test"), ("asan", f"{A.build}/tc_test_asan")]

# ---------------------------------------------------------------- HF reference renderer
from jinja2 import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
def raise_exception(msg): raise TemplateError(msg)
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
TMPL = env.from_string(open(A.template, encoding="utf-8").read())

def hf_render(req, gen, think):
    msgs = copy.deepcopy(req["messages"]) if "messages" in req else []
    for m in msgs if isinstance(msgs, list) else []:
        for tc in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
            f = tc.get("function", tc)
            if isinstance(f.get("arguments"), str):  # OpenAI wire form -> mapping (what vLLM/HF servers do)
                f["arguments"] = json.loads(f["arguments"]) if f["arguments"].strip() else {}
    kw = {"messages": msgs, "add_generation_prompt": gen}
    if "tools" in req: kw["tools"] = req["tools"]
    if think is not None: kw["enable_thinking"] = think
    try: return TMPL.render(**kw), None
    except Exception as e: return None, f"{type(e).__name__}: {e}"

def llb_render(binp, raw, gen, think):
    r = subprocess.run([binp, "render", str(int(gen)), "0" if think is False else "1"], input=raw.encode(), capture_output=True)
    out = r.stdout.decode("utf-8", "replace")
    if r.returncode == 3: return None, out
    if r.returncode != 0: return None, f"CRASH rc={r.returncode} {r.stderr.decode()[-600:]}"
    return out, None

# ---------------------------------------------------------------- fixtures
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather — «now» in °C",
    "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City, e.g. \"Paris\""},
    "unit": {"type": "string", "enum": ["c", "f"]}, "days": {"type": "integer"}}, "required": ["location"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "description": "Search\ttabs\nand\\slashes \u0001",
    "parameters": {"properties": {"query": {"type": "string"}, "filters": {"type": "object"}, "tags": {"type": "array", "items": {"type": "string"}},
    "limit": {"type": "integer"}, "exact": {"type": "boolean"}, "score": {"type": "number"}, "note": {"type": ["string", "null"]}},
    "type": "object", "required": ["query"]}}}
EVENT = {"type": "function", "function": {"name": "make_event", "parameters": {"type": "object", "properties": {
    "title": {"type": "string"}, "when": {"type": "object", "properties": {"start": {"type": "string"}}},
    "attendees": {"type": "array"}, "priority": {"type": "integer"}, "meta": {"type": "dict"}}}}}
BARE = {"name": "run_code", "description": "no type/function wrapper", "parameters": {"type": "object",
    "properties": {"code": {"type": "string"}, "timeout": {"type": "number"}}}}
TOOLS = [WEATHER, SEARCH, EVENT]
ALLT = [WEATHER, SEARCH, EVENT, BARE]

def tc(name, args, wrap=True, as_str=False, cid="call_1"):
    a = json.dumps(args, ensure_ascii=False) if as_str else args
    f = {"name": name, "arguments": a}
    return {"id": cid, "type": "function", "function": f} if wrap else f
U = lambda c: {"role": "user", "content": c}
S = lambda c: {"role": "system", "content": c}
T = lambda c: {"role": "tool", "content": c}
def AS(c=None, calls=None, rc=None):
    m = {"role": "assistant", "content": c}
    if calls is not None: m["tool_calls"] = calls
    if rc is not None: m["reasoning_content"] = rc
    return m

C = []  # (name, request or raw-json string, gen, think)
def add(name, req, gen=True, think=None): C.append((name, req, gen, think))

add("user only", {"messages": [U("hi")]})
add("user only thinking off", {"messages": [U("hi")]}, think=False)
add("user only thinking on", {"messages": [U("hi")]}, think=True)
add("no gen prompt", {"messages": [U("hi")]}, gen=False)
add("system+user", {"messages": [S("You are terse."), U("2+2?")]})
add("system empty no tools", {"messages": [S(""), U("x")]})
add("list content", {"messages": [S([{"type": "text", "text": "  sys A "}, {"type": "text", "text": "B\n"}]), U([{"type": "text", "text": "part1 "}, {"text": "part2"}])]})
add("tools no system", {"messages": [U("weather in Paris?")], "tools": TOOLS})
add("tools + system", {"messages": [S("Be helpful."), U("weather?")], "tools": TOOLS})
add("tools + whitespace system", {"messages": [S(" \n\t "), U("weather?")], "tools": TOOLS})
add("tools bare + key order", {"messages": [U("run it")], "tools": [BARE, {"z": 1, "a": [True, None, 1.5], "name": "odd"}]})
add("empty tools list", {"messages": [S("s"), U("q")], "tools": []}, think=False)
add("multi-turn plain", {"messages": [U("a"), AS("b"), U("c")]})
add("reasoning_content before and after lqi", {"messages": [U("q1"), AS("a1", rc="old thought"), U("q2"), AS("a2", rc="  new thought\n")]}, gen=False)
add("think split in content", {"messages": [U("q"), AS("<think>\nI ponder\n</think>\n\nThe answer")]}, gen=False)
add("think split multi tags", {"messages": [U("q"), AS("pre<think>a</think>mid<think>\nb\n</think>\n\nend")]}, gen=False)
add("reasoning_content wins over split", {"messages": [U("q"), AS("x</think>y", rc="R")]}, gen=False)
add("reasoning null -> split", {"messages": [U("q"), AS("\n\nr1\n</think>\nc1", rc=None)]}, gen=False)
add("tool loop wrapped", {"messages": [U("weather Paris"), AS("", [tc("get_weather", {"location": "Paris", "unit": "c"})]),
    T('{"temp": 21}'), AS("It is 21C.")], "tools": TOOLS}, think=True)
add("args as string", {"messages": [U("weather Oslo"), AS(None, [tc("get_weather", {"location": "Oslo", "days": 3}, as_str=True)])], "tools": TOOLS}, gen=False)
add("no function wrapper", {"messages": [U("weather"), AS("", [tc("get_weather", {"location": "Rome"}, wrap=False)])], "tools": TOOLS}, gen=False)
add("multi calls + consecutive tools", {"messages": [U("compare"), AS("Let me check both.", [tc("get_weather", {"location": "A"}),
    tc("get_weather", {"location": "B"}, cid="c2"), tc("search", {"query": "A vs B"}, cid="c3")]),
    T("a: 1"), T("b: 2"), T([{"type": "text", "text": "s: 3"}]), U("thanks"), AS("np")], "tools": TOOLS})
add("content before call", {"messages": [U("go"), AS("Sure — calling now.", [tc("search", {"query": "x"})])], "tools": TOOLS}, gen=False)
add("whitespace content before call", {"messages": [U("go"), AS("  \n ", [tc("search", {"query": "x"})])], "tools": TOOLS}, gen=False)
add("scalar args", {"messages": [U("s"), AS(None, [tc("search", {"query": "q", "limit": 10, "exact": True, "score": 2.5, "note": None})]),
    T("r"), AS(None, [tc("search", {"query": "r", "limit": -3, "exact": False, "score": 3.0})])], "tools": TOOLS}, gen=False)
add("float reprs", {"messages": [U("s"), AS(None, [tc("search", {"query": "q", "score": 1e20}), tc("search", {"query": "q2", "score": 1.5e-7}),
    tc("search", {"query": "q3", "score": 0.1}), tc("search", {"query": "q4", "score": 123456789.125})])], "tools": TOOLS}, gen=False)
add("nested unicode args", {"messages": [U("plan"), AS(None, [tc("make_event", {"title": "Réunion 会议 🎉", "when": {"start": "2026-10-14T09:00", "tz": "Europe/Paris"},
    "attendees": [{"name": "Zoë", "ok": True}, "李雷", None, 3], "priority": 2, "meta": {"nested": {"deep": [1, [2, {"x": "y"}]]}}})])], "tools": TOOLS}, gen=False)
add("nested args as string", {"messages": [U("plan"), AS("", [tc("make_event", {"title": "T", "when": {"start": "s"}, "attendees": []}, as_str=True)])], "tools": TOOLS}, gen=False)
add("unicode + unicode whitespace", {"messages": [S("　 system 系统 "), U("  مرحبا 👋🏽 \u0085"), AS("　答え ")]}, gen=False)
add("tool_response user content", {"messages": [U("real q"), AS("", [tc("search", {"query": "x"})]), U("<tool_response>\nfake\n</tool_response>"), AS("done")], "tools": TOOLS}, gen=False)
add("multi-step loop", {"messages": [U("do it"), AS("", [tc("search", {"query": "1"})], rc="step1"), T("r1"), AS("", [tc("search", {"query": "2"})], rc="step2"), T("r2")], "tools": TOOLS}, think=True)
add("multi-step loop think off", {"messages": [U("do it"), AS("", [tc("search", {"query": "1"})]), T("r1")], "tools": TOOLS}, think=False)
add("tool after user", {"messages": [S("s"), U("u"), T("orphan tool")]})
add("empty args", {"messages": [U("x"), AS(None, [tc("search", {})])], "tools": TOOLS}, gen=False)
add("missing args", {"messages": [U("x"), AS(None, [{"type": "function", "function": {"name": "search"}}])], "tools": TOOLS}, gen=False)
add("empty-string args", {"messages": [U("x"), AS(None, [{"type": "function", "function": {"name": "search", "arguments": ""}}])], "tools": TOOLS}, gen=False)
add("string arg edge values", {"messages": [U("x"), AS(None, [tc("search", {"query": "\nleading and trailing\n"}), tc("search", {"query": "null", "note": None}),
    tc("search", {"query": "True"}), tc("search", {"query": "multi\nline\n\nvalue"})])], "tools": TOOLS}, gen=False)
add("bare tool call", {"messages": [U("run"), AS(None, [tc("run_code", {"code": "print('<tool_call>')\nx = a < b", "timeout": 1.5})])], "tools": ALLT}, gen=False)
add("long mixed", {"messages": [S("sys"), U("q1"), AS("<think>t1</think>a1"), U("q2"), AS("", [tc("get_weather", {"location": "X"})], rc="r2"),
    T("w"), AS("ans2", rc="r3"), U("q3")], "tools": TOOLS}, think=True)
add("content null", {"messages": [U("x"), AS(None)]}, gen=False)
add("assistant content list", {"messages": [U("x"), AS([{"type": "text", "text": "list reply"}])]}, gen=False)
add("tools as mapping (ignored)", {"messages": [U("x")], "tools": {"a": 1}})
# raw JSON: number lexemes and duplicate keys only expressible as text
add("raw number lexemes", '{"messages":[{"role":"user","content":"n"},{"role":"assistant","content":null,"tool_calls":[{"function":'
    '{"name":"f","arguments":{"a":1.0e2,"b":-0,"c":12345678901234567890123,"d":1E-7,"e":[1.50,2E+3],"f":-0.0}}}]}],'
    '"tools":[{"type":"function","function":{"name":"f","parameters":{"x":1.0e2}}}]}', gen=False)
add("raw duplicate keys", '{"messages":[{"role":"user","content":"d"},{"role":"assistant","content":"","tool_calls":[{"function":'
    '{"name":"f","arguments":{"k":1,"j":2,"k":{"z":1,"z":2}}}}]}]}', gen=False)
add("raw escapes", '{"messages":[{"role":"user","content":"tab\\there \\u00e9 \\ud83d\\ude00 \\u001f end"}],'
    '"tools":[{"type":"function","function":{"name":"g","description":"q\\"uote \\u007f \\u2028"}}]}')
# error cases (both sides must raise)
add("ERR no messages", {"messages": []})
add("ERR system not first", {"messages": [U("a"), S("late")]})
add("ERR bad role", {"messages": [U("a"), {"role": "robot", "content": "x"}]})
add("ERR no user query", {"messages": [S("s"), U("<tool_response>\nx\n</tool_response>")]})
add("UNSUP image", {"messages": [U([{"type": "image", "image": "x.png"}, {"type": "text", "text": "what"}])]})
add("ERR system image", {"messages": [S([{"type": "image_url", "image_url": {"url": "u"}}]), U("q")]})
add("ERR content number", {"messages": [U(5)]})
add("ERR item without text", {"messages": [U([{"type": "text"}])]})
add("ERR call without name", {"messages": [U("x"), AS(None, [{"function": {"arguments": {}}}])]})
add("ERR args null", {"messages": [U("x"), AS(None, [{"function": {"name": "f", "arguments": None}}])]})

# ---------------------------------------------------------------- 1. render parity
print(f"\n== render parity: {len(C)} conversations x {len(BINS)} builds ==")
n_err_cases = 0
for name, req, gen, think in C:
    raw = req if isinstance(req, str) else json.dumps(req, ensure_ascii=False)
    obj = json.loads(raw)
    want, werr = hf_render(obj, gen, think)
    if name.startswith("ERR"): n_err_cases += 1
    for bname, binp in BINS:
        got, gerr = llb_render(binp, raw, gen, think)
        if name.startswith("UNSUP"):  # vision items: HF renders pads, llb refuses (documented)
            ok(got is None and "unsupported" in (gerr or ""), f"render[{bname}] {name}: llb refuses", f"llb: {gerr} / {got!r}")
        elif want is None:
            ok(got is None and not (gerr or "").startswith("CRASH"), f"render[{bname}] {name}: both raise", f"hf: {werr}\nllb: {gerr} / {got!r}")
        else:
            d = ""
            if got != want:
                i = next((k for k in range(min(len(got or ""), len(want))) if got[k] != want[k]), min(len(got or ""), len(want)))
                d = f"llb err: {gerr}\nfirst diff at {i}:\n hf : {want[max(0,i-60):i+60]!r}\n llb: {(got or '')[max(0,i-60):i+60]!r}"
            ok(got == want, f"render[{bname}] {name} ({len(want)} B)", d)
print(f"render cases: {len(C)} ({len(C) - n_err_cases} renders, {n_err_cases} raise cases)")

# ---------------------------------------------------------------- 2. round trip
def jeq(a, b):
    if isinstance(a, bool) or isinstance(b, bool): return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)): return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a.keys()) == list(b.keys()) and all(jeq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list): return len(a) == len(b) and all(jeq(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b

def tstr(v):  # the template's `args_value | string` / tojson
    if isinstance(v, (dict, list)): return json.dumps(v, ensure_ascii=False)
    return v if isinstance(v, str) else str(v)

def parse(binp, stream, tools, reasoning, seed):
    tf = "-"
    if tools is not None:
        tf = f"{A.build}/tools.json"
        open(tf, "w", encoding="utf-8").write(json.dumps(tools, ensure_ascii=False))
    r = subprocess.run([binp, "parse", tf, str(int(reasoning)), str(A.nrandom), str(seed)], input=stream.encode(), capture_output=True)
    return r.returncode, r.stdout.decode(), r.stderr.decode()

def coalesce(evs):
    out = []
    for e in evs:
        if out and e["type"] in ("content", "reasoning") and out[-1]["type"] == e["type"]: out[-1]["text"] += e["text"]
        else: out.append(dict(e))
    return out

print("\n== round trip (render -> stream parse at every chunking) ==")
nrt = 0
for ci, (name, req, gen, think) in enumerate(C):
    if name.startswith(("ERR", "UNSUP")): continue
    obj = json.loads(req) if isinstance(req, str) else req
    msgs = obj["messages"]
    idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("tool_calls")]
    if not idx: continue
    cut = dict(obj, messages=msgs[: idx[-1] + 1])
    text, err = hf_render(cut, False, None)
    if text is None: continue
    nrt += 1
    m = msgs[idx[-1]]
    body = text[text.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n"):]
    assert body.endswith("<|im_end|>\n"); body = body[: -len("<|im_end|>\n")]
    reasoning = body.startswith("<think>\n")
    if reasoning: body = body[len("<think>\n"):]
    want_calls = []
    for c in m["tool_calls"]:
        f = c.get("function", c); a = f.get("arguments", {})
        if isinstance(a, str): a = json.loads(a) if a.strip() else {}
        props = {}
        for t in obj.get("tools") or []:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            if isinstance(fn, dict) and fn.get("name") == f["name"]:
                props = ((fn.get("parameters") or {}).get("properties")) or {}
        # a param without a schema entry comes back as the template's text (vLLM behaviour)
        a = {k: (v if k in props else tstr(v)) for k, v in a.items()}
        want_calls.append((f["name"], a))
    content = m.get("content") if isinstance(m.get("content"), str) else "".join(x["text"] for x in (m.get("content") or []))
    rc = m.get("reasoning_content")
    want_reason = rc.strip() if isinstance(rc, str) else (content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip() if "</think>" in content else "")
    if not isinstance(rc, str) and "</think>" in content: content = content.split("</think>")[-1].lstrip("\n")
    for bname, binp in BINS:
        rcode, out, errs = parse(binp, body, obj.get("tools"), reasoning, ci)
        what = f"roundtrip[{bname}] {name} ({len(body)} B, {errs.strip()})"
        if rcode != 0: ok(False, what, f"rc={rcode} {errs[-800:]}"); continue
        evs = coalesce(json.loads(out))
        calls = [(e["name"], e["arguments"]) for e in evs if e["type"] == "tool_call"]
        got_c = "".join(e["text"] for e in evs if e["type"] == "content")
        got_r = "".join(e["text"] for e in evs if e["type"] == "reasoning")
        errs_ev = [e for e in evs if e["type"] == "error"]
        good = (not errs_ev and len(calls) == len(want_calls) and all(a[0] == b[0] and jeq(a[1], b[1]) for a, b in zip(calls, want_calls))
                and got_c.strip() == content.strip() and (not reasoning or got_r.strip() == want_reason))
        ok(good, what, f"want calls {want_calls}\ngot  {evs}\nwant content {content.strip()!r} reasoning {want_reason!r}")
print(f"round-trip conversations: {nrt}")

# ---------------------------------------------------------------- 3. adversarial streams
TYPES = [{"type": "function", "function": {"name": "types", "parameters": {"type": "object", "properties": {
    "s": {"type": "string"}, "i": {"type": "integer"}, "i2": {"type": "integer"}, "n": {"type": "number"}, "n2": {"type": "number"},
    "b": {"type": "boolean"}, "b2": {"type": "boolean"}, "o": {"type": "object"}, "a": {"type": "array"},
    "nl": {"type": ["integer", "null"]}, "any": {"type": "whatever"}, "notype": {"description": "no type"}}}}}]
def P(params, fn="f"):
    return "<tool_call>\n<function=" + fn + ">\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params) + "</function>\n</tool_call>"
CT = lambda t: {"type": "content", "text": t}
RS = lambda t: {"type": "reasoning", "text": t}
ER = lambda t: {"type": "error", "text": t}
CALL = lambda n, a: {"type": "tool_call", "name": n, "arguments": a}
UNT = "<tool_call>\n<function=get_weather>\n<parameter=location>\nPar"
ADV = [
    ("'<tool' at end", None, 0, "Hello <tool", [CT("Hello <tool")]),
    ("'<tool_cal' then 'l>'", None, 0, "abc<tool_cal" + "l>\n<function=f>\n</function>\n</tool_call>", [CT("abc"), CALL("f", {})]),
    ("text with <tool_x>", None, 0, "x <tool_x> y <tool_call_not> z <", [CT("x <tool_x> y <tool_call_not> z <")]),
    ("unterminated call", TOOLS, 0, "pre " + UNT, [CT("pre "), ER("unterminated <tool_call>"), CT(UNT)]),
    ("utf8 everywhere", TOOLS, 0, "héllo 世界 🎉 " + P([("location", "東京 🗼")], "get_weather"), [CT("héllo 世界 🎉 "), CALL("get_weather", {"location": "東京 🗼"})]),
    ("two calls back to back", TOOLS, 0, P([("location", "A")], "get_weather") + P([("query", "B")], "search"),
        [CALL("get_weather", {"location": "A"}), CALL("search", {"query": "B"})]),
    ("two calls newline-separated", TOOLS, 0, "\n" + P([("location", "A")], "get_weather") + "\n" + P([("query", "B")], "search") + "\n\n",
        [CT("\n"), CALL("get_weather", {"location": "A"}), CALL("search", {"query": "B"})]),
    ("malformed body", None, 0, "<tool_call>garbage</tool_call>after", [ER("tool call: expected <function="), CT("<tool_call>garbage</tool_call>after")]),
    ("empty call", None, 0, "<tool_call></tool_call>", [ER("tool call: no <function=...> block"), CT("<tool_call></tool_call>")]),
    ("stray text in function", None, 0, "<tool_call>\n<function=f>\nhello\n</function>\n</tool_call>",
        [ER("tool call: unexpected text in function f"), CT("<tool_call>\n<function=f>\nhello\n</function>\n</tool_call>")]),
    ("text after call kept", None, 0, P([("a", "1")]) + "\nTrailing words", [CALL("f", {"a": "1"}), CT("\nTrailing words")]),
    ("reasoning split", None, 1, "think about <tool_call> stuff\n</think>\n\nanswer " + P([]),
        [RS("think about <tool_call> stuff\n"), CT("answer "), CALL("f", {})]),
    ("reasoning unterminated", None, 1, "pondering </thi", [RS("pondering </thi")]),
    ("reasoning then only newlines", None, 1, "r</think>\n\n\n", [RS("r")]),
    ("content-mode leading newlines kept", None, 0, "\n\nhi", [CT("\n\nhi")]),
    ("schema typing", TYPES, 0, P([("s", "null"), ("i", " -007 "), ("i2", "1.5"), ("n", "2.50"), ("n2", "3.0"), ("b", "TRUE"), ("b2", "yes"),
        ("o", "{bad"), ("a", '[1, {"x": null}]'), ("nl", "null"), ("any", "True"), ("notype", "42"), ("extra", "42")], "types"),
        [CALL("types", {"s": "null", "i": -7, "i2": "1.5", "n": 2.5, "n2": 3, "b": True, "b2": "yes", "o": "{bad", "a": [1, {"x": None}],
                        "nl": None, "any": True, "notype": "42", "extra": "42"})]),
    ("nullable string", [SEARCH], 0, P([("query", "null"), ("note", "None")], "search") + P([("query", "x"), ("note", "null")], "search"),
        [CALL("search", {"query": "null", "note": None}), CALL("search", {"query": "x", "note": None})]),
    ("missing </parameter>", None, 0, "<tool_call>\n<function=f>\n<parameter=a>\nx\n<parameter=b>\ny\n</function>\n</tool_call>", [CALL("f", {"a": "x", "b": "y"})]),
    ("missing </function>", None, 0, "<tool_call>\n<function=f>\n<parameter=a>\nx\n</parameter>\n</tool_call>", [CALL("f", {"a": "x"})]),
    ("two functions in one call", None, 0, "<tool_call>\n<function=f>\n</function>\n<function=g>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>",
        [CALL("f", {}), CALL("g", {"k": "v"})]),
    ("value with < and partial tags", None, 0, P([("code", "if a<b: print('</par')\n<tool_call x")]), [CALL("f", {"code": "if a<b: print('</par')\n<tool_call x"})]),
    ("multiline value strip one newline", None, 0, P([("v", "\nline1\nline2\n")]), [CALL("f", {"v": "\nline1\nline2\n"})]),
    ("duplicate params", None, 0, P([("a", "1"), ("b", "2"), ("a", "3")]), [CALL("f", {"a": "3", "b": "2"})]),
    ("empty stream", None, 0, "", []),
    ("utf8 split near tag", None, 0, "é<tool_call" + "é", [CT("é<tool_callé")]),
    ("call then unterminated", None, 0, P([]) + "\n<tool_call>\n<function=g>", [CALL("f", {}), ER("unterminated <tool_call>"), CT("<tool_call>\n<function=g>")]),
]
print(f"\n== adversarial streams: {len(ADV)} ==")
for ai, (name, tools, reasoning, stream, want) in enumerate(ADV):
    for bname, binp in BINS:
        rcode, out, errs = parse(binp, stream, tools, reasoning, 1000 + ai)
        what = f"adversarial[{bname}] {name} ({errs.strip()})"
        if rcode != 0: ok(False, what, f"rc={rcode} {errs[-800:]}"); continue
        got = [{k: v for k, v in e.items() if k != "index"} for e in coalesce(json.loads(out))]
        idx_ok = [e["index"] for e in json.loads(out) if e["type"] == "tool_call"] == list(range(sum(e["type"] == "tool_call" for e in want)))
        good = idx_ok and len(got) == len(want) and all(g["type"] == w["type"] and (jeq(g.get("arguments"), w.get("arguments")) if w["type"] == "tool_call" and g["name"] == w["name"] else g == w) for g, w in zip(got, want))
        ok(good, what, f"want {want}\ngot  {got}")

print(f"\nSUMMARY: {PASS} PASS, {FAIL} FAIL  (render cases {len(C)}, round-trip convs {nrt}, adversarial {len(ADV)}; each x2 builds: -O2 -Werror and -fsanitize=address,undefined)")
sys.exit(1 if FAIL else 0)
