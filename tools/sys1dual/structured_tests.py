#!/usr/bin/env python3
"""tools/sys1dual/structured_tests.py - G16 end-to-end tests of llb-serve --structured (CPU).

  python structured_tests.py --bin DIR --tokenizer TOKENIZER.BIN --work DIR [--port 18600]

DIR holds llb-serve and fake_engine (tools/sys1dual/fake_engine.c: dual.c's stream protocol
with the real tokenizer and G1's real grammar mask, no model). Two servers run side by side:
S = --structured, P = the flag off. Checks:
  goldens   plain requests: S == P byte for byte (bar id/created/timings), and the engine
            request lines S sent == P's (the flag changes nothing for a plain request)
  flag-off  structured fields on P: 400 (tools: W1-SERVER's own message)
  schema    response_format json_schema / json_object, top-level json_schema, GBNF passthrough;
            content validates (jsonschema); finish_reason stop; usage counts up to the EOG id
  thinking  reasoning_content / content split; constrained only after </think>
  tools     auto (one call, two calls + content, plain answer), required, named; arguments
            typed by the schema; finish_reason tool_calls; content null when only calls
  fuzz      tool_choice required and json_schema with seeded random constrained decodes
            (off-script): every finished response parses and validates
  stream    for every case: the SSE deltas reassemble to exactly the non-stream message
  errors    the 400 surface
Exit code = number of failures.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request, urllib.error

import jsonschema

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--port", type=int, default=18600)
ap.add_argument("--fuzz", type=int, default=24)
A = ap.parse_args()
os.makedirs(A.work, exist_ok=True)
PASS = FAIL = 0


def ok(c, what, detail=""):
    global PASS, FAIL
    if c:
        PASS += 1
        print("PASS", what)
    else:
        FAIL += 1
        print("FAIL", what)
        if detail:
            print("     " + str(detail)[:2000].replace("\n", "\n     "))


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
         "days": {"type": "integer"}, "detail": {"type": "boolean"}}, "required": ["city", "unit"]}}},
    {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
        "q": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "limit": {"type": ["integer", "null"]}}, "required": ["q"]}}},
]
CALL_W = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>"
CALL_S = "<tool_call>\n<function=search>\n<parameter=q>\nbest café 🙂\n</parameter>\n<parameter=tags>\n[\"food\", \"paris\"]\n</parameter>\n</function>\n</tool_call>"
SCHEMA = {"type": "object", "properties": {"name": {"type": "string", "maxLength": 40}, "age": {"type": "integer"},
          "languages": {"type": "array", "items": {"type": "string", "enum": ["English", "French", "Italian"]}},
          "retired": {"type": "boolean"}}, "required": ["name", "age", "languages", "retired"]}
SCRIPT = [
    {"match": "capital of France", "text": "Paris.<|im_end|>"},
    {"match": "Water boils", "text": " 100 degrees Celsius at sea level."},
    {"match": "THINK-WEATHER", "text": "The user wants weather.\n</think>\n\n" + CALL_W + "<|im_end|>"},
    {"match": "WEATHER", "text": CALL_W + "<|im_end|>"},
    {"match": "TWO-CALLS", "text": "Checking both.\n\n" + CALL_W + "\n" + CALL_S + "<|im_end|>"},
    {"match": "NO-TOOL", "text": "Hello! No tool needed.<|im_end|>"},
    {"match": "BAD-CALL", "text": "<tool_call>\n<function=get_weather>\nnot a parameter\n</function>\n</tool_call><|im_end|>"},
    {"match": "THINK-ADA", "text": "Ada was born 1815, died at 36.\n</think>\n\n{\"name\": \"Ada Lovelace\", \"age\": 36, \"languages\": [\"English\", \"French\"], \"retired\": false}"},
    {"match": "ADA", "text": "{\"name\": \"Ada Lovelace\", \"age\": 36, \"languages\": [\"English\", \"French\"], \"retired\": false}"},
    {"match": "UTF", "text": "{\"name\": \"Zoë 🙂 café\", \"age\": 7, \"languages\": [\"French\"], \"retired\": true}"},
    {"match": "OFF-SCRIPT", "text": "{\"name\": 12"},
    {"match": "YESNO", "text": " yes."},
    {"match": "FUZZ-THINK", "text": "ok\n</think>\n\n"},
]
with open(f"{A.work}/script.jsonl", "w") as f:
    for s in SCRIPT:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")


def start(name, port, flag):
    log = f"{A.work}/{name}.engine.jsonl"
    open(log, "w").close()
    env = dict(os.environ, LLB_GGUF="/fake/model.gguf", FAKE_SCRIPT=f"{A.work}/script.jsonl", FAKE_LOG=log,
               LLB_TOKENIZER=A.tokenizer)
    args = [f"{A.bin}/llb-serve", "--port", str(port), "--alias", "fake"]
    if flag:
        args += ["--structured", A.tokenizer]
    args += ["--engine", f"{A.bin}/fake_engine", "--", "x"]
    p = subprocess.Popen(args, env=env, stderr=open(f"{A.work}/{name}.stderr", "w"))
    for _ in range(200):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health").status == 200:
                return p, log
        except Exception:
            time.sleep(0.05)
    raise SystemExit(f"{name} did not come up")


S, SLOG = start("S", A.port, True)
P, PLOG = start("P", A.port + 1, False)


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req)
        return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def norm(text):  # drop the run-varying fields
    text = re.sub(r'"(id|created)":("[^"]*"|\d+)', r'"\1":X', text)
    return re.sub(r'"timings":\{[^}]*\}', '"timings":X', text)


def sse(text):
    ev = [l[6:] for l in text.split("\n") if l.startswith("data: ")]
    return [json.loads(e) for e in ev if e != "[DONE]"], bool(ev) and ev[-1] == "[DONE]"


def chat(port, body, stream=False):
    b = dict(body, stream=stream)
    return post(port, "/v1/chat/completions", b)


def assemble(chunks):  # SSE chat deltas -> one message (+ finish_reason, usage)
    m = {"role": None, "content": "", "reasoning_content": "", "tool_calls": []}
    fr = usage = None
    for c in chunks:
        if "usage" in c and c.get("choices"):
            usage = c["usage"]
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            if "role" in d:
                m["role"] = d["role"]
            m["content"] += d.get("content", "")
            m["reasoning_content"] += d.get("reasoning_content", "")
            for t in d.get("tool_calls", []):
                m["tool_calls"].append({k: v for k, v in t.items() if k != "index"})
            fr = ch.get("finish_reason") or fr
    return m, fr, usage


def same_as_stream(name, body):
    """non-stream vs stream of the same request: the message, finish_reason and usage agree"""
    c1, b1 = chat(A.port, body)
    c2, b2 = chat(A.port, body, True)
    if c1 != 200 or c2 != 200:
        ok(False, f"{name}: stream == non-stream", f"{c1} {b1}\n{c2} {b2}")
        return None
    r = json.loads(b1)
    ch, done = sse(b2)
    m, fr, usage = assemble(ch)
    msg = r["choices"][0]["message"]
    want = {"role": "assistant", "content": msg["content"] or "", "reasoning_content": msg.get("reasoning_content", ""),
            "tool_calls": [{k: v for k, v in t.items() if k != "id"} for t in msg.get("tool_calls", [])]}
    got = dict(m, tool_calls=[{k: v for k, v in t.items() if k != "id"} for t in m["tool_calls"]])
    good = done and got == want and fr == r["choices"][0]["finish_reason"] and usage == r["usage"]
    ok(good, f"{name}: stream deltas reassemble to the non-stream message", f"{got}\n{want}\n{fr} {usage} {r['usage']}")
    return r


U = lambda t: [{"role": "user", "content": t}]
NOTHINK = {"enable_thinking": False}

# ---------------------------------------------------------------- goldens (plain path)
plain = [
    ("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 16, "return_tokens": True}),
    ("/v1/completions", {"prompt": "Water boils at a temperature of", "max_tokens": 16, "stream": True}),
    ("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 16, "temperature": 0.7, "top_p": 0.9, "seed": 42, "stop": ["\n"]}),
    ("/v1/chat/completions", {"messages": [{"role": "system", "content": "You are terse."}] + U("What is the capital of France?"),
                              "max_tokens": 24, "chat_template_kwargs": NOTHINK}),
    ("/v1/chat/completions", {"messages": U("What is the capital of France?"), "max_tokens": 24, "stream": True,
                              "stream_options": {"include_usage": True}}),
    ("/v1/chat/completions", {"messages": U("capital of France, café 🙂"), "max_tokens": 32, "response_format": {"type": "text"}}),
]
for i, (path, body) in enumerate(plain):
    cs, bs = post(A.port, path, body)
    cp, bp = post(A.port + 1, path, body)
    ok(cs == cp == 200 and norm(bs) == norm(bp), f"golden {i}: plain request, flag on == flag off ({path})", f"{bs[:400]}\n{bp[:400]}")
ls, lp = open(SLOG).read().splitlines(), open(PLOG).read().splitlines()
ls = ls[:len(plain)]  # S also ran the tools: [] request below... (read before it)
ok(ls == lp and len(ls) == len(plain), f"golden: the {len(ls)} engine request lines are byte-identical", f"{ls}\n{lp}")

body = {"messages": U("capital of France"), "max_tokens": 12}
cs, bs = post(A.port, "/v1/chat/completions", dict(body, tools=[], response_format={"type": "text"}))
cp, bp = post(A.port + 1, "/v1/chat/completions", body)
ok(cs == cp == 200 and norm(bs) == norm(bp), "tools: [] + response_format text on S == the bare request on P", bs)

# ---------------------------------------------------------------- flag off
c, b = chat(A.port + 1, {"messages": U("ADA"), "response_format": {"type": "json_object"}})
ok(c == 400 and "--structured" in b, "flag off: response_format -> 400 naming --structured", b)
c, b = post(A.port + 1, "/v1/completions", {"prompt": "x", "grammar": "root ::= \"a\""})
ok(c == 400 and "--structured" in b, "flag off: grammar -> 400", b)
c, b = chat(A.port + 1, {"messages": U("hi"), "tools": TOOLS})
ok(c == 400 and "tools/function calling is not supported yet" in b, "flag off: tools -> W1-SERVER's own 400", b)

# ---------------------------------------------------------------- schema / grammar
r = same_as_stream("json_schema (thinking off)", {"messages": U("ADA"), "max_tokens": 64, "chat_template_kwargs": NOTHINK,
                   "response_format": {"type": "json_schema", "json_schema": {"name": "p", "strict": True, "schema": SCHEMA}}})
if r:
    msg = r["choices"][0]["message"]
    try:
        jsonschema.validate(json.loads(msg["content"]), SCHEMA)
        valid = True
    except Exception as e:
        valid = e
    ok(valid is True and r["choices"][0]["finish_reason"] == "stop" and "<|im_end|>" not in msg["content"],
       "json_schema: content validates, finish stop, nothing past EOG", f"{valid} {msg}")
    # usage = the tokens up to and including the EOG id (the engine ran all 64)
    c, b = chat(A.port, {"messages": U("ADA"), "max_tokens": 64, "chat_template_kwargs": NOTHINK, "return_tokens": True,
                         "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}})
    rr = json.loads(b)
    ids = rr["llb"]["ids"]
    ok(rr["usage"]["completion_tokens"] == len(ids) and ids[-1] == 248046 and 248046 not in ids[:-1],
       "EOG stop: completion_tokens / ids end at the first EOG id", rr["usage"])
line = open(SLOG).read().splitlines()[-1]
ok('"grammar":' in line and '"json_schema"' not in line, "engine line: the schema went as GBNF in dual.c's grammar field")

r = same_as_stream("json_schema + thinking", {"messages": U("THINK-ADA"), "max_tokens": 120, "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}}})
if r:
    msg = r["choices"][0]["message"]
    ok(msg.get("reasoning_content") == "Ada was born 1815, died at 36.\n" and json.loads(msg["content"])["age"] == 36,
       "thinking: reasoning_content split off, content = the JSON", msg)
r = same_as_stream("json_schema UTF-8", {"messages": U("UTF"), "max_tokens": 64, "chat_template_kwargs": NOTHINK, "json_schema": SCHEMA})
ok(r and json.loads(r["choices"][0]["message"]["content"])["name"] == "Zoë 🙂 café", "top-level json_schema, multi-byte strings")
r = same_as_stream("json_object", {"messages": U("ADA"), "max_tokens": 64, "chat_template_kwargs": NOTHINK, "response_format": {"type": "json_object"}})
ok(r and isinstance(json.loads(r["choices"][0]["message"]["content"]), dict), "json_object: an object")
# off script: the scripted text breaks the schema at "12"; the mask forces a valid continuation
r = same_as_stream("off-script", {"messages": U("OFF-SCRIPT"), "max_tokens": 200, "chat_template_kwargs": NOTHINK, "json_schema": SCHEMA, "seed": 3})
if r and r["choices"][0]["finish_reason"] == "stop":
    try:
        jsonschema.validate(json.loads(r["choices"][0]["message"]["content"]), SCHEMA)
        ok(True, "off-script: the mask forced a valid instance")
    except Exception as e:
        ok(False, "off-script: the mask forced a valid instance", e)
else:
    ok(False, "off-script: finished", r)
c, b = post(A.port, "/v1/completions", {"prompt": "YESNO", "max_tokens": 8, "grammar": 'root ::= " " ("yes" | "no") "."'})
rr = json.loads(b)
ok(c == 200 and rr["choices"][0]["text"] == " yes." and rr["choices"][0]["finish_reason"] == "stop", "completions GBNF passthrough", b)
c, b = post(A.port, "/v1/completions", {"prompt": "YESNO", "max_tokens": 8, "stream": True, "grammar": 'root ::= " " ("yes" | "no") "."'})
ch, done = sse(b)
ok(done and "".join(x["choices"][0]["text"] for x in ch if x["choices"]) == " yes." and ch[-1]["choices"][0]["finish_reason"] == "stop",
   "completions GBNF, streamed")
c, b = post(A.port, "/v1/completions", {"prompt": "ADA", "max_tokens": 3, "json_schema": SCHEMA})
rr = json.loads(b)
ok(c == 200 and rr["choices"][0]["finish_reason"] == "length" and rr["usage"]["completion_tokens"] == 3, "max_tokens before the end: finish length", b)

# ---------------------------------------------------------------- tools
r = same_as_stream("tools auto (thinking off)", {"messages": U("WEATHER in Paris?"), "tools": TOOLS, "max_tokens": 80, "chat_template_kwargs": NOTHINK})
if r:
    m, ch = r["choices"][0]["message"], r["choices"][0]
    tc = m.get("tool_calls", [])
    ok(ch["finish_reason"] == "tool_calls" and m["content"] is None and len(tc) == 1 and tc[0]["type"] == "function" and
       tc[0]["function"]["name"] == "get_weather" and json.loads(tc[0]["function"]["arguments"]) == {"city": "Paris", "unit": "celsius"} and
       tc[0]["id"].startswith("call_"), "tools auto: one call, typed arguments, content null, finish tool_calls", m)
line = open(SLOG).read().splitlines()[-1]
ok('"grammar"' not in line and "<tools>" in json.loads(line)["prompt"], "tools auto: unconstrained, tools block in the prompt")
r = same_as_stream("tools auto + thinking", {"messages": U("THINK-WEATHER"), "tools": TOOLS, "max_tokens": 120})
ok(r and r["choices"][0]["message"]["reasoning_content"] == "The user wants weather.\n" and r["choices"][0]["message"]["tool_calls"],
   "tools + thinking: reasoning split, then the call", r and r["choices"][0]["message"])
r = same_as_stream("two calls + content", {"messages": U("TWO-CALLS"), "tools": TOOLS, "max_tokens": 160, "chat_template_kwargs": NOTHINK})
if r:
    m = r["choices"][0]["message"]
    args = [json.loads(t["function"]["arguments"]) for t in m.get("tool_calls", [])]
    ok(m["content"] == "Checking both.\n\n" and args == [{"city": "Paris", "unit": "celsius"}, {"q": "best café 🙂", "tags": ["food", "paris"]}],
       "two calls + leading content (array arg parsed, UTF-8 kept)", m)
r = same_as_stream("no tool", {"messages": U("NO-TOOL"), "tools": TOOLS, "max_tokens": 40, "chat_template_kwargs": NOTHINK})
ok(r and r["choices"][0]["message"]["content"] == "Hello! No tool needed." and r["choices"][0]["finish_reason"] == "stop" and
   "tool_calls" not in r["choices"][0]["message"], "tools offered, plain answer: content, finish stop")
r = same_as_stream("malformed call", {"messages": U("BAD-CALL"), "tools": TOOLS, "max_tokens": 60, "chat_template_kwargs": NOTHINK})
ok(r and "not a parameter" in r["choices"][0]["message"]["content"] and "tool_calls" not in r["choices"][0]["message"],
   "malformed call: no tool_calls, raw text kept as content")
r = same_as_stream("tool_choice required", {"messages": U("WEATHER"), "tools": TOOLS, "tool_choice": "required", "max_tokens": 80, "chat_template_kwargs": NOTHINK})
line = open(SLOG).read().splitlines()[-1]
ok(r and r["choices"][0]["finish_reason"] == "tool_calls" and '"grammar"' in line, "tool_choice required: grammar sent, call made")
r = same_as_stream("tool_choice named", {"messages": U("NO-TOOL"), "tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "search"}},
                   "max_tokens": 120, "chat_template_kwargs": NOTHINK, "seed": 5})
ok(r and [t["function"]["name"] for t in r["choices"][0]["message"].get("tool_calls", [])] == ["search"],
   "tool_choice named: the scripted plain answer is overridden by the forced search call", r and r["choices"][0]["message"])
r = same_as_stream("tool history", {"messages": U("WEATHER?") + [
    {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\",\"unit\":\"celsius\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"}], "tools": TOOLS, "max_tokens": 30, "chat_template_kwargs": NOTHINK})
line = json.loads(open(SLOG).read().splitlines()[-1])["prompt"]
ok("<tool_response>\n18C sunny\n</tool_response>" in line and "<parameter=city>\nParis\n</parameter>" in line, "tool history rendered (tool role + prior call)")

# ---------------------------------------------------------------- fuzz (off-script, seeded random constrained decodes)
fin = val = 0
bad = []
for seed in range(A.fuzz):
    think = seed % 2 == 1
    body = {"messages": U("FUZZ-THINK" if think else "fuzz"), "tools": TOOLS, "tool_choice": "required", "max_tokens": 400, "seed": seed,
            "parallel_tool_calls": seed % 3 != 0}
    if not think:
        body["chat_template_kwargs"] = NOTHINK
    c, b = chat(A.port, body)
    r = json.loads(b)
    if c != 200:
        bad.append(b)
        continue
    ch = r["choices"][0]
    if ch["finish_reason"] != "tool_calls":
        continue
    fin += 1
    good = True
    for t in ch["message"]["tool_calls"]:
        sch = next(x["function"]["parameters"] for x in TOOLS if x["function"]["name"] == t["function"]["name"])
        try:
            jsonschema.validate(json.loads(t["function"]["arguments"]), sch)
        except Exception as e:
            good = False
            bad.append((t, str(e)[:200]))
    if seed % 3 == 0 and len(ch["message"]["tool_calls"]) != 1:
        good = False
        bad.append(("parallel_tool_calls false gave", len(ch["message"]["tool_calls"])))
    val += good
ok(fin >= A.fuzz * 2 // 3 and val == fin and not bad, f"fuzz tools required: {fin}/{A.fuzz} finished, {val}/{fin} validate", bad[:3])
fin = val = 0
bad = []
schemas = [SCHEMA, {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 4},
           {"anyOf": [{"type": "null"}, {"type": "object", "properties": {"k": {"enum": ["a", "b"]}}, "required": ["k"]}]},
           {"type": "object"}]
for seed in range(A.fuzz):
    sch = schemas[seed % len(schemas)]
    c, b = post(A.port, "/v1/completions", {"prompt": "fuzz", "max_tokens": 300, "seed": seed, "json_schema": sch})
    r = json.loads(b)
    if c != 200 or r["choices"][0]["finish_reason"] != "stop":
        continue
    fin += 1
    try:
        jsonschema.validate(json.loads(r["choices"][0]["text"]), sch)
        val += 1
    except Exception as e:
        bad.append((r["choices"][0]["text"][:200], str(e)[:200]))
ok(fin >= A.fuzz // 2 and val == fin, f"fuzz json_schema: {fin}/{A.fuzz} finished, {val}/{fin} validate", bad[:3])

# ---------------------------------------------------------------- errors
errs = [
    ({"messages": U("x"), "response_format": {"type": "json_schema"}}, "json_schema.schema"),
    ({"messages": U("x"), "response_format": {"type": "xml"}}, "response_format type"),
    ({"messages": U("x"), "json_schema": {"type": "string", "format": "email"}}, "unsupported keyword"),
    ({"messages": U("x"), "grammar": "root ::= undefined-rule"}, "grammar:"),
    ({"messages": U("x"), "grammar": "root ::= \"a\"", "response_format": {"type": "json_object"}}, "at most one"),
    ({"messages": U("x"), "tools": TOOLS, "json_schema": {}}, "cannot be combined"),
    ({"messages": U("x"), "tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "nope"}}}, "not in tools"),
    ({"messages": U("x"), "tools": TOOLS, "tool_choice": "sometimes"}, "tool_choice must be"),
    ({"messages": U("x"), "tools": {"a": 1}}, "tools must be an array"),
    ({"messages": U("x"), "functions": [{"name": "f"}]}, "deprecated"),
]
for i, (body, frag) in enumerate(errs):
    c, b = chat(A.port, body)
    ok(c == 400 and frag in b and json.loads(b)["error"]["type"] == "invalid_request_error", f"error {i}: 400 '{frag}'", b)
c, b = post(A.port, "/v1/completions", {"prompt": "x", "tools": TOOLS})
ok(c == 400 and "chat" in b, "completions + tools: 400", b)
c, b = chat(A.port, {"messages": U("ADA"), "max_tokens": 8})
ok(c == 200, "the server still serves after the error surface")

for p in (S, P):
    p.terminate()
    p.wait(timeout=60)
ok(S.returncode == 0 and P.returncode == 0, "both servers shut down cleanly (rc 0)")
print(f"structured_tests: {PASS} PASS / {FAIL} FAIL")
sys.exit(FAIL)
