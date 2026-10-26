#!/usr/bin/env python3
"""check_no_gpl_tts.py: CI gate against GPL espeak-ng / phonemizer / Piper in the shipping path.

Why: espeak-ng is GPL-3.0-or-later with no linking exception. Linking libespeak-ng (ctypes,
cffi, dlopen, -lespeak-ng, vendored source) relicense the engine as GPLv3 (GPLv3 §5c).
The PyPI `phonemizer` package is GPL-3.0-or-later because it binds libespeak-ng in-process,
and `piper` / `piper-phonemize` wheels embed espeak-ng, so a Piper wheel is a GPLv3
distribution. Only an arm's-length espeak-ng CLI (fork/exec, text in / phonemes out) is
acceptable, and it does not belong inside this repository's shipping code.


Scanned: shipping roots (src/, engine/, apps/, scripts/, kernels/, host/, vision/,
backends/, release/, tools/), the engine .bend sources, root
build/manifest files, and .github/workflows/.
Skipped: docs/, tests/, reports/, receipts/, tasks/, markdown, test_*.py, *_test.py,
build artefacts, and every path matched by tools/gpl_tts_allowlist.txt.

The check is deliberately line-oriented and precision-tuned so prose such as
"no espeak-ng" and the local method name `Kokoro.phonemizer()` do not fire:
  * comments and Python triple-quoted docstrings are not executable -> ignored
  * a token preceded by a negation ("no espeak-ng", "without espeak") is ignored
  * `phonemizer` fires on imports, dependency declarations and module use, not on
    an identifier that merely happens to be named phonemizer
Everything else that names espeak/libespeak/EspeakFallback/piper in code fails.

Exit 0 = clean, 1 = violations, 2 = usage or read error.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import os
import re
import sys
import tokenize
from pathlib import Path

# ---------------------------------------------------------------- configuration

# Directories/files that are shipped, built into the product, or are the product.
SHIP_DIRS = [
    "src", "engine", "apps", "scripts",
    "kernels", "host", "vision", "backends", "release",
    "tools", ".github",
]
SHIP_ROOT_FILES = [
    "build.sh", "engine.bend", "engine_gpu.bend", "engine_qwen35.bend",
    "engine_qwen35_gpu.bend", "gguf.bend",
    "requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py",
    "setup.cfg", "package.json", "Cargo.toml", "Dockerfile",
]

# Dev-only/evidence trees. Nothing here is linked into the shipped engine.
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "build",
    "docs", "tests", "reports", "receipts", "tasks",
}
SKIP_FILE_GLOBS = ["*.md", "test_*.py", "*_test.py", "*.log", "*.pyc"]

# The scanner and its allowlist necessarily contain the forbidden words.
SELF_EXEMPT = {"tools/check_no_gpl_tts.py", "tools/gpl_tts_allowlist.txt"}

# Files whose whole job is dependency declarations: bare quoted names count.
MANIFEST_GLOBS = ["requirements*.txt", "pyproject.toml", "setup.py", "setup.cfg",
                  "package.json", "Cargo.toml", "*.cmake", "CMakeLists.txt"]

# rule id, compiled regex, human description, manifest-only?
# "manifest-only" rules apply to dependency declarations (requirements.txt,
# pyproject.toml, ...) where a bare quoted name is a dependency, not an identifier.
RULES = [
    ("libespeak", re.compile(r"libespeak", re.I),
     "links/names libespeak-ng (GPL-3.0-or-later)", False),
    ("espeak", re.compile(r"(?<![\w.])espeak(?:[-_]?ng)?(?![\w.])", re.I),
     "references espeak-ng in executable code (link, include, dlopen or subprocess)", False),
    ("espeak-flag", re.compile(r"-lespeak", re.I),
     "links espeak-ng via the linker flag", False),
    ("EspeakFallback", re.compile(r"\bEspeakFallback\b"),
     "uses misaki's opt-in espeak backend (reintroduces GPL)", False),
    ("phonemizer-import", re.compile(r"(?<![\w.])(?:import|from)\s+phonemizer\b", re.I),
     "imports the GPL-3.0-or-later phonemizer package (binds libespeak-ng)", False),
    ("phonemizer-require", re.compile(r"\brequire\s*\(\s*[\"']phonemizer\b", re.I),
     "requires the GPL-3.0-or-later phonemizer package", False),
    ("phonemizer-use", re.compile(r"\bphonemizer\s*\.\s*(?:phonemize|Phonemizer|EspeakBackend)\b", re.I),
     "calls the GPL phonemizer package", False),
    ("phonemizer-install", re.compile(
        r"\b(?:pip3?|uv\s+pip|uv\s+add|poetry\s+add|conda\s+install)\b[^\n]*\bphonemizer\b", re.I),
     "installs the GPL phonemizer package", False),
    ("phonemizer-dep", re.compile(
        r"""(?i)(?:"|')?phonemizer(?:"|')?\s*(?:[<>=~!]|$|,|\])"""),
     "declares phonemizer as a dependency (GPL-3.0-or-later)", True),
    ("piper", re.compile(r"\bpiper(?:-phonemize|-tts|_phonemize|_tts)?\b", re.I),
     "references Piper/Piper wheels, which embed espeak-ng", False),
]


NEGATED = re.compile(
    r"(?i)\b(?:no|not|never|without|zero|free\s+of)\s+(?:espeak|phonemizer|piper)"
    r"|\b(?:espeak|phonemizer|piper)[-_ ]free\b")

HASH_COMMENT_EXTS = {".py", ".bend", ".sh", ".bash", ".zsh", ".yml", ".yaml",
                     ".toml", ".cfg", ".txt", ".mk", ""}
C_COMMENT_EXTS = {".c", ".h", ".cc", ".cpp", ".cxx", ".cu", ".m", ".mm",
                  ".js", ".ts", ".go", ".rs", ".java", ".swift"}
PY_EXTS = {".py"}


# ---------------------------------------------------------------- comment stripping

class _BlockState:
    __slots__ = ("in_block",)

    def __init__(self) -> None:
        self.in_block = False


def _strip_generic(line: str, ext: str, state: _BlockState) -> str:
    """Remove // /* */ comments and (where applicable) # comments, preserving strings."""
    hash_comments = ext in HASH_COMMENT_EXTS
    c_comments = ext in C_COMMENT_EXTS or ext == ""
    out: list[str] = []
    quote: str | None = None
    i = 0
    n = len(line)
    while i < n:
        if state.in_block:
            j = line.find("*/", i)
            if j < 0:
                return "".join(out)
            state.in_block = False
            i = j + 2
            continue
        ch = line[i]
        if quote:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(line[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'`":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if c_comments and ch == "/" and i + 1 < n:
            if line[i + 1] == "/":
                break
            if line[i + 1] == "*":
                state.in_block = True
                i += 2
                continue
        if hash_comments and ch == "#":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _python_code_lines(text: str) -> dict[int, str]:
    """Map line number -> executable code, blanking comments and triple-quoted strings."""
    lines = text.splitlines()
    chars = [list(line) for line in lines]

    def blank(start_line: int, start_col: int, end_line: int, end_col: int) -> None:
        for ln in range(start_line, end_line + 1):
            if ln - 1 >= len(chars):
                return
            row = chars[ln - 1]
            lo = start_col if ln == start_line else 0
            hi = end_col if ln == end_line else len(row)
            for c in range(max(0, lo), min(len(row), hi)):
                row[c] = " "

    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                blank(tok.start[0], tok.start[1], tok.end[0], tok.end[1])
            elif tok.type == tokenize.STRING:
                s = tok.string.lstrip("rRbBuUfF")
                if s.startswith('"""') or s.startswith("'''"):
                    blank(tok.start[0], tok.start[1], tok.end[0], tok.end[1])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Fall back to the generic stripper rather than failing open.
        state = _BlockState()
        return {i: _strip_generic(line, ".py", state)
                for i, line in enumerate(text.splitlines(), 1)}
    return {i + 1: "".join(row) for i, row in enumerate(chars)}


def executable_lines(path: Path, text: str) -> dict[int, str]:
    ext = path.suffix.lower()
    if ext in PY_EXTS:
        return _python_code_lines(text)
    state = _BlockState()
    out: dict[int, str] = {}
    for i, line in enumerate(text.splitlines(), 1):
        out[i] = _strip_generic(line, ext, state)
    return out


# ---------------------------------------------------------------- allowlist / walk

def load_allowlist(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if not path.is_file():
        return entries
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        glob, _, reason = line.partition("#")
        glob = glob.strip()
        if glob:
            entries.append((glob, reason.strip() or "no reason given"))
    return entries


def allowlisted(rel: str, entries: list[tuple[str, str]]) -> tuple[bool, str]:
    for glob, reason in entries:
        if fnmatch.fnmatchcase(rel, glob):
            return True, reason
    return False, ""


def iter_shipping_files(root: Path):
    seen: set[Path] = set()
    for d in SHIP_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(x for x in dirnames if x not in SKIP_DIRS)
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if p in seen:
                    continue
                rel = p.relative_to(root).as_posix()
                if any(fnmatch.fnmatchcase(name, g) for g in SKIP_FILE_GLOBS):
                    continue
                seen.add(p)
                yield p, rel
    for name in SHIP_ROOT_FILES:
        p = root / name
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if p not in seen and not any(fnmatch.fnmatchcase(name, g) for g in SKIP_FILE_GLOBS):
                seen.add(p)
                yield p, rel


def looks_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


# ---------------------------------------------------------------- main

def scan(root: Path, allowlist_path: Path) -> tuple[list[tuple[str, int, str, str, str]], int, int, int]:
    allow = load_allowlist(allowlist_path)
    violations: list[tuple[str, int, str, str, str]] = []
    scanned = skipped_allow = skipped_self = 0
    for path, rel in iter_shipping_files(root):
        if rel in SELF_EXEMPT:
            skipped_self += 1
            continue
        ok, _reason = allowlisted(rel, allow)
        if ok:
            skipped_allow += 1
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            print(f"ERROR: cannot read {rel}: {exc}", file=sys.stderr)
            raise
        if looks_binary(data):
            continue
        text = data.decode("utf-8", errors="replace")
        code = executable_lines(path, text)
        scanned += 1
        is_manifest = any(fnmatch.fnmatchcase(path.name, g) for g in MANIFEST_GLOBS)
        for ln in sorted(code):
            line = code[ln]
            if not line.strip():
                continue
            for rule_id, rx, why, manifest_only in RULES:
                if manifest_only and not is_manifest:
                    continue
                m = rx.search(line)
                if not m:
                    continue
                if rule_id in ("espeak", "piper") and NEGATED.search(line):
                    continue
                snippet = line.strip()
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                violations.append((rel, ln, rule_id, snippet, why))
                break  # one finding per line; later rules are the same defect
    return violations, scanned, skipped_allow, skipped_self


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fail if the shipping path uses GPL espeak-ng/phonemizer/Piper.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]),
                    help="repository root (default: parent of tools/)")
    ap.add_argument("--allowlist", default=None,
                    help="allowlist path (default: <root>/tools/gpl_tts_allowlist.txt)")
    ap.add_argument("--quiet", action="store_true", help="only print the verdict")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not (root / "tools").is_dir():
        print(f"ERROR: {root} does not look like the llama-bend repo (no tools/)", file=sys.stderr)
        return 2
    allow_path = Path(args.allowlist) if args.allowlist else root / "tools" / "gpl_tts_allowlist.txt"

    try:
        violations, scanned, skipped_allow, skipped_self = scan(root, allow_path)
    except OSError:
        return 2

    if not args.quiet:
        print(f"GPL-TTS gate: root={root}")
        print(f"  shipping roots scanned: {scanned} files "
              f"({skipped_allow} allowlisted, {skipped_self} self-exempt)")
        print(f"  allowlist: {allow_path.relative_to(root) if allow_path.is_relative_to(root) else allow_path}")

    if violations:
        print(f"\nFAIL: {len(violations)} GPL espeak/phonemizer/Piper hit(s) in the shipping path\n")
        for rel, ln, rule_id, snippet, why in violations:
            print(f"  {rel}:{ln}: {snippet}")
            print(f"      rule={rule_id}: {why}")
        print("\nBlocked: espeak-ng is GPL-3.0-or-later with no linking exception; linking it")
        print("relicense the engine as GPLv3. phonemizer and Piper wheels embed/bind espeak-ng.")
        print("Only an arm's-length CLI outside the shipped code is permissible, with mitigation")
        print("(LICENSING-MEMO.md §2). Remove the dependency, or add a justified dev-only path")
        print("pattern to tools/gpl_tts_allowlist.txt.")
        return 1

    print(f"PASS: no GPL espeak-ng/phonemizer/Piper usage in the shipping path "
          f"({scanned} files scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
