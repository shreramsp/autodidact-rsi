"""Run LLM-written redactor code safely.

The code Gemini writes is untrusted. Before running it we:
  1. parse it and reject anything outside a small allow-list
     (only pure-text modules may be imported; no open/exec/eval/dunders);
  2. run it in a separate Python process (`python -I`) with a timeout,
     a memory cap, an empty working directory and no API key in its env.

A crash on one message only costs that message (it is scored as "found
nothing"); a timeout or a rejected program scores zero on everything.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .generator import Message
from .metrics import Report, score

ALLOWED_IMPORTS = {"re", "string", "itertools", "collections", "functools", "math",
                   "typing", "unicodedata", "dataclasses", "enum", "__future__"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview"}

RUNNER = r'''
import json, sys, importlib.util
spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
texts = json.load(sys.stdin)
out = []
for t in texts:
    try:
        spans = mod.find_pii(t)
        out.append([[int(s), int(e), str(l)] for (s, e, l) in spans])
    except Exception as exc:
        out.append(None)
json.dump(out, sys.stdout)
'''


class RejectedCode(Exception):
    pass


def check_code(code: str) -> None:
    """Static safety check. Raises RejectedCode with a reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise RejectedCode(f"syntax error: {e}") from None
    has_fn = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise RejectedCode(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise RejectedCode(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise RejectedCode(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RejectedCode(f"dunder attribute not allowed: {node.attr}")
        elif isinstance(node, ast.FunctionDef) and node.name == "find_pii":
            has_fn = True
    if not has_fn:
        raise RejectedCode("no find_pii(text) function defined")


def _limit_memory():  # runs in the child process before exec
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    except Exception:
        pass


@dataclass
class RunResult:
    predictions: list[list | None]
    status: str            # "ok" | "rejected: ..." | "timeout" | "crash: ..."


def run_code(code: str, texts: list[str], timeout: float = 30.0) -> RunResult:
    try:
        check_code(code)
    except RejectedCode as e:
        return RunResult([None] * len(texts), f"rejected: {e}")
    with tempfile.TemporaryDirectory() as d:
        cand = Path(d) / "candidate.py"
        runner = Path(d) / "runner.py"
        cand.write_text(code)
        runner.write_text(RUNNER)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        try:
            p = subprocess.run([sys.executable, "-I", str(runner), str(cand)],
                               input=json.dumps(texts), capture_output=True, text=True,
                               timeout=timeout, cwd=d, env=env,
                               preexec_fn=_limit_memory if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return RunResult([None] * len(texts), "timeout")
        if p.returncode != 0:
            err = (p.stderr.strip().splitlines() or ["?"])[-1][:200]
            return RunResult([None] * len(texts), f"crash: {err}")
        try:
            return RunResult(json.loads(p.stdout), "ok")
        except json.JSONDecodeError:
            return RunResult([None] * len(texts), "crash: bad output")


def evaluate(code: str, messages: list[Message], keep_failures: bool = True) -> tuple[Report, str]:
    """Run a redactor on messages and score it. Returns (report, status)."""
    res = run_code(code, [m.text for m in messages])
    return score(messages, res.predictions, keep_failures), res.status
