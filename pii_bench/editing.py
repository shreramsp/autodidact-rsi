"""Shared machinery for "ask Gemini to propose an edit to the redactor".

propose_edit()   build the prompt, call Gemini, pull out the new code + intent
failure_digest() turn a scoring Report into a short list of concrete mistakes
diff_features()  the 8 numbers Budget RSI uses to describe an edit without running it
"""
from __future__ import annotations

import ast
import difflib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from . import LABELS
from .llm import Gemini
from .metrics import Report

SYSTEM = """You are an expert Python engineer improving a rule-based PII redactor for
customer-support messages. The program must stay deterministic and offline.

Contract
- Define find_pii(text) -> list of (start, end, label) tuples (character offsets into text).
- Labels: NAME, EMAIL, PHONE, CREDIT_CARD, SSN, ADDRESS.
- Allowed imports ONLY: re, string, itertools, collections, functools, math, typing, unicodedata.
  No NLP libraries, no dateparser/presidio/spacy, no file or network access, no eval/exec.

What counts as PII
- NAME: a person's name (full name, first name in a sign-off, or a third person like "my wife Ana Ruiz").
- EMAIL: personal email, including obfuscated forms like "ana [at] gmail [dot] com".
- PHONE: US or international phone numbers, with or without an extension.
- CREDIT_CARD: payment card numbers (they pass the Luhn checksum).
- SSN: US social security numbers.
- ADDRESS: a full postal address (street through ZIP) as ONE span, even across a line break.
NOT PII (do not redact): order numbers, tracking numbers, part/model numbers, prices, dates,
product names (e.g. "Apple Watch"), store/company/city names on their own.

Output format (exactly):
INTENT: {"target": "<LABEL or GENERAL>", "change": "<one sentence>", "expected": "<one sentence on which metric should move>"}
```python
<the COMPLETE new file>
```"""


@dataclass
class Edit:
    code: str
    intent: dict
    raw: str
    cost_usd: float
    cached: bool
    ok: bool = True
    error: str = ""


def failure_digest(report: Report, k: int = 24, focus: str | None = None, seed: int = 0,
                   kind: str | None = None) -> str:
    """A balanced sample of concrete failures, as plain text for the prompt.
    kind="FN" shows only misses, kind="FP" only false alarms (used by RSIFlow nodes)."""
    rng = random.Random(seed)
    fails = [f for f in report.failures if focus in (None, "GENERAL", f.label)
             and kind in (None, f.kind)]
    by_key: dict[tuple, list] = {}
    for f in fails:
        by_key.setdefault((f.kind, f.label), []).append(f)
    picked = []
    keys = sorted(by_key)
    while len(picked) < k and any(by_key.values()):
        for key in keys:
            if by_key[key] and len(picked) < k:
                picked.append(by_key[key].pop(rng.randrange(len(by_key[key]))))
    lines = []
    for f in picked:
        if f.kind == "FN":
            lines.append(f"MISSED {f.label}: {f.snippet!r}  (redactor found: {f.got})  context: {f.context}")
        else:
            lines.append(f"WRONGLY REDACTED as {f.label}: {f.snippet!r}  context: {f.context}")
    return "\n".join(lines) or "(no failures)"


def score_table(report: Report) -> str:
    rows = [f"overall  F1 {report.f1:.3f}  precision {report.precision:.3f}  recall {report.recall:.3f}"]
    for lab in LABELS:
        d = report.per_label[lab]
        rows.append(f"{lab:12s} F1 {d['f1']:.3f}  P {d['precision']:.3f}  R {d['recall']:.3f}  "
                    f"(tp {d['tp']}, fp {d['fp']}, fn {d['fn']})")
    return "\n".join(rows)


def build_prompt(code: str, report: Report, *, focus: str | None = None, extra: str = "",
                 digest: str | None = None, max_changed_lines: int = 40) -> str:
    focus_line = (f"Focus this edit on {focus}." if focus and focus != "GENERAL"
                  else "Pick the single most valuable improvement.")
    return f"""Current redactor:
```python
{code}
```

Scores on the feedback set:
{score_table(report)}

Sample of its mistakes (<<...>> marks the span in question):
{digest if digest is not None else failure_digest(report, focus=focus)}
{extra}
Make ONE focused improvement (change at most ~{max_changed_lines} lines). Keep everything that already works.
{focus_line}"""


_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
_INTENT_RE = re.compile(r"INTENT:\s*(\{.*?\})", re.S)


def parse_response(text: str) -> tuple[str | None, dict]:
    blocks = _CODE_RE.findall(text)
    code = max(blocks, key=len).strip() + "\n" if blocks else None
    intent = {}
    m = _INTENT_RE.search(text)
    if m:
        try:
            intent = json.loads(m.group(1))
        except json.JSONDecodeError:
            intent = {"change": m.group(1)[:200]}
    return code, intent


def propose_edit(llm: Gemini, code: str, report: Report, *, focus: str | None = None,
                 extra: str = "", digest: str | None = None, variant: int = 0,
                 temperature: float = 0.8) -> Edit:
    prompt = build_prompt(code, report, focus=focus, extra=extra, digest=digest)
    r = llm.generate(prompt, system=SYSTEM, temperature=temperature, variant=variant)
    new_code, intent = parse_response(r.text)
    if not new_code:
        return Edit(code, intent, r.text, r.cost_usd, r.cached, ok=False, error="no code block")
    return Edit(new_code, intent, r.text, r.cost_usd, r.cached)


# ----------------------------------------------------- Budget RSI features ---

FEATURE_NAMES = ["lines_added", "lines_deleted", "regex_added", "regex_removed",
                 "funcs_added", "funcs_removed", "labels_touched", "context_words_added"]

_REGEX_HINT = re.compile(r"\\[dbswDSW]|\[[^\]]+\]|\{\d|[+*?]\)?|\(\?")


def _strings(code: str) -> tuple[Counter, Counter, set]:
    """(regex-like string literals, plain keyword literals, function names)."""
    regexes, words, funcs = Counter(), Counter(), set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return regexes, words, funcs
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if _REGEX_HINT.search(s):
                regexes[s] += 1
            elif re.fullmatch(r"[a-z][a-z ]{1,25}", s):
                words[s] += 1
        elif isinstance(node, ast.FunctionDef):
            funcs.add(node.name)
    return regexes, words, funcs


def diff_features(parent: str, child: str) -> list[float]:
    """Describe an edit with 8 numbers, WITHOUT running it."""
    added = deleted = 0
    changed_text = []
    for line in difflib.unified_diff(parent.splitlines(), child.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
            changed_text.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
            changed_text.append(line)
    pr, pw, pf = _strings(parent)
    cr, cw, cf = _strings(child)
    blob = "\n".join(changed_text)
    labels = sum(1 for lab in LABELS if lab in blob)
    return [added, deleted,
            sum((cr - pr).values()), sum((pr - cr).values()),
            len(cf - pf), len(pf - cf),
            labels, sum((cw - pw).values())]
