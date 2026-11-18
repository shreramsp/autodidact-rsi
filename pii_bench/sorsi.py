"""Second-Order RSI (arXiv:2610.05701, laptop adaptation).

First-order RSI: see failures -> patch -> repeat.
Second-order RSI: ALSO watch your own history of attempts. When the history
shows a structural anomaly, stop patching and run a diagnostic probe that asks
WHY, then remember the answer (inquiry memory) for every future edit.

Three anomalies (detected by plain code, no LLM):
  recurrence   the same target was attempted 2+ times recently without real gain
  opposing     an edit undoes an earlier one: precision and recall swap direction
               on the same label, or a regex added earlier gets removed
  mismatch     the outcome contradicts the intent: the target label did not
               improve, or another label got noticeably worse
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from . import LABELS
from .editing import SYSTEM, _strings, failure_digest
from .llm import Gemini
from .metrics import Report

EPS = 0.01


@dataclass
class Attempt:
    round: int
    target: str
    change: str
    expected: str
    accepted: bool
    steering_before: float
    steering_after: float
    label_deltas: dict            # label -> {"dp","dr","df1"} on the feedback set
    regex_added: list
    regex_removed: list
    anomalies: list = field(default_factory=list)

    def line(self) -> str:
        d = self.label_deltas.get(self.target, {})
        return (f"round {self.round}: target {self.target} | change: {self.change[:110]} | "
                f"{'ACCEPTED' if self.accepted else 'rejected'} | steering {self.steering_before:.3f}->{self.steering_after:.3f} | "
                f"{self.target} dP {d.get('dp', 0):+.3f} dR {d.get('dr', 0):+.3f}")


def label_deltas(before: Report, after: Report) -> dict:
    out = {}
    for lab in LABELS:
        b, a = before.per_label[lab], after.per_label[lab]
        out[lab] = {"dp": a["precision"] - b["precision"], "dr": a["recall"] - b["recall"],
                    "df1": a["f1"] - b["f1"]}
    return out


def regex_changes(parent: str, child: str) -> tuple[list, list]:
    pr, _, _ = _strings(parent)
    cr, _, _ = _strings(child)
    return sorted((cr - pr).keys()), sorted((pr - cr).keys())


def detect(cur: Attempt, history: list[Attempt], version: int = 2) -> list[str]:
    """version 1 = first design (lab 10.06 v1); version 2 = after the v1 post-mortem."""
    found = []
    tgt = cur.target
    dcur = cur.label_deltas.get(tgt, {})

    # recurrence: >=2 of the last 4 attempts (incl. this one) hit the same target with no real gain
    recent = (history[-3:] + [cur])
    stuck = [a for a in recent if a.target == tgt and (not a.accepted or a.label_deltas.get(tgt, {}).get("df1", 0) <= 0.005)]
    if len(stuck) >= 2:
        found.append("recurrence")

    # opposing: precision/recall swap direction vs an earlier attempt on the same label,
    #           or this edit removes a regex an earlier ACCEPTED edit added
    for prev in reversed(history):
        if prev.target != tgt:
            continue
        dp0, dr0 = prev.label_deltas[tgt]["dp"], prev.label_deltas[tgt]["dr"]
        dp1, dr1 = dcur.get("dp", 0), dcur.get("dr", 0)
        if (dp0 > EPS and dr0 < -EPS and dp1 < -EPS and dr1 > EPS) or \
           (dp0 < -EPS and dr0 > EPS and dp1 > EPS and dr1 < -EPS):
            found.append("opposing")
            break
    # v1 also flagged "removes a regex an earlier accepted edit added". Post-mortem: every
    # REFINEMENT of a regex looks like a removal, so it fired on 9/10 attempts. Dropped in v2.
    if version == 1 and "opposing" not in found:
        undone = set(cur.regex_removed) & {r for a in history if a.accepted for r in a.regex_added}
        if undone:
            found.append("opposing")

    # mismatch: target did not improve, or a different label dropped by > 0.03 F1
    collateral = [l for l, d in cur.label_deltas.items() if l != tgt and d["df1"] < -0.03]
    if (tgt in LABELS and dcur.get("df1", 0) <= 0) or collateral:
        found.append("mismatch")
    return found


PROBE_QUESTIONS = {
    "recurrence": ["Is there an underlying constraint or assumption the repeated fixes ignore?",
                   "Are the fixes treating a symptom (one pattern) instead of the cause?",
                   "What evidence in the examples has not been used yet (e.g. surrounding words)?"],
    "opposing": ["Which two requirements are pulling in opposite directions?",
                 "What distinguishes the true cases from the false ones, if not the digit pattern itself?",
                 "Is the search space (regex shape only) mis-specified for this label?"],
    "mismatch": ["Why did the outcome differ from what the edit intended?",
                 "Is our model of the data wrong (formats, contexts) or is the metric misaligned?",
                 "Which other label did the edit interfere with, and how?"],
}

PROBE_SYSTEM = """You are the diagnostic module of a self-improving PII redactor.
You do NOT write code. You investigate WHY repeated edits are failing and return a short,
evidence-based diagnosis that will be stored in the agent's inquiry memory.

Answer in exactly this format:
ROOT_CAUSE: <one or two sentences, cite the evidence>
RULE: <one concrete, testable rule future edits must follow>
NEXT_EDIT: <the single edit you recommend next, one sentence>"""


# v3: the diagnosis may change the WORKFLOW, not only add a code rule.
PROBE_SYSTEM_V3 = PROBE_SYSTEM.replace(
    "NEXT_EDIT: <the single edit you recommend next, one sentence>",
    """NEXT_EDIT: <the single edit you recommend next, one sentence>
ACTION: <exactly one of SWITCH_TARGET | RESAMPLE_EVIDENCE | ADD_PRESERVE_EXAMPLES | KEEP_GOING>

Workflow actions (pick the one that addresses the root cause):
- SWITCH_TARGET: this label is at a precision/recall balance point; more patches will thrash. Park it and work on other labels.
- RESAMPLE_EVIDENCE: the agent keeps seeing the same few failure examples; show different/more examples each round.
- ADD_PRESERVE_EXAMPLES: fixes keep breaking cases that used to work; show must-still-catch examples in every edit prompt.
- KEEP_GOING: the diagnosis gives a clear code fix; no workflow change needed.""")
ACTIONS = ("SWITCH_TARGET", "RESAMPLE_EVIDENCE", "ADD_PRESERVE_EXAMPLES", "KEEP_GOING")


def preserve_examples(messages, target: str, per_fmt: int = 1) -> str:
    """One real example of every surface format of `target` in the feedback set.
    v2: the probe must explain failures WITHOUT contradicting these (v1 proposed
    'names must be capitalized' while lowercase names exist in the data)."""
    seen, lines = {}, []
    for m in messages:
        for s in m.spans:
            if s.label == target and seen.get(s.fmt, 0) < per_fmt:
                seen[s.fmt] = seen.get(s.fmt, 0) + 1
                ctx = m.text[max(0, s.start - 30):s.start] + "<<" + m.text[s.start:s.end] + ">>" + m.text[s.end:s.end + 20]
                lines.append(f"  ({s.fmt}) {ctx!r}")
    return "\n".join(lines)


def run_probe(llm: Gemini, anomaly: str, target: str, history: list[Attempt], code: str,
              report: Report, variant: int = 0, messages=None, refuted: list[dict] | None = None,
              version: int = 2) -> dict:
    trace = "\n".join(a.line() for a in history[-6:])
    qs = "\n".join(f"- {q}" for q in PROBE_QUESTIONS[anomaly])
    extra = ""
    if messages is not None:
        extra += (f"\nReal {target} examples that MUST still be caught (one per format) - your RULE "
                  f"must be consistent with ALL of them:\n{preserve_examples(messages, target)}\n")
    if refuted:
        extra += "\nRefuted hypotheses (rules that were tried and failed twice - do NOT repeat them):\n" + \
                 "\n".join(f"- {r['rule']}" for r in refuted) + "\n"
    prompt = f"""Anomaly detected: {anomaly.upper()} on label {target}.

Recent attempts (the agent's own trace):
{trace}

Current redactor:
```python
{code}
```

Evidence - current mistakes for {target} on the feedback set (<<...>> marks the span):
{failure_digest(report, k=20, focus=target)}
{extra}
Diagnostic questions:
{qs}"""
    system = PROBE_SYSTEM_V3 if version >= 3 else PROBE_SYSTEM
    r = llm.generate(prompt, system=system, temperature=0.3, variant=variant)
    out = {"anomaly": anomaly, "target": target, "raw": r.text}
    for key in ("ROOT_CAUSE", "RULE", "NEXT_EDIT", "ACTION"):
        m = re.search(rf"{key}:\s*(.+?)(?=\n[A-Z_]+:|\Z)", r.text, re.S)
        out[key.lower()] = m.group(1).strip() if m else ""
    if version >= 3:
        out["action"] = next((a for a in ACTIONS if a in out["action"].upper()), "KEEP_GOING")
    return out


def memory_block(memory: list[dict]) -> str:
    active = [m for m in memory if m.get("status", "active") != "refuted"]
    refuted = [m for m in memory if m.get("status") == "refuted"]
    out = ""
    if active:
        out += "\nInquiry memory (diagnosed root causes - your edit MUST respect these rules):\n" + "\n".join(
            f"- [{m['target']} / {m['anomaly']}] root cause: {m['root_cause']} RULE: {m['rule']}" for m in active) + "\n"
    if refuted:
        out += "\nRefuted hypotheses (followed twice, both edits rejected - do NOT follow these):\n" + "\n".join(
            f"- [{m['target']}] {m['rule']}" for m in refuted) + "\n"
    return out


def update_memory(memory: list[dict], att: Attempt, refute_after: int = 2) -> list[dict]:
    """v2 falsification: an active rule on att.target gains a strike for every rejected
    edit made while it was in force; at `refute_after` strikes it is marked refuted.
    Returns the entries that were refuted by this attempt."""
    newly = []
    for m in memory:
        if m.get("status", "active") != "active" or m["target"] != att.target or m["round"] >= att.round:
            continue
        if att.accepted:
            m["status"], m["strikes"] = "supported", 0
        else:
            m["strikes"] = m.get("strikes", 0) + 1
            if m["strikes"] >= refute_after:
                m["status"] = "refuted"
                newly.append(m)
    return newly


def to_dict(a: Attempt) -> dict:
    return asdict(a)
