"""Scoring a redactor's predicted spans against the gold spans.

Matching rule (per message):
  a predicted span is a TRUE POSITIVE for a gold span if
    * the labels are equal, and
    * the prediction covers at least 80% of the gold span's characters
      (a redaction that leaves half a card number visible is a leak).
  Each gold span can be matched once. Unmatched predictions are FALSE
  POSITIVES (over-redaction); unmatched gold spans are FALSE NEGATIVES.

Reported numbers
  precision  of everything redacted, how much was really PII
  recall     of all real PII, how much was caught
  f1         harmonic mean of the two - the main score
  leak_rate  share of gold PII characters left visible by ANY prediction
             (label-agnostic: the privacy view of the world)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import LABELS
from .generator import Message

COVER = 0.8


@dataclass
class Failure:
    kind: str          # "FN" (missed PII) or "FP" (redacted something that isn't PII)
    label: str
    msg_id: str
    snippet: str       # the text in question
    context: str       # a window of surrounding text
    fmt: str = ""      # gold format (FN only)
    got: str = ""      # what the redactor predicted instead (FN only)


@dataclass
class Report:
    precision: float
    recall: float
    f1: float
    leak_rate: float
    per_label: dict[str, dict[str, float]]
    tp: int
    fp: int
    fn: int
    n_messages: int
    errors: int = 0                       # messages where the redactor crashed
    failures: list[Failure] = field(default_factory=list)

    def summary(self) -> dict:
        return {"f1": round(self.f1, 4), "precision": round(self.precision, 4),
                "recall": round(self.recall, 4), "leak_rate": round(self.leak_rate, 4),
                "per_label_f1": {k: round(v["f1"], 3) for k, v in self.per_label.items()},
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "errors": self.errors}

    def line(self) -> str:
        return (f"F1 {self.f1:6.1%}  P {self.precision:6.1%}  R {self.recall:6.1%}  "
                f"leak {self.leak_rate:5.1%}  errors {self.errors}")


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _ctx(text: str, s: int, e: int, w: int = 45) -> str:
    left = text[max(0, s - w):s]
    right = text[e:e + w]
    return (left + "<<" + text[s:e] + ">>" + right).replace("\n", "\\n")


def score(messages: list[Message], predictions: list[list | None], keep_failures: bool = True) -> Report:
    counts = {lab: [0, 0, 0] for lab in LABELS}  # tp, fp, fn
    gold_chars = leaked_chars = 0
    errors = 0
    failures: list[Failure] = []

    for msg, preds in zip(messages, predictions):
        if preds is None:          # redactor crashed on this message
            errors += 1
            preds = []
        clean = []
        for p in preds:
            try:
                s, e, lab = int(p[0]), int(p[1]), str(p[2]).upper()
            except Exception:
                continue
            if 0 <= s < e <= len(msg.text):
                clean.append((s, e, lab))

        used = set()
        matched_pred = set()
        for g in msg.spans:
            glen = g.end - g.start
            gold_chars += glen
            covered = set()
            for (s, e, _lab) in clean:
                covered.update(range(max(s, g.start), min(e, g.end)))
            leaked_chars += glen - len(covered)

            hit = None
            for j, (s, e, lab) in enumerate(clean):
                if j in matched_pred or lab != g.label:
                    continue
                ov = max(0, min(e, g.end) - max(s, g.start))
                if ov / glen >= COVER:
                    hit = j
                    break
            if hit is None:
                counts.setdefault(g.label, [0, 0, 0])[2] += 1
                if keep_failures:
                    got = [f"{lab}:{msg.text[s:e]!r}" for (s, e, lab) in clean
                           if min(e, g.end) > max(s, g.start)]
                    failures.append(Failure("FN", g.label, msg.id, msg.text[g.start:g.end],
                                            _ctx(msg.text, g.start, g.end), g.fmt, ", ".join(got) or "nothing"))
            else:
                matched_pred.add(hit)
                used.add(hit)
                counts[g.label][0] += 1

        for j, (s, e, lab) in enumerate(clean):
            if j not in matched_pred:
                counts.setdefault(lab, [0, 0, 0])[1] += 1
                if keep_failures:
                    failures.append(Failure("FP", lab, msg.id, msg.text[s:e], _ctx(msg.text, s, e)))

    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    fn = sum(c[2] for c in counts.values())
    p, r, f = _prf(tp, fp, fn)
    per_label = {}
    for lab in LABELS:
        lp, lr, lf = _prf(*counts.get(lab, [0, 0, 0]))
        per_label[lab] = {"precision": lp, "recall": lr, "f1": lf,
                          "tp": counts[lab][0], "fp": counts[lab][1], "fn": counts[lab][2]}
    leak = leaked_chars / gold_chars if gold_chars else 0.0
    return Report(p, r, f, leak, per_label, tp, fp, fn, len(messages), errors, failures)
