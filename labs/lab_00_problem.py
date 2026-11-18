"""Lab 00 - Meet the problem: synthetic support messages and the generation-0 redactor.  ($0)

Shows (1) what the data looks like, (2) the answer key, (3) how the deliberately
weak baseline redactor scores. Every RSI method in this repo starts from here.

Run:  python labs/lab_00_problem.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pii_bench import LABELS
from pii_bench.labkit import BASELINE, banner, data, step
from pii_bench.sandbox import evaluate, run_code


def show(m, preds=None):
    print("  " + m.text.replace("\n", "\n  "))
    for s in m.spans:
        print(f"    gold  {s.label:12s} {m.text[s.start:s.end]!r:40s} format={s.fmt}")
    if m.decoys:
        print(f"    decoys (look like PII, are NOT): {m.decoys}")
    if preds is not None:
        print(f"    baseline found: {[(m.text[s:e], lab) for s, e, lab in preds] or 'nothing'}")


def main():
    banner("Lab 00 - The PII redaction problem")
    feedback, steering, heldout = data()
    step("the three splits")
    for name, msgs, role in [("feedback", feedback, "shown to Gemini (its mistakes are quoted in prompts)"),
                             ("steering", steering, "small + cheap: decides whether an edit is kept"),
                             ("heldout", heldout, "never shown to Gemini: the honest final score")]:
        c = Counter(s.label for m in msgs for s in m.spans)
        print(f"  {name:9s} {len(msgs):3d} messages, {sum(c.values()):4d} PII spans  - {role}")
    print(f"  labels: {', '.join(LABELS)}")

    step("three example messages with their answer key")
    picks = [m for m in feedback if m.decoys and len(m.spans) >= 2][:3]
    preds = run_code(BASELINE, [m.text for m in picks]).predictions
    for m, p in zip(picks, preds):
        show(m, p)
        print()

    step("the generation-0 redactor (pii_bench/baseline_redactor.py)")
    print("  " + BASELINE.strip().replace("\n", "\n  "))
    rep, _ = evaluate(BASELINE, heldout)
    step("its score on the held-out split")
    print(f"  {rep.line()}")
    for lab in LABELS:
        d = rep.per_label[lab]
        print(f"  {lab:12s} precision {d['precision']:.2f}  recall {d['recall']:.2f}  F1 {d['f1']:.2f}")
    print("\n  -> no names or addresses at all, and it redacts order/part numbers by mistake.\n"
          "     The three RSI methods each try to improve this program with Gemini's help.")


if __name__ == "__main__":
    main()
