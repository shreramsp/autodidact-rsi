"""Lab 10.08 - Apply SO-RSI to one debugging task: phone numbers vs order numbers.

The hardest bug left in our redactor: an order number like 467-683-9760 has the
exact shape of a US phone number. This lab follows the course recipe:

  1. a small failing unit test (8 real phones that MUST be caught,
     6 look-alikes that must NOT be)
  2. up to 4 first-order fix attempts; for each we record what changed,
     what was expected, and what happened
  3. a retrospective: run the anomaly detector over those attempts
  4. one diagnostic probe -> root cause + rule
  5. one more fix, guided by the diagnosis

Run:  python labs/02_second_order_rsi/lab_10_08_debug_session.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pii_bench.editing import propose_edit
from pii_bench.generator import Message, _Builder
from pii_bench.labkit import banner, run_dir, save_json, spend_line, step
from pii_bench.llm import Gemini
from pii_bench.sandbox import evaluate
from pii_bench.sorsi import Attempt, detect, label_deltas, memory_block, regex_changes, run_probe, to_dict

OUT = run_dir("10_08_debug_session")
ATTEMPTS = 4

START = '''import re

PHONE = re.compile(r"\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}")


def find_pii(text):
    """This debugging session only cares about PHONE."""
    return [(m.start(), m.end(), "PHONE") for m in PHONE.finditer(text)]
'''

# (text before, phone or None, text after) - phone=None means the message is a look-alike decoy
CASES = [
    ("Call me at ", "(408) 555-0199", "."),
    ("My number is ", "408-555-0199", ", thanks."),
    ("Text me on ", "+1 650 555 0123", " after 5pm."),
    ("Best phone: ", "4155550188", "."),
    ("Reach me at ", "+44 20 7946 0958", " (UK)."),
    ("Call ", "212.555.0144", " any time."),
    ("My cell is ", "+91 98765 43210", "."),
    ("Office line ", "510-555-0100 ext. 42", "."),
    ("My order number is 467-683-9760.", None, ""),
    ("Order #4085550199 is the one I'm talking about.", None, ""),
    ("The replacement part number is 171-07-6651.", None, ""),
    ("The tracking number is 9400 1234 5678 9012.", None, ""),
    ("I paid $1453.25 on 02/16/2025.", None, ""),
    ("Reference code 555-0199-77 for your records.", None, ""),
]


def unit_test() -> list[Message]:
    msgs = []
    for i, (pre, phone, post) in enumerate(CASES):
        b = _Builder()
        b.add(pre)
        if phone:
            b.pii(phone, "PHONE", "unit")
        b.add(post)
        msgs.append(Message(f"ut-{i:02d}", b.text(), b.spans, [] if phone else ["lookalike"]))
    return msgs


def describe(rep) -> str:
    d = rep.per_label["PHONE"]
    return f"caught {d['tp']}/8 phones, {d['fp']} false alarms on look-alikes (F1 {d['f1']:.3f})"


def main():
    banner("Lab 10.08 - Debugging session: phones vs look-alike numbers")
    tests = unit_test()
    llm = Gemini(tag="10.08")
    code = START
    rep, _ = evaluate(code, tests)
    step(f"failing test, before any fix: {describe(rep)}")
    for f in rep.failures:
        print(f"  {'MISSED' if f.kind == 'FN' else 'FALSE ALARM'}: {f.context}")

    step(f"first-order attempts (up to {ATTEMPTS}): patch -> test -> repeat")
    history = []
    for r in range(1, ATTEMPTS + 1):
        e = propose_edit(llm, code, rep, focus="PHONE", variant=r,
                         extra="\nThis is a unit-test debugging session: only PHONE matters.\n")
        new_rep, status = evaluate(e.code, tests)
        added, removed = regex_changes(code, e.code)
        att = Attempt(r, "PHONE", e.intent.get("change", ""), e.intent.get("expected", ""),
                      new_rep.f1 > rep.f1, rep.f1, new_rep.f1, label_deltas(rep, new_rep), added, removed)
        history.append(att)
        print(f"\n  attempt {r}")
        print(f"    changed  : {att.change[:150]}")
        print(f"    expected : {att.expected[:150]}")
        print(f"    happened : {describe(new_rep)}  [{status}]  -> {'KEEP' if att.accepted else 'discard'}")
        if att.accepted:
            code, rep = e.code, new_rep
        if rep.f1 == 1.0:
            print("    all tests pass - stopping early")
            break

    step("retrospective: run the anomaly detector over the attempts")
    found = []
    for i, att in enumerate(history):
        att.anomalies = detect(att, history[:i], version=2)
        found += att.anomalies
        print(f"  attempt {att.round}: {att.anomalies or 'no anomaly'}")

    result = {"attempts": [to_dict(a) for a in history], "after_attempts": describe(rep)}
    if rep.f1 < 1.0:
        kind = "opposing" if "opposing" in found else ("recurrence" if "recurrence" in found else "mismatch")
        step(f"diagnostic probe ({kind})")
        probe = run_probe(llm, kind, "PHONE", history, code, rep, variant=1, messages=tests, version=3)
        probe["round"] = 0
        print(f"  root cause : {probe['root_cause']}")
        print(f"  rule       : {probe['rule']}")
        print(f"  next edit  : {probe['next_edit']}")
        step("one fix guided by the diagnosis")
        e = propose_edit(llm, code, rep, focus="PHONE", variant=99,
                         extra=memory_block([probe]) + f"Diagnosis recommends this next edit: {probe['next_edit']}\n"
                               "\nThis is a unit-test debugging session: only PHONE matters.\n")
        final, status = evaluate(e.code, tests)
        print(f"  {describe(final)}  [{status}]")
        for f in final.failures:
            print(f"    {'MISSED' if f.kind == 'FN' else 'FALSE ALARM'}: {f.context}")
        print("  Note: F1 weighs a missed phone (a privacy LEAK) the same as a false alarm on an\n"
              "  order number (over-redaction). The 'is the metric misaligned?' probe question applies.")
        (OUT / "diagnosis_guided_fix.py").write_text(e.code)
        if final.f1 > rep.f1:
            code, rep = e.code, final
        result.update({"probe": probe, "after_diagnosis": describe(final)})
    else:
        print("  tests already pass; no probe needed")

    step("final state")
    print(f"  {describe(rep)}")
    for f in rep.failures:
        print(f"  still {'MISSED' if f.kind == 'FN' else 'FALSE ALARM'}: {f.context}")
    (OUT / "final_phone_fix.py").write_text(code)
    save_json(OUT / "session.json", result)
    print("\n" + spend_line(llm))


if __name__ == "__main__":
    main()
