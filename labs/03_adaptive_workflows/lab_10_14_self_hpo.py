"""Lab 10.14 - Self-HPO: alternate HARNESS optimization and POLICY optimization.

  policy   what the model produces: the redactor code (as in every other lab)
  harness  the environment the policy improves in:
             digest_size   how many failure examples each edit prompt shows
             group_by      "none" | "label"   (how the failures are laid out)
             focus_mode    "worst_label" | "rotate" | "misses_first" | "false_alarms_first"
             extra_tests   NEW labeled messages the policy writes for itself

  H-step   the SAME model ("policy as harness proposer") reads how the last
           P-steps went and proposes a new harness (JSON). Its test cases are
           validated (every PII value must really occur in its message).
  P-step   one ordinary edit of the redactor under the current harness.

Schedule: H, P, P, P, H, P, P, P, H, P, P, P, P  ->  10 P-steps, 3 H-steps.
Acceptance stays the fixed steering split, so a harness can change what the
policy SEES but can never move the goalposts. Held-out is reporting only.

Baseline: the fixed harness of lab 10.06 naive (same program, 10 edits, $0 replay).

Run:  python labs/03_adaptive_workflows/lab_10_14_self_hpo.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pii_bench import LABELS
from pii_bench.editing import failure_digest, propose_edit, score_table
from pii_bench.generator import Message, Span
from pii_bench.labkit import BASELINE, RUNS, banner, data, load_json, run_dir, save_json, spend_line, step
from pii_bench.llm import Gemini
from pii_bench.sandbox import evaluate

P_STEPS = 10
H_AT = {1, 4, 7}                     # an H-step runs before these P-steps
OUT = run_dir("10_14_self_hpo")
DEFAULT_HARNESS = {"digest_size": 24, "group_by": "none", "focus_mode": "worst_label", "extra_tests": []}

H_SYSTEM = """You are optimizing the HARNESS that a code-editing agent works in. The agent improves
a rule-based PII redactor (labels: NAME, EMAIL, PHONE, CREDIT_CARD, SSN, ADDRESS).
You do not edit the redactor. You choose how the agent is shown its mistakes and you may write
NEW labeled test messages for PII formats you suspect the redactor misses or over-redacts.

Respond with ONLY a JSON object:
{"digest_size": <int 12-48>,
 "group_by": "none" | "label",
 "focus_mode": "worst_label" | "rotate" | "misses_first" | "false_alarms_first",
 "new_tests": [{"text": "<a realistic customer-support message>",
                "pii": [["<exact substring of text>", "<LABEL>"], ...]}],   (0-6 items)
 "rationale": "<one sentence>"}
Remember: order numbers, tracking numbers, part numbers, prices and product names are NOT PII;
include some of those as traps in your tests (with no pii entry for them)."""


def h_step(llm, r, harness, history, code, rep, n_extra):
    recent = "\n".join(f"  P-step {h['p']}: focus {h['focus']} -> {'ACCEPTED' if h['accepted'] else 'rejected'} "
                       f"(steering {h['before']:.3f}->{h['after']:.3f})" for h in history[-4:]) or "  (none yet)"
    show = {k: v for k, v in harness.items() if k != "extra_tests"}
    prompt = f"""Current harness: {json.dumps(show)}  (+ {n_extra} self-written test messages so far)

Recent policy steps under this harness:
{recent}

Current redactor:
```python
{code}
```

Scores on the training-feedback set:
{score_table(rep)}

Sample of current mistakes:
{failure_digest(rep, k=20, seed=r)}

Propose the next harness."""
    resp = llm.generate(prompt, system=H_SYSTEM, temperature=0.5, variant=r)
    m = re.search(r"\{.*\}", resp.text, re.S)
    try:
        cfg = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        cfg = {}
    new = dict(harness)
    if isinstance(cfg.get("digest_size"), int):
        new["digest_size"] = max(12, min(48, cfg["digest_size"]))
    if cfg.get("group_by") in ("none", "label"):
        new["group_by"] = cfg["group_by"]
    if cfg.get("focus_mode") in ("worst_label", "rotate", "misses_first", "false_alarms_first"):
        new["focus_mode"] = cfg["focus_mode"]
    tests, rejected = validate_tests(cfg.get("new_tests") or [], f"h{r}")
    return new, tests, rejected, cfg.get("rationale", "")


def validate_tests(raw, prefix):
    """Keep a self-written test only if every PII value occurs in its text with a known label."""
    ok, bad = [], 0
    for i, t in enumerate(raw[:6]):
        try:
            text, spans, pos = str(t["text"]), [], 0
            for value, label in t.get("pii", []):
                label = str(label).upper()
                at = text.find(str(value))
                if label not in LABELS or at < 0 or not value:
                    raise ValueError
                spans.append(Span(at, at + len(value), label, "self_written"))
            ok.append(Message(f"{prefix}-{i}", text, spans, []))
        except Exception:
            bad += 1
    return ok, bad


def digest_for(rep, harness, focus, r):
    kind = {"misses_first": "FN", "false_alarms_first": "FP"}.get(harness["focus_mode"])
    text = failure_digest(rep, k=harness["digest_size"], focus=focus, seed=r, kind=kind)
    if harness["group_by"] == "label":
        lines = text.splitlines()
        text = "\n".join(sorted(lines, key=lambda l: next((lab for lab in LABELS if lab in l.split(":")[0]), "")))
    return text


def pick_focus(harness, rep, p):
    mode = harness["focus_mode"]
    if mode == "rotate":
        return LABELS[(p - 1) % len(LABELS)]
    if mode in ("misses_first", "false_alarms_first"):
        return "GENERAL"
    return max(LABELS, key=lambda l: rep.per_label[l]["fp"] + rep.per_label[l]["fn"])


def main():
    banner("Lab 10.14 - Self-HPO: alternating harness (H) and policy (P) steps")
    feedback, steering, heldout = data()
    llm = Gemini(tag="10.14")
    harness = dict(DEFAULT_HARNESS)
    code = BASELINE
    st = evaluate(code, steering, keep_failures=False)[0].f1
    curve = [{"p": 0, "heldout": evaluate(code, heldout, keep_failures=False)[0].f1}]
    history, h_log = [], []
    for p in range(1, P_STEPS + 1):
        train = feedback + harness["extra_tests"]
        rep, _ = evaluate(code, train)
        if p in H_AT:
            new, tests, bad, why = h_step(llm, p, harness, history, code, rep, len(harness["extra_tests"]))
            new["extra_tests"] = harness["extra_tests"] + tests
            changed = {k: (harness[k], new[k]) for k in ("digest_size", "group_by", "focus_mode") if harness[k] != new[k]}
            print(f"  H-step before P{p}: {changed or 'no setting changed'}; +{len(tests)} self-written tests "
                  f"({bad} rejected by validation)\n      rationale: {why[:160]}")
            if tests:
                print(f"      e.g. {tests[0].text[:120]!r} -> {[(tests[0].text[s.start:s.end], s.label) for s in tests[0].spans]}")
            h_log.append({"before_p": p, "changed": changed, "tests_added": len(tests), "tests_rejected": bad,
                          "rationale": why, "examples": [t.to_json() for t in tests[:3]]})
            harness = new
            train = feedback + harness["extra_tests"]
            rep, _ = evaluate(code, train)
        focus = pick_focus(harness, rep, p)
        e = propose_edit(llm, code, rep, focus=focus, variant=p, digest=digest_for(rep, harness, focus, p))
        new_st = evaluate(e.code, steering, keep_failures=False)[0].f1
        ok = new_st > st
        history.append({"p": p, "focus": focus, "accepted": ok, "before": st, "after": new_st})
        print(f"  P{p:<2d} focus {focus:11s} [{harness['focus_mode']}, show {harness['digest_size']}, "
              f"group {harness['group_by']}]  steering {st:.3f}->{new_st:.3f}  {'ACCEPT' if ok else 'reject'}")
        if ok:
            code, st = e.code, new_st
        curve.append({"p": p, "heldout": evaluate(code, heldout, keep_failures=False)[0].f1})

    final = evaluate(code, heldout)[0]
    naive = load_json(RUNS / "10_06_naive_vs_sorsi" / "v4" / "results.json")["naive"]["summary"]
    step("comparison (same baseline, 10 policy edits each)")
    print(f"  fixed harness (10.06 naive): accepted {naive['accepted']}/10  held-out F1 {naive['final_heldout']['f1']:.3f}")
    print(f"  Self-HPO                   : accepted {sum(h['accepted'] for h in history)}/10  held-out F1 {final.f1:.3f}  "
          f"(+{len(H_AT)} harness calls)")
    print(f"  per label: {final.summary()['per_label_f1']}")
    (OUT / "final_redactor.py").write_text(code)
    (OUT / "self_written_tests.jsonl").write_text("\n".join(json.dumps(t.to_json()) for t in harness["extra_tests"]))
    save_json(OUT / "results.json", {"p_steps": history, "h_steps": h_log, "curve": curve,
                                      "final_harness": {k: v for k, v in harness.items() if k != "extra_tests"},
                                      "final_heldout": final.summary(),
                                      "accepted": sum(h["accepted"] for h in history)})
    print("\n" + spend_line(llm))


if __name__ == "__main__":
    main()
