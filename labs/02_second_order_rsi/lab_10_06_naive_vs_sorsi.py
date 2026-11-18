"""Lab 10.06 - Second-Order RSI: naive (first-order) loop vs SO-RSI loop, live.

Both loops start from the same baseline redactor and get the same number of
edit attempts. Each round: pick the label with the most mistakes, ask Gemini
for one edit, keep it only if the steering F1 improves.

  naive   just repeats that. It still RECORDS anomalies (so we can count them),
          but never acts on them.
  SO-RSI  after every attempt runs the anomaly detector on its own trace.
          On an anomaly it runs a diagnostic probe (one extra Gemini call that
          writes no code), stores the root cause in inquiry memory, and every
          later edit prompt includes that memory.

Two versions of SO-RSI (same naive baseline):
  --version 1  the first design. It LOST to naive; see the post-mortem in the README.
  --version 2  after the post-mortem: (a) "opposing" = real precision/recall flip-flops only,
               (b) probes must stay consistent with real examples of the label,
               (c) memory rules are refuted after 2 failed edits that followed them.
               Still stuck: the trace showed the WORKFLOW was the problem (same evidence
               every round, a label sitting at a precision/recall balance point).
  --version 3  the probe may also change the workflow: SWITCH_TARGET (park the label),
               RESAMPLE_EVIDENCE (rotate failure examples), ADD_PRESERVE_EXAMPLES.
               Observed: the probe model picked KEEP_GOING every time.
  --version 4  adds a code-level escalation rule (from the course's lab 10.07 answer:
               "if enough inconclusive probes accumulate ... try broader exploration"):
               once 2 rules for a label are refuted, the label is parked for 4 rounds.

Run:  python labs/02_second_order_rsi/lab_10_06_naive_vs_sorsi.py            (v4)
      python labs/02_second_order_rsi/lab_10_06_naive_vs_sorsi.py --version 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import Counter

from pii_bench import LABELS
from pii_bench.editing import failure_digest, propose_edit
from pii_bench.labkit import BASELINE, banner, data, run_dir, save_json, spend_line, step
from pii_bench.llm import Gemini
from pii_bench.sandbox import evaluate
from pii_bench.sorsi import (Attempt, detect, label_deltas, memory_block, preserve_examples, regex_changes,
                             run_probe, to_dict, update_memory)

ROUNDS = 10
MAX_PROBES = 4
PRIORITY = ["opposing", "recurrence", "mismatch"]
OUT = None  # set in main() per version


def worst_label(rep, parked=()):
    options = [l for l in LABELS if l not in parked] or list(LABELS)
    return max(options, key=lambda l: rep.per_label[l]["fp"] + rep.per_label[l]["fn"])


def run_loop(mode, llm, feedback, steering, heldout, version):
    step(f"{mode.upper()} loop: {ROUNDS} edit attempts")
    code = BASELINE
    fb_rep, _ = evaluate(code, feedback)
    st = evaluate(code, steering, keep_failures=False)[0].f1
    history, memory, probes, curve = [], [], [], []
    pending, probed = None, set()
    parked, preserve, resample = {}, set(), False     # v3 workflow state
    curve.append({"round": 0, "steering": st, "heldout": evaluate(code, heldout, keep_failures=False)[0].f1})
    for r in range(1, ROUNDS + 1):
        active_parks = {l for l, until in parked.items() if r <= until}
        if pending and pending["target"] in active_parks:   # a parked label beats a pending probe
            pending = None
        focus = pending["target"] if pending else worst_label(fb_rep, active_parks)
        extra, digest = "", None
        if mode == "sorsi":
            extra = memory_block(memory)
            if pending:
                extra += f"Diagnosis recommends this next edit: {pending['next_edit']}\n"
            if focus in preserve:
                extra += f"\nReal {focus} examples that must STILL be caught after your edit:\n" \
                         f"{preserve_examples(feedback, focus)}\n"
            if resample:
                digest = failure_digest(fb_rep, k=36, focus=focus, seed=r)
        e = propose_edit(llm, code, fb_rep, focus=focus, extra=extra, digest=digest, variant=r)
        new_fb, _ = evaluate(e.code, feedback)
        new_st = evaluate(e.code, steering, keep_failures=False)[0].f1
        added, removed = regex_changes(code, e.code)
        att = Attempt(r, focus, e.intent.get("change", "") or e.error, e.intent.get("expected", ""),
                      new_st > st, st, new_st, label_deltas(fb_rep, new_fb), added, removed)
        att.anomalies = detect(att, history, version)
        history.append(att)
        flag = f"  anomalies: {att.anomalies}" if att.anomalies else ""
        print(f"  {att.line()}{flag}")
        if mode == "sorsi" and version >= 2:
            for m in update_memory(memory, att):
                print(f"    REFUTED rule (2 failed edits followed it): {m['rule'][:120]}")
            n_refuted = sum(1 for m in memory if m["target"] == focus and m.get("status") == "refuted")
            if version >= 4 and n_refuted >= 2 and parked.get(focus, 0) < r:
                parked[focus] = r + 4
                pending = None
                print(f"    ESCALATION: {n_refuted} refuted rules on {focus} -> park {focus} until round {r + 4}")
        if att.accepted:
            code, fb_rep, st = e.code, new_fb, new_st
        pending = None

        if mode == "sorsi" and att.anomalies and len(probes) < MAX_PROBES:
            for a in sorted(att.anomalies, key=PRIORITY.index):
                if (a, focus) not in probed:
                    if version >= 2:
                        probe = run_probe(llm, a, focus, history, code, fb_rep, variant=r, messages=feedback,
                                          refuted=[m for m in memory if m.get("status") == "refuted"],
                                          version=version)
                    else:
                        probe = run_probe(llm, a, focus, history, code, fb_rep, variant=r)
                    probe["round"] = r
                    probed.add((a, focus))
                    memory.append(probe)
                    probes.append(probe)
                    pending = probe
                    print(f"    PROBE[{a} on {focus}] root cause: {probe['root_cause'][:150]}")
                    print(f"    RULE: {probe['rule'][:150]}")
                    if version >= 3:
                        act = probe["action"]
                        print(f"    WORKFLOW ACTION: {act}")
                        if act == "SWITCH_TARGET":
                            parked[focus] = r + 3
                            pending = None
                        elif act == "RESAMPLE_EVIDENCE":
                            resample = True
                        elif act == "ADD_PRESERVE_EXAMPLES":
                            preserve.add(focus)
                    break
        curve.append({"round": r, "steering": st, "heldout": evaluate(code, heldout, keep_failures=False)[0].f1})

    final = evaluate(code, heldout)[0]
    summary = {"mode": mode, "attempts": len(history), "accepted": sum(a.accepted for a in history),
               "success_rate": sum(a.accepted for a in history) / len(history),
               "final_steering": st, "final_heldout": final.summary(),
               "anomalies": dict(Counter(x for a in history for x in a.anomalies)),
               "probes": len(probes)}
    return {"summary": summary, "trace": [to_dict(a) for a in history], "probes": probes,
            "memory": memory, "curve": curve, "final_code": code}


def plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, res in runs.items():
        ax.plot([c["round"] for c in res["curve"]], [c["heldout"] for c in res["curve"]], "o-", label=name)
        if name == "SO-RSI":
            lo = min(c["heldout"] for c in res["curve"])
            for p in res["probes"]:
                ax.axvline(p["round"], color="#a33", ls=":", lw=0.8)
                ax.text(p["round"] + 0.08, lo + 0.02, f"probe: {p['anomaly']} ({p['target']})",
                        rotation=90, fontsize=7.5, color="#a33", va="bottom")
    ax.set_xlabel("edit attempt")
    ax.set_ylabel("held-out F1 of current program")
    ax.set_title("First-order (naive) vs Second-Order RSI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "naive_vs_sorsi.png", dpi=130)
    print(f"saved {OUT.relative_to(OUT.parents[2]) / 'naive_vs_sorsi.png'}")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=4, choices=[1, 2, 3, 4])
    version = ap.parse_args().version
    OUT = run_dir(f"10_06_naive_vs_sorsi/v{version}")
    banner(f"Lab 10.06 - Naive RSI vs Second-Order RSI (SO-RSI v{version})")
    feedback, steering, heldout = data()
    llm = Gemini(tag="10.06")
    runs = {"naive": run_loop("naive", llm, feedback, steering, heldout, version),
            "SO-RSI": run_loop("sorsi", llm, feedback, steering, heldout, version)}
    step("comparison")
    print(f"{'':8s} {'accepted':>9s} {'success':>8s} {'held-out F1':>12s} {'P':>6s} {'R':>6s}  anomalies seen        probes")
    for name, res in runs.items():
        s = res["summary"]
        h = s["final_heldout"]
        print(f"{name:8s} {s['accepted']:>5d}/{s['attempts']:<3d} {s['success_rate']:8.0%} {h['f1']:12.3f} "
              f"{h['precision']:6.3f} {h['recall']:6.3f}  {str(s['anomalies']):22s} {s['probes']}")
    for name, res in runs.items():
        (OUT / f"final_{name.lower().replace('-', '')}.py").write_text(res.pop("final_code"))
    save_json(OUT / "results.json", runs)
    plot(runs)
    print("\n" + spend_line(llm))


if __name__ == "__main__":
    main()
