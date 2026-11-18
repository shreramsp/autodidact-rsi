"""Summary - compare every method on one chart and print the spend per lab.  ($0)

All "10-attempt" methods start from the same baseline redactor and get exactly
10 edit attempts, so their held-out F1 is directly comparable:
  fixed workflow (naive), SO-RSI v1-v4, RSIFlow, Self-HPO.
Budget RSI is a different experiment (it saves EVALUATIONS, not attempts), so it
is reported on its own row.

Run:  python labs/summary.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pii_bench.labkit import RUNS, banner, load_json, run_dir, save_json, step
from pii_bench.llm import LEDGER

OUT = run_dir("summary")
BASELINE_F1 = None


def main():
    banner("Summary - three RSI methods on one PII redaction problem")
    so = {v: load_json(RUNS / "10_06_naive_vs_sorsi" / f"v{v}" / "results.json") for v in (1, 2, 3, 4)}
    flow = load_json(RUNS / "10_13_rsiflow" / "results.json")
    hpo = load_json(RUNS / "10_14_self_hpo" / "results.json")
    budget = load_json(RUNS / "10_05_budget_loop" / "results.json")
    pool = load_json(RUNS / "10_03_edit_pool" / "pool.json")
    base = so[4]["naive"]["curve"][0]["heldout"]

    rows = [("baseline (gen 0)", 0, base, None)]
    n = so[4]["naive"]["summary"]
    rows.append(("fixed workflow (naive)", n["accepted"], n["final_heldout"]["f1"], n["final_heldout"]["recall"]))
    for v in (1, 2, 3, 4):
        s = so[v]["SO-RSI"]["summary"]
        rows.append((f"Second-Order RSI v{v}", s["accepted"], s["final_heldout"]["f1"], s["final_heldout"]["recall"]))
    rows.append(("RSIFlow", flow["accepted"], flow["final_heldout"]["f1"], flow["final_heldout"]["recall"]))
    rows.append(("Self-HPO", hpo["accepted"], hpo["final_heldout"]["f1"], hpo["final_heldout"]["recall"]))

    step("10 edit attempts each, same starting program")
    print(f"  {'method':26s} {'accepted':>9s} {'held-out F1':>12s} {'recall':>8s}")
    for name, acc, f1, rec in rows:
        print(f"  {name:26s} {acc:>6d}/10 {f1:12.3f} {'' if rec is None else f'{rec:8.3f}'}")

    step("Budget RSI (different question: how many EVALUATIONS does selection cost?)")
    h = budget["history"][-1]
    best_pool = max(r["child_heldout_f1"] for r in pool if r["generation"] == 6)
    print(f"  live Budget RSI loop: held-out F1 {h['heldout_f1']:.3f} with {h['evals_used']:,} selection evals")
    print(f"  lab 10.03 search    : held-out F1 {best_pool:.3f} with {6 * 8 * 60:,} selection evals")

    step("Gemini spend by lab (from runs/cost_ledger.jsonl)")
    spend = defaultdict(lambda: [0, 0.0])
    for line in LEDGER.read_text().splitlines():
        d = json.loads(line)
        spend[d["tag"]][0] += 1
        spend[d["tag"]][1] += d["usd"]
    for tag, (calls, usd) in sorted(spend.items()):
        print(f"  {tag:8s} {calls:3d} calls  ${usd:.4f}")
    total = sum(v[1] for v in spend.values())
    print(f"  {'total':8s} {sum(v[0] for v in spend.values()):3d} calls  ${total:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    curves = [("fixed workflow (naive)", [c["heldout"] for c in so[4]["naive"]["curve"]], "#888"),
              ("Second-Order RSI v4", [c["heldout"] for c in so[4]["SO-RSI"]["curve"]], "#c63"),
              ("RSIFlow", [c["heldout"] for c in flow["curve"]], "#36c"),
              ("Self-HPO", [c["heldout"] for c in hpo["curve"]], "#4a7")]
    for name, ys, col in curves:
        a.plot(range(len(ys)), ys, "o-", label=name, color=col, ms=4)
    a.set_xlabel("edit attempt")
    a.set_ylabel("held-out F1 (400 unseen messages)")
    a.set_title("Same start, same 10 attempts")
    a.legend()
    names = [r[0] for r in rows]
    vals = [r[2] for r in rows]
    colors = ["#bbb", "#888", "#e9b", "#e9b", "#e9b", "#c63", "#36c", "#4a7"]
    b.barh(names[::-1], vals[::-1], color=colors[::-1])
    for i, v in enumerate(vals[::-1]):
        b.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    b.set_xlim(0, 1.08)
    b.set_xlabel("held-out F1 after 10 attempts")
    b.set_title("Final score by method")
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=130)
    print("\nsaved runs/summary/overview.png")
    save_json(OUT / "summary.json", {"ten_attempt_methods": [dict(zip(("method", "accepted", "heldout_f1", "recall"), r)) for r in rows],
                                     "budget_rsi": {"heldout_f1": h["heldout_f1"], "evals": h["evals_used"],
                                                    "search_heldout_f1": best_pool, "search_evals": 6 * 8 * 60},
                                     "spend_usd": {k: v[1] for k, v in spend.items()}, "total_usd": total})


if __name__ == "__main__":
    main()
