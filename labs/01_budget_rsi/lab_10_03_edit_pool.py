"""Lab 10.03 - Budget RSI, part 1: build a pool of REAL edits and compare selection strategies.

The course demo uses 128 synthetic (made-up) edits. Here every edit is a real
rewrite of our PII redactor proposed by Gemini.

What this script does
  1. Search for G generations. In each one Gemini proposes K edits to the
     current parent, each with a different focus (NAME, ADDRESS, PHONE, ...).
  2. Score every edit on the small steering split (60 msgs) AND, only for this
     study, on the large held-out split (400 msgs) so we know the truth.
  3. Ask: if we had to pick ONE child per generation, how good is the pick of
       oracle         - fully evaluate all K children (expensive upper bound)
       steering-only  - evaluate all K children on the small split
       budget readout - evaluate NONE; predict with the linear readout
       random         - evaluate none; pick at random
     The readout is trained leave-one-generation-out, so it never sees the
     generation it is choosing in.

Run:  python labs/01_budget_rsi/lab_10_03_edit_pool.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from pii_bench.budget import LinearReadout, spearman
from pii_bench.editing import FEATURE_NAMES, diff_features, propose_edit
from pii_bench.labkit import BASELINE, banner, data, run_dir, save_json, spend_line, step
from pii_bench.llm import Gemini
from pii_bench.sandbox import evaluate

G = 6                     # generations
FOCI = ["NAME", "ADDRESS", "PHONE", "EMAIL", "CREDIT_CARD", "SSN", "GENERAL", "GENERAL"]  # K = 8
OUT = run_dir("10_03_edit_pool")


def build_pool(llm, feedback, steering, heldout):
    parent, parent_id = BASELINE, "g0"
    programs = {"g0": BASELINE}
    pool = []
    for g in range(1, G + 1):
        fb_rep, _ = evaluate(parent, feedback)
        parent_steer, _ = evaluate(parent, steering, keep_failures=False)
        step(f"generation {g}: parent {parent_id} (steering F1 {parent_steer.f1:.3f}) -> {len(FOCI)} edits")
        for k, focus in enumerate(FOCI):
            e = propose_edit(llm, parent, fb_rep, focus=focus, variant=k)
            st, st_status = evaluate(e.code, steering, keep_failures=False)
            he, _ = evaluate(e.code, heldout, keep_failures=False)
            cid = f"g{g}c{k}"
            programs[cid] = e.code
            row = {"id": cid, "generation": g, "parent": parent_id, "focus": focus,
                   "intent": e.intent.get("change", ""), "status": st_status if e.ok else e.error,
                   "parent_rate": parent_steer.f1,
                   **dict(zip(FEATURE_NAMES, diff_features(parent, e.code))),
                   "child_steering_f1": st.f1, "child_heldout_f1": he.f1}
            pool.append(row)
            print(f"  {cid} focus={focus:11s} steering {st.f1:.3f}  heldout {he.f1:.3f}  "
                  f"+{row['lines_added']}/-{row['lines_deleted']} lines  [{row['status']}]")
        # next parent: best child by the CHEAP steering score (never by held-out)
        best = max((r for r in pool if r["generation"] == g), key=lambda r: r["child_steering_f1"])
        if best["child_steering_f1"] > parent_steer.f1:
            parent, parent_id = programs[best["id"]], best["id"]
    return pool, programs


def compare(pool, n_steer, n_held):
    gens = sorted({r["generation"] for r in pool})
    rng = random.Random(0)
    picks = {"oracle": [], "steering-only": [], "budget readout": [], "random": []}
    corr = []
    for g in gens:
        sib = [r for r in pool if r["generation"] == g]
        train = [r for r in pool if r["generation"] != g]
        model = LinearReadout().fit(train)
        pred = model.predict(sib)
        corr.append(spearman(pred, [r["child_heldout_f1"] for r in sib]))
        picks["oracle"].append(max(sib, key=lambda r: r["child_heldout_f1"]))
        picks["steering-only"].append(max(sib, key=lambda r: r["child_steering_f1"]))
        picks["budget readout"].append(sib[int(np.argmax(pred))])
        picks["random"].append(rng.choice(sib))
    k = len(FOCI)
    # "random": use the exact expected value (mean over siblings) instead of one lucky draw
    expected_random = float(np.mean([np.mean([r["child_heldout_f1"] for r in pool if r["generation"] == g])
                                     for g in gens]))
    # cost of ONE selection decision, in message-evaluations
    cost = {"oracle": k * n_held, "steering-only": k * n_steer + n_held,
            "budget readout": n_held, "random": n_held}
    oracle = np.mean([r["child_heldout_f1"] for r in picks["oracle"]])
    table = []
    for name, rows in picks.items():
        m = float(np.mean([r["child_heldout_f1"] for r in rows]))
        if name == "random":
            m = expected_random
        table.append({"strategy": name, "mean_heldout_f1_of_pick": m, "regret_vs_oracle": float(oracle - m),
                      "evals_per_decision": cost[name], "picks": [r["id"] for r in rows]})
    return table, float(np.nanmean(corr))


def plot(table, pool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [t["strategy"] for t in table]
    a.bar(names, [t["mean_heldout_f1_of_pick"] for t in table], color=["#888", "#4a7", "#36c", "#c63"])
    for i, t in enumerate(table):
        a.text(i, t["mean_heldout_f1_of_pick"] + 0.005, f"{t['evals_per_decision']:,} evals", ha="center", fontsize=9)
    a.set_ylabel("held-out F1 of the chosen edit (mean)")
    a.set_title("Which edit does each strategy pick?")
    a.set_ylim(0, max(t["mean_heldout_f1_of_pick"] for t in table) * 1.2)
    b.scatter([r["child_steering_f1"] for r in pool], [r["child_heldout_f1"] for r in pool], c=[r["generation"] for r in pool], cmap="viridis")
    b.plot([0, 1], [0, 1], "k--", lw=0.8)
    b.set_xlabel("steering F1 (60 msgs, cheap)")
    b.set_ylabel("held-out F1 (400 msgs, expensive)")
    b.set_title("Is the cheap split a good proxy? (color = generation)")
    fig.tight_layout()
    fig.savefig(OUT / "selection.png", dpi=130)
    print(f"saved {(OUT / 'selection.png').relative_to(OUT.parents[1])}")


def main():
    banner("Lab 10.03 - Budget RSI: real edit pool + selection strategies")
    feedback, steering, heldout = data()
    llm = Gemini(tag="10.03")
    pool, programs = build_pool(llm, feedback, steering, heldout)
    save_json(OUT / "pool.json", pool)
    save_json(OUT / "programs.json", programs)

    step("selection comparison (leave-one-generation-out)")
    table, rho = compare(pool, len(steering), len(heldout))
    print(f"{'strategy':16s} {'held-out F1 of pick':>20s} {'regret':>8s} {'evals/decision':>15s}")
    for t in table:
        print(f"{t['strategy']:16s} {t['mean_heldout_f1_of_pick']:20.3f} {t['regret_vs_oracle']:8.3f} {t['evals_per_decision']:15,d}")
    full = table[0]["evals_per_decision"]
    bud = next(t for t in table if t["strategy"] == "budget readout")
    print(f"\nreadout rank-correlation with held-out truth (mean over generations): {rho:.2f}")
    print(f"budget readout uses {bud['evals_per_decision']:,} instead of {full:,} evals per decision "
          f"({1 - bud['evals_per_decision'] / full:.0%} fewer)")
    print(f"training cost of the readout: {len(pool)} edits x {len(steering)} steering msgs = "
          f"{len(pool) * len(steering):,} evals, paid once and reused")
    save_json(OUT / "selection.json", {"table": table, "readout_spearman": rho})
    plot(table, pool)
    print("\n" + spend_line(llm))


if __name__ == "__main__":
    main()
