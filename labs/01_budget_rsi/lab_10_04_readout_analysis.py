"""Lab 10.04 - Budget RSI, part 2: open up the linear readout.  (no API calls, $0)

Uses the 48 real edits saved by lab 10.03 and answers the lab's questions:

  1. What did the readout learn? -> print every coefficient, ranked
  2. Why did it pick WORSE than random in 10.03?  -> a confounding check
  3. What if we keep only 4 of the 8 diff features?
  4. How does the readout change with fewer training edits?

Run:  python labs/01_budget_rsi/lab_10_04_readout_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from pii_bench.budget import READOUT_FEATURES, LinearReadout, spearman
from pii_bench.labkit import RUNS, banner, load_json, run_dir, save_json, step

OUT = run_dir("10_04_readout_analysis")


def loo(pool, features, target="child_steering_f1", train_frac=1.0, seed=0):
    """Leave-one-generation-out: train on the other generations, pick in this one."""
    rng = np.random.default_rng(seed)
    picks, rhos = [], []
    for g in sorted({r["generation"] for r in pool}):
        sib = [r for r in pool if r["generation"] == g]
        train = [r for r in pool if r["generation"] != g]
        if train_frac < 1.0:
            idx = rng.choice(len(train), max(4, int(len(train) * train_frac)), replace=False)
            train = [train[i] for i in idx]
        m = LinearReadout(features=features).fit(train, target=target)
        p = m.predict(sib)
        best = np.flatnonzero(np.isclose(p, p.max()))       # break ties at random,
        picks.append(sib[int(rng.choice(best))]["child_heldout_f1"])  # not "first child wins"
        rhos.append(spearman(p, [s["child_heldout_f1"] for s in sib]))
    return float(np.mean(picks)), float(np.nanmean(rhos))


def main():
    banner("Lab 10.04 - What did the Budget RSI readout learn?")
    pool = load_json(RUNS / "10_03_edit_pool" / "pool.json")
    print(f"{len(pool)} real edits from lab 10.03")

    step("1. coefficients of the readout trained on all edits (standardized)")
    model = LinearReadout().fit(pool)
    coefs = model.coefficients()
    for name, w in coefs:
        bar = "#" * int(abs(w) * 200)
        print(f"  {name:22s} {w:+.4f}  {bar}")
    print("  (+ means 'more of this -> higher predicted steering F1')")

    step("2. confounding check: does edit size mean the same thing early and late?")
    early = [r for r in pool if r["generation"] <= 2]
    late = [r for r in pool if r["generation"] >= 4]
    for name, grp in [("early (gen 1-2)", early), ("late  (gen 4-6)", late)]:
        print(f"  {name}: avg lines added {np.mean([r['lines_added'] for r in grp]):5.1f}   "
              f"avg parent F1 {np.mean([r['parent_rate'] for r in grp]):.3f}")
    gain = [r["child_steering_f1"] - r["parent_rate"] for r in pool]
    rho_abs = spearman([r["lines_added"] for r in pool], [r["child_steering_f1"] for r in pool])
    rho_gain = spearman([r["lines_added"] for r in pool], gain)
    print(f"  lines_added vs absolute score : rho {rho_abs:+.2f}")
    print(f"  lines_added vs GAIN over parent: rho {rho_gain:+.2f}")
    print("  -> Big edits happened early, when the program was weak. A model of the ABSOLUTE\n"
          "     score learns 'big edit = low score', but among siblings bigger edits help.")

    step("3. 8 diff features vs 4")
    four = ["parent_rate", "lines_added", "lines_deleted", "regex_added", "funcs_added"]
    rows = []
    for name, feats in [("parent_rate + 8 diff features", READOUT_FEATURES),
                        ("parent_rate + 4 diff features", four),
                        ("parent_rate only (ablation)", ["parent_rate"])]:
        runs = [loo(pool, feats, seed=s) for s in range(20)]
        m, rho = float(np.mean([r[0] for r in runs])), float(np.mean([r[1] for r in runs]))
        rows.append({"features": name, "mean_heldout_f1_of_pick": m, "spearman": rho})
        print(f"  {name:32s} pick held-out F1 {m:.3f}   rank corr {rho:+.2f}")
    print("  (parent_rate is identical for siblings, so 'parent_rate only' = a random tie-break)")

    step("4. fewer training edits")
    curve = []
    for frac in [0.25, 0.5, 0.75, 1.0]:
        ms = [loo(pool, READOUT_FEATURES, train_frac=frac, seed=s)[0] for s in range(5)]
        curve.append({"train_fraction": frac, "mean_heldout_f1_of_pick": float(np.mean(ms))})
        print(f"  {int(frac * 100):3d}% of training edits -> pick held-out F1 {np.mean(ms):.3f}")

    save_json(OUT / "analysis.json", {"coefficients": coefs, "rho_lines_vs_abs": rho_abs,
                                      "rho_lines_vs_gain": rho_gain, "feature_sets": rows,
                                      "training_curve": curve})
    print("\nTakeaway: the readout is only as good as its features. Lab 10.05 adds\n"
          "task-aware features and predicts the GAIN instead of the absolute score.")


if __name__ == "__main__":
    main()
