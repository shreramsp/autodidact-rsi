"""Lab 10.05 - Budget RSI, part 3: extend the scorer to OUR task and run it live.

Part A ($0) - fix the readout using what lab 10.04 found:
   * predict the GAIN over the parent, not the absolute score (removes confounding)
   * add two task-aware features (see pii_bench/budget.py: TASK_FEATURES)
   and compare feature sets leave-one-generation-out on the 10.03 pool.

Part B (live, ~24 Gemini calls) - a real Budget RSI improvement loop:
   each generation
     1. Gemini proposes K=4 edits, each aimed at one of the parent's 4 weakest labels
     2. the readout scores all 4 WITHOUT evaluating them (1 smoke-test eval each)
     3. only the top-scored edit is evaluated on the steering split
     4. keep it if it beats the parent; add it to the readout's training data
   For honesty we ALSO score every child on steering in the background ("shadow
   evaluation") - never used for decisions, only to measure how good the picks were.

Run:  python labs/01_budget_rsi/lab_10_05_budget_loop.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from pii_bench import LABELS
from pii_bench.budget import READOUT_FEATURES, TASK_FEATURES, LinearReadout, spearman, task_features
from pii_bench.editing import FEATURE_NAMES, diff_features, propose_edit
from pii_bench.labkit import BASELINE, RUNS, banner, data, load_json, run_dir, save_json, spend_line, step
from pii_bench.llm import Gemini
from pii_bench.sandbox import evaluate

GENERATIONS, K = 6, 4
SELECTED = ["parent_rate"] + TASK_FEATURES       # the feature set Part A picks
OUT = run_dir("10_05_budget_loop")


def part_a(feedback):
    step("Part A: add task-aware features to the 10.03 pool and compare readouts ($0)")
    pool = load_json(RUNS / "10_03_edit_pool" / "pool.json")
    progs = load_json(RUNS / "10_03_edit_pool" / "programs.json")
    reports = {}
    for r in pool:
        if r["parent"] not in reports:
            reports[r["parent"]], _ = evaluate(progs[r["parent"]], feedback)
        r.update(task_features(progs[r["parent"]], progs[r["id"]], reports[r["parent"]]))
        r["gain"] = r["child_steering_f1"] - r["parent_rate"]

    def loo(features, target):
        picks, rhos = [], []
        for g in sorted({r["generation"] for r in pool}):
            sib = [r for r in pool if r["generation"] == g]
            m = LinearReadout(features=features).fit([r for r in pool if r["generation"] != g], target=target)
            p = m.predict(sib)
            picks.append(sib[int(np.argmax(p))]["child_heldout_f1"])
            rhos.append(spearman(p, [s["child_heldout_f1"] for s in sib]))
        return float(np.mean(picks)), float(np.nanmean(rhos))

    rows = []
    for name, feats, target in [("paper: 8 diff features, absolute score", READOUT_FEATURES, "child_steering_f1"),
                                ("paper: 8 diff features, predict gain", READOUT_FEATURES, "gain"),
                                ("paper + task features, predict gain", READOUT_FEATURES + TASK_FEATURES, "gain"),
                                ("task features only, predict gain", SELECTED, "gain")]:
        m, rho = loo(feats, target)
        rows.append({"readout": name, "mean_heldout_f1_of_pick": m, "spearman": rho})
        print(f"  {name:42s} pick held-out F1 {m:.3f}   rank corr {rho:+.2f}")
    rand = np.mean([np.mean([r["child_heldout_f1"] for r in pool if r["generation"] == g]) for g in range(1, 7)])
    print(f"  {'random pick (expected value)':42s} pick held-out F1 {rand:.3f}")
    return pool, rows


def part_b(llm, pool, feedback, steering, heldout):
    step(f"Part B: live Budget RSI loop - {GENERATIONS} generations x {K} edits")
    readout_rows = [r for r in pool]                       # warm start from the 10.03 pool
    code, prog_id = BASELINE, "g0"
    parent_steer = evaluate(code, steering, keep_failures=False)[0].f1
    evals_used, stall = 0, 0
    history =[{"generation": 0, "program": prog_id, "steering_f1": parent_steer,
                "heldout_f1": evaluate(code, heldout, keep_failures=False)[0].f1, "evals_used": 0}]
    decisions = []
    print(f"  gen 0  steering {parent_steer:.3f}  held-out {history[0]['heldout_f1']:.3f}")
    for g in range(1, GENERATIONS + 1):
        fb_rep, _ = evaluate(code, feedback)
        weakest = sorted(LABELS, key=lambda l: -(fb_rep.per_label[l]["fp"] + fb_rep.per_label[l]["fn"]))[:K]
        model = LinearReadout(features=SELECTED).fit(readout_rows, target="gain")
        kids = []
        for k, focus in enumerate(weakest):
            # `stall` = rejections in a row. Without it a rejected generation re-sends the
            # identical prompt and gets the identical (cached) edits back forever.
            e = propose_edit(llm, code, fb_rep, focus=focus, variant=1000 + k + 10 * stall)
            row = {"id": f"b{g}c{k}", "focus": focus, "parent_rate": parent_steer,
                   **dict(zip(FEATURE_NAMES, diff_features(code, e.code))),
                   **task_features(code, e.code, fb_rep), "code": e.code}
            evals_used += 1                                    # the smoke test
            kids.append(row)
        pred = model.predict(kids)
        chosen = kids[int(np.argmax(pred))]
        st, _ = evaluate(chosen["code"], steering, keep_failures=False)
        evals_used += len(steering)
        chosen["child_steering_f1"] = st.f1
        chosen["gain"] = st.f1 - parent_steer
        readout_rows.append(chosen)                            # online learning

        # shadow evaluation (NOT used for the decision; only to grade it)
        shadow = [evaluate(c["code"], steering, keep_failures=False)[0].f1 for c in kids]
        best_possible = int(np.argmax(shadow))
        accepted = st.f1 > parent_steer
        if accepted:
            code, prog_id, parent_steer = chosen["code"], chosen["id"], st.f1
            stall = 0
        else:
            stall += 1
        he = evaluate(code, heldout, keep_failures=False)[0].f1
        decisions.append({"generation": g, "foci": weakest, "predicted": pred.round(4).tolist(),
                          "shadow_steering": shadow, "chosen": chosen["id"], "chosen_focus": chosen["focus"],
                          "best_by_shadow": kids[best_possible]["id"],
                          "regret": float(max(shadow) - shadow[int(np.argmax(pred))]), "accepted": accepted})
        history.append({"generation": g, "program": prog_id, "steering_f1": parent_steer,
                        "heldout_f1": he, "evals_used": evals_used})
        print(f"  gen {g}  foci {weakest}  picked {chosen['focus']:11s} "
              f"(shadow best: {kids[best_possible]['focus']:11s}, regret {decisions[-1]['regret']:.3f})  "
              f"{'ACCEPT' if accepted else 'reject'}  steering {parent_steer:.3f}  held-out {he:.3f}")
    final_code = code
    return history, decisions, final_code


def plot(history, pool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # the 10.03 run (steering-greedy, all 8 children evaluated each generation) for reference
    ref_x, ref_y, used = [0], [history[0]["heldout_f1"]], 0   # same baseline program
    for g in range(1, 7):
        used += 8 * 60
        sib = [r for r in pool if r["generation"] == g]
        ref_x.append(used)
        ref_y.append(max(sib, key=lambda r: r["child_steering_f1"])["child_heldout_f1"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([h["evals_used"] for h in history], [h["heldout_f1"] for h in history], "o-", label="Budget RSI (readout picks, 1 of 4 evaluated)")
    ax.plot(ref_x, ref_y, "s--", alpha=0.7, label="Lab 10.03 search (all 8 children evaluated)")
    ax.set_xlabel("message-evaluations spent on selection")
    ax.set_ylabel("held-out F1 of current program")
    ax.set_title("Improvement per evaluation budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "budget_loop.png", dpi=130)
    print(f"saved runs/10_05_budget_loop/budget_loop.png")


def main():
    banner("Lab 10.05 - Budget RSI on our task: better features + a live loop")
    feedback, steering, heldout = data()
    pool, rows = part_a(feedback)
    llm = Gemini(tag="10.05")
    history, decisions, final_code = part_b(llm, pool, feedback, steering, heldout)
    agree = sum(d["chosen"] == d["best_by_shadow"] for d in decisions)
    print(f"\n  readout picked the same edit as full evaluation in {agree}/{len(decisions)} generations; "
          f"mean regret {np.mean([d['regret'] for d in decisions]):.3f} steering F1")
    print(f"  final held-out F1 {history[-1]['heldout_f1']:.3f} using {history[-1]['evals_used']:,} selection evals "
          f"(lab 10.03 used {6 * 8 * 60:,} to reach {max(r['child_heldout_f1'] for r in pool if r['generation'] == 6):.3f})")
    save_json(OUT / "results.json", {"part_a": rows, "history": history, "decisions": decisions})
    (OUT / "final_redactor.py").write_text(final_code)
    plot(history, pool)
    print("\n" + spend_line(llm))


if __name__ == "__main__":
    main()
