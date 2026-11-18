"""Lab 10.12 - Compare workflow adaptation: where are OUR bottlenecks?  (no API calls, $0)

RSIFlow and Self-HPO both adapt the improvement PROCESS, but different parts of it:

  RSIFlow   a meta-agent re-routes work between workflow steps (nodes)
            toward whatever is currently the bottleneck
  Self-HPO  alternates two optimizations: the HARNESS (the environment the
            model improves in: tests, feedback format) and the POLICY
            (what the model produces: here, the redactor code)

Before building either, this script profiles the runs we already have
(labs 10.03 and 10.06) to find the real bottlenecks, then maps each one to
the method that targets it.

Run:  python labs/03_adaptive_workflows/lab_10_12_workflow_profile.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import Counter, defaultdict

from pii_bench.labkit import RUNS, banner, load_json, run_dir, save_json, step

OUT = run_dir("10_12_workflow_profile")


def main():
    banner("Lab 10.12 - Profile the improvement workflow")
    pool = load_json(RUNS / "10_03_edit_pool" / "pool.json")
    naive = load_json(RUNS / "10_06_naive_vs_sorsi" / "v4" / "results.json")["naive"]
    findings = {}

    step("1. edit outcomes by focus (lab 10.03, 48 edits)")
    by_focus = defaultdict(list)
    for r in pool:
        by_focus[r["focus"]].append(r["child_steering_f1"] - r["parent_rate"])
    for f, gains in sorted(by_focus.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        wins = sum(g > 0.001 for g in gains)
        print(f"  {f:12s} edits {len(gains):2d}  improved {wins:2d}  mean gain {sum(gains) / len(gains):+.3f}")
    findings["gain_by_focus"] = {f: sum(g) / len(g) for f, g in by_focus.items()}

    step("2. where the fixed 'worst label first' workflow spent its attempts (lab 10.06 naive)")
    spent = Counter(a["target"] for a in naive["trace"])
    won = Counter(a["target"] for a in naive["trace"] if a["accepted"])
    for lab, n in spent.most_common():
        print(f"  {lab:12s} attempts {n}  accepted {won.get(lab, 0)}")
    findings["naive_attempts"] = dict(spent)
    findings["naive_accepted"] = dict(won)

    step("3. failure modes")
    crashes = sum(1 for r in pool if r["status"] != "ok")
    print(f"  crashed/rejected programs in 10.03: {crashes}/{len(pool)}")
    final = naive["summary"]["final_heldout"]
    print(f"  naive final held-out: P {final['precision']:.3f}  R {final['recall']:.3f}  "
          f"(fp {final['fp']}, fn {final['fn']})")
    print(f"  per-label F1: {final['per_label_f1']}")
    findings["naive_final"] = final

    step("4. bottlenecks -> which method targets them")
    table = [
        ("Attempts pile onto one label while others never get a turn (ADDRESS F1 0 in naive)",
         "RSIFlow", "route work by expected gain x node success rate, not just 'worst label'"),
        ("Misses and false alarms need opposite fixes, but one prompt mixes them",
         "RSIFlow", "separate recall-hunter / precision-guard nodes; route by FN vs FP share"),
        ("The model always sees the same fixed failure sample from a fixed test set",
         "Self-HPO", "H-step: policy rewrites its own harness (sample size, grouping, focus mode)"),
        ("Rare formats (obfuscated emails, +91 phones) have few or no examples",
         "Self-HPO", "H-step: policy writes NEW labeled test cases for formats it suspects"),
    ]
    for bottleneck, method, fix in table:
        print(f"  [{method:8s}] {bottleneck}\n             -> {fix}")

    step("RSIFlow vs Self-HPO at a glance")
    print("  aspect        RSIFlow                          Self-HPO")
    print("  focus         workflow orchestration           harness + policy")
    print("  adapts        which node runs next             the environment the policy learns in")
    print("  iteration     continuous re-routing            alternating H-step / P-step")
    print("  prefer when   bottleneck moves between steps   feedback/tests are the bottleneck")
    save_json(OUT / "profile.json", {"findings": findings,
                                      "bottlenecks": [dict(zip(("bottleneck", "method", "fix"), t)) for t in table]})


if __name__ == "__main__":
    main()
