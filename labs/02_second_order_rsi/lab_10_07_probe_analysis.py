"""Lab 10.07 - Analyze the diagnostic probes.  (no API calls, $0)

Reads the saved traces of lab 10.06 (all four SO-RSI versions) and answers:
  1. How often did each anomaly type fire, and did the detector get noisier/cleaner?
  2. For every probe: what was diagnosed, what rule/action came out, and what
     happened to the next edit on that label?
  3. How often did several anomalies fire on the same attempt?

Run:  python labs/02_second_order_rsi/lab_10_07_probe_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import Counter

from pii_bench.labkit import RUNS, banner, load_json, run_dir, save_json, step

OUT = run_dir("10_07_probe_analysis")
WHY = {
    "recurrence": "stops repeating a fix that already failed; asks which assumption the fixes share",
    "opposing": "names the two requirements pulling in opposite directions (precision vs recall)",
    "mismatch": "checks whether our model of the data or the metric is wrong, not just the code",
}


def main():
    banner("Lab 10.07 - What did the diagnostic probes find?")
    versions = {}
    for v in (1, 2, 3, 4):
        path = RUNS / "10_06_naive_vs_sorsi" / f"v{v}" / "results.json"
        if path.exists():
            versions[v] = load_json(path)
    if not versions:
        sys.exit("run lab 10.06 first")

    step("1. anomaly counts per version (naive loop only RECORDS them, SO-RSI acts on them)")
    print(f"  {'version':8s} {'loop':7s} {'recurrence':>10s} {'opposing':>9s} {'mismatch':>9s} {'accepted':>9s} {'held-out F1':>12s}")
    rows = []
    for v, res in versions.items():
        for loop in ("naive", "SO-RSI"):
            s = res[loop]["summary"]
            a = s["anomalies"]
            rows.append({"version": v, "loop": loop, **{k: a.get(k, 0) for k in WHY},
                         "accepted": s["accepted"], "heldout_f1": s["final_heldout"]["f1"]})
            print(f"  v{v:<7d} {loop:7s} {a.get('recurrence', 0):10d} {a.get('opposing', 0):9d} {a.get('mismatch', 0):9d} "
                  f"{s['accepted']:>6d}/10 {s['final_heldout']['f1']:12.3f}")
    print("  v1 -> v2: 'opposing' on the naive loop drops 9 -> 1 once regex REFINEMENTS stop counting as undo.")

    latest = max(versions)
    res = versions[latest]["SO-RSI"]
    trace = res["trace"]
    step(f"2. every probe in SO-RSI v{latest}")
    table = []
    for p in res["probes"]:
        nxt = next((a for a in trace if a["round"] > p["round"] and a["target"] == p["target"]), None)
        mem = next((m for m in res["memory"] if m["round"] == p["round"]), {})
        row = {"round": p["round"], "anomaly": p["anomaly"], "label": p["target"],
               "root_cause": p["root_cause"], "rule": p["rule"], "action": p.get("action", ""),
               "next_edit_on_label": (f"round {nxt['round']}: " + ("ACCEPTED" if nxt["accepted"] else "rejected")) if nxt else "none",
               "rule_status": mem.get("status", "active"), "why_it_helps": WHY[p["anomaly"]]}
        table.append(row)
        print(f"\n  round {row['round']} | {row['anomaly'].upper()} on {row['label']} | action {row['action'] or '-'}")
        print(f"    root cause : {row['root_cause'][:200]}")
        print(f"    rule       : {row['rule'][:200]}")
        print(f"    next edit  : {row['next_edit_on_label']}   rule ended up: {row['rule_status'].upper()}")

    step("3. simultaneous anomalies")
    multi = Counter(len(a["anomalies"]) for a in trace)
    for n in sorted(multi):
        print(f"  {multi[n]} attempts had {n} anomal{'y' if n == 1 else 'ies'} at once")

    step("summary table (anomaly -> count -> typical recommendation -> why it helps)")
    counts = Counter(x for a in trace for x in a["anomalies"])
    for kind in WHY:
        rec = next((t["rule"][:90] for t in table if t["anomaly"] == kind), "(no probe run)")
        print(f"  {kind:10s} x{counts.get(kind, 0):<3d} rec: {rec}")
        print(f"  {'':10s}      why: {WHY[kind]}")

    print("\nKey observation: every probe found the RIGHT symptom ('talking about' captured as a name,\n"
          "order numbers matched as phones), but the cheap model always answered KEEP_GOING and\n"
          "proposed rules that broke other cases. Refuting rules (v2) and escalating to 'park the\n"
          "label' in plain code (v4) is what turned correct diagnoses into progress.")
    save_json(OUT / "probe_table.json", {"anomaly_counts": rows, "probes": table,
                                          "simultaneous": dict(multi)})


if __name__ == "__main__":
    main()
