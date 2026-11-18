"""Lab 10.13 - RSIFlow: a meta-agent that re-routes the workflow to the bottleneck.

Workflow nodes (each is one kind of edit request to Gemini):
  NAME, EMAIL, PHONE, CREDIT_CARD, SSN, ADDRESS   label specialists
  recall_hunter     only shown MISSES, asked to catch more without new false alarms
  precision_guard   only shown FALSE ALARMS, asked to remove them without new misses

Meta-agent (plain code, runs before every attempt):
  bottleneck(node) = share of current mistakes that node is responsible for
  success(node)    = (wins + 1) / (tries + 2)        - learned from its own history
  route to         argmax  bottleneck x success
  evolve           insert precision_guard when false alarms > 35% of mistakes,
                   insert recall_hunter   when misses       > 65% of mistakes,
                   cool a node down for 2 rounds after 2 failures in a row

Baseline: the fixed workflow from lab 10.06 ("always fix the worst label"),
same baseline program, same 10 attempts. It replays from cache for $0.

Run:  python labs/03_adaptive_workflows/lab_10_13_rsiflow.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pii_bench import LABELS
from pii_bench.editing import failure_digest, propose_edit
from pii_bench.labkit import BASELINE, RUNS, banner, data, load_json, run_dir, save_json, spend_line, step
from pii_bench.llm import Gemini
from pii_bench.sandbox import evaluate

ROUNDS = 10
OUT = run_dir("10_13_rsiflow")
NODE_PROMPTS = {
    "recall_hunter": "\nYou are the RECALL node: every example below is MISSED PII. Catch more of it "
                     "without creating new false alarms.\n",
    "precision_guard": "\nYou are the PRECISION node: every example below is a FALSE ALARM. Stop "
                       "redacting these without losing any real PII.\n",
}


class MetaAgent:
    def __init__(self):
        self.nodes = list(LABELS)                 # the workflow starts with specialists only
        self.stats = {}                           # node -> [tries, wins, fails_in_a_row]
        self.cooldown = {}
        self.events = []

    def bottleneck(self, node, rep):
        total = (rep.fp + rep.fn) or 1
        if node == "recall_hunter":
            return rep.fn / total
        if node == "precision_guard":
            return rep.fp / total
        d = rep.per_label[node]
        return (d["fp"] + d["fn"]) / total

    def evolve(self, r, rep):
        total = (rep.fp + rep.fn) or 1
        for node, share, cond in [("precision_guard", rep.fp / total, rep.fp / total > 0.35),
                                  ("recall_hunter", rep.fn / total, rep.fn / total > 0.65)]:
            if cond and node not in self.nodes:
                self.nodes.append(node)
                self.events.append(f"round {r}: insert {node} (share {share:.0%})")
                print(f"    EVOLVE: insert node {node} ({share:.0%} of mistakes)")

    def route(self, r, rep):
        self.evolve(r, rep)
        scored = []
        for n in self.nodes:
            if self.cooldown.get(n, 0) >= r:
                continue
            tries, wins, _ = self.stats.get(n, [0, 0, 0])
            scored.append((self.bottleneck(n, rep) * (wins + 1) / (tries + 2), n))
        scored.sort(reverse=True)
        top = ", ".join(f"{n} {s:.3f}" for s, n in scored[:4])
        return scored[0][1], top

    def update(self, r, node, won):
        s = self.stats.setdefault(node, [0, 0, 0])
        s[0] += 1
        s[1] += int(won)
        s[2] = 0 if won else s[2] + 1
        if s[2] >= 2:
            self.cooldown[node] = r + 2
            s[2] = 0
            self.events.append(f"round {r}: cool down {node} for 2 rounds")
            print(f"    EVOLVE: cool down {node} (2 failures in a row)")


def main():
    banner("Lab 10.13 - RSIFlow meta-agent vs a fixed workflow")
    feedback, steering, heldout = data()
    llm = Gemini(tag="10.13")
    agent = MetaAgent()
    code = BASELINE
    fb_rep, _ = evaluate(code, feedback)
    st = evaluate(code, steering, keep_failures=False)[0].f1
    curve = [{"round": 0, "heldout": evaluate(code, heldout, keep_failures=False)[0].f1}]
    trace = []
    for r in range(1, ROUNDS + 1):
        node, top = agent.route(r, fb_rep)
        if node in LABELS:
            e = propose_edit(llm, code, fb_rep, focus=node, variant=r)
        else:
            kind = "FN" if node == "recall_hunter" else "FP"
            e = propose_edit(llm, code, fb_rep, focus="GENERAL", variant=r, extra=NODE_PROMPTS[node],
                             digest=failure_digest(fb_rep, k=24, kind=kind, seed=r))
        new_fb, _ = evaluate(e.code, feedback)
        new_st = evaluate(e.code, steering, keep_failures=False)[0].f1
        won = new_st > st
        agent.update(r, node, won)
        print(f"  round {r:2d}  route -> {node:15s} (top: {top})  steering {st:.3f}->{new_st:.3f}  "
              f"{'ACCEPT' if won else 'reject'}")
        if won:
            code, fb_rep, st = e.code, new_fb, new_st
        trace.append({"round": r, "node": node, "candidates": top, "accepted": won, "steering": st})
        curve.append({"round": r, "heldout": evaluate(code, heldout, keep_failures=False)[0].f1})

    final = evaluate(code, heldout)[0]
    naive = load_json(RUNS / "10_06_naive_vs_sorsi" / "v4" / "results.json")["naive"]
    step("comparison (same baseline, 10 attempts each)")
    n = naive["summary"]
    print(f"  fixed workflow (10.06 naive): accepted {n['accepted']}/10  held-out F1 {n['final_heldout']['f1']:.3f}  "
          f"per label {n['final_heldout']['per_label_f1']}")
    print(f"  RSIFlow                     : accepted {sum(t['accepted'] for t in trace)}/10  held-out F1 {final.f1:.3f}  "
          f"per label {final.summary()['per_label_f1']}")
    print(f"  node usage: { {k: v[0] for k, v in agent.stats.items()} }")
    (OUT / "final_redactor.py").write_text(code)
    save_json(OUT / "results.json", {"trace": trace, "curve": curve, "events": agent.events,
                                      "node_stats": agent.stats, "final_heldout": final.summary(),
                                      "accepted": sum(t["accepted"] for t in trace)})
    print("\n" + spend_line(llm))


if __name__ == "__main__":
    main()
