# Method 3 · Adaptive workflows: RSIFlow and Self-HPO

Course labs 10.12–10.14 · papers: RSIFlow (arXiv:2610.21791), Self-HPO (arXiv:2610.22791)

Both methods improve the *process* rather than a single edit.

| | RSIFlow | Self-HPO |
|---|---|---|
| Adapts | which workflow node runs next | the harness the policy learns in |
| Mechanism | a meta-agent routes to the bottleneck and evolves the node set | alternating H-step (harness) and P-step (policy) |
| Best when | the bottleneck moves between steps | feedback and tests are the bottleneck |

Both are compared against the fixed workflow from lab 10.06: same starting program, 10 edit attempts, **0.741** held-out F1.

## Lab 10.12: profile the bottlenecks ([script](lab_10_12_workflow_profile.py), $0)

The script reads the earlier runs to find out where the fixed workflow wasted effort:
- It spent **all 10 attempts on NAME and PHONE**. ADDRESS never got a turn and ended at **0.0 F1**.
- In lab 10.03, edits focused on NAME gained the most on average (+0.095) and edits focused on ADDRESS lost the most (−0.040).
- Misses and false alarms need opposite fixes, but a single prompt mixes them together.

It then maps each bottleneck to the method that targets it.

## Lab 10.13: RSIFlow meta-agent ([script](lab_10_13_rsiflow.py))

**Nodes.** Six label specialists, plus two nodes the meta-agent can insert:
- **recall_hunter**: shown only *missed* PII
- **precision_guard**: shown only *false alarms*

**Meta-agent** (plain code, runs before every attempt):
- routes to `argmax bottleneck(node) × success(node)`, where success is `(wins+1)/(tries+2)` learned from the node's own history
- inserts precision_guard when false alarms exceed 35% of mistakes, and recall_hunter when misses exceed 65%
- cools a node down after 2 failures in a row

| | Kept | Held-out F1 | Lowest label |
|---|---|---|---|
| fixed workflow | 4/10 | 0.741 | ADDRESS 0.0 |
| **RSIFlow** | **8/10** | **0.944** | NAME 0.914 |

In round 1 the meta-agent inserted recall_hunter (83% of mistakes were misses), and in round 4 it inserted precision_guard (false alarms had grown to 56%). It then alternated as the bottleneck shifted.

**Caveat:** it never used a label specialist. Most of the gain probably comes from splitting misses and false alarms into separate requests, not from the routing alone. A fixed recall/precision alternation is the ablation to run next.

## Lab 10.14: Self-HPO ([script](lab_10_14_self_hpo.py))

- **Policy** = the redactor code.
- **Harness** = what the policy sees while improving:
  - `digest_size`: how many failures are shown
  - `group_by`: none or by label
  - `focus_mode`: worst label / rotate / misses first / false alarms first
  - **self-written test messages**
- **H-step:** the same model reads how the last P-steps went and returns a new harness as JSON. Self-written tests are kept only if every PII value really occurs in the text.
- **P-step:** one ordinary edit under the current harness.

The schedule is H, P, P, P, H, P, P, P, H, P, P, P, P (10 P-steps). Edits are still **accepted on the fixed steering split**, so a harness can change what the policy *sees* but can never move the goalposts.

| | Kept | Held-out F1 |
|---|---|---|
| fixed harness | 4/10 | 0.741 |
| **Self-HPO** | **7/10** | **0.863** |

Harness changes the model chose: show 24 → 28 → 32 failures, group them by label, and add 8 self-written tests with trap order and part numbers. The tests are saved in [self_written_tests.jsonl](../../runs/10_14_self_hpo/self_written_tests.jsonl).

**Caveat:** validation checks that a PII value is present, not that its label is right; one test labels `support@neovibe.org`, a company address, as personal EMAIL.

## Takeaway

The biggest gains in the whole project came from changing the **process**: separating recall work from precision work (RSIFlow), and letting the model shape its own feedback (Self-HPO). Both beat every single-loop variant from Methods 1 and 2 under the same 10-attempt budget.
