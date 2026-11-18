# Method 2 · Second-Order RSI: diagnose why before patching

Course labs 10.06–10.08 · paper: SO-RSI (arXiv:2610.05701)

## The idea

A **first-order** loop sees failures, patches them and repeats. A **second-order** loop also watches *its own history of attempts*. When that history looks wrong, it stops patching and asks why.

| Anomaly (plain-code detector) | Looks like | Probe asks |
|---|---|---|
| **recurrence** | the same target attempted 2+ times without gain | which assumption do the failed fixes share? |
| **opposing** | precision and recall swap direction on the same label | which two requirements conflict? |
| **mismatch** | target didn't improve, or another label dropped | is our model of the data, or the metric, wrong? |

A **diagnostic probe** is a separate Gemini call that writes no code. It returns `ROOT_CAUSE`, `RULE` and `NEXT_EDIT`, and the result is stored in **inquiry memory**, which every later edit prompt includes. The code is in [pii_bench/sorsi.py](../../pii_bench/sorsi.py).

## Lab 10.06: naive vs SO-RSI, four versions ([script](lab_10_06_naive_vs_sorsi.py))

Both loops start from the same program and get 10 edit attempts. The naive loop records anomalies but never acts on them.

| Version | What changed | Kept | Held-out F1 |
|---|---|---|---|
| naive | – | 4/10 | 0.741 |
| SO-RSI v1 | first design | 2/10 | 0.700 |
| SO-RSI v2 | fixed noisy detector; probes see must-keep examples; rules refuted after 2 failures | 2/10 | 0.700 |
| SO-RSI v3 | probe may also change the *workflow* (switch target / resample evidence / preserve examples) | 2/10 | 0.700 |
| **SO-RSI v4** | **plain-code escalation: 2 refuted rules on a label → park that label for 4 rounds** | **5/10** | **0.791** |

**Post-mortem (the most useful part of this method):**

- **v1.** The "opposing" detector also fired whenever a regex was *refined*, because a refinement looks like "an earlier regex was removed". It fired on 9 of 10 attempts. Meanwhile the probe found the real bug (`I'm <<talking about>>` captured as a name) but proposed the rule "names must be capitalized". That contradicts the data, which contains lowercase names like `hey its sarah chen`. Memory kept that rule, so SO-RSI spent 8 rounds on NAME.
- **v2.** False alarms on the naive loop dropped from 9 to 1, and bad rules were refuted. It was still stuck: the same 24 failure examples appeared every round, and the "talking about" fix trades +0.20 precision for −0.07 recall on NAME, so it nets −0.001 and is always rejected.
- **v3.** Offered workflow actions, but the small probe model answered `KEEP_GOING` every time.
- **v4.** Followed the course's own answer to lab 10.07 Q2: *"if enough inconclusive probes accumulate … try broader exploration."* Once NAME was parked, the loop fixed PHONE and then ADDRESS (+0.07 in one edit), which naive never reached.

All four versions replay from cache: `--version 1|2|3|4`. Chart: [v4/naive_vs_sorsi.png](../../runs/10_06_naive_vs_sorsi/v4/naive_vs_sorsi.png).

## Lab 10.07: what the probes found ([script](lab_10_07_probe_analysis.py), $0)

The script prints anomaly counts per version, then every probe with its root cause, rule, workflow action, what happened to the next edit, and whether the rule ended up *refuted* or *supported*.

The key observation: **every probe identified the right symptom**, both the "talking about" name and order numbers matched as phones. Diagnosis quality was not the bottleneck. Turning a diagnosis into a different *action* was, and that had to be enforced in code.

## Lab 10.08: one debugging session ([script](lab_10_08_debug_session.py))

The unit test has 8 real phones that must be caught and 6 look-alikes that must not be (order, part and tracking numbers, a price, a reference code).

| Step | Result |
|---|---|
| start | 5/8 phones, 3 false alarms |
| attempt 1 | program rejected by the sandbox (syntax error) |
| attempt 2 | **8/8 phones, 1 false alarm** → kept |
| attempts 3, 4 | the same idea twice, each losing a real phone → *recurrence + mismatch* |
| probe | root cause: the code looks back only a few characters for "order", which never reaches past `order number is` |
| guided fix | 0 false alarms, but it now misses `My number is 408-555-0199` |

The last row is over-correction: `My number is …` and `My order number is …` differ by one word. It also raises the third probe question, *is the metric misaligned?* F1 scores the leaked phone (a privacy failure) as roughly equal to the masked order number (an inconvenience).

## Takeaway

Second-order inquiry helped only once the agent could act on it structurally: refute rules, and park a label that keeps thrashing. With a small model, the diagnosis was the easy part; the escalation logic had to live in code.
