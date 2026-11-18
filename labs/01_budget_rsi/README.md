# Method 1 · Budget RSI: score edits before running them

Course labs 10.03–10.05 · paper: Budget RSI (arXiv:2610.29121)

## The idea

Each round, Gemini proposes several rewrites of the redactor. Fully evaluating every rewrite is the expensive step; in production it would mean people reviewing redacted tickets. Budget RSI avoids most of that cost:

1. Keep a small, cheap **steering split** (60 messages) next to the large **held-out split** (400).
2. Describe each proposed edit with numbers that are known *before running it*: how good the parent already is (`parent_rate`) plus 8 features of the code diff (lines added/deleted, regexes added/removed, functions added/removed, labels touched, keyword strings added).
3. Fit a **linear readout**, `predicted = w · features`, and fully evaluate only the edit it ranks highest.

## Lab 10.03: a pool of real edits ([script](lab_10_03_edit_pool.py))

The course demo uses 128 made-up edits. This lab collects **48 real Gemini edits**: 6 generations × 8 edits, each aimed at a different label, and scores all of them on both splits so the true answer is known. It then asks which child each strategy would have picked, training the readout leave-one-generation-out.

| Strategy | Held-out F1 of pick | Evals per decision |
|---|---|---|
| oracle (fully evaluate all 8) | 0.815 | 3,200 |
| steering-only (all 8 on the small split) | 0.814 | 880 |
| budget readout (paper features) | 0.706 | 400 |
| random | 0.730 | 400 |

- The search itself works: **0.271 → 0.912 held-out F1 in 6 generations**.
- The steering split is a near-perfect proxy for the held-out split (right panel of [selection.png](../../runs/10_03_edit_pool/selection.png)).
- The paper's readout does **worse than random** on this task, with rank correlation −0.31.

## Lab 10.04: why the readout failed ([script](lab_10_04_readout_analysis.py), $0)

- **Coefficients.** `parent_rate` dominates (+0.22), and every diff feature is close to zero.
- **Confounding.** Early edits were large (17.9 lines on average) because the program was weak; late edits were small (6.7 lines). Lines added correlates **−0.49 with the absolute score** but **+0.34 with the gain over the parent**. A model of the absolute score learns the wrong sign.
- **8 vs 4 features.** Both are near random (0.706 / 0.723 vs 0.727 for `parent_rate` only, which ties for all siblings and so amounts to a random pick).
- **More training data made the readout worse** (0.729 with 25% of the data, 0.706 with 100%), because more data means more confounded examples.

## Lab 10.05: fix the features, then run it live ([script](lab_10_05_budget_loop.py))

**Part A ($0).** Predict the **gain** instead of the absolute score, and add two task-aware features that still need no evaluation on any split:
- `smoke_ok`: does the new program run on one fixed sentence?
- `targeted_error_mass`: what share of the parent's known mistakes belong to the labels this diff touches?

| Readout | Pick F1 | Rank corr |
|---|---|---|
| paper features, absolute | 0.706 | −0.31 |
| paper features, gain | 0.718 | −0.26 |
| **task features, gain** | **0.783** | **+0.28** |
| random | 0.730 | |

**Part B (live).** 6 generations × 4 edits. The readout picks one edit and only that one is evaluated. All 4 are also scored in the background ("shadow evaluation") to grade the pick.
- **Result: 0.271 → 0.817 held-out F1 using 384 selection evals**; the full search had used 480 to reach 0.658 ([budget_loop.png](../../runs/10_05_budget_loop/budget_loop.png)).
- **Honest grade:** the readout matched the best child in 2 of 6 generations, with a mean regret of 0.019 F1. It under-rates NAME edits.
- **A bug worth knowing about:** the first version of this loop got stuck. A rejected generation re-sent the identical prompt and got the identical cached edit back three times in a row. That is the *recurrence* anomaly that Method 2 is built to detect. The fix asks for fresh samples after each rejection.

## Takeaway

The mechanism transfers; the features don't. A linear readout can save about 90% of evaluation cost, but only when its features describe *where* an edit lands for this task, and only when it predicts gain rather than absolute score.
