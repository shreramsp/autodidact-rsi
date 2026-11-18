"""Budget RSI's linear readout scorer (arXiv:2610.29121, laptop adaptation).

Idea: running every proposed edit through a full evaluation is expensive.
Instead, describe each edit with cheap numbers it already has *before*
running it, and fit a linear model that predicts how well it will do:

    predicted = w0 + w1*parent_rate + w2*lines_added + ... + w9*context_words_added

  parent_rate   how good the program being edited already is (steering F1)
  diff features 8 numbers computed from the code diff alone (see editing.py)

Training labels come from the small, cheap *steering* split. The large
*held-out* split is only used to check whether the chosen edit really is good.
"""
from __future__ import annotations

import numpy as np

from .editing import FEATURE_NAMES

READOUT_FEATURES = ["parent_rate"] + FEATURE_NAMES


class LinearReadout:
    def __init__(self, ridge: float = 1.0, features: list[str] | None = None):
        self.ridge = ridge
        self.features = features or READOUT_FEATURES
        self.mean = self.std = self.w = None
        self.b = 0.0

    def _X(self, rows: list[dict]) -> np.ndarray:
        return np.array([[float(r[f]) for f in self.features] for r in rows])

    def fit(self, rows: list[dict], target: str = "child_steering_f1") -> "LinearReadout":
        X = self._X(rows)
        y = np.array([float(r[target]) for r in rows])
        self.mean, self.std = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mean) / self.std
        self.b = float(y.mean())
        A = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ (y - self.b))
        return self

    def predict(self, rows: list[dict]) -> np.ndarray:
        Z = (self._X(rows) - self.mean) / self.std
        return Z @ self.w + self.b

    def coefficients(self) -> list[tuple[str, float]]:
        """Standardized weights: change in predicted F1 for a 1-std change in the feature."""
        return sorted(zip(self.features, self.w.tolist()), key=lambda t: -abs(t[1]))


# ------------------------------------------------ task-aware features (10.05) ---
#
# The paper's 8 diff features describe the SHAPE of an edit. For our task they
# barely predict quality (lab 10.04 shows why). Lab 10.05 adds two features that
# describe WHERE the edit lands, still without evaluating it on any data split:
#
#   smoke_ok             does the new program run on ONE fixed sentence? (1 eval)
#   targeted_error_mass  share of the parent's feedback-set mistakes that belong to
#                        the labels this diff touches (the parent's mistakes are
#                        already known: they were computed to write the prompt)

TASK_FEATURES = ["smoke_ok", "targeted_error_mass"]
SMOKE_TEXT = "Hi, I'm Ana Ruiz, call me at 408-555-0199 or ana.ruiz@gmail.com."


def task_features(parent_code: str, child_code: str, parent_feedback_report) -> dict:
    import difflib
    from . import LABELS
    from .sandbox import run_code
    diff = "\n".join(l for l in difflib.unified_diff(parent_code.splitlines(), child_code.splitlines(),
                                                     lineterm="", n=0) if l[:1] in "+-")
    rep = parent_feedback_report
    touched = [lab for lab in LABELS if lab in diff]
    total = (rep.fp + rep.fn) or 1
    mass = sum(rep.per_label[l]["fp"] + rep.per_label[l]["fn"] for l in touched) / total
    smoke = float(run_code(child_code, [SMOKE_TEXT]).status == "ok")
    return {"smoke_ok": smoke, "targeted_error_mass": mass}


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])
