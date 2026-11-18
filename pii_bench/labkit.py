"""Small helpers shared by the lab scripts (printing, saving, loading)."""
from __future__ import annotations

import json
from pathlib import Path

from . import generator
from .llm import total_spent, BUDGET_USD

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
BASELINE = (Path(__file__).resolve().parent / "baseline_redactor.py").read_text()


def banner(title: str):
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


def step(msg: str):
    print(f"\n--- {msg}")


def data():
    """(feedback, steering, heldout) message lists."""
    return generator.load("feedback"), generator.load("steering"), generator.load("heldout")


def run_dir(name: str) -> Path:
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=1))
    print(f"saved {path.relative_to(ROOT)}")


def load_json(path: Path):
    return json.loads(path.read_text())


def spend_line(llm) -> str:
    return f"{llm.meter()}  |  total spent all runs: ${total_spent():.4f} of ${BUDGET_USD:.2f} cap"
