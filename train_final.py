"""Train the scorer on ALL clean prompts and save it for the demo app.
Use this model for the demo only. Reported metrics must come from evaluate.py (cross-validated)."""
import argparse
from pathlib import Path

import pandas as pd

from firewall import HarmScorer

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data")
ap.add_argument("--out", default="models/scorer.joblib")
a = ap.parse_args()
P = pd.read_csv(Path(a.data_dir) / "prompts.csv")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
HarmScorer(True).fit(P.text, P.label).save(a.out)
print(f"trained on {len(P)} prompts -> {a.out}")
