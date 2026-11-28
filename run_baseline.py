import json
from src.data import load_sold
from src.baseline import run_baselines

tr, va, te = load_sold()
print(len(tr), len(va), len(te), "| offensive share train:", round(tr.y.mean(), 3))
sel = run_baselines(tr, va, te)["selected"]
print(json.dumps(sel, indent=2))
print("Saved results/baseline.json and results/baseline_test_preds.npy")
