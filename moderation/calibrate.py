r"""Choose the two thresholds from VALIDATION data (not test) and save moderation/thresholds.json.
  remove = lowest threshold at which >= 90% of items scoring above it are truly offensive (precision)
  flag   = highest threshold at which >= 90% of all truly offensive items score at/above it (recall)
Run from repo root:  python -m moderation.calibrate --model "D:\path\to\xlmr" """
import argparse, json
import numpy as np
from src.data import load_sold
from src.preprocess import clean
from .scanner import TransformerScorer, THRESHOLD_FILE

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--precision", type=float, default=0.90)
ap.add_argument("--recall", type=float, default=0.90)
a = ap.parse_args()

_, va, _ = load_sold()
sc = TransformerScorer(a.model)
p = np.array(sc.probs([clean(t) for t in va.text]))
y = va.y.values
grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
rows = []
for t in grid:
    sel = p >= t
    prec = y[sel].mean() if sel.any() else float("nan")
    rows.append((float(t), float(prec), float((sel & (y == 1)).sum() / (y == 1).sum()), int(sel.sum())))
print("threshold  precision  recall  n_flagged")
for t, pr, rc, n in rows:
    print(f"   {t:.2f}      {pr:.3f}    {rc:.3f}    {n}")

ok_remove = [t for t, pr, rc, n in rows if n >= 20 and pr >= a.precision]
t_remove = min(ok_remove) if ok_remove else 0.8
ok_flag = [t for t, pr, rc, n in rows if rc >= a.recall]
t_flag = max(ok_flag) if ok_flag else 0.5
if t_flag >= t_remove:
    t_flag = round(max(0.05, t_remove - 0.1), 2)
src = (f"calibrated on validation set (n={len(y)}): remove>={t_remove} for precision>={a.precision}, "
       f"flag>={t_flag} for recall>={a.recall}, model={sc.name}")
THRESHOLD_FILE.write_text(json.dumps({"flag": t_flag, "remove": t_remove, "source": src, "table": rows}, indent=1))
print("\nSaved", THRESHOLD_FILE, "\n", src)
