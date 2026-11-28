"""Significance + multi-seed summary from saved test predictions (CPU only, takes seconds).

Reads  results/<name>_test_preds.npy   e.g. baseline, xlmr, sinbert (seed 42)  and  xlmr_s1, xlmr_s2, sinbert_s1 ... (extra seeds)
Writes results/significance.json and results/significance.md (tables you can paste into the report).

  python significance.py

What it reports
  1. Per-model macro-F1 for every seed, with mean +- std (sample std) across seeds.
  2. Paired comparisons on the seed-42 runs: exact McNemar test (+ Holm correction for the number of comparisons)
     and a paired bootstrap 95% CI for the macro-F1 difference.
"""
import itertools, json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

RES = Path("results")
B, RNG_SEED = 10_000, 42


def load_labels():
    f = RES / "test_labels.npy"
    if f.exists():
        return np.load(f)
    from src.data import load_sold            # needs the dataset (cached after the first download)
    y = load_sold()[2]["y"].values
    np.save(f, y)
    return y


def macro_f1_fast(y, p):
    """Macro-F1 for binary labels from confusion counts (used inside the bootstrap)."""
    tp1 = ((p == 1) & (y == 1)).sum(-1); fp1 = ((p == 1) & (y == 0)).sum(-1); fn1 = ((p == 0) & (y == 1)).sum(-1)
    tp0 = ((p == 0) & (y == 0)).sum(-1); fp0 = ((p == 0) & (y == 1)).sum(-1); fn0 = ((p == 1) & (y == 0)).sum(-1)
    f = lambda tp, fp, fn: np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return (f(tp1, fp1, fn1) + f(tp0, fp0, fn0)) / 2


def mcnemar_exact(y, pa, pb):
    ca, cb = pa == y, pb == y
    b = int((ca & ~cb).sum())   # A right, B wrong
    c = int((~ca & cb).sum())   # A wrong, B right
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, p


def bootstrap_diff(y, pa, pb, rng):
    n = len(y); diffs = []
    for _ in range(B // 1000):
        idx = rng.integers(0, n, size=(1000, n))
        diffs.append(macro_f1_fast(y[idx], pa[idx]) - macro_f1_fast(y[idx], pb[idx]))
    d = np.concatenate(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    files = sorted(RES.glob("*_test_preds.npy"))
    if not files:
        sys.exit("No results/*_test_preds.npy files found. Run run_baseline.py / finetune.py first.")
    y = load_labels()
    fam = {}   # family -> {seed_label: preds}
    for f in files:
        stem = f.name[:-len("_test_preds.npy")]
        m = re.fullmatch(r"(.+?)_s(\d+)", stem)
        family, seed = (m.group(1), int(m.group(2))) if m else (stem, 42)
        p = np.load(f)
        if len(p) != len(y):
            sys.exit(f"{f.name} has {len(p)} predictions but the test set has {len(y)} rows.")
        fam.setdefault(family, {})[seed] = p

    out = {"n_test": int(len(y)), "bootstrap_resamples": B, "per_model": {}, "pairwise_seed42": []}
    md = ["## Macro-F1 across seeds (test)\n", "| Model | Seeds | Macro-F1 per seed | Mean ± std |", "|---|---|---|---|"]
    for family, runs in fam.items():
        seeds = sorted(runs)
        f1s = [f1_score(y, runs[s], average="macro") for s in seeds]
        mean = float(np.mean(f1s)); std = float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None
        out["per_model"][family] = dict(seeds=seeds, macro_f1=f1s, mean=mean, std=std)
        per = ", ".join(f"{s}: {v:.4f}" for s, v in zip(seeds, f1s))
        md.append(f"| {family} | {len(seeds)} | {per} | {mean:.4f}" + (f" ± {std:.4f}" if std is not None else " (single run)") + " |")

    main_preds = {k: v[42] for k, v in fam.items() if 42 in v}
    names = list(main_preds)
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for a, b in itertools.combinations(names, 2):
        pa, pb = main_preds[a], main_preds[b]
        n_a_only, n_b_only, p = mcnemar_exact(y, pa, pb)
        lo, hi = bootstrap_diff(y, pa, pb, rng)
        diff = f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")
        rows.append(dict(a=a, b=b, f1_diff=float(diff), a_right_b_wrong=n_a_only, b_right_a_wrong=n_b_only,
                         mcnemar_p=float(p), ci95=[lo, hi]))
    if rows:
        adj = holm([r["mcnemar_p"] for r in rows])
        for r, q in zip(rows, adj):
            r["mcnemar_p_holm"] = q
            r["significant_at_0.05"] = bool(q < 0.05)
    out["pairwise_seed42"] = rows

    md += ["", "## Pairwise comparison, seed-42 runs (A − B)\n",
           "| A | B | ΔMacro-F1 | A right / B wrong | B right / A wrong | McNemar p | Holm-adj. p | 95% bootstrap CI of Δ |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['a']} | {r['b']} | {r['f1_diff']:+.4f} | {r['a_right_b_wrong']} | {r['b_right_a_wrong']} | "
                  f"{r['mcnemar_p']:.4f} | {r['mcnemar_p_holm']:.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] |")
    md += ["", "Reading guide: a difference is only claimed if Holm-adjusted p < 0.05 AND the bootstrap interval excludes 0. "
           "Otherwise write \"comparable\"."]

    (RES / "significance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RES / "significance.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nSaved results/significance.json and results/significance.md")


if __name__ == "__main__":
    main()