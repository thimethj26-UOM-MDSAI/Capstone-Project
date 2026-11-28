r"""Policy analysis for the 3-way moderation policy (remove / flag for review / publish).

Run from the repo root (needs the fine-tuned model the FIRST time only; scores are cached afterwards):
    python policy_analysis.py --model "D:\UoM MDSAI\COURSE DATA\S3\Capstone\BaseCode\xlmr"

What it does
  1. Scores validation + test tweets with the model (cached in results/policy_probs_<name>_<split>.npy).
  2. VALIDATION ONLY: sensitivity table - how the thresholds and the human review workload change when the
     precision target for auto-removal is 0.85 / 0.90 / 0.95 (same rule as moderation/calibrate.py).
  3. TEST SET, ONE TIME: evaluates the thresholds currently saved in moderation/thresholds.json
     (true label x decision table, false-removal rate, share of offensive posts caught, review workload).
     Thresholds are never tuned on the test set. If results/policy_test_eval.json already exists the test section is
     skipped unless you pass --force (so the final numbers can't be quietly re-picked).
  4. Figures: precision-recall curve with the operating points, test decision breakdown, reliability diagram.
Outputs: results/policy_analysis.md, results/policy_analysis.json, results/policy_*.png
"""
import argparse, json, sys, datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

RES = Path("results")
THRESH_FILE = Path("moderation/thresholds.json")
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)       # identical to moderation/calibrate.py
TARGETS = [0.85, 0.90, 0.95]
RECALL_TARGET = 0.90

# colours: validated categorical slots 1-3 of the reference palette; text in neutral inks
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, INK2, GRIDC = "#191919", "#666666", "#e6e6e6"


# ---------------------------------------------------------------- data
def get_labels(split):
    f = RES / f"{split}_labels.npy"
    if not f.exists():
        from src.data import load_sold
        _, va, te = load_sold()
        np.save(RES / "val_labels.npy", va.y.values)
        np.save(RES / "test_labels.npy", te.y.values)
    return np.load(f)


def get_probs(name, model_path, rescore):
    files = {s: RES / f"policy_probs_{name}_{s}.npy" for s in ("val", "test")}
    if rescore or not all(f.exists() for f in files.values()):
        if not model_path:
            sys.exit("Scores are not cached yet: run with --model \"path\\to\\model folder\".")
        from src.data import load_sold
        from src.preprocess import clean
        from moderation.scanner import TransformerScorer
        _, va, te = load_sold()
        sc = TransformerScorer(model_path)
        for split, df in (("val", va), ("test", te)):
            print(f"Scoring {split} ({len(df)} tweets) ...")
            np.save(files[split], np.array(sc.probs([clean(t) for t in df.text])))
        np.save(RES / "val_labels.npy", va.y.values)
        np.save(RES / "test_labels.npy", te.y.values)
    return {s: np.load(f) for s, f in files.items()}


# ---------------------------------------------------------------- policy maths
def grid_rows(p, y):
    rows = []
    for t in GRID:
        sel = p >= t
        prec = float(y[sel].mean()) if sel.any() else float("nan")
        rows.append((float(t), prec, float((sel & (y == 1)).sum() / (y == 1).sum()), int(sel.sum())))
    return rows


def pick_thresholds(rows, target_p, target_r=RECALL_TARGET):
    """Same rule as moderation/calibrate.py. Returns (flag, remove); remove=None if no threshold reaches the target."""
    ok_r = [t for t, pr, rc, n in rows if n >= 20 and pr >= target_p]
    remove = min(ok_r) if ok_r else None
    ok_f = [t for t, pr, rc, n in rows if rc >= target_r]
    flag = max(ok_f) if ok_f else 0.5
    if remove is not None and flag >= remove:
        flag = round(max(0.05, remove - 0.1), 2)
    return float(flag), (float(remove) if remove is not None else None)


def decisions(p, flag, remove):
    d = np.zeros(len(p), dtype=int)                    # 0 published, 1 flagged, 2 removed
    d[p >= flag] = 1
    if remove is not None:
        d[p >= remove] = 2
    return d


def outcome(p, y, flag, remove):
    d = decisions(p, flag, remove)
    n, off, notoff = len(y), int((y == 1).sum()), int((y == 0).sum())
    rem, fl, pub = d == 2, d == 1, d == 0
    safe = lambda a, b: float(a / b) if b else None
    return dict(
        flag=flag, remove=remove, n=n,
        removed=int(rem.sum()), flagged=int(fl.sum()), published=int(pub.sum()),
        share_removed=safe(rem.sum(), n), share_flagged=safe(fl.sum(), n), share_published=safe(pub.sum(), n),
        removal_precision=safe(y[rem].sum(), rem.sum()),
        false_removals=int(((y == 0) & rem).sum()), false_removal_rate=safe(((y == 0) & rem).sum(), notoff),
        off_total=off, off_auto_removed=int(((y == 1) & rem).sum()), off_in_review=int(((y == 1) & fl).sum()),
        off_published_missed=int(((y == 1) & pub).sum()),
        auto_removed_rate=safe(((y == 1) & rem).sum(), off), caught_rate=safe(((y == 1) & (rem | fl)).sum(), off),
        missed_rate=safe(((y == 1) & pub).sum(), off),
        review_queue_offensive_share=safe(y[fl].sum(), fl.sum()),
        notoff_sent_to_review=int(((y == 0) & fl).sum()), notoff_sent_to_review_rate=safe(((y == 0) & fl).sum(), notoff),
        notoff_published_rate=safe(((y == 0) & pub).sum(), notoff),
        confusion={"NOT": [int(((y == 0) & rem).sum()), int(((y == 0) & fl).sum()), int(((y == 0) & pub).sum())],
                   "OFF": [int(((y == 1) & rem).sum()), int(((y == 1) & fl).sum()), int(((y == 1) & pub).sum())]})


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, e = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(dict(bin=[float(edges[b]), float(edges[b + 1])], n=int(m.sum()),
                             mean_prob=float(p[m].mean()), frac_offensive=float(y[m].mean())))
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e), rows


# ---------------------------------------------------------------- plots
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRIDC, linewidth=0.8); ax.set_axisbelow(True)


def plot_pr(pv, yv, flag, remove, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pr, rc, _ = precision_recall_curve(yv, pv)
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=200)
    ax.plot(rc, pr, color=BLUE, linewidth=2)
    for (t, lab), ty in (((remove, "auto-remove"), 0.62), ((flag, "flag for review"), 0.50)):
        if t is None:
            continue
        sel = pv >= t
        pt = (float((sel & (yv == 1)).sum() / (yv == 1).sum()), float(yv[sel].mean()) if sel.any() else 0)
        ax.scatter(*pt, s=60, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(f"{lab}: p >= {t:.2f}\nprecision {pt[1]:.2f}, recall {pt[0]:.2f}", pt, xytext=(0.04, ty),
                    fontsize=8.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.8, shrinkA=2, shrinkB=4))
    ax.set_xlim(0, 1.0); ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("Recall (share of offensive posts caught)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Precision (share of actioned posts that are offensive)", color=INK2, fontsize=9.5)
    ax.set_title("Validation set: precision-recall curve with the two operating points", loc="left", color=INK, fontsize=10.5)
    style(ax); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_decisions(conf, path, flag, remove):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Auto-removed", "Sent to review", "Published"]; colors = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    for yi, (row, name) in enumerate((("NOT", "Truly NOT offensive"), ("OFF", "Truly offensive"))):
        counts = conf[row]; tot = sum(counts); left = 0
        for k, c in enumerate(counts):
            w = c / tot if tot else 0
            ax.barh(yi, w, left=left, color=colors[k], edgecolor="white", linewidth=2, height=0.55, label=labels[k] if yi == 0 else None)
            if w >= 0.07:
                ax.text(left + w / 2, yi, f"{c}\n{w * 100:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if k == 0 else INK, fontweight="bold" if k == 0 else None)
            elif c > 0:
                ax.text(left + w / 2, yi - 0.36, f"{c} ({w * 100:.1f}%)", ha="center", va="bottom", fontsize=8, color=INK)
            left += w
        ax.text(-0.01, yi, f"{name}\n(n={tot})", ha="right", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 1.6); ax.invert_yaxis(); ax.set_yticks([])
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb"); ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(ncol=3, frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(f"Test set: what the policy did (flag >= {flag:.2f}, remove >= {remove:.2f})", loc="left", color=INK, fontsize=10.5, pad=26)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.72, bottom=0.14); fig.savefig(path); plt.close(fig)


def plot_reliability(rows, e, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=200)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=1.2, linestyle="--")
    ax.text(0.62, 0.5, "perfectly calibrated", color=INK2, fontsize=8, rotation=38)
    xs = [r["mean_prob"] for r in rows]; ys = [r["frac_offensive"] for r in rows]
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    for r in rows:
        ax.annotate(str(r["n"]), (r["mean_prob"], r["frac_offensive"]), xytext=(-6, 7), textcoords="offset points", ha="right", fontsize=7, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Predicted probability (mean per bin)", color=INK2, fontsize=9.5)
    ax.set_ylabel("Observed share offensive", color=INK2, fontsize=9.5)
    ax.set_title(f"Test set: reliability (ECE = {e:.3f}); numbers = posts per bin", loc="left", color=INK, fontsize=9.5)
    style(ax); fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------- main
def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path to the fine-tuned model folder (only needed the first time)")
    ap.add_argument("--name", default="xlmr", help="short name used in cache file names")
    ap.add_argument("--rescore", action="store_true", help="ignore cached scores and score again")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results/policy_test_eval.json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    RES.mkdir(exist_ok=True)
    if not THRESH_FILE.exists():
        sys.exit("moderation/thresholds.json not found. Run: python -m moderation.calibrate --model ...")
    th = json.loads(THRESH_FILE.read_text(encoding="utf-8"))
    flag, remove = float(th["flag"]), float(th["remove"])

    P = get_probs(a.name, a.model, a.rescore)
    yv, yt = get_labels("val"), get_labels("test")
    pv, pt = P["val"], P["test"]
    if len(pv) != len(yv) or len(pt) != len(yt):
        sys.exit("Score/label length mismatch: run again with --rescore.")

    out = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), model=a.name,
               chosen_thresholds=dict(flag=flag, remove=remove, source=th.get("source")), validation={}, test=None)
    md = []

    # --- 1. sensitivity on VALIDATION --------------------------------------------------------------------
    rows = grid_rows(pv, yv)
    sens = []
    for tp in TARGETS:
        f, r = pick_thresholds(rows, tp)
        o = outcome(pv, yv, f, r); o["precision_target"] = tp
        o["is_chosen"] = bool(r is not None and abs(f - flag) < 1e-9 and abs(r - remove) < 1e-9)
        sens.append(o)
    out["validation"]["sensitivity"] = sens
    md += ["## 1. Sensitivity of the policy to the precision target (validation set, n=%d)" % len(yv), "",
           f"Auto-remove threshold = lowest grid threshold whose precision >= target; flag threshold = highest threshold "
           f"with recall >= {RECALL_TARGET:.2f}. Validation data only.", "",
           "| Precision target | Flag >= | Remove >= | Auto-removed | Sent to review | Published | Removal precision | False removals (of NOT) | Offensive posts caught (removed + review) | Offensive posts missed |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for o in sens:
        star = " (chosen)" if o["is_chosen"] else ""
        rm = "none reaches target" if o["remove"] is None else f"{o['remove']:.2f}"
        md.append(f"| {o['precision_target']:.2f}{star} | {o['flag']:.2f} | {rm} | {pct(o['share_removed'])} | {pct(o['share_flagged'])} | "
                  f"{pct(o['share_published'])} | {pct(o['removal_precision'])} | {o['false_removals']} ({pct(o['false_removal_rate'])}) | "
                  f"{pct(o['caught_rate'])} | {pct(o['missed_rate'])} |")
    md += ["", "Shares of auto-removed / review / published are fractions of ALL validation posts, so the review column is the human workload.", ""]

    # --- 2. one-time TEST evaluation ---------------------------------------------------------------------
    done = RES / "policy_test_eval.json"
    if done.exists() and not a.force:
        print(f"NOTE: {done} already exists, so the test-set section was NOT recomputed (the thresholds must not be re-picked "
              "after seeing test results). Use --force only for a deliberate re-run.")
        out["test"] = json.loads(done.read_text(encoding="utf-8")).get("test")
    else:
        t = outcome(pt, yt, flag, remove)
        mf1_05 = float(f1_score(yt, (pt >= 0.5).astype(int), average="macro"))
        e, rel_rows = ece(pt, yt)
        ref = None
        if (RES / f"{a.name}.json").exists():
            ref = json.loads((RES / f"{a.name}.json").read_text(encoding="utf-8"))["test"]["macro_f1"]
        t.update(macro_f1_at_0_5=mf1_05, training_run_macro_f1=ref, ece=e, reliability=rel_rows)
        out["test"] = t
    t = out["test"]
    if t:
        c = t["confusion"]
        md += ["## 2. Test-set evaluation of the saved thresholds (evaluated once; n=%d)" % t["n"], "",
               f"Thresholds from moderation/thresholds.json: flag >= {t['flag']:.2f}, remove >= {t['remove']:.2f}. "
               "They were chosen on validation data only.", "",
               "| True label | Auto-removed | Sent to review | Published | Total |", "|---|---|---|---|---|"]
        for lab, name in (("NOT", "NOT offensive"), ("OFF", "Offensive")):
            tot = sum(c[lab])
            md.append(f"| {name} | " + " | ".join(f"{v} ({v / tot * 100:.1f}%)" for v in c[lab]) + f" | {tot} |")
        md += ["", "| Measure | Value |", "|---|---|",
               f"| Removal precision (auto-removed posts that are truly offensive) | {pct(t['removal_precision'])} |",
               f"| Wrongly auto-removed posts (NOT offensive) | {t['false_removals']} ({pct(t['false_removal_rate'])} of NOT posts) |",
               f"| Offensive posts auto-removed | {t['off_auto_removed']} ({pct(t['auto_removed_rate'])}) |",
               f"| Offensive posts sent to human review | {t['off_in_review']} |",
               f"| Offensive posts caught in total (removed + review) | {pct(t['caught_rate'])} |",
               f"| Offensive posts published (missed) | {t['off_published_missed']} ({pct(t['missed_rate'])}) |",
               f"| Review queue: share that is truly offensive | {pct(t['review_queue_offensive_share'])} |",
               f"| NOT-offensive posts sent to review | {t['notoff_sent_to_review']} ({pct(t['notoff_sent_to_review_rate'])} of NOT posts) |",
               f"| Human review workload (share of all posts) | {pct(t['share_flagged'])} |",
               f"| Expected calibration error (ECE) | {t['ece']:.3f} |", ""]
        ref = t.get("training_run_macro_f1")
        chk = f"Sanity check: macro-F1 at the usual 0.5 cut-off from these scores = {t['macro_f1_at_0_5']:.4f}"
        if ref is not None:
            chk += f" (training run reported {ref:.4f}; small differences are normal because training used GPU fp16 and this uses CPU fp32)"
            if abs(ref - t["macro_f1_at_0_5"]) > 0.01:
                chk += "  ** LARGE DIFFERENCE: check that --model points at the same model **"
        md += [chk + ".", ""]
        plot_decisions(c, RES / "policy_test_decisions.png", t["flag"], t["remove"])
        plot_reliability(t["reliability"], t["ece"], RES / "policy_reliability.png")
    plot_pr(pv, yv, flag, remove, RES / "policy_pr_curve.png")

    (RES / "policy_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RES / "policy_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if out["test"] and (not done.exists() or a.force):
        done.write_text(json.dumps({"created": out["created"], "thresholds": out["chosen_thresholds"], "test": out["test"]}, indent=2),
                        encoding="utf-8")
    print("\n".join(md))
    print("Saved results/policy_analysis.md/.json and results/policy_pr_curve.png, policy_test_decisions.png, policy_reliability.png")


if __name__ == "__main__":
    main()