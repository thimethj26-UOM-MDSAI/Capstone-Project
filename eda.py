"""EDA for SOLD -> results/eda/*.png + eda_summary.json. Run: python eda.py"""
import json, re
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import load_sold
from src.config import RESULTS_DIR
from src.preprocess import clean, URL, MENTION, EMOJI

out = RESULTS_DIR / "eda"; out.mkdir(parents=True, exist_ok=True)
tr, va, te = load_sold()
S = {}
for name, df in [("train", tr), ("val", va), ("test", te)]:
    S[name] = dict(n=len(df), offensive_share=round(df.y.mean(), 4),
                   url_rate=round(df.text.str.contains(URL).mean(), 4),
                   mention_rate=round(df.text.str.contains(MENTION).mean(), 4),
                   emoji_rate=round(df.text.str.contains(EMOJI).mean(), 4),
                   duplicate_texts=int(df.text.duplicated().sum()))
S["train_test_text_overlap"] = len(set(tr.text) & set(te.text))

# class balance
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(["NOT", "OFF"], [(tr.y == 0).sum(), (tr.y == 1).sum()], color=["#4C78A8", "#E45756"])
ax.set_title("Class balance (train)"); fig.tight_layout(); fig.savefig(out / "class_balance.png", dpi=150)

# length distributions (words) by class, after cleaning
tr["n_words"] = tr.text.map(lambda t: len(clean(t).split()))
fig, ax = plt.subplots(figsize=(5, 3))
for y, c, l in [(0, "#4C78A8", "NOT"), (1, "#E45756", "OFF")]:
    ax.hist(tr[tr.y == y].n_words, bins=30, alpha=.6, color=c, label=l)
ax.set_xlabel("words per tweet"); ax.legend(); fig.tight_layout(); fig.savefig(out / "length.png", dpi=150)
S["mean_words"] = {l: round(tr[tr.y == y].n_words.mean(), 2) for y, l in [(0, "NOT"), (1, "OFF")]}

# most class-associated words (smoothed log-odds)
import math
cnt = {y: Counter(w for t in tr[tr.y == y].text for w in set(clean(t).split())) for y in (0, 1)}
n = {y: (tr.y == y).sum() for y in (0, 1)}
vocab = [w for w in set(cnt[0]) | set(cnt[1]) if cnt[0][w] + cnt[1][w] >= 10]
lo = {w: math.log((cnt[1][w] + 1) / (n[1] + 2)) - math.log((cnt[0][w] + 1) / (n[0] + 2)) for w in vocab}
S["top_offensive_words"] = sorted(lo, key=lo.get, reverse=True)[:25]
S["top_clean_words"] = sorted(lo, key=lo.get)[:25]

# does the dataset carry token-level labels? (record column names for the explainability step)
S["columns"] = [c for c in tr.columns]
(out / "eda_summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=2))
print(json.dumps(S, ensure_ascii=False, indent=2))
