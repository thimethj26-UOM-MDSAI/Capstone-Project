"""TF-IDF baselines (word + char n-grams) with LR / linear SVM. Selects on validation, reports test once."""
import json
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from .config import SEED, RESULTS_DIR
from .preprocess import clean


def metrics(y, p):
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, p), macro_f1=f, macro_precision=pr, macro_recall=rc,
                weighted_f1=f1_score(y, p, average="weighted"), off_f1=f1_score(y, p, pos_label=1))


def build_features(tr, va, te, keep_emoji=False):
    prep = lambda df: [clean(t, keep_emoji=keep_emoji) for t in df["text"]]
    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    f = lambda fit, X: hstack([(w.fit_transform if fit else w.transform)(X),
                               (c.fit_transform if fit else c.transform)(X)]).tocsr()
    return f(True, Xtr), f(False, Xva), f(False, Xte), (w, c)


def run_baselines(tr, va, te, keep_emoji=False, tag="baseline"):
    Xtr, Xva, Xte, _ = build_features(tr, va, te, keep_emoji)
    models = {
        "LR": lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
        "LinearSVM": lambda C: LinearSVC(C=C, class_weight="balanced", random_state=SEED),
    }
    out, best = {}, None
    for name, mk in models.items():
        for C in (0.1, 1, 10):
            m = mk(C).fit(Xtr, tr["y"])
            vf = metrics(va["y"], m.predict(Xva))["macro_f1"]
            if best is None or vf > best[0]:
                best = (vf, name, C, m)
            out[f"{name}_C{C}"] = dict(val_macro_f1=vf)
    vf, name, C, m = best  # test set touched once, with the validation-selected config
    out["selected"] = dict(model=name, C=C, val_macro_f1=vf, test=metrics(te["y"], m.predict(Xte)))
    (RESULTS_DIR / f"{tag}.json").write_text(json.dumps(out, indent=2))
    return out
