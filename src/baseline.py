"""TF-IDF baselines (word + char n-grams) with LR / linear SVM. Selects on validation, reports test once.
Also saves test predictions (for McNemar) and deployment cost (size, CPU latency) like the transformers."""
import json, os, tempfile, time
import numpy as np
import joblib
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


def deployment_cost(w, c, model, texts, keep_emoji=False, n=100):
    """Disk size of the whole pipeline (vectorizers + classifier) and CPU latency per tweet (batch=1),
    measured the same way as the transformers: cleaning + features + prediction, one tweet at a time."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pipeline.joblib")
        joblib.dump((w, c, model), path)
        size_mb = os.path.getsize(path) / 1e6
    sample = list(texts[:n])
    t0 = time.time()
    for s in sample:
        x = clean(s, keep_emoji=keep_emoji)
        model.predict(hstack([w.transform([x]), c.transform([x])]).tocsr())
    return dict(disk_mb=round(size_mb, 1), cpu_latency_ms=round((time.time() - t0) / len(sample) * 1000, 2))


def run_baselines(tr, va, te, keep_emoji=False, tag="baseline"):
    t_start = time.time()
    Xtr, Xva, Xte, (w, c) = build_features(tr, va, te, keep_emoji)
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
    test_pred = m.predict(Xte)
    out["selected"] = dict(model=name, C=C, val_macro_f1=vf, test=metrics(te["y"], test_pred),
                           cost=deployment_cost(w, c, m, list(te["text"]), keep_emoji),
                           total_seconds=round(time.time() - t_start))
    (RESULTS_DIR / f"{tag}.json").write_text(json.dumps(out, indent=2))
    np.save(RESULTS_DIR / f"{tag}_test_preds.npy", np.asarray(test_pred))  # same row order as the test set; for McNemar
    return out
