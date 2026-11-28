"""Scoring: probability a post is offensive + per-word explanation (leave-one-word-out occlusion).
Real scorer = fine-tuned transformer. StubScorer exists ONLY for testing the plumbing without a model."""
import json
from pathlib import Path
from src.preprocess import clean

DEFAULT_THRESHOLDS = {"flag": 0.5, "remove": 0.8}
THRESHOLD_FILE = Path("moderation/thresholds.json")


def load_thresholds():
    if THRESHOLD_FILE.exists():
        d = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
        return {"flag": float(d["flag"]), "remove": float(d["remove"]),
                "source": d.get("source", "thresholds.json")}
    return {**DEFAULT_THRESHOLDS, "source": "defaults (run moderation/calibrate.py to calibrate on validation data)"}


class TransformerScorer:
    def __init__(self, model_path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
        self.name = Path(model_path).name

    def probs(self, texts, batch=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], return_tensors="pt", padding=True,
                               truncation=True, max_length=128)
                out += self.torch.softmax(self.model(**enc).logits, -1)[:, 1].tolist()
        return out

    def score(self, text):
        words = clean(text).split()[:60]
        if not words:
            return {"prob": 0.0, "words": []}
        variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        p = self.probs(variants)
        base = p[0]
        return {"prob": base, "words": [{"word": w, "impact": round(base - pv, 4)}
                                         for w, pv in zip(words, p[1:])]}


class StubScorer:
    """Keyword rule for testing only. NOT a model. Never use for the demo or any results."""
    name = "STUB-keyword-rule"
    HARD = ("මෝඩයෙක්", "බල්ලෙක්", "පිස්සුද", "පිළිකුලයි")
    SOFT = ("මෝඩ",)

    def score(self, text):
        words = clean(text).split()
        hard = [w for w in words if any(h in w for h in self.HARD)]
        soft = [w for w in words if any(s in w for s in self.SOFT)]
        prob = 0.93 if hard else 0.62 if soft else 0.04
        hit = set(hard or soft)
        return {"prob": prob, "words": [{"word": w, "impact": round(prob - 0.04, 4) if w in hit else 0.0}
                                         for w in words]}
