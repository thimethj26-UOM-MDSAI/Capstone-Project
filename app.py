"""Demo: paste a Sinhala post -> offensive / not + words that drove the decision.
Explanation = leave-one-word-out occlusion (local, no external API). Run: python app.py --model models/xlmr"""
import argparse, torch, gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocess import clean

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="models/xlmr"); a, _ = ap.parse_known_args()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model).eval()

@torch.no_grad()
def p_off(texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return torch.softmax(model(**enc).logits, -1)[:, 1].tolist()

def explain(text):
    words = clean(text).split()
    if not words: return "Enter some text.", []
    base = p_off([" ".join(words)])[0]
    drops = [base - p for p in p_off([" ".join(words[:i] + words[i + 1:]) for i in range(len(words))])]
    hl = [(w, round(d, 3) if d > 0.02 else None) for w, d in zip(words, drops)]  # only words that raise P(offensive)
    label = "OFFENSIVE" if base >= 0.5 else "NOT OFFENSIVE"
    return f"{label}  (P offensive = {base:.2f})", hl

gr.Interface(explain, gr.Textbox(lines=3, label="Sinhala post"),
             [gr.Textbox(label="Verdict"), gr.HighlightedText(label="Words pushing toward 'offensive'")],
             title="Sinhala Offensive Language Detector").launch()
