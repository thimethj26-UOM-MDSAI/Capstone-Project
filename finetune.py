"""Fine-tune a transformer classifier on SOLD.
Usage:  python finetune.py --model xlm-roberta-base --out models/xlmr                 (seed 42, the main run)
        python finetune.py --model xlm-roberta-base --out models/xlmr_s1 --seed 1    (extra seed)
The data split is ALWAYS the same (fixed in src/config.py); --seed only changes weight init, dropout and batch order.
Selects best epoch on validation; evaluates test once at the end; records size + latency.
Results go to results/<basename of --out>.json and results/<basename of --out>_test_preds.npy"""
import argparse, json, time, os
import numpy as np, torch
from torch import nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, set_seed)
from src.data import load_sold
from src.preprocess import clean
from src.baseline import metrics

p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=4); p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--bs", type=int, default=16); p.add_argument("--max_len", type=int, default=128)
p.add_argument("--seed", type=int, default=42); p.add_argument("--keep_emoji", action="store_true")
a = p.parse_args()
set_seed(a.seed)
name = os.path.basename(a.out.rstrip("/\\"))

tr, va, te = load_sold()
tok = AutoTokenizer.from_pretrained(a.model)

class DS(torch.utils.data.Dataset):
    def __init__(s, df):
        s.enc = tok([clean(t, keep_emoji=a.keep_emoji) for t in df.text], truncation=True, max_length=a.max_len)
        s.y = df.y.tolist()
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        d = {k: v[i] for k, v in s.enc.items()}; d["labels"] = s.y[i]; return d

dtr, dva, dte = DS(tr), DS(va), DS(te)
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
w = torch.tensor([1.0, (tr.y == 0).sum() / (tr.y == 1).sum()], dtype=torch.float)  # class weights

class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels"); out = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=w.to(out.logits.device))(out.logits, labels)
        return (loss, out) if return_outputs else loss

def cm(ep):
    pr = ep.predictions.argmax(-1)
    return dict(macro_f1=f1_score(ep.label_ids, pr, average="macro"), accuracy=accuracy_score(ep.label_ids, pr))

args = TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
    per_device_train_batch_size=a.bs, per_device_eval_batch_size=64, eval_strategy="epoch",
    save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
    save_total_limit=1, seed=a.seed, fp16=torch.cuda.is_available(), report_to="none", weight_decay=0.01)
t = WTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, processing_class=tok, compute_metrics=cm)
t0 = time.time(); t.train(); train_s = time.time() - t0
val_m = t.evaluate()  # must run while the model is still on the GPU (before the CPU latency test below)

pred = t.predict(dte); te_m = metrics(te.y.values, pred.predictions.argmax(-1))
t.save_model(a.out); tok.save_pretrained(a.out)

# deployment cost: params, disk size, CPU latency per tweet (batch=1)
model.cpu().eval(); sample = [clean(x) for x in te.text[:100]]
t1 = time.time()
with torch.no_grad():
    for s in sample: model(**tok(s, return_tensors="pt", truncation=True, max_length=a.max_len))
lat_ms = (time.time() - t1) / len(sample) * 1000
size_mb = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)
              if f.endswith((".safetensors", ".bin"))) / 1e6
res = dict(model=a.model, args=vars(a), train_seconds=round(train_s), params_m=round(sum(x.numel() for x in model.parameters()) / 1e6, 1),
           disk_mb=round(size_mb, 1), cpu_latency_ms=round(lat_ms, 1), val=val_m, test=te_m,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/{name}.json", "w"), indent=2)
np.save(f"results/{name}_test_preds.npy", pred.predictions.argmax(-1))  # for McNemar
print(json.dumps(res, indent=2))
