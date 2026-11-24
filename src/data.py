"""Load SOLD and build fixed train/val/test splits (official test set untouched until final eval)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import SEED, VAL_FRACTION, HF_DATASET


def _norm_label(x) -> int:
    """OFF/OFFENSIVE/1 -> 1, NOT/NOT OFFENSIVE/0 -> 0."""
    s = str(x).strip().upper()
    if s in {"OFF", "OFFENSIVE", "1"}:
        return 1
    if s in {"NOT", "NOT OFFENSIVE", "0"}:
        return 0
    raise ValueError(f"Unrecognised label: {x!r}")


def load_sold(text_col="text", label_col="label"):
    """Returns (train_df, val_df, test_df) with columns text, y. Needs internet access to Hugging Face."""
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET)
    train = ds["train"].to_pandas()
    test = ds["test"].to_pandas()
    print("Columns:", list(train.columns))  # check once: token-level label column names live here
    return split_frames(train, test, text_col, label_col)


def split_frames(train, test, text_col="text", label_col="label"):
    def prep(df):
        out = df.copy()
        out["y"] = out[label_col].map(_norm_label)
        out["text"] = out[text_col].astype(str)
        return out
    train, test = prep(train), prep(test)
    tr, va = train_test_split(train, test_size=VAL_FRACTION, stratify=train["y"], random_state=SEED)
    return tr.reset_index(drop=True), va.reset_index(drop=True), test.reset_index(drop=True)
