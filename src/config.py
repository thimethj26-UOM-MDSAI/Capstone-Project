"""Central config: seeds, split sizes, paths. Report these in Appendix A."""
from pathlib import Path

SEED = 42
VAL_FRACTION = 0.10          # stratified slice of the official 7,500 train set
HF_DATASET = "sinhala-nlp/SOLD"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
