"""Apply a trained LayerGap model to TCGA-KIRC and report AUC, C-index and per-patient outputs.

    python scripts/evaluate.py --model results/models/layergap_cptac.pkl
"""
import argparse, pickle, sys, warnings
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); warnings.simplefilter("ignore")
from sklearn.metrics import roc_auc_score
from src.data_access import load, tcga_blocks
from src.evaluation.metrics import harrell_c
ap = argparse.ArgumentParser(); ap.add_argument("--model", default=str(ROOT / "results/models/layergap_cptac.pkl")); a = ap.parse_args()
data, kb = load(); m = pickle.load(open(a.model, "rb")); st = data["tcga"]["samples"]
d = m.predict(tcga_blocks(data), list(st.index), return_details=True)
out = pd.DataFrame({"prob_high_grade": d["prob"], "uncertainty": d["uncertainty"], "layers_used": d["n_layers"]}, index=st.index)
out = out.join(d["contrib"].add_prefix("contrib_"))
out.to_csv(ROOT / "results/tables/tcga_patient_outputs.csv")
ok = st.os_days > 0
print(f"TCGA AUC = {roc_auc_score(st.high_grade.astype(int), d['prob']):.3f}; "
      f"OS C-index = {harrell_c(st.os_days[ok], st.os_event[ok], d['prob'][ok.values]):.3f}")
