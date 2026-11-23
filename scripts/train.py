"""Train LayerGap on all CPTAC patients and save the model, block weights and MAP ranking.

    python scripts/train.py --out results/models/layergap_cptac.pkl
"""
import argparse, pickle, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); warnings.simplefilter("ignore")
from src.data_access import load, cptac_blocks
from src.proposed_method.layergap import LayerGap
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results/models/layergap_cptac.pkl"))
ap.add_argument("--with-metabolites", action="store_true"); a = ap.parse_args()
data, kb = load(); s = data["cptac"]["samples"]
m = LayerGap(kb["pathways"], seed=2026).fit(cptac_blocks(data, a.with_metabolites), list(s.index), s.high_grade.values)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pickle.dump(m, open(a.out, "wb"))
print("block weights:", dict(zip(m.active, m.alpha.round(3))))
print(m.map_scores().head(10)[["pathway", "MAP", "concordance"]].to_string(index=False))
