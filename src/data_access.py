"""Loading helpers shared by all experiment scripts."""
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(processed=ROOT / "data/processed/cohorts.pkl", knowledge=ROOT / "data/knowledge/knowledge.pkl"):
    with open(processed, "rb") as fh:
        data = pickle.load(fh)
    with open(knowledge, "rb") as fh:
        kb = pickle.load(fh)
    return data, kb


def cptac_blocks(data, with_metabolites=True):
    c = data["cptac"]
    D = {"G": c["genomic"], "MUT": c["mut"], "T": c["rna"], "P": c["prot"]}
    if with_metabolites:
        D["M"] = c["met"]
    return D


def tcga_blocks(data):
    t = data["tcga"]
    return {"G": t["genomic"], "MUT": t["mut"], "T": t["rna"]}
