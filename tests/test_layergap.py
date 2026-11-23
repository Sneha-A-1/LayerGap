"""Unit tests for LayerGap on small simulated inputs (no downloads needed).

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.proposed_method.layergap import LayerGap, PathwayEncoder  # noqa: E402


def toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(60)]
    idx = [f"s{i}" for i in range(n)]
    y = rng.integers(0, 2, n)
    T = pd.DataFrame(rng.normal(size=(n, 60)), index=idx, columns=genes)
    P = 0.6 * T + 0.8 * pd.DataFrame(rng.normal(size=(n, 60)), index=idx, columns=genes)
    P.iloc[:, :10] += 1.0 * (y - 0.5)[:, None]  # protein-level (uncoupling) signal in pathway A
    G = pd.DataFrame(rng.integers(0, 2, (n, 3)).astype(float), index=idx, columns=["mut_A", "mut_B", "gain_7"])
    M = pd.DataFrame(rng.normal(size=(n, 6)), index=idx, columns=[f"m{i}" for i in range(6)])
    pw = {"A": {"genes": genes[:10], "mets": ["m0", "m1"], "kind": "metabolic"},
          "B": {"genes": genes[10:30], "mets": ["m2", "m3"], "kind": "metabolic"},
          "C": {"genes": genes[30:60], "mets": [], "kind": "hallmark"}}
    return {"G": G, "T": T, "P": P, "M": M}, idx, y, pw


def test_encoder_shapes_and_uncoupling():
    D, idx, y, pw = toy()
    e = PathwayEncoder(pw).fit(D, idx)
    F = e.transform(D, idx)
    assert set(F) >= {"G", "T", "P", "U", "M", "V"}
    assert F["U"].shape == (len(idx), 3)
    # uncoupling of pathway A separates classes better than its transcript score
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, F["U"]["A"]) > roc_auc_score(y, F["T"]["A"])


def test_fit_predict_and_missing_layer():
    D, idx, y, pw = toy()
    m = LayerGap(pw, n_boot=5, n_boot_inner=3, seed=1).fit(D, idx[:90], y[:90])
    p = m.predict(D, idx[90:])
    assert p.shape == (30,) and np.all((p > 0) & (p < 1))
    Dm = dict(D)
    Dm["P"] = D["P"].copy()
    Dm["P"].iloc[90:] = np.nan  # protein missing for all test patients
    det = m.predict(Dm, idx[90:], return_details=True)
    assert np.all(det["n_layers"] < len(m.active))
    assert np.all(np.isfinite(det["prob"])) and np.all(det["uncertainty"] >= 0)


def test_map_scores_rank_signal_pathway_first():
    D, idx, y, pw = toy(n=200)
    m = LayerGap(pw, n_boot=8, n_boot_inner=3, seed=2).fit(D, idx, y)
    assert m.map_scores().pathway.iloc[0] == "A"


def test_alpha_nonnegative():
    D, idx, y, pw = toy()
    m = LayerGap(pw, n_boot=5, n_boot_inner=3).fit(D, idx, y)
    assert np.all(m.alpha >= 0)
