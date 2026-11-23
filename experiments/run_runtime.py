"""Runtime profile of LayerGap stages and of end-to-end processing (E10).

    python experiments/run_runtime.py
"""
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from src.data_access import cptac_blocks, load  # noqa: E402
from src.proposed_method.layergap import BlockModel, LayerGap, PathwayEncoder  # noqa: E402

SEED = 2026
OUT = ROOT / "results/metrics"


def t(fn, reps=5):
    vals = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        vals.append(time.perf_counter() - t0)
    return np.median(vals), r


def main():
    t0 = time.perf_counter()
    data, kb = load()
    t_load = time.perf_counter() - t0
    D = cptac_blocks(data, with_metabolites=False)
    s = data["cptac"]["samples"]
    idx, y = list(s.index), s.high_grade.values
    tr, te = idx[:170], idx[170:]
    rows = [{"stage": "Load harmonised cohorts", "seconds": t_load}]
    sec, enc = t(lambda: PathwayEncoder(kb["pathways"]).fit(D, tr))
    rows.append({"stage": "Pathway encoder fit (b_g estimation, gene sets)", "seconds": sec})
    sec, F = t(lambda: enc.transform(D, tr))
    rows.append({"stage": "Pathway encoding of training samples (G,T,P,U)", "seconds": sec})
    X = F["U"].values
    sec, _ = t(lambda: BlockModel(n_boot=1).fit(X, y[:170]), reps=3)
    rows.append({"stage": "Single block model, C selection + 1 fit", "seconds": sec})
    sec, _ = t(lambda: BlockModel(Cs=(0.01,), n_boot=20).fit(X, y[:170], C=0.01), reps=3)
    rows.append({"stage": "Bootstrap ensemble, 20 fits (uncertainty estimate)", "seconds": sec})
    sec, m = t(lambda: LayerGap(kb["pathways"], seed=SEED).fit(D, tr, y[:170]), reps=3)
    rows.append({"stage": "Full LayerGap training (4 blocks, OOF stacking)", "seconds": sec})
    sec, _ = t(lambda: m.predict(D, te), reps=20)
    rows.append({"stage": f"Inference, {len(te)} samples (encoding + ensemble + fusion)", "seconds": sec,
                 "ms_per_sample": sec / len(te) * 1000})
    m1 = LayerGap(kb["pathways"], seed=SEED, n_boot=1, n_boot_inner=1).fit(D, tr, y[:170])
    sec1, _ = t(lambda: m1.predict(D, te), reps=20)
    rows.append({"stage": "Inference without bootstrap ensemble (n_boot=1)", "seconds": sec1, "ms_per_sample": sec1 / len(te) * 1000})
    one = [te[0]]
    sec, _ = t(lambda: m.predict(D, one, return_details=True), reps=50)
    rows.append({"stage": "Single-patient report (probability, uncertainty, block contributions)", "seconds": sec,
                 "ms_per_sample": sec * 1000})
    tracemalloc.start()
    LayerGap(kb["pathways"], seed=SEED).fit(D, tr, y[:170])
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    rows.append({"stage": "Peak Python memory during training (MB)", "seconds": np.nan, "MB": peak})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E10_runtime_layergap.csv", index=False)
    print(df.round(4))


if __name__ == "__main__":
    main()
