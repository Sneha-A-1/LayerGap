"""Synthetic stress tests with known ground truth (E6).

Synthetic cohorts are generated from the real CPTAC data so that gene-gene correlation, gene-wise
transcript-protein coupling (b_g) and genomic feature distributions are realistic:
  * T = low-rank (30 PCs) Gaussian draw from the real transcript matrix + gene-wise residual noise
  * P = b_g * T + sqrt(1 - b_g^2) * E, with E a low-rank draw from the real protein residual matrix
  * G, MUT = rows resampled from the real genomic tables (carry no signal)
  * outcome-linked signal is planted in k_U pathways at the protein level only (uncoupling signal)
    and in k_T other pathways at the transcript level (propagating to protein through b_g).
Because the signal pathways are known, both predictive accuracy and pathway recovery can be scored.

    python experiments/run_synthetic.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")
import torch  # noqa: E402

torch.set_num_threads(2)
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.baselines.baselines import TMC, ElasticNetLR, LateFusionLR, XGBoost  # noqa: E402
from src.data_access import cptac_blocks, load  # noqa: E402
from src.proposed_method.layergap import LayerGap, PathwayEncoder  # noqa: E402

SEED = 2026
OUT = ROOT / "results/metrics"


class SyntheticCohort:
    def __init__(self, data, kb, n_pc=30, seed=SEED):
        D = cptac_blocks(data, with_metabolites=False)
        s = data["cptac"]["samples"]
        enc = PathwayEncoder(kb["pathways"]).fit(D, list(s.index))
        self.genes = enc.shared
        self.beta = enc.beta.values
        T = D["T"].loc[s.index, self.genes].values
        R = D["P"].loc[s.index, self.genes].values - T * self.beta
        self.T_mu, self.R_mu = T.mean(0), R.mean(0)
        self.T_load, self.T_res = self._lowrank(T, n_pc)
        self.R_load, self.R_res = self._lowrank(R, n_pc)
        self.G, self.MUT = D["G"].loc[s.index], D["MUT"].loc[s.index]
        rng = np.random.default_rng(seed)
        gi = {g: i for i, g in enumerate(self.genes)}
        cand = [k for k, v in enc.gene_sets.items() if 10 <= len(v) <= 60]
        pick = rng.choice(cand, 10, replace=False)
        self.u_paths, self.t_paths = list(pick[:5]), list(pick[5:])
        self.u_idx = np.unique(np.concatenate([[gi[g] for g in enc.gene_sets[k]] for k in self.u_paths]))
        self.t_idx = np.unique(np.concatenate([[gi[g] for g in enc.gene_sets[k]] for k in self.t_paths]))

    @staticmethod
    def _lowrank(X, k):
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        load = (S[:k, None] * Vt[:k]) / np.sqrt(len(X))
        res = (Xc - U[:, :k] * S[:k] @ Vt[:k]).std(0)
        return load, res

    def sample(self, n, rng, prev=0.4, d_u=0.5, d_t=0.25, noise=1.0, prefix="s"):
        y = (rng.random(n) < prev).astype(int)
        k = self.T_load.shape[0]
        T = self.T_mu + rng.standard_normal((n, k)) @ self.T_load + rng.standard_normal((n, len(self.genes))) * self.T_res
        T[:, self.t_idx] += d_t * (y - prev)[:, None] * 2
        E = self.R_mu + rng.standard_normal((n, k)) @ self.R_load + rng.standard_normal((n, len(self.genes))) * self.R_res
        P = self.beta * T + E
        P[:, self.u_idx] += d_u * (y - prev)[:, None] * 2
        if noise > 1:
            sd = np.sqrt(noise ** 2 - 1)
            T += rng.standard_normal(T.shape) * sd
            P += rng.standard_normal(P.shape) * sd
        idx = [f"{prefix}{i}" for i in range(n)]
        rows = rng.integers(0, len(self.G), n)
        Tz = pd.DataFrame(T, index=idx, columns=self.genes)
        Pz = pd.DataFrame(P, index=idx, columns=self.genes)
        Tz, Pz = (Tz - Tz.mean()) / Tz.std(), (Pz - Pz.mean()) / Pz.std()
        G = self.G.iloc[rows].set_axis(idx)
        MUT = self.MUT.iloc[rows].set_axis(idx)
        return {"G": G, "MUT": MUT, "T": Tz, "P": Pz}, idx, y


def pathway_recovery(model, truth, kb):
    if isinstance(model, LayerGap):
        top = list(model.map_scores().pathway[:10])
    else:  # EN-concat: pathway-aggregated absolute coefficients
        coef = pd.Series(np.abs(model.model.coef_[0]), index=[f"{b}:{c}" for b in model.blocks for c in model.cols[b]])
        agg = {}
        for k, v in kb["pathways"].items():
            if len(v["genes"]) >= 5:
                agg[k] = sum(coef.get(f"T:{g}", 0) + coef.get(f"P:{g}", 0) for g in v["genes"]) / np.sqrt(len(v["genes"]))
        top = list(pd.Series(agg).sort_values(ascending=False).index[:10])
    return len(set(top) & set(truth)) / 10.0


def main():
    data, kb = load()
    gen = SyntheticCohort(data, kb)
    truth = gen.u_paths + gen.t_paths
    pd.Series({"uncoupling_signal_pathways": "; ".join(gen.u_paths),
               "transcript_signal_pathways": "; ".join(gen.t_paths)}).to_csv(OUT / "E6_synthetic_truth.csv")
    # effect sizes chosen so that a reference elastic-net model reaches AUC ~0.78 at n=240 (avoids ceiling effects)
    default = dict(n=240, prev=0.4, d_u=0.15, d_t=0.075, noise=1.0)
    scen = [("sample size", f"n={n}", {**default, "n": n}) for n in [60, 120, 240, 480]]
    scen += [("noise", f"x{v}", {**default, "noise": v}) for v in [1.5, 2.0]]
    scen += [("class imbalance", "prevalence 0.15", {**default, "prev": 0.15})]
    scen += [("signal type", "transcript-only signal", {**default, "d_u": 0.0, "d_t": 0.2})]
    rows = []
    for seed in range(3):
        for group, label, cfg in scen:
            rng = np.random.default_rng(SEED + 100 * seed + hash(label) % 97)
            Dtr, itr, ytr = gen.sample(cfg["n"], rng, cfg["prev"], cfg["d_u"], cfg["d_t"], cfg["noise"], prefix="tr")
            Dte, ite, yte = gen.sample(600, rng, cfg["prev"], cfg["d_u"], cfg["d_t"], cfg["noise"], prefix="te")
            models = [LayerGap(kb["pathways"], seed=SEED), ElasticNetLR(seed=SEED), LateFusionLR(seed=SEED),
                      XGBoost(seed=SEED), TMC(seed=SEED)]
            models[0].name = "LayerGap"
            for m in models:
                t0 = time.perf_counter()
                m.fit(Dtr, itr, ytr)
                fit_s = time.perf_counter() - t0
                p = m.predict(Dte, ite)
                rec = pathway_recovery(m, truth, kb) if isinstance(m, (LayerGap, ElasticNetLR)) and group != "signal type" else np.nan
                rows.append({"seed": seed, "group": group, "scenario": label, "method": m.name,
                             "AUC": roc_auc_score(yte, p), "pathway_precision_at_10": rec, "fit_s": fit_s})
                if group == "sample size" and cfg["n"] == 240:  # missing-protein scenario on the default cohort
                    for frac in [0.25, 0.5, 1.0]:
                        Dm = dict(Dte)
                        drop = rng.choice(ite, int(frac * len(ite)), replace=False)
                        Dm["P"] = Dte["P"].copy()
                        Dm["P"].loc[drop] = np.nan
                        rows.append({"seed": seed, "group": "missing protein (test)", "scenario": f"{int(frac*100)}% missing",
                                     "method": m.name, "AUC": roc_auc_score(yte, m.predict(Dm, ite))})
            print(time.strftime("%H:%M:%S"), seed, label, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "E6_synthetic.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["group", "scenario", "method"])[["AUC", "pathway_precision_at_10"]].mean().round(3))


if __name__ == "__main__":
    main()
