"""Run all experiments of the study.

    python experiments/run_experiments.py --exp main        # E1 repeated CV, all methods
    python experiments/run_experiments.py --exp all         # everything in order

Every experiment writes CSV files to results/metrics/ and a log to results/logs/.
Random seeds are fixed (SEED below); fold assignments are stored with the predictions.
"""
import argparse
import json
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

from sklearn.model_selection import StratifiedKFold  # noqa: E402

from src.baselines.baselines import (TMC, ElasticNetLR, LateFusionLR, XGBoost,  # noqa: E402
                                     all_baselines)
from src.data_access import cptac_blocks, load, tcga_blocks  # noqa: E402
from src.evaluation.metrics import (bootstrap_ci, classification_metrics,  # noqa: E402
                                    corrected_resampled_ttest, harrell_c, holm, paired_bootstrap_p)
from src.proposed_method.layergap import LayerGap  # noqa: E402

SEED = 2026
OUT = ROOT / "results/metrics"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "results/logs").mkdir(parents=True, exist_ok=True)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lg(kb, **kw):
    m = LayerGap(kb["pathways"], seed=SEED, **kw)
    m.name = "LayerGap"
    return m


def folds(s, n_splits=5, n_repeats=5, seed=SEED):
    strata = s["high_grade"].astype(str) + "_" + s["cohort"].astype(str)
    out = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for f, (a, b) in enumerate(skf.split(s.index, strata)):
            out.append((r, f, a, b))
    return out


def timed_fit_predict(m, D, tr, ytr, te, Dte=None):
    t0 = time.perf_counter()
    m.fit(D, tr, ytr)
    t1 = time.perf_counter()
    p = m.predict(Dte if Dte is not None else D, te)
    t2 = time.perf_counter()
    return p, t1 - t0, (t2 - t1) / len(te) * 1000.0


# ------------------------------------------------------------------------- E1 main CV
def exp_main(data, kb):
    D = cptac_blocks(data, with_metabolites=False)
    s = data["cptac"]["samples"]
    y, idx = s.high_grade.values, np.array(s.index)
    rows, preds = [], []
    for r, f, a, b in folds(s):
        methods = [lg(kb)] + all_baselines(seed=SEED)
        for m in methods:
            p, tfit, tinf = timed_fit_predict(m, D, list(idx[a]), y[a], list(idx[b]))
            met = classification_metrics(y[b], p)
            rows.append({"method": m.name, "repeat": r, "fold": f, **met, "fit_s": tfit, "infer_ms_per_sample": tinf})
            preds += [{"method": m.name, "repeat": r, "fold": f, "case": c, "y": yy, "p": pp}
                      for c, yy, pp in zip(idx[b], y[b], p)]
        say(f"E1 repeat {r} fold {f} done")
        pd.DataFrame(rows).to_csv(OUT / "E1_main_cv_folds.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "E1_main_cv_predictions.csv", index=False)
    summarise_main(len(s))


def summarise_main(n):
    df = pd.read_csv(OUT / "E1_main_cv_folds.csv")
    metrics = ["AUC", "PR_AUC", "BalAcc", "MCC", "Brier"]
    summ = df.groupby("method")[metrics + ["fit_s", "infer_ms_per_sample"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    ref = df[df.method == "LayerGap"].sort_values(["repeat", "fold"])
    n_test = n / 5
    tests = []
    for m in df.method.unique():
        if m == "LayerGap":
            continue
        o = df[df.method == m].sort_values(["repeat", "fold"])
        for k in ["AUC", "MCC", "BalAcc"]:
            t, p = corrected_resampled_ttest(ref[k].values, o[k].values, n - n_test, n_test)
            tests.append({"baseline": m, "metric": k, "mean_diff": (ref[k].values - o[k].values).mean(), "t": t, "p": p})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = np.nan
    for k in tests.metric.unique():
        sel = tests.metric == k
        tests.loc[sel, "p_holm"] = holm(tests.loc[sel, "p"].values)
    summ.to_csv(OUT / "E1_main_cv_summary.csv")
    tests.to_csv(OUT / "E1_main_cv_tests.csv", index=False)
    print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(3).sort_values("AUC_mean", ascending=False))


# ------------------------------------------------------------------------- E2 cross-cohort
def exp_cross(data, kb):
    D = cptac_blocks(data, with_metabolites=False)
    s = data["cptac"]["samples"]
    rows, preds = [], []
    for src, dst in [("INI", "EXP"), ("EXP", "INI")]:
        tr, te = list(s.index[s.cohort == src]), list(s.index[s.cohort == dst])
        ytr, yte = s.loc[tr, "high_grade"].values, s.loc[te, "high_grade"].values
        res = {}
        for m in [lg(kb)] + all_baselines(seed=SEED):
            p, tfit, tinf = timed_fit_predict(m, D, tr, ytr, te)
            res[m.name] = p
            lo, hi = bootstrap_ci(yte, p)
            rows.append({"train": src, "test": dst, "method": m.name, **classification_metrics(yte, p),
                         "AUC_lo": lo, "AUC_hi": hi})
            preds += [{"train": src, "method": m.name, "case": c, "y": a, "p": b} for c, a, b in zip(te, yte, p)]
        for k, p in res.items():
            if k != "LayerGap":
                d, pv = paired_bootstrap_p(yte, res["LayerGap"], p)
                rows[-len(res) + list(res).index(k)].update({"dAUC_vs_LayerGap": -d, "p_vs_LayerGap": pv})
        say(f"E2 {src}->{dst} done")
    pd.DataFrame(rows).to_csv(OUT / "E2_cross_cohort.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "E2_cross_cohort_predictions.csv", index=False)
    print(pd.DataFrame(rows)[["train", "test", "method", "AUC", "AUC_lo", "AUC_hi", "MCC"]].round(3))


# ------------------------------------------------------------------------- E3 TCGA external
def exp_tcga(data, kb):
    D = cptac_blocks(data, with_metabolites=False)
    Dt = tcga_blocks(data)
    s, st = data["cptac"]["samples"], data["tcga"]["samples"]
    tr, te = list(s.index), list(st.index)
    ytr, yte = s.high_grade.values, st.high_grade.values.astype(int)
    methods = [lg(kb)] + all_baselines(blocks=("G", "T"), seed=SEED)
    m2 = lg(kb, blocks=("G", "T"))
    m2.name = "LayerGap (G+T only)"
    methods.insert(1, m2)
    rows, risk = [], {}
    for m in methods:
        p, tfit, tinf = timed_fit_predict(m, D, tr, ytr, te, Dte=Dt)
        risk[m.name] = p
        lo, hi = bootstrap_ci(yte, p)
        ok = st.os_days.notna().values & (st.os_days.values > 0)
        c = harrell_c(st.os_days.values[ok], st.os_event.values[ok], p[ok])
        rows.append({"method": m.name, **classification_metrics(yte, p), "AUC_lo": lo, "AUC_hi": hi, "OS_Cindex": c})
        say(f"E3 {m.name} AUC={rows[-1]['AUC']:.3f} C={c:.3f}")
    for r in rows:
        if r["method"] != "LayerGap":
            d, pv = paired_bootstrap_p(yte, risk["LayerGap"], risk[r["method"]])
            r.update({"dAUC_LayerGap_minus": d, "p_paired": pv})
    pd.DataFrame(rows).to_csv(OUT / "E3_tcga_external.csv", index=False)
    pd.DataFrame(risk, index=te).assign(y=yte, os_days=st.os_days.values, os_event=st.os_event.values,
                                       stage=st.stage.values).to_csv(OUT / "E3_tcga_predictions.csv")
    survival_tcga()
    print(pd.DataFrame(rows).round(3))


def survival_tcga():
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    d = pd.read_csv(OUT / "E3_tcga_predictions.csv", index_col=0)
    d = d[d.os_days > 0].dropna(subset=["os_days", "os_event"])
    d["advanced_stage"] = d.stage.astype(str).str.contains("III|IV").astype(int)
    d["score"] = (d["LayerGap"] - d["LayerGap"].mean()) / d["LayerGap"].std()
    d["high_score"] = (d["LayerGap"] >= d["LayerGap"].median()).astype(int)
    lr = logrank_test(d.os_days[d.high_score == 1], d.os_days[d.high_score == 0],
                      d.os_event[d.high_score == 1], d.os_event[d.high_score == 0])
    out = {"n": len(d), "events": int(d.os_event.sum()), "logrank_p": lr.p_value}
    for name, cols in [("uni", ["score"]), ("multi", ["score", "advanced_stage"]), ("multi_grade", ["score", "advanced_stage", "y"])]:
        cph = CoxPHFitter().fit(d[cols + ["os_days", "os_event"]], "os_days", "os_event")
        sm = cph.summary.loc["score"]
        out.update({f"{name}_HR": sm["exp(coef)"], f"{name}_HR_lo": sm["exp(coef) lower 95%"],
                    f"{name}_HR_hi": sm["exp(coef) upper 95%"], f"{name}_p": sm["p"]})
    pd.Series(out).to_csv(OUT / "E3_tcga_survival.csv")
    print(pd.Series(out).round(4))


# ------------------------------------------------------------------------- E4 four-layer core
def exp_core(data, kb):
    D = cptac_blocks(data, with_metabolites=True)
    s = data["cptac"]["samples"]
    core = set(data["cptac"]["core"])
    y, idx = s.high_grade.values, np.array(s.index)
    Dm = dict(D)
    Dm["M"] = D["M"].reindex(s.index)  # NaN rows outside the core
    mk = {
        "LayerGap (G,T,P,U,M,V)": lambda: lg(kb),
        "LayerGap (G,T,P,U)": lambda: lg(kb, blocks=("G", "T", "P", "U")),
        "EN-concat (+M)": lambda: ElasticNetLR(("G", "T", "P", "M"), seed=SEED),
        "Late-fusion LR (+M)": lambda: LateFusionLR(("G", "T", "P", "M"), seed=SEED),
        "TMC (+M)": lambda: TMC(("G", "T", "P", "M"), seed=SEED),
    }
    preds = []
    for r, f, a, b in folds(s, n_repeats=5):
        for name, make in mk.items():
            m = make()
            tr = list(idx[a])
            if name.startswith(("EN", "Late", "TMC")):  # these need complete blocks: metabolites outside the core are mean-imputed (0 after z-scoring)
                Dtr = {k: (v if k != "M" else v.fillna(0.0)) for k, v in Dm.items()}
                m.fit(Dtr, tr, y[a])
                p = m.predict(Dtr, list(idx[b]))
            else:
                m.fit(Dm, tr, y[a])
                p = m.predict(Dm, list(idx[b]))
            preds += [{"method": name, "repeat": r, "case": c, "y": yy, "p": pp, "core": c in core}
                      for c, yy, pp in zip(idx[b], y[b], p)]
        say(f"E4 repeat {r} fold {f}")
    P = pd.DataFrame(preds)
    P.to_csv(OUT / "E4_core_predictions.csv", index=False)
    from sklearn.metrics import roc_auc_score
    res = P[P.core].groupby(["method", "repeat"]).apply(lambda g: roc_auc_score(g.y, g.p)).rename("AUC_core").reset_index()
    summ = res.groupby("method").AUC_core.agg(["mean", "std"]).reset_index()
    summ.to_csv(OUT / "E4_core_summary.csv", index=False)
    print(summ.round(3))


# ------------------------------------------------------------------------- E5 missing layers
def exp_missing(data, kb):
    D = cptac_blocks(data, with_metabolites=False)
    s = data["cptac"]["samples"]
    y, idx = s.high_grade.values, np.array(s.index)
    rows = []
    rng = np.random.default_rng(SEED)
    for r, f, a, b in folds(s, n_repeats=2):
        models = [lg(kb), ElasticNetLR(seed=SEED), LateFusionLR(seed=SEED), XGBoost(seed=SEED), TMC(seed=SEED)]
        for m in models:
            m.fit(D, list(idx[a]), y[a])
        te = idx[b]
        for layer in ["P", "T"]:
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                drop = set(rng.choice(te, int(round(frac * len(te))), replace=False))
                Dm = dict(D)
                Dm[layer] = D[layer].copy()
                Dm[layer].loc[Dm[layer].index.isin(drop)] = np.nan
                for m in models:
                    t0 = time.perf_counter()
                    p = m.predict(Dm, list(te))
                    rows.append({"method": m.name, "layer_masked": layer, "fraction": frac, "repeat": r, "fold": f,
                                 **classification_metrics(y[b], p),
                                 "infer_ms_per_sample": (time.perf_counter() - t0) / len(te) * 1000})
        say(f"E5 repeat {r} fold {f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E5_missing_layers.csv", index=False)
    print(df.groupby(["layer_masked", "fraction", "method"]).AUC.mean().unstack().round(3))


# ------------------------------------------------------------------------- E7 ablation and E8 sensitivity
def exp_ablation(data, kb):
    D = cptac_blocks(data, with_metabolites=False)
    s = data["cptac"]["samples"]
    y, idx = s.high_grade.values, np.array(s.index)
    variants = {
        "Full LayerGap": {},
        "- uncoupling block (U)": {"blocks": ("G", "T", "P")},
        "- precision weighting": {"use_precision": False},
        "- pathway constraint (gene-level)": {"pathway_level": False, "blocks": ("G", "T", "P")},
        "- genomic block (G)": {"blocks": ("T", "P", "U")},
        "- transcript block (T)": {"blocks": ("G", "P", "U")},
        "- protein block (P)": {"blocks": ("G", "T", "U")},
        "U block only": {"blocks": ("U",)},
        "T block only": {"blocks": ("T",)},
        "P block only": {"blocks": ("P",)},
    }
    rows = []
    for r, f, a, b in folds(s):
        for name, kw in variants.items():
            m = lg(kb, **kw)
            p, tfit, tinf = timed_fit_predict(m, D, list(idx[a]), y[a], list(idx[b]))
            rows.append({"variant": name, "repeat": r, "fold": f, **classification_metrics(y[b], p), "fit_s": tfit})
        say(f"E7 repeat {r} fold {f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E7_ablation_folds.csv", index=False)
    full = df[df.variant == "Full LayerGap"].sort_values(["repeat", "fold"])
    out = []
    for v in variants:
        o = df[df.variant == v].sort_values(["repeat", "fold"])
        t, p = corrected_resampled_ttest(full.AUC.values, o.AUC.values, len(s) * 0.8, len(s) * 0.2)
        out.append({"variant": v, **{k: o[k].mean() for k in ["AUC", "PR_AUC", "BalAcc", "MCC", "Brier"]},
                    "AUC_sd": o.AUC.std(), "dAUC_vs_full": o.AUC.mean() - full.AUC.mean(), "p_vs_full": p if v != "Full LayerGap" else np.nan})
    pd.DataFrame(out).to_csv(OUT / "E7_ablation_summary.csv", index=False)
    print(pd.DataFrame(out).round(3))


def exp_sensitivity(data, kb):
    D = cptac_blocks(data, with_metabolites=False)
    s = data["cptac"]["samples"]
    y, idx = s.high_grade.values, np.array(s.index)
    grid = [("n_boot", v, {"n_boot": v}) for v in [5, 10, 20, 40]]
    grid += [("tau", v, {"tau": v}) for v in [0.01, 0.1, 1.0, 10.0]]
    grid += [("min_genes", v, {"min_genes": v}) for v in [3, 5, 10, 20]]
    grid += [("C (fixed)", v, {"Cs": (v,)}) for v in [0.001, 0.01, 0.1, 1.0]]
    grid += [("stack_l2", v, {"stack_l2": v}) for v in [0.1, 1.0, 10.0]]
    rows = []
    rng = np.random.default_rng(SEED)
    for r, f, a, b in folds(s, n_repeats=2):
        for par, val, kw in grid:
            p, tfit, _ = timed_fit_predict(lg(kb, **kw), D, list(idx[a]), y[a], list(idx[b]))
            rows.append({"parameter": par, "value": val, "repeat": r, "fold": f, "AUC": classification_metrics(y[b], p)["AUC"], "fit_s": tfit})
        for frac in [0.4, 0.6, 0.8, 1.0]:  # training-set size
            sub = rng.choice(a, int(round(frac * len(a))), replace=False)
            for m in [lg(kb), ElasticNetLR(seed=SEED), LateFusionLR(seed=SEED)]:
                p, tfit, _ = timed_fit_predict(m, D, list(idx[sub]), y[sub], list(idx[b]))
                rows.append({"parameter": f"train_fraction::{m.name}", "value": frac, "repeat": r, "fold": f,
                             "AUC": classification_metrics(y[b], p)["AUC"], "fit_s": tfit})
        say(f"E8 repeat {r} fold {f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E8_sensitivity.csv", index=False)
    print(df.groupby(["parameter", "value"]).AUC.agg(["mean", "std"]).round(3))


EXPS = {"main": exp_main, "cross": exp_cross, "tcga": exp_tcga, "core": exp_core,
        "missing": exp_missing, "ablation": exp_ablation, "sensitivity": exp_sensitivity}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=list(EXPS) + ["all", "summarise_main", "survival"])
    a = ap.parse_args()
    data, kb = load()
    if a.exp == "summarise_main":
        summarise_main(len(data["cptac"]["samples"]))
    elif a.exp == "survival":
        survival_tcga()
    else:
        for k in (EXPS if a.exp == "all" else [a.exp]):
            say(f"=== {k} ===")
            EXPS[k](data, kb)
