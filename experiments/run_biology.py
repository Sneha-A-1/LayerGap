"""Biological analyses: transcript-protein coupling, pathway prioritisation, stability and
cross-cohort / cross-layer replication of prioritised pathways and genes.

    python experiments/run_biology.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score, silhouette_score  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from src.baselines.baselines import ElasticNetLR  # noqa: E402
from src.data_access import cptac_blocks, load, tcga_blocks  # noqa: E402
from src.proposed_method.layergap import LayerGap, PathwayEncoder  # noqa: E402

SEED = 2026
OUT = ROOT / "results/metrics"


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def colwise_spearman(A, B):
    """Spearman correlation between matching columns of A and B (samples x genes)."""
    ra, rb = A.rank().values, B.rank().values
    ra = (ra - ra.mean(0)) / (ra.std(0) + 1e-12)
    rb = (rb - rb.mean(0)) / (rb.std(0) + 1e-12)
    return (ra * rb).mean(0)


# ----------------------------------------------------------------------------- batch structure
def batch_analysis(data):
    c = data["cptac"]
    coh = c["samples"].cohort
    rows = []
    for name, raw, z in [("RNA", c["rna_raw"], c["rna"]), ("Protein", c["prot_raw"], c["prot"])]:
        for stage, X in [("before", raw), ("after", z)]:
            X = X.loc[X.index.isin(coh.index)].dropna(axis=1)
            pcs = PCA(10, random_state=SEED).fit_transform((X - X.mean()).values)
            rows.append({"layer": name, "stage": stage, "silhouette_by_subcohort": silhouette_score(pcs, coh.loc[X.index]),
                         "n_features": X.shape[1]})
            pd.DataFrame(pcs[:, :2], index=X.index, columns=["PC1", "PC2"]).assign(cohort=coh.loc[X.index].values).to_csv(
                OUT / f"E0_pca_{name}_{stage}.csv")
    pd.DataFrame(rows).to_csv(OUT / "E0_batch_silhouette.csv", index=False)
    print(pd.DataFrame(rows).round(3))


# ----------------------------------------------------------------------------- coupling analysis
def coupling_analysis(data, kb, n_perm=1000):
    c = data["cptac"]
    s = c["samples"]
    shared = sorted(set(c["rna"].columns) & set(c["prot"].columns))
    T, P = c["rna"].loc[s.index, shared], c["prot"].loc[s.index, shared]
    hg = s.high_grade.values.astype(bool)
    r_all = colwise_spearman(T, P)
    r_hi, r_lo = colwise_spearman(T[hg], P[hg]), colwise_spearman(T[~hg], P[~hg])
    d = r_hi - r_lo
    genes = pd.DataFrame({"r_all": r_all, "r_low": r_lo, "r_high": r_hi, "delta": d}, index=shared)
    # per-cohort replication
    for coh in ["INI", "EXP"]:
        m = (s.cohort == coh).values
        genes[f"delta_{coh}"] = colwise_spearman(T[m & hg], P[m & hg]) - colwise_spearman(T[m & ~hg], P[m & ~hg])
    # permutation null for the median delta (grade labels shuffled within sub-cohort)
    rng = np.random.default_rng(SEED)
    sets = {k: [g for g in v["genes"] if g in genes.index] for k, v in kb["pathways"].items()}
    sets = {k: v for k, v in sets.items() if len(v) >= 10}
    gi = {g: i for i, g in enumerate(shared)}
    set_idx = {k: np.array([gi[g] for g in v]) for k, v in sets.items()}
    obs_pw = {k: d[v].mean() for k, v in set_idx.items()}
    null_med, null_pw = [], {k: [] for k in sets}
    coh = s.cohort.values
    for _ in range(n_perm):
        perm = hg.copy()
        for cc in ["INI", "EXP"]:
            m = coh == cc
            perm[m] = rng.permutation(hg[m])
        dp = colwise_spearman(T[perm], P[perm]) - colwise_spearman(T[~perm], P[~perm])
        null_med.append(np.median(dp))
        for k, v in set_idx.items():
            null_pw[k].append(dp[v].mean())
    null_med = np.array(null_med)
    obs = np.median(d)
    summary = {"n_genes": len(shared), "median_r_all": float(np.median(r_all)), "median_r_low": float(np.median(r_lo)),
               "median_r_high": float(np.median(r_hi)), "median_delta": float(obs),
               "frac_genes_lower_coupling_in_high": float((d < 0).mean()),
               "perm_p_two_sided": float((np.sum(np.abs(null_med - null_med.mean()) >= abs(obs - null_med.mean())) + 1) / (n_perm + 1)),
               "null_median_delta_mean": float(null_med.mean()),
               "median_sd_ratio_RNA_high_vs_low": float(np.median(T[hg].std() / T[~hg].std())),
               "median_sd_ratio_protein_high_vs_low": float(np.median(P[hg].std() / P[~hg].std())),
               "median_delta_INI": float(genes.delta_INI.median()), "median_delta_EXP": float(genes.delta_EXP.median()),
               "wilcoxon_p_INI": float(stats.wilcoxon(genes.delta_INI).pvalue),
               "wilcoxon_p_EXP": float(stats.wilcoxon(genes.delta_EXP).pvalue)}
    pw = pd.DataFrame({"pathway": list(obs_pw), "n_genes": [len(sets[k]) for k in obs_pw],
                       "mean_delta": list(obs_pw.values()),
                       "perm_p": [(np.sum(np.abs(np.array(null_pw[k]) - np.mean(null_pw[k])) >= abs(obs_pw[k] - np.mean(null_pw[k]))) + 1) / (n_perm + 1) for k in obs_pw]})
    pw["fdr"] = multipletests(pw.perm_p, method="fdr_bh")[1]
    pw = pw.sort_values("mean_delta")
    genes.to_csv(OUT / "E9_coupling_genes.csv")
    pw.to_csv(OUT / "E9_coupling_pathways.csv", index=False)
    pd.Series(summary).to_csv(OUT / "E9_coupling_summary.csv")
    print(pd.Series(summary).round(4)); print(pw.head(12).round(4))


# ----------------------------------------------------------------------------- prioritisation
def prioritisation(data, kb, n_boot=30):
    D = cptac_blocks(data, with_metabolites=False)
    s = data["cptac"]["samples"]
    idx, y = list(s.index), s.high_grade.values
    full = LayerGap(kb["pathways"], seed=SEED).fit(D, idx, y)
    mp = full.map_scores()
    mp.to_csv(OUT / "E9_map_scores.csv", index=False)
    pd.Series(dict(zip(full.active, full.alpha))).to_csv(OUT / "E9_block_weights_full.csv")
    # stability: patient bootstrap, top-20 overlap (Jaccard) for MAP vs EN-concat vs univariate ranking
    rng = np.random.default_rng(SEED)
    tops = {"LayerGap MAP": [], "EN-concat (pathway-aggregated)": [], "Univariate U-score": []}
    enc_sets = None
    for bi in range(n_boot):
        b = rng.choice(len(idx), len(idx), replace=True)
        bidx = [idx[i] for i in b]
        bidx_u = list(dict.fromkeys(bidx))
        m = LayerGap(kb["pathways"], seed=SEED + bi, n_boot=10, n_boot_inner=5).fit(D, bidx_u, s.loc[bidx_u, "high_grade"].values)
        tops["LayerGap MAP"].append(set(m.map_scores().pathway[:20]))
        en = ElasticNetLR(seed=SEED, grid=(0.1,)).fit(D, bidx_u, s.loc[bidx_u, "high_grade"].values)
        coef = pd.Series(np.abs(en.model.coef_[0]), index=[f"{blk}:{c}" for blk in en.blocks for c in en.cols[blk]])
        enc_sets = enc_sets or {k: v["genes"] for k, v in kb["pathways"].items()}
        agg = {}
        for k, gs in enc_sets.items():
            vals = [coef.get(f"T:{g}", 0) + coef.get(f"P:{g}", 0) for g in gs]
            if len(gs) >= 5:
                agg[k] = np.sum(vals) / np.sqrt(len(gs))
        tops["EN-concat (pathway-aggregated)"].append(set(pd.Series(agg).sort_values(ascending=False).index[:20]))
        U = m.encoder.transform(D, bidx_u)["U"]
        yy = s.loc[bidx_u, "high_grade"].values
        au = U.apply(lambda col: abs(roc_auc_score(yy, col) - 0.5))
        tops["Univariate U-score"].append(set(au.sort_values(ascending=False).index[:20]))
        say(f"stability bootstrap {bi}")
    rows = []
    for k, L in tops.items():
        jac = [len(a & b) / len(a | b) for i, a in enumerate(L) for b in L[i + 1:]]
        rows.append({"ranking": k, "mean_jaccard_top20": np.mean(jac), "sd": np.std(jac)})
    pd.DataFrame(rows).to_csv(OUT / "E9_stability.csv", index=False)
    freq = pd.Series([p for L in tops["LayerGap MAP"] for p in L]).value_counts() / n_boot
    freq.rename("selection_frequency").to_csv(OUT / "E9_map_selection_frequency.csv")
    print(pd.DataFrame(rows).round(3))
    return full


# ----------------------------------------------------------------------------- replication
def replication(data, kb, full):
    c, s = data["cptac"], data["cptac"]["samples"]
    D = cptac_blocks(data, with_metabolites=False)
    mp = pd.read_csv(OUT / "E9_map_scores.csv")
    top = list(mp.pathway[:25])
    enc = full.encoder
    F = enc.transform(D, list(s.index))
    rows = []
    for pw in top:
        r = {"pathway": pw, "MAP": mp.set_index("pathway").loc[pw, "MAP"]}
        for blk in ["T", "P", "U"]:
            if pw in F[blk]:
                for coh in ["INI", "EXP"]:
                    m = s.cohort == coh
                    x, yy = F[blk].loc[m, pw], s.loc[m, "high_grade"]
                    r[f"AUC_{blk}_{coh}"] = roc_auc_score(yy, x)
                    r[f"p_{blk}_{coh}"] = stats.mannwhitneyu(x[yy == 1], x[yy == 0]).pvalue
        # tumour vs adjacent normal (protein, raw log2 on shared genes, paired cases)
        genes = [g for g in enc.gene_sets.get(pw, []) if g in c["prot_normal"].columns]
        pair = sorted(set(c["prot_normal"].index) & set(c["prot_raw"].index))
        if len(genes) >= 5 and len(pair) > 10:
            t = c["prot_raw"].loc[pair, genes].mean(axis=1)
            n = c["prot_normal"].loc[pair, genes].mean(axis=1)
            r["prot_T_minus_N"] = float((t - n).median())
            r["p_T_vs_N"] = stats.wilcoxon(t, n).pvalue
        rows.append(r)
    rep = pd.DataFrame(rows)
    # replication: same direction in both sub-cohorts for U (AUC on the same side of 0.5)
    rep["U_replicates"] = ((rep.AUC_U_INI - 0.5) * (rep.AUC_U_EXP - 0.5) > 0) & (rep.p_U_INI < 0.05) & (rep.p_U_EXP < 0.05)
    rep.to_csv(OUT / "E9_replication_top_pathways.csv", index=False)
    print(rep[["pathway", "MAP", "AUC_U_INI", "AUC_U_EXP", "U_replicates"]].round(3).head(25))

    # TCGA: prognostic value of the transcript-level score of the top pathways
    from lifelines import CoxPHFitter
    st = data["tcga"]["samples"]
    Ft = enc.transform(tcga_blocks(data), list(st.index))["T"]
    ok = (st.os_days > 0).values
    trows = []
    for pw in top:
        if pw not in Ft or Ft[pw].isna().mean() > 0.2:
            continue
        d = pd.DataFrame({"x": (Ft[pw] - Ft[pw].mean()) / Ft[pw].std(), "t": st.os_days, "e": st.os_event})[ok].dropna()
        cph = CoxPHFitter().fit(d, "t", "e")
        sm = cph.summary.loc["x"]
        trows.append({"pathway": pw, "HR_per_SD": sm["exp(coef)"], "lo": sm["exp(coef) lower 95%"],
                      "hi": sm["exp(coef) upper 95%"], "p": sm["p"],
                      "AUC_grade_TCGA": roc_auc_score(st.high_grade.astype(int)[Ft[pw].notna()], Ft[pw].dropna())})
    tt = pd.DataFrame(trows)
    if len(tt):
        tt["fdr"] = multipletests(tt.p, method="fdr_bh")[1]
    tt.to_csv(OUT / "E9_tcga_top_pathways.csv", index=False)
    print(tt.round(4))

    # gene-level candidates: uncoupling residual of genes in the top pathways, replicated in both cohorts
    shared = enc.shared
    R = c["prot"].loc[s.index, shared] - c["rna"].loc[s.index, shared] * enc.beta.values
    cand = sorted(set(g for pw in top[:10] for g in enc.gene_sets.get(pw, [])))
    grows = []
    for g in cand:
        r = {"gene": g, "beta_TP": enc.beta[g]}
        for coh in ["INI", "EXP"]:
            m = s.cohort == coh
            r[f"AUC_{coh}"] = roc_auc_score(s.loc[m, "high_grade"], R.loc[m, g])
            r[f"p_{coh}"] = stats.mannwhitneyu(R.loc[m & (s.high_grade == 1), g], R.loc[m & (s.high_grade == 0), g]).pvalue
        r["pathways"] = "; ".join(pw for pw in top[:10] if g in enc.gene_sets.get(pw, []))
        grows.append(r)
    G = pd.DataFrame(grows)
    G["fdr_INI"] = multipletests(G.p_INI, method="fdr_bh")[1]
    G["fdr_EXP"] = multipletests(G.p_EXP, method="fdr_bh")[1]
    G["replicated"] = (G.fdr_INI < 0.1) & (G.fdr_EXP < 0.1) & ((G.AUC_INI - 0.5) * (G.AUC_EXP - 0.5) > 0)
    G["mean_AUC"] = (G.AUC_INI + G.AUC_EXP) / 2
    G = G.sort_values(["replicated", "mean_AUC"], key=lambda x: x if x.dtype == bool else (x - 0.5).abs(), ascending=False)
    # TCGA RNA survival and tumour-vs-normal protein change for replicated genes
    rna_t = data["tcga"]["rna"]
    extra = []
    for g in G.gene[G.replicated]:
        e = {"gene": g}
        if g in rna_t:
            d = pd.DataFrame({"x": rna_t[g], "t": st.os_days, "e": st.os_event})[ok].dropna()
            sm = CoxPHFitter().fit(d, "t", "e").summary.loc["x"]
            e.update({"TCGA_RNA_HR": sm["exp(coef)"], "TCGA_RNA_p": sm["p"]})
        if g in c["prot_normal"]:
            pair = sorted(set(c["prot_normal"].index) & set(c["prot_raw"].index))
            dd = c["prot_raw"].loc[pair, g] - c["prot_normal"].loc[pair, g]
            e.update({"log2FC_T_vs_N": dd.median(), "p_T_vs_N": stats.wilcoxon(dd).pvalue})
        extra.append(e)
    if extra:
        G = G.merge(pd.DataFrame(extra), on="gene", how="left")
    G.to_csv(OUT / "E9_gene_candidates.csv", index=False)
    print(G.head(20).round(4))


# ----------------------------------------------------------------------------- metabolite layer
def metabolite_layer(data, kb, full):
    c, s = data["cptac"], data["cptac"]["samples"]
    core = c["core"]
    D = cptac_blocks(data, with_metabolites=True)
    enc = PathwayEncoder(kb["pathways"]).fit(D, list(s.index))
    F = enc.transform(D, core)
    yy = s.loc[core, "high_grade"]
    rows = []
    for blk in ["M", "V"]:
        if blk not in F:
            continue
        for pw in F[blk].columns:
            x = F[blk][pw]
            rows.append({"block": blk, "pathway": pw, "n_mets": len(enc.met_sets[pw]), "AUC": roc_auc_score(yy, x),
                         "p": stats.mannwhitneyu(x[yy == 1], x[yy == 0]).pvalue,
                         "rho_PM": enc.gamma.get(pw, np.nan)})
    M = pd.DataFrame(rows)
    M["fdr"] = M.groupby("block").p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    M.sort_values("p").to_csv(OUT / "E9_core_metabolite_pathways.csv", index=False)
    # single metabolites vs grade in the core, tumour vs NAT
    met_raw = c["met_raw"]
    mrows = []
    for m in met_raw.columns:
        x = met_raw.loc[core, m]
        mrows.append({"metabolite": m, "AUC_grade": roc_auc_score(yy, x), "p_grade": stats.mannwhitneyu(x[yy == 1], x[yy == 0]).pvalue,
                      "log2FC_T_vs_N": met_raw[m].median() - c["met_normal"][m].median(),
                      "p_T_vs_N": stats.mannwhitneyu(met_raw[m], c["met_normal"][m]).pvalue})
    mm = pd.DataFrame(mrows)
    mm["fdr_grade"] = multipletests(mm.p_grade, method="fdr_bh")[1]
    mm.sort_values("p_grade").to_csv(OUT / "E9_core_metabolites.csv", index=False)
    print(M.sort_values("p").head(10).round(4)); print(mm.sort_values("p_grade").head(10).round(4))

    # Atlas: RNA-metabolite coupling of pathway scores and tumour-vs-normal shifts
    arows = []
    for ds, A in data["atlas"].items():
        Da = {"T": A["rna"], "P": A["rna"], "M": A["met"]}  # P placeholder: atlas has no proteome
        tum = list(A["tn"].index[A["tn"] == "Tumor"])
        nor = list(A["tn"].index[A["tn"] == "Normal"])
        e2 = PathwayEncoder(kb["pathways"], use_uncoupling=False)
        e2.shared = sorted(A["rna"].columns)
        sh = set(e2.shared)
        e2.gene_sets = {k: [g for g in v["genes"] if g in sh] for k, v in kb["pathways"].items()}
        e2.gene_sets = {k: v for k, v in e2.gene_sets.items() if len(v) >= 5}
        e2.met_sets = {k: [m for m in v["mets"] if m in A["met"].columns] for k, v in kb["pathways"].items()}
        e2.met_sets = {k: v for k, v in e2.met_sets.items() if len(v) >= 2 and k in e2.gene_sets}
        allx = list(A["tn"].index)
        Tm = e2._mean_scores(A["rna"].loc[allx], e2.gene_sets)
        Mm = e2._met_scores(A["met"].loc[allx])
        for pw in Mm.columns:
            r = {"dataset": ds, "pathway": pw, "n_tumour": len(tum), "n_normal": len(nor),
                 "rho_TM_tumour": stats.spearmanr(Tm.loc[tum, pw], Mm.loc[tum, pw])[0]}
            if len(nor) >= 10:
                r["rho_TM_normal"] = stats.spearmanr(Tm.loc[nor, pw], Mm.loc[nor, pw])[0]
                r["M_T_minus_N"] = Mm.loc[tum, pw].median() - Mm.loc[nor, pw].median()
                r["p_M_T_vs_N"] = stats.mannwhitneyu(Mm.loc[tum, pw], Mm.loc[nor, pw]).pvalue
            arows.append(r)
    At = pd.DataFrame(arows)
    At.to_csv(OUT / "E9_atlas_pathways.csv", index=False)
    # Fisher-z meta-analysis of tumour RNA-metabolite coupling per pathway
    def fz(g):
        z = np.arctanh(np.clip(g.rho_TM_tumour, -0.999, 0.999))
        w = g.n_tumour - 3
        zm = (z * w).sum() / w.sum()
        se = 1 / np.sqrt(w.sum())
        return pd.Series({"rho_meta": np.tanh(zm), "p_meta": 2 * stats.norm.sf(abs(zm / se)), "k": len(g)})
    meta = At.groupby("pathway").apply(fz).reset_index()
    meta["fdr"] = multipletests(meta.p_meta, method="fdr_bh")[1]
    meta.sort_values("p_meta").to_csv(OUT / "E9_atlas_coupling_meta.csv", index=False)
    print(meta.sort_values("p_meta").head(10).round(4))


if __name__ == "__main__":
    data, kb = load()
    say("batch"); batch_analysis(data)
    say("coupling"); coupling_analysis(data, kb)
    say("prioritisation"); full = prioritisation(data, kb)
    say("replication"); replication(data, kb, full)
    say("metabolite layer"); metabolite_layer(data, kb, full)
