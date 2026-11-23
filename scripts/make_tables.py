"""Write LaTeX result tables (manuscript/tables/*.tex) directly from results/metrics/*.csv.

Every number in a result table is a single value; variability is given in its own column.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results/metrics"
T = ROOT / "manuscript/tables"
T.mkdir(parents=True, exist_ok=True)


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("HALLMARK\\_", "")


def pw(p, n=44):
    p = p.replace("HALLMARK_", "").replace("_", " ")
    p = p.title() if p.isupper() else p
    p = p if len(p) <= n else p[: n - 1] + "..."
    return esc(p)


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def fp(p):
    if pd.isna(p):
        return "--"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def write(name, caption, label, header, rows, colspec, size=r"\scriptsize", note=None, pre=None):
    lines = [r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
             r"\begin{adjustbox}{max width=\textwidth}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"] + ([pre] if pre else []) + [" & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, r)) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines += [rf"\par\vspace{{2pt}}\parbox{{0.95\textwidth}}{{\tiny {note}}}"]
    lines += [r"\end{table}"]
    (T / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("wrote", name)


def t_main():
    f = M / "E1_main_cv_folds.csv"
    if not (f.exists() and (M / "E1_main_cv_tests.csv").exists()):
        return
    df = pd.read_csv(f)
    tests = pd.read_csv(M / "E1_main_cv_tests.csv")
    g = df.groupby("method")
    order = g.AUC.mean().sort_values(ascending=False).index
    rows = []
    for m in order:
        d = g.get_group(m)
        tt = tests[(tests.baseline == m) & (tests.metric == "AUC")]
        rows.append([r"\textbf{LayerGap}" if m == "LayerGap" else esc(m), f3(d.AUC.mean()), f3(d.AUC.std()), f3(d.PR_AUC.mean()),
                     f3(d.BalAcc.mean()), f3(d.MCC.mean()), f3(d.MCC.std()), f3(d.Brier.mean()),
                     "--" if m == "LayerGap" else fp(tt.p_holm.iloc[0]),
                     f"{d.fit_s.mean():.2f}", f"{d.infer_ms_per_sample.mean():.2f}"])
    write("tab_main_cv", "Repeated five-fold cross-validation (five repeats, 25 test folds) on 212 CPTAC patients.", "tab:main",
          ["Method", "AUC", "AUC SD", "PR-AUC", "Bal. acc.", "MCC", "MCC SD", "Brier", r"$P_{\mathrm{Holm}}$ (AUC)", "Train (s)", "Infer (ms/pt)"],
          rows, "lcccccccccc",
          note=r"Values are means over 25 test folds; SD, standard deviation across folds. $P_{\mathrm{Holm}}$: corrected resampled $t$-test of the AUC difference between LayerGap and each baseline, Holm-adjusted over eight baselines. Training and inference times were measured on two CPU cores and include inner tuning.")


def t_cross():
    f = M / "E2_cross_cohort.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    rows = []
    for m in d[d.train == "INI"].sort_values("AUC", ascending=False).method:
        a = d[(d.train == "INI") & (d.method == m)].iloc[0]
        b = d[(d.train == "EXP") & (d.method == m)].iloc[0]
        rows.append([r"\textbf{LayerGap}" if m == "LayerGap" else esc(m), f3(a.AUC), f"{a.AUC_lo:.3f}--{a.AUC_hi:.3f}", f3(a.MCC), f3(a.BalAcc),
                     f3(b.AUC), f"{b.AUC_lo:.3f}--{b.AUC_hi:.3f}", f3(b.MCC), f3(b.BalAcc)])
    write("tab_cross", "Cross-cohort transfer between the two CPTAC sub-cohorts.", "tab:cross",
          ["Method", "AUC", "95\\% CI", "MCC", "Bal. acc.", "AUC", "95\\% CI", "MCC", "Bal. acc."], rows, "lcccccccc",
          pre=r" & \multicolumn{4}{c}{Train INI ($n$=103) $\rightarrow$ test EXP ($n$=109)} & \multicolumn{4}{c}{Train EXP $\rightarrow$ test INI} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
          note=r"Columns 2--5: training on INI, testing on EXP; columns 6--9: training on EXP, testing on INI. CI, 2,000 bootstrap resamples of test patients.")


def t_tcga():
    f = M / "E3_tcga_external.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).sort_values("AUC", ascending=False)
    rows = [[r"\textbf{" + esc(r.method) + "}" if r.method.startswith("LayerGap") else esc(r.method), f3(r.AUC), f"{r.AUC_lo:.3f}--{r.AUC_hi:.3f}",
             f3(r.PR_AUC), f3(r.MCC), f3(r.Brier), f3(r.OS_Cindex), "--" if r.method == "LayerGap" else fp(r.p_paired)] for r in d.itertuples()]
    write("tab_tcga", "External test in TCGA-KIRC (502 tumours; genomic and transcript layers only) after training on 212 CPTAC patients.", "tab:tcga",
          ["Method", "AUC", "95\\% CI", "PR-AUC", "MCC", "Brier", "OS C-index", r"$P$ vs LayerGap"], rows, "lccccccc",
          note=r"LayerGap was trained with all CPTAC blocks; at TCGA only its $G$ and $T$ blocks are available. LayerGap ($G$+$T$ only) was trained with those two blocks. $P$: paired bootstrap test of the AUC difference.")
    s = pd.read_csv(M / "E3_tcga_survival.csv", index_col=0).iloc[:, 0]
    rows = [["Univariable", f"{s['uni_HR']:.2f}", f"{s['uni_HR_lo']:.2f}--{s['uni_HR_hi']:.2f}", fp(s["uni_p"])],
            ["Adjusted for stage (III/IV vs I/II)", f"{s['multi_HR']:.2f}", f"{s['multi_HR_lo']:.2f}--{s['multi_HR_hi']:.2f}", fp(s["multi_p"])],
            ["Adjusted for stage and grade", f"{s['multi_grade_HR']:.2f}", f"{s['multi_grade_HR_lo']:.2f}--{s['multi_grade_HR_hi']:.2f}", fp(s["multi_grade_p"])]]
    write("tab_survival", f"Cox regression of overall survival on the LayerGap risk score in TCGA-KIRC ({int(s['n'])} patients, {int(s['events'])} deaths).",
          "tab:survival", ["Model", "HR per SD", "95\\% CI", "$P$"], rows, "lccc",
          note=f"Log-rank test for median split of the score: $P$ = {fp(s['logrank_p'])}.")


def t_general():
    if not ((M / "E2_cross_cohort.csv").exists() and (M / "E3_tcga_external.csv").exists()):
        return
    cc = pd.read_csv(M / "E2_cross_cohort.csv")
    tc = pd.read_csv(M / "E3_tcga_external.csv").set_index("method")
    methods = ["LayerGap", "LayerGap (G+T only)"] + [m for m in tc.sort_values("AUC", ascending=False).index if not m.startswith("LayerGap")]
    rows = []
    for m in methods:
        r = [r"\textbf{" + esc(m) + "}" if m.startswith("LayerGap") else esc(m)]
        for tr in ["INI", "EXP"]:
            x = cc[(cc.train == tr) & (cc.method == m)]
            r += [f"{x.AUC.iloc[0]:.3f} ({x.AUC_lo.iloc[0]:.3f}--{x.AUC_hi.iloc[0]:.3f})"] if len(x) else ["--"]
        t = tc.loc[m]
        r += [f"{t.AUC:.3f} ({t.AUC_lo:.3f}--{t.AUC_hi:.3f})", f3(t.OS_Cindex)]
        rows.append(r)
    write("tab_general", "Generalisation across cohorts: AUC (95\\% bootstrap CI) for cross-cohort transfer within CPTAC and for external testing in TCGA-KIRC.",
          "tab:general", ["Method", "INI $\\rightarrow$ EXP", "EXP $\\rightarrow$ INI", "CPTAC $\\rightarrow$ TCGA", "TCGA OS C-index"], rows, "lcccc",
          note="INI$\\rightarrow$EXP: trained on 103 INI patients, tested on 109 EXP patients, and vice versa. CPTAC$\\rightarrow$TCGA: trained on all 212 CPTAC patients, tested on 502 TCGA-KIRC tumours without proteome. LayerGap ($G$+$T$ only) was trained with the genomic and transcript blocks only. MCC, balanced accuracy and paired tests are given in Tables~\\ref{tab:cross} and \\ref{tab:tcga}.")


def t_ablation():
    f = M / "E7_ablation_summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    rows = [[esc(r.variant), f3(r.AUC), f3(r.AUC_sd), f3(r.PR_AUC), f3(r.MCC), f3(r.Brier), "--" if r.variant == "Full LayerGap" else f"{r.dAUC_vs_full:+.3f}",
             fp(r.p_vs_full)] for r in d.itertuples()]
    write("tab_ablation", "Ablation of LayerGap components under the folds of the primary evaluation.", "tab:ablation",
          ["Variant", "AUC", "AUC SD", "PR-AUC", "MCC", "Brier", r"$\Delta$AUC", "$P$"], rows, "lccccccc",
          note=r"$\Delta$AUC relative to the full model; $P$: corrected resampled $t$-test over 25 paired folds.")


def t_core():
    f = M / "E4_core_summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).sort_values("mean", ascending=False)
    rows = [[esc(r.method), f3(r["mean"]), f3(r["std"])] for _, r in d.iterrows()]
    write("tab_core", "AUC in the four-layer core (50 patients with matched metabolomics) under pooled repeated cross-validation.", "tab:core",
          ["Method", "AUC", "SD (5 repeats)"], rows, "lcc",
          note="Models were trained on all training patients in each fold; metabolomics was available only for core patients. AUC was computed on core patients pooled over the five folds of each repeat.")


def t_missing():
    f = M / "E5_missing_layers.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    g = d.groupby(["method", "layer_masked", "fraction"]).AUC.mean().unstack([1, 2])
    order = ["LayerGap", "Late-fusion LR", "TMC", "XGB-concat", "EN-concat"]
    rows = []
    for m in order:
        rows.append([r"\textbf{LayerGap}" if m == "LayerGap" else esc(m)] + [f3(g.loc[m, (l, fr)]) for l in ["P", "T"] for fr in [0.0, 0.5, 1.0]])
    write("tab_missing", "AUC when the protein or RNA layer is unavailable for a fraction of test patients (10 folds).", "tab:missing",
          ["Method", "P 0\\%", "P 50\\%", "P 100\\%", "T 0\\%", "T 50\\%", "T 100\\%"], rows, "lcccccc",
          note="Models were trained on complete data; the indicated fraction of test patients had the layer removed. Baselines fill a missing layer with the training mean; LayerGap gives the block zero weight.")


def t_pathways():
    f = M / "E9_map_scores.csv"
    if not f.exists():
        return
    mp = pd.read_csv(f).head(12)
    rep = pd.read_csv(M / "E9_replication_top_pathways.csv").set_index("pathway")
    tc = pd.read_csv(M / "E9_tcga_top_pathways.csv").set_index("pathway")
    freq = pd.read_csv(M / "E9_map_selection_frequency.csv", index_col=0).iloc[:, 0]
    rows = []
    for k, r in enumerate(mp.itertuples(), start=1):
        rr = rep.loc[r.pathway] if r.pathway in rep.index else None
        t = tc.loc[r.pathway] if r.pathway in tc.index else None
        rows.append([k, pw(r.pathway, 40), f3(r.MAP), f"{r.concordance:.2f}", f"{freq.get(r.pathway, 0):.2f}",
                     f3(rr.AUC_U_INI) if rr is not None else "--", f3(rr.AUC_U_EXP) if rr is not None else "--",
                     "yes" if rr is not None and bool(rr.U_replicates) else "no",
                     f"{t.HR_per_SD:.2f}" if t is not None else "--", fp(t.fdr) if t is not None else "--"])
    write("tab_pathways", "Top pathways ranked by the MAP score of the full CPTAC model, with replication and prognostic evidence.", "tab:pathways",
          ["Rank", "Pathway", "MAP", "Conc.", "Sel. freq.", "AUC$_U$ INI", "AUC$_U$ EXP", "Replicated", "TCGA HR", "TCGA FDR"],
          rows, "clcccccccc",
          note="Conc., cross-layer sign concordance; Sel. freq., fraction of 30 patient bootstraps in which the pathway entered the MAP top 20; AUC$_U$, AUC of the uncoupling score for high grade (values below 0.5 indicate lower uncoupling in high grade); Replicated, $P<0.05$ in both sub-cohorts with the same direction; TCGA HR, hazard ratio per SD of the transcript pathway score for overall survival.")


def t_genes():
    f = M / "E9_gene_candidates.csv"
    if not f.exists():
        return
    G = pd.read_csv(f)
    G = G[(G.p_INI < 0.05) & (G.p_EXP < 0.05) & ((G.AUC_INI - 0.5) * (G.AUC_EXP - 0.5) > 0)].copy()
    G["eff"] = (G.mean_AUC - 0.5).abs()
    G = G.sort_values("eff", ascending=False).head(14)
    rows = [[r.gene, f"{r.beta_TP:.2f}", f3(r.AUC_INI), fp(r.p_INI), f3(r.AUC_EXP), fp(r.p_EXP), pw(str(r.pathways).split(";")[0], 30)] for r in G.itertuples()]
    write("tab_genes", "Genes whose protein-level uncoupling was associated with high grade in both CPTAC sub-cohorts.", "tab:genes",
          ["Gene", "$b_g$", "AUC INI", "$P$ INI", "AUC EXP", "$P$ EXP", "Pathway"], rows, "lcccccl",
          note="Genes from the ten top-ranked pathways with nominal $P<0.05$ (Mann--Whitney) in both sub-cohorts and concordant direction. $b_g$, transcript--protein coupling coefficient. AUC above 0.5: protein exceeds the transcript-based expectation in high-grade tumours.")


def t_metabolites():
    f = M / "E9_core_metabolites.csv"
    if not f.exists():
        return
    mm = pd.read_csv(f).head(10)
    try:
        import pickle
        nm = pickle.load(open(ROOT / "data/processed/cohorts.pkl", "rb"))["cptac"]["met_name_map"]
        lookup = nm.drop_duplicates("normalised").set_index("normalised").original
        mm["metabolite"] = mm.metabolite.map(lambda x: lookup.get(x, x))
    except Exception:
        pass
    rows = [[esc(r.metabolite), f3(r.AUC_grade), fp(r.p_grade), fp(r.fdr_grade), f"{r.log2FC_T_vs_N:+.2f}", fp(r.p_T_vs_N)] for r in mm.itertuples()]
    write("tab_metabolites", "Metabolites most strongly associated with high grade in the four-layer core (50 tumours) and their tumour--normal shift.", "tab:metabolites",
          ["Metabolite", "AUC (grade)", "$P$", "FDR", "log$_2$FC T vs NAT", "$P$ T vs NAT"], rows, "lccccc",
          note="AUC above 0.5: higher in high-grade tumours. Tumour--NAT comparison uses the 50 tumours and 7 adjacent normal tissues (Mann--Whitney).")
    meta = pd.read_csv(M / "E9_atlas_coupling_meta.csv").sort_values("p_meta").head(8)
    rows = [[pw(r.pathway, 48), f"{r.rho_meta:+.3f}", fp(r.p_meta), fp(r.fdr), int(r.k)] for r in meta.itertuples()]
    write("tab_atlas", "Pathway-level RNA--metabolite coupling in tumours of the four CAMP ccRCC cohorts (Fisher $z$ meta-analysis).", "tab:atlas",
          ["Pathway", r"$\rho_{\mathrm{meta}}$", "$P$", "FDR", "Cohorts"], rows, "lcccc")


def t_sensitivity():
    f = M / "E8_sensitivity.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    names = {"n_boot": "Bootstrap fits $B$", "tau": r"Precision floor $\tau$", "min_genes": "Minimum pathway genes",
             "C (fixed)": "Fixed block $C$", "stack_l2": "Stacker penalty"}
    rows = []
    for p, lab in names.items():
        g = s[s.parameter == p].groupby("value").AUC.agg(["mean", "std"])
        rows.append([lab, ", ".join(f"{v:g}" for v in g.index), ", ".join(f"{x:.3f}" for x in g["mean"]),
                     f"{g['mean'].max() - g['mean'].min():.3f}"])
    tf = s[s.parameter.str.startswith("train_fraction")].copy()
    for m, g in tf.groupby(tf.parameter.str.split("::").str[1]):
        gg = g.groupby("value").AUC.mean()
        rows.append([f"Training fraction ({esc(m)})", ", ".join(f"{v:g}" for v in gg.index), ", ".join(f"{x:.3f}" for x in gg.values),
                     f"{gg.max() - gg.min():.3f}"])
    write("tab_sensitivity", "Sensitivity of the cross-validated AUC to LayerGap parameters and to training-set size (10 folds).", "tab:sensitivity",
          ["Parameter", "Values", "Mean AUC per value", "Range"], rows, "llll")


def t_runtime():
    f = M / "E10_runtime_layergap.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    rows = []
    for r in d.itertuples():
        val = f"{r.MB:.1f} MB" if "memory" in r.stage.lower() else (f"{r.seconds * 1000:.1f} ms" if r.seconds < 1 else f"{r.seconds:.2f} s")
        per = "--" if pd.isna(getattr(r, "ms_per_sample", np.nan)) else f"{r.ms_per_sample:.2f}"
        rows.append([esc(r.stage), val, per])
    write("tab_runtime", "Runtime profile of LayerGap stages (median of repeated runs, two CPU cores, 170 training and 42 test patients).", "tab:runtime",
          ["Stage", "Time / memory", "ms per patient"], rows, "lcc")


def t_synthetic():
    f = M / "E6_synthetic.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    methods = ["LayerGap", "EN-concat", "Late-fusion LR", "XGB-concat", "TMC"]
    rows = []
    for (grp, sc), g in d.groupby(["group", "scenario"], sort=False):
        a = g.groupby("method").AUC.mean()
        rec = g.groupby("method").pathway_precision_at_10.mean()
        rows.append([esc(grp), esc(sc)] + [f3(a.get(m)) for m in methods] +
                    [f"{rec.get('LayerGap'):.2f}" if not pd.isna(rec.get("LayerGap", np.nan)) else "--",
                     f"{rec.get('EN-concat'):.2f}" if not pd.isna(rec.get("EN-concat", np.nan)) else "--"])
    write("tab_synthetic", "Synthetic stress tests derived from the CPTAC data with planted pathway signals (mean of three seeds; 600 synthetic test patients).",
          "tab:synthetic", ["Group", "Scenario", "LayerGap", "EN-concat", "Late-fusion", "XGB", "TMC", r"Prec@10 LG", r"Prec@10 EN"], rows, "llccccccc",
          note="Synthetic data were generated from the real CPTAC transcript and protein matrices (low-rank Gaussian models with real gene-wise coupling); signal was planted at the protein level in five pathways and at the transcript level in five others. Prec@10: fraction of the ten true signal pathways among the top ten ranked pathways (LayerGap MAP; pathway-aggregated elastic-net coefficients).")


if __name__ == "__main__":
    for fn in [t_main, t_general, t_cross, t_tcga, t_ablation, t_core, t_missing, t_pathways, t_genes, t_metabolites, t_sensitivity, t_runtime, t_synthetic]:
        fn()
