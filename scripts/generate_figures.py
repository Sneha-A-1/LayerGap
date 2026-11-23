"""Generate all result figures from results/metrics/*.csv (run after the experiments).

    python scripts/generate_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results/metrics"
FIG = ROOT / "manuscript/figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#6b6b6b", "axes.labelcolor": "#222222", "xtick.color": "#444444",
                     "ytick.color": "#444444", "axes.grid": False, "legend.frameon": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PROPOSED, BASE = "#2a78d6", "#9a9994"
LAYER = {"G": "#2a78d6", "T": "#1baf7a", "P": "#eb6834", "U": "#6250d6", "M": "#e87ba4", "V": "#eda100"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def panel(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=11, weight="bold", va="bottom")


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def short(p, n=46):
    p = p.replace("HALLMARK_", "").replace("_", " ")
    p = p.title() if p.isupper() else p
    return p if len(p) <= n else p[: n - 1] + "…"


# ----------------------------------------------------------------------------- Fig 2 study design
def fig_design():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    cohorts = [("CPTAC INI (discovery)", "n = 103", "GTP"), ("CPTAC EXP (confirmatory)", "n = 109", "GTP"),
               ("CPTAC four-layer core (EXP)", "n = 50", "GTPM"), ("TCGA-KIRC", "n = 502", "GT"),
               ("CAMP ccRCC1–4", "n = 242 (172 T, 70 N)", "TM")]
    layers = ["G", "T", "P", "M"]
    names = {"G": "Genomics", "T": "Transcriptomics", "P": "Proteomics", "M": "Metabolomics"}
    x0, y0, cw, rh = 3.15, 4.3, 0.62, 0.68
    ax.text(0.15, 4.85, "Cohort", fontsize=9, weight="bold")
    for j, l in enumerate(layers):
        ax.text(x0 + j * cw + cw / 2 - 0.1, 4.62, names[l], rotation=28, ha="left", va="bottom", fontsize=7.2, weight="bold", color=LAYER[l])
    for i, (c, n, have) in enumerate(cohorts):
        y = y0 - i * rh
        ax.text(0.15, y - 0.12, c, fontsize=8.2, color=INK, weight="bold")
        ax.text(0.15, y - 0.36, n, fontsize=7.2, color=INK2)
        for j, l in enumerate(layers):
            on = l in have
            ax.add_patch(Rectangle((x0 + j * cw + 0.06, y - 0.5), cw - 0.12, rh - 0.18, fc=LAYER[l] if on else "#f1f0ec",
                                   ec="white", lw=2))
            ax.text(x0 + j * cw + cw / 2, y - 0.2, "✓" if on else "–", ha="center", va="center", fontsize=9,
                    color="white" if on else "#9a9994")
    # experiments
    exps = [("E1, E7, E8", "Repeated 5×5 CV, ablation,\nsensitivity", [0, 1]), ("E2", "INI ↔ EXP transfer", [0, 1]),
            ("E4, E9", "Four-layer contribution,\nmetabolite biology", [2]), ("E3", "External test, survival,\nmissing P", [3]),
            ("E9", "Gene–metabolite\nreplication", [4]), ("E5, E6", "Masked layers; synthetic\ncohorts from CPTAC", [0, 1])]
    ex, ew = 6.25, 2.55
    for k, (lab, desc, rows) in enumerate(exps):
        col = k % 2
        r = k // 2
        bx, by = ex + col * (ew + 0.25), 4.25 - r * 1.2
        ax.add_patch(FancyBboxPatch((bx, by - 0.78), ew, 0.88, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#f7f6f2", ec="#bdbcb6"))
        ax.text(bx + 0.12, by - 0.05, lab, fontsize=8.2, weight="bold", color=PROPOSED, va="top")
        ax.text(bx + 0.12, by - 0.33, desc, fontsize=7.2, color=INK, va="top")
    ax.annotate("", xy=(6.15, 3.2), xytext=(5.75, 3.2), arrowprops=dict(arrowstyle="-|>", color="#555555"))
    ax.text(0.15, 0.55, "Knowledge: Human-GEM v2.1.1 (113 metabolic subsystems, 26,372 enzyme–metabolite links) and 50 MSigDB Hallmark sets → 163 pathways.\n"
            "Task: high grade (G3/G4) vs low grade (G1/G2). All supervised steps inside patient-level training folds.", fontsize=7.6, color=INK2)
    save(fig, "fig2_study_design")


# ----------------------------------------------------------------------------- Fig 3 main CV
def fig_main():
    f = M / "E1_main_cv_folds.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    order = df.groupby("method").AUC.mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    data = [df[df.method == m].AUC.values for m in order]
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color="#777"), capprops=dict(color="#777"))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(PROPOSED if m == "LayerGap" else "#dcdbd6")
        patch.set_edgecolor("#555555")
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(d, i + 1 + rng.uniform(-0.15, 0.15, len(d)), s=6, color=INK2, alpha=0.5, zorder=3, lw=0)
    for i, m in enumerate(order):
        ax.text(1.005, i + 1, f"{np.mean(data[i]):.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK)
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.set_xlabel("AUC (25 test folds)")
    xgrid(ax)
    panel(ax, "A")
    ax = axes[1]
    mets = ["AUC", "PR_AUC", "BalAcc", "MCC"]
    g = df.groupby("method")[mets].agg(["mean", "std"])
    for k, met in enumerate(mets):
        for i, m in enumerate(order):
            mu, sd = g.loc[m, (met, "mean")], g.loc[m, (met, "std")]
            y = i + (k - 1.5) * 0.18
            col = CAT[k]
            ax.errorbar(mu, y, xerr=sd / np.sqrt(25), fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=0.8,
                        label=met.replace("_", "-").replace("BalAcc", "Balanced acc.") if i == 0 else None)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Mean ± s.e. across folds")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=7)
    xgrid(ax)
    panel(ax, "B")
    fig.tight_layout()
    save(fig, "fig3_main_cv")


# ----------------------------------------------------------------------------- Fig 4 generalisation
def km(ax, t, e, label, col):
    from lifelines import KaplanMeierFitter
    k = KaplanMeierFitter().fit(t / 365.25, e, label=label)
    k.plot_survival_function(ax=ax, ci_show=True, color=col, lw=1.6, ci_alpha=0.12)


def fig_generalisation():
    if not (M / "E2_cross_cohort.csv").exists():
        return
    cc = pd.read_csv(M / "E2_cross_cohort.csv")
    tc = pd.read_csv(M / "E3_tcga_external.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    methods = cc[cc.train == "INI"].sort_values("AUC").method.tolist()
    for k, (tr, te, col, mk) in enumerate([("INI", "EXP", CAT[0], "o"), ("EXP", "INI", CAT[1], "s")]):
        sub = cc[cc.train == tr].set_index("method").loc[methods]
        y = np.arange(len(methods)) + (k - 0.5) * 0.28
        ax.errorbar(sub.AUC, y, xerr=[sub.AUC - sub.AUC_lo, sub.AUC_hi - sub.AUC], fmt=mk, ms=4, color=col,
                    elinewidth=0.9, label=f"train {tr} → test {te}")
    ax.set_yticks(range(len(methods)), methods)
    for t in ax.get_yticklabels():
        if t.get_text() == "LayerGap":
            t.set_weight("bold")
    ax.set_xlabel("AUC (95% bootstrap CI)")
    ax.legend(loc="lower left", fontsize=7)
    xgrid(ax)
    panel(ax, "A")
    ax = axes[1]
    tc = tc.sort_values("AUC")
    cols = [PROPOSED if m.startswith("LayerGap") else BASE for m in tc.method]
    ax.errorbar(tc.AUC, range(len(tc)), xerr=[tc.AUC - tc.AUC_lo, tc.AUC_hi - tc.AUC], fmt="none", ecolor="#888", elinewidth=0.9)
    ax.scatter(tc.AUC, range(len(tc)), c=cols, s=26, zorder=3)
    ax.set_yticks(range(len(tc)), tc.method)
    ax.set_xlabel("TCGA-KIRC AUC (95% CI), no proteome")
    xgrid(ax)
    panel(ax, "B")
    ax = axes[2]
    p = pd.read_csv(M / "E3_tcga_predictions.csv", index_col=0)
    p = p[p.os_days > 0]
    hi = p.LayerGap >= p.LayerGap.median()
    km(ax, p.os_days[~hi], p.os_event[~hi], f"Low score (n={int((~hi).sum())})", CAT[0])
    km(ax, p.os_days[hi], p.os_event[hi], f"High score (n={int(hi.sum())})", CAT[1])
    sv = pd.read_csv(M / "E3_tcga_survival.csv", index_col=0).iloc[:, 0]
    ax.text(0.03, 0.06, f"log-rank P = {sv['logrank_p']:.1e}\nHR per SD = {sv['uni_HR']:.2f} "
            f"({sv['uni_HR_lo']:.2f}–{sv['uni_HR_hi']:.2f})\nstage-adjusted HR = {sv['multi_HR']:.2f}", transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Years")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=7)
    ygrid(ax)
    panel(ax, "C")
    fig.tight_layout()
    save(fig, "fig4_generalisation")


# ----------------------------------------------------------------------------- Fig 5 robustness
def fig_robustness():
    if not (M / "E5_missing_layers.csv").exists():
        return
    d = pd.read_csv(M / "E5_missing_layers.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    methods = ["LayerGap", "EN-concat", "Late-fusion LR", "XGB-concat", "TMC"]
    cols = dict(zip(methods, [PROPOSED, CAT[1], CAT[2], CAT[3], CAT[4]]))
    for k, layer in enumerate(["P", "T"]):
        ax = axes[k]
        g = d[d.layer_masked == layer].groupby(["method", "fraction"]).AUC.agg(["mean", "std", "count"]).reset_index()
        for m in methods:
            s = g[g.method == m]
            ax.plot(s.fraction * 100, s["mean"], marker="o", ms=4, lw=2 if m == "LayerGap" else 1.4, color=cols[m], label=m)
            ax.fill_between(s.fraction * 100, s["mean"] - s["std"] / np.sqrt(s["count"]), s["mean"] + s["std"] / np.sqrt(s["count"]),
                            color=cols[m], alpha=0.1, lw=0)
        ax.set_xlabel(f"Test patients without {'proteome' if layer == 'P' else 'transcriptome'} (%)")
        ax.set_ylabel("AUC")
        ax.set_xticks([0, 25, 50, 75, 100])
        ygrid(ax)
        panel(ax, "AB"[k])
        if k == 0:
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.36, -0.04))
    ax = axes[2]
    w = pd.read_csv(M / "E9_block_weights_full.csv", index_col=0).iloc[:, 0]
    ax.bar(w.index, w.values, color=[LAYER[b] for b in w.index], width=0.6)
    for i, v in enumerate(w.values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(w)), [{"G": "Genomic", "T": "Transcript", "P": "Protein", "U": "Uncoupling"}[b] for b in w.index])
    ax.set_ylabel("Fusion weight α_b (full CPTAC model)")
    ygrid(ax)
    panel(ax, "C")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig5_robustness")


# ----------------------------------------------------------------------------- Fig 6 ablation & sensitivity
def fig_ablation():
    if not (M / "E7_ablation_folds.csv").exists():
        return
    d = pd.read_csv(M / "E7_ablation_folds.csv")
    full = d[d.variant == "Full LayerGap"].sort_values(["repeat", "fold"]).AUC.values
    rows = []
    for v in d.variant.unique():
        x = d[d.variant == v].sort_values(["repeat", "fold"]).AUC.values - full
        rows.append((v, x.mean(), x.std(ddof=1) / np.sqrt(len(x)), d[d.variant == v].AUC.mean()))
    r = pd.DataFrame(rows, columns=["variant", "d", "se", "auc"]).sort_values("d")
    sens = M / "E8_sensitivity.csv"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    cols = [PROPOSED if v == "Full LayerGap" else ("#c8553d" if dd < 0 else "#7a9e7e") for v, dd in zip(r.variant, r.d)]
    ax.barh(r.variant, r.d, xerr=1.96 * r.se, color=cols, height=0.55, error_kw=dict(ecolor="#555", lw=0.8))
    for i, (dd, a) in enumerate(zip(r.d, r.auc)):
        ax.text(1.01, i, f"AUC {a:.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.8, color=INK2)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("ΔAUC vs full LayerGap (paired folds, 95% CI)")
    xgrid(ax)
    panel(ax, "A")
    ax = axes[1]
    if sens.exists():
        s = pd.read_csv(sens)
        tf = s[s.parameter.str.startswith("train_fraction")].copy()
        tf["method"] = tf.parameter.str.split("::").str[1]
        cm = {"LayerGap": PROPOSED, "EN-concat": CAT[1], "Late-fusion LR": CAT[2]}
        for m, g in tf.groupby("method"):
            gg = g.groupby("value").AUC.agg(["mean", "std", "count"])
            ax.errorbar(gg.index * 100 + {"LayerGap": 0, "EN-concat": -1.2, "Late-fusion LR": 1.2}.get(m, 0), gg["mean"],
                        yerr=gg["std"] / np.sqrt(gg["count"]), marker="o", ms=4, color=cm.get(m, BASE),
                        lw=2 if m == "LayerGap" else 1.3, elinewidth=0.8, capsize=2, label=m)
        ax.set_xlabel("Training set used (% of training fold)")
        ax.set_ylabel("AUC (mean ± s.e., 10 folds)")
        ax.legend(fontsize=7, loc="lower right")
        ygrid(ax)
    panel(ax, "B")
    fig.tight_layout()
    save(fig, "fig6_ablation")


def fig_sensitivity():
    if not (M / "E8_sensitivity.csv").exists():
        return
    s = pd.read_csv(M / "E8_sensitivity.csv")
    s = s[~s.parameter.str.startswith("train_fraction")]
    pars = ["n_boot", "tau", "min_genes", "C (fixed)", "stack_l2"]
    labels = {"n_boot": "Bootstrap fits B", "tau": "Precision floor τ", "min_genes": "Minimum pathway genes",
              "C (fixed)": "Fixed block C", "stack_l2": "Stacker penalty"}
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8), sharey=True)
    for ax, p in zip(axes, pars):
        g = s[s.parameter == p].groupby("value").AUC.agg(["mean", "std", "count"])
        x = np.arange(len(g))
        ax.errorbar(x, g["mean"], yerr=g["std"] / np.sqrt(g["count"]), marker="o", ms=4, color=PROPOSED, capsize=2, lw=1.6)
        ax.set_xticks(x, [f"{v:g}" for v in g.index])
        ax.set_title(labels[p], fontsize=8)
        ygrid(ax)
    axes[0].set_ylabel("AUC (mean ± s.e.)")
    fig.tight_layout()
    save(fig, "fig_s1_sensitivity")


# ----------------------------------------------------------------------------- Fig 7 biology
def fig_biology():
    if not (M / "E9_map_scores.csv").exists():
        return
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1], height_ratios=[1, 1.05], hspace=0.42, wspace=0.55)
    # A coupling
    ax = fig.add_subplot(gs[0, 0])
    gcp = pd.read_csv(M / "E9_coupling_genes.csv", index_col=0)
    bins = np.linspace(-0.6, 1, 50)
    ax.hist(gcp.r_low, bins=bins, histtype="step", lw=1.6, color=CAT[0], label=f"Low grade (median {gcp.r_low.median():.2f})")
    ax.hist(gcp.r_high, bins=bins, histtype="step", lw=1.6, color=CAT[1], label=f"High grade (median {gcp.r_high.median():.2f})")
    sm = pd.read_csv(M / "E9_coupling_summary.csv", index_col=0).iloc[:, 0]
    ax.text(0.02, 0.62, f"{int(sm['n_genes'])} genes\npermutation\nP = {sm['perm_p_two_sided']:.3f}", transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlabel("Gene-wise RNA–protein Spearman ρ")
    ax.set_ylabel("Genes")
    ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ygrid(ax)
    panel(ax, "A")
    # B MAP top 15 with block coefficients
    ax = fig.add_subplot(gs[0, 1:])
    mp = pd.read_csv(M / "E9_map_scores.csv").head(15).iloc[::-1]
    rep = pd.read_csv(M / "E9_replication_top_pathways.csv").set_index("pathway")
    y = np.arange(len(mp))
    for j, b in enumerate(["T", "P", "U"]):
        ax.scatter(np.full(len(mp), j), y, s=np.clip(np.abs(mp[f"w_{b}"].fillna(0)) * 1600, 4, 260),
                   c=np.where(mp[f"w_{b}"].fillna(0) > 0, "#c8553d", "#3b6ea8"), alpha=0.85, lw=0)
    ax.barh(y, mp.MAP / mp.MAP.max() * 2.4, left=3.4, color=PROPOSED, height=0.55)
    for yy, (pw, sc) in enumerate(zip(mp.pathway, mp.MAP)):
        star = " ★" if pw in rep.index and bool(rep.loc[pw, "U_replicates"]) else ""
        ax.text(5.95, yy, f"{sc:.3f}{star}", va="center", fontsize=6.8)
    ax.set_yticks(y, [short(p, 36) for p in mp.pathway], fontsize=7)
    ax.set_xticks([0, 1, 2, 4.6], ["T coef.", "P coef.", "U coef.", "MAP score"])
    ax.set_xlim(-0.6, 6.9)
    ax.text(0, len(mp) + 0.3, "red: higher in high grade; blue: lower", fontsize=6.6, color=INK2)
    ax.text(5.95, -1.6, "★ U-score replicated in INI and EXP", fontsize=6.6, color=INK2)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_visible(False)
    ax.tick_params(length=0)
    panel(ax, "B")
    # C replication scatter for all pathways
    ax = fig.add_subplot(gs[1, 0])
    enc_rep = pd.read_csv(M / "E9_replication_top_pathways.csv")
    ax.axhline(0.5, color="#bbb", lw=0.7)
    ax.axvline(0.5, color="#bbb", lw=0.7)
    ok = enc_rep.U_replicates.astype(bool)
    ax.scatter(enc_rep.AUC_U_INI[~ok], enc_rep.AUC_U_EXP[~ok], s=18, color=BASE, label="Top-25 MAP pathway")
    ax.scatter(enc_rep.AUC_U_INI[ok], enc_rep.AUC_U_EXP[ok], s=26, color=PROPOSED, label="Replicated (P<0.05 in both)")
    key = []
    for k, (_, r) in enumerate(enc_rep[ok].sort_values("AUC_U_EXP", ascending=False).iterrows(), start=1):
        ax.annotate(str(k), (r.AUC_U_INI, r.AUC_U_EXP), fontsize=6.5, xytext=(4, 3), textcoords="offset points", weight="bold")
        key.append(f"{k}  {short(r.pathway, 34)}")
    ax.text(0.03, 0.97, "\n".join(key), transform=ax.transAxes, va="top", fontsize=5.8, color=INK)
    ax.set_xlabel("Uncoupling-score AUC, INI")
    ax.set_ylabel("Uncoupling-score AUC, EXP")
    ax.legend(fontsize=6.5, loc="lower right")
    panel(ax, "C")
    # D gene candidates
    ax = fig.add_subplot(gs[1, 1])
    G = pd.read_csv(M / "E9_gene_candidates.csv")
    G["nominal_rep"] = (G.p_INI < 0.05) & (G.p_EXP < 0.05) & ((G.AUC_INI - 0.5) * (G.AUC_EXP - 0.5) > 0)
    top = G[G.nominal_rep].assign(eff=lambda x: (x.mean_AUC - 0.5).abs()).sort_values("eff", ascending=False).head(14).iloc[::-1]
    yy = np.arange(len(top))
    ax.scatter(top.AUC_INI, yy, color=CAT[0], s=22, label="INI", zorder=3)
    ax.scatter(top.AUC_EXP, yy, color=CAT[1], marker="s", s=22, label="EXP", zorder=3)
    for i, (a, b) in enumerate(zip(top.AUC_INI, top.AUC_EXP)):
        ax.plot([a, b], [i, i], color="#bbb", lw=1, zorder=1)
    ax.axvline(0.5, color="#888", lw=0.7)
    ax.set_yticks(yy, top.gene, fontsize=7)
    ax.set_xlabel("Gene-level uncoupling AUC for high grade")
    ax.legend(fontsize=7, loc="center", bbox_to_anchor=(0.5, 0.5))
    xgrid(ax)
    panel(ax, "D")
    # E TCGA prognostic value of top pathways (transcript level)
    ax = fig.add_subplot(gs[1, 2])
    t = pd.read_csv(M / "E9_tcga_top_pathways.csv").sort_values("HR_per_SD").tail(12)
    yy = np.arange(len(t))
    cols = [PROPOSED if f < 0.05 else BASE for f in t.fdr]
    ax.errorbar(t.HR_per_SD, yy, xerr=[t.HR_per_SD - t.lo, t.hi - t.HR_per_SD], fmt="none", ecolor="#999", elinewidth=0.8)
    ax.scatter(t.HR_per_SD, yy, c=cols, s=20, zorder=3)
    ax.axvline(1, color="#666", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1, 1.25, 1.6], ["0.6", "0.8", "1", "1.25", "1.6"])
    ax.minorticks_off()
    ax.set_yticks(yy, [short(p, 30) for p in t.pathway], fontsize=6.6)
    ax.set_xlabel("TCGA overall-survival HR per SD\n(transcript score; blue: FDR<0.05)")
    panel(ax, "E")
    save(fig, "fig7_biology")


# ----------------------------------------------------------------------------- Fig 8 metabolites
def fig_metabolites():
    if not (M / "E9_core_metabolites.csv").exists():
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    ax = axes[0]
    core = M / "E4_core_summary.csv"
    if core.exists():
        c = pd.read_csv(core).sort_values("mean")
        cols = [PROPOSED if m.startswith("LayerGap (G,T,P,U,M,V)") else ("#8fb7e6" if m.startswith("LayerGap") else BASE) for m in c.method]
        ax.barh(c.method, c["mean"], xerr=c["std"] / np.sqrt(5), color=cols, height=0.55, error_kw=dict(ecolor="#555", lw=0.8))
        for i, v in enumerate(c["mean"]):
            ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=7)
        ax.set_xlim(0.4, 0.95)
        ax.set_xlabel("AUC in the four-layer core (n=50)")
        xgrid(ax)
    panel(ax, "A")
    ax = axes[1]
    mm = pd.read_csv(M / "E9_core_metabolites.csv").head(12).iloc[::-1]
    names = pd.Series(mm.metabolite)
    try:
        import pickle
        nm = pickle.load(open(ROOT / "data/processed/cohorts.pkl", "rb"))["cptac"]["met_name_map"]
        lookup = nm.drop_duplicates("normalised").set_index("normalised").original
        names = names.map(lambda x: lookup.get(x, x))
    except Exception:
        pass
    cols = ["#c8553d" if a > 0.5 else "#3b6ea8" for a in mm.AUC_grade]
    ax.barh(range(len(mm)), mm.AUC_grade - 0.5, left=0.5, color=cols, height=0.6)
    for i, (a, f) in enumerate(zip(mm.AUC_grade, mm.fdr_grade)):
        ax.text(0.5 + (a - 0.5) + (0.01 if a > 0.5 else -0.01), i, f"FDR {f:.2f}", va="center", ha="left" if a > 0.5 else "right", fontsize=6.4)
    ax.axvline(0.5, color="#555", lw=0.8)
    ax.set_yticks(range(len(mm)), names, fontsize=7)
    ax.set_xlim(0.1, 0.95)
    ax.set_xlabel("Metabolite AUC for high grade (core)")
    panel(ax, "B")
    ax = axes[2]
    meta = pd.read_csv(M / "E9_atlas_coupling_meta.csv").sort_values("p_meta").head(12).iloc[::-1]
    cols = [PROPOSED if f < 0.05 else BASE for f in meta.fdr]
    ax.barh(range(len(meta)), meta.rho_meta, color=cols, height=0.6)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(range(len(meta)), [short(p, 36) for p in meta.pathway], fontsize=7)
    ax.set_xlabel("Tumour RNA–metabolite ρ, meta-analysis of CAMP ccRCC1–4\n(blue: FDR<0.05)")
    xgrid(ax)
    panel(ax, "C")
    fig.tight_layout()
    save(fig, "fig8_metabolites")


# ----------------------------------------------------------------------------- Fig 9 synthetic
def fig_synthetic():
    f = M / "E6_synthetic.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    methods = ["LayerGap", "EN-concat", "Late-fusion LR", "XGB-concat", "TMC"]
    cols = dict(zip(methods, [PROPOSED, CAT[1], CAT[2], CAT[3], CAT[4]]))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    ax = axes[0]
    s = d[d.group == "sample size"].copy()
    s["n"] = s.scenario.str.replace("n=", "").astype(int)
    for m in methods:
        g = s[s.method == m].groupby("n").AUC.agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=4, color=cols[m], lw=2 if m == "LayerGap" else 1.3, capsize=2, label=m)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([60, 120, 240, 480], ["60", "120", "240", "480"])
    ax.set_xlabel("Synthetic training size")
    ax.set_ylabel("Test AUC (600 synthetic patients)")
    ax.legend(fontsize=6.5)
    ygrid(ax)
    panel(ax, "A")
    ax = axes[1]
    sc = d[d.group.isin(["noise", "class imbalance", "signal type"])]
    labs = sc.scenario.unique().tolist()
    w = 0.16
    for k, m in enumerate(methods):
        g = sc[sc.method == m].groupby("scenario").AUC.mean().reindex(labs)
        ax.bar(np.arange(len(labs)) + (k - 2) * w, g.values, width=w, color=cols[m])
    ax.set_xticks(range(len(labs)), [l.replace(" signal", "\nsignal").replace("prevalence ", "prevalence\n") for l in labs], fontsize=6.6)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Test AUC")
    ygrid(ax)
    panel(ax, "B")
    ax = axes[2]
    mi = d[d.group == "missing protein (test)"]
    order = ["25% missing", "50% missing", "100% missing"]
    base = d[(d.group == "sample size") & (d.scenario == "n=240")].groupby("method").AUC.mean()
    for m in methods:
        g = mi[mi.method == m].groupby("scenario").AUC.mean().reindex(order)
        ax.plot([0, 25, 50, 100], [base[m]] + list(g.values), marker="o", ms=4, color=cols[m], lw=2 if m == "LayerGap" else 1.3)
    ax.set_xlabel("Synthetic test patients without proteome (%)")
    ax.set_ylabel("Test AUC")
    ygrid(ax)
    panel(ax, "C")
    ax = axes[3]
    r = d[d.pathway_precision_at_10.notna()]
    labs = r.scenario.unique().tolist()
    for k, m in enumerate(["LayerGap", "EN-concat"]):
        g = r[r.method == m].groupby("scenario").pathway_precision_at_10.agg(["mean", "std"]).reindex(labs)
        ax.bar(np.arange(len(labs)) + (k - 0.5) * 0.38, g["mean"], yerr=g["std"], width=0.38, color=cols[m], label=m,
               error_kw=dict(ecolor="#555", lw=0.7))
    ax.set_xticks(range(len(labs)), [l.replace("prevalence ", "prev.\n") for l in labs], fontsize=6.4, rotation=0)
    ax.set_ylabel("Signal pathways in top 10 (precision@10)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5)
    ygrid(ax)
    panel(ax, "D")
    fig.tight_layout()
    save(fig, "fig9_synthetic")


def fig_batch():
    if not (M / "E0_pca_RNA_before.csv").exists():
        return
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    sil = pd.read_csv(M / "E0_batch_silhouette.csv")
    for ax, (layer, stage) in zip(axes, [("RNA", "before"), ("RNA", "after"), ("Protein", "before"), ("Protein", "after")]):
        d = pd.read_csv(M / f"E0_pca_{layer}_{stage}.csv", index_col=0)
        for k, c in enumerate(["INI", "EXP"]):
            s = d[d.cohort == c]
            ax.scatter(s.PC1, s.PC2, s=8, color=CAT[k], alpha=0.75, label=c, lw=0)
        v = sil[(sil.layer == layer) & (sil.stage == stage)].silhouette_by_subcohort.iloc[0]
        ax.set_title(f"{layer}, {stage} z-scoring\nsilhouette = {v:.3f}", fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig_s2_batch")


if __name__ == "__main__":
    fig_design()
    fig_main()
    fig_generalisation()
    fig_robustness()
    fig_ablation()
    fig_sensitivity()
    fig_biology()
    fig_metabolites()
    fig_synthetic()
    fig_batch()
