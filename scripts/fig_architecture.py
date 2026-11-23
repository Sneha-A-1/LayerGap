"""Figure 1: multi-layer architecture and dataflow of LayerGap (drawn programmatically)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Ellipse

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "manuscript/figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

C = {"G": "#3b6ea8", "T": "#2a9d8f", "P": "#c8553d", "M": "#8e6bb8", "K": "#6c757d",
     "proc": "#f4f1ea", "edge": "#3d3d3d", "val": "#e9c46a", "out": "#264653"}


def dna(ax, x, y, s=1.0):
    t = np.linspace(0, 2 * np.pi, 60)
    ax.plot(x + 0.18 * s * np.sin(t * 1.5), y + 0.25 * s * (t / np.pi - 1), color=C["G"], lw=1.3)
    ax.plot(x - 0.18 * s * np.sin(t * 1.5), y + 0.25 * s * (t / np.pi - 1), color=C["G"], lw=1.3)
    for tt in t[::6]:
        ax.plot([x + 0.18 * s * np.sin(tt * 1.5), x - 0.18 * s * np.sin(tt * 1.5)], [y + 0.25 * s * (tt / np.pi - 1)] * 2,
                color=C["G"], lw=0.6, alpha=0.7)


def rna(ax, x, y, s=1.0):
    t = np.linspace(0, 2 * np.pi, 60)
    xs, ys = x + 0.17 * s * np.sin(t * 1.5), y + 0.25 * s * (t / np.pi - 1)
    ax.plot(xs, ys, color=C["T"], lw=1.5)
    for tt in t[::6]:
        ax.plot([x + 0.17 * s * np.sin(tt * 1.5), x + 0.17 * s * np.sin(tt * 1.5) + 0.09 * s], [y + 0.25 * s * (tt / np.pi - 1)] * 2,
                color=C["T"], lw=0.8)


def protein(ax, x, y, s=1.0):
    rng = np.random.default_rng(3)
    pts = np.cumsum(rng.normal(0, 0.06 * s, (14, 2)), axis=0)
    pts -= pts.mean(0)
    for (a, b) in pts:
        ax.add_patch(Circle((x + a, y + b), 0.045 * s, color=C["P"], alpha=0.85, lw=0))
    ax.plot(x + pts[:, 0], y + pts[:, 1], color=C["P"], lw=0.8)


def metabolite(ax, x, y, s=1.0):
    ang = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    ax.add_patch(Polygon(np.c_[x + 0.13 * s * np.cos(ang), y + 0.13 * s * np.sin(ang)], closed=True, fill=False, ec=C["M"], lw=1.4))
    ax.plot([x + 0.13 * s * np.cos(ang[0]), x + 0.24 * s * np.cos(ang[0])], [y + 0.13 * s * np.sin(ang[0]), y + 0.24 * s * np.sin(ang[0])], color=C["M"], lw=1.2)
    ax.add_patch(Circle((x + 0.27 * s * np.cos(ang[0]), y + 0.27 * s * np.sin(ang[0])), 0.03 * s, color=C["M"]))


def database(ax, x, y, s=1.0, col=C["K"]):
    w, h = 0.32 * s, 0.34 * s
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, color=col, alpha=0.25, lw=0))
    for yy in [y - h / 2, y, y + h / 2]:
        ax.add_patch(Ellipse((x, yy), w, 0.1 * s, fc="white" if yy == y + h / 2 else "none", ec=col, lw=1))
    ax.plot([x - w / 2] * 2, [y - h / 2, y + h / 2], color=col, lw=1)
    ax.plot([x + w / 2] * 2, [y - h / 2, y + h / 2], color=col, lw=1)


def cohort(ax, x, y, s=1.0, col=C["out"]):
    for dx in [-0.12, 0, 0.12]:
        ax.add_patch(Circle((x + dx * s, y + 0.09 * s), 0.045 * s, color=col))
        ax.add_patch(FancyBboxPatch((x + dx * s - 0.05 * s, y - 0.12 * s), 0.1 * s, 0.15 * s,
                                    boxstyle="round,pad=0.01,rounding_size=0.04", color=col))


def network(ax, x, y, s=1.0, col=C["K"]):
    rng = np.random.default_rng(7)
    pts = rng.uniform(-0.2, 0.2, (7, 2)) * s
    for i in range(7):
        for j in range(i + 1, 7):
            if np.linalg.norm(pts[i] - pts[j]) < 0.22 * s:
                ax.plot([x + pts[i, 0], x + pts[j, 0]], [y + pts[i, 1], y + pts[j, 1]], color=col, lw=0.7)
    cols = [C["T"], C["P"], C["M"], C["T"], C["P"], C["M"], C["G"]]
    for (a, b), c in zip(pts, cols):
        ax.add_patch(Circle((x + a, y + b), 0.035 * s, color=c, zorder=3))


def box(ax, x, y, w, h, title, lines, fc=C["proc"], ec=C["edge"], tcol="black", fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=fs + 0.6, weight="bold", color=tcol)
    ax.text(x + w / 2, y + h - 0.38, "\n".join(lines), ha="center", va="top", fontsize=fs, color=tcol, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, col=C["edge"], lw=1.1, ls="-", rad=0.0, label=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=col, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label, fontsize=6.3, ha="center", color=col, style="italic")


fig, ax = plt.subplots(figsize=(17, 9.4))
ax.set_xlim(0, 17)
ax.set_ylim(0, 9.4)
ax.axis("off")

layers = [(0.15, "L1  Multi-omics data\nacquisition"), (2.75, "L2  Harmonisation and\nquality control"),
          (5.05, "L3  Pathway-constrained\nencoding"), (7.45, "L4  Cascade\nuncoupling"),
          (9.75, "L5  Precision-weighted\nadaptive fusion"), (12.15, "L6  Interpretation and\nvalidation"),
          (14.65, "L7  Precision-oncology\noutputs")]
widths = [2.35, 2.1, 2.2, 2.1, 2.2, 2.3, 2.2]
for (x, t), w in zip(layers, widths):
    ax.add_patch(Rectangle((x, 0.2), w, 8.55, fc="#fbfbfb", ec="#cfcfcf", lw=0.8, ls="--"))
    ax.text(x + w / 2, 9.05, t, ha="center", va="center", fontsize=8.6, weight="bold", color="#222222")

# L1 inputs
inputs = [("Genomics", "somatic mutations, arm-level\nCNAs, wGII, TMB", dna, C["G"], 7.6),
          ("Transcriptomics", "RNA-seq (FPKM / TPM / RSEM)", rna, C["T"], 6.35),
          ("Proteomics", "DIA-MS global proteome", protein, C["P"], 5.1),
          ("Metabolomics", "LC-MS metabolite abundance", metabolite, C["M"], 3.85)]
for name, sub, icon, col, y in inputs:
    ax.add_patch(FancyBboxPatch((0.3, y - 0.48), 2.05, 0.96, boxstyle="round,pad=0.02,rounding_size=0.06", fc="white", ec=col, lw=1.2))
    icon(ax, 0.62, y, 0.9)
    ax.text(0.95, y + 0.17, name, fontsize=7.8, weight="bold", color=col, va="center")
    ax.text(0.95, y - 0.17, sub, fontsize=6.0, va="center", color="#333333")
cohort(ax, 0.62, 2.55, 1.0)
ax.text(0.95, 2.72, "Clinical cohorts", fontsize=7.8, weight="bold", color=C["out"])
ax.text(0.95, 2.28, "CPTAC ccRCC n=212 (G,T,P)\n4-layer core n=50 (+M)\nTCGA-KIRC n=502 (G,T)\nCAMP atlas n=242 (T,M)", fontsize=5.8, va="center")
database(ax, 0.62, 1.05, 1.0)
ax.text(0.95, 1.22, "Prior knowledge", fontsize=7.8, weight="bold", color=C["K"])
ax.text(0.95, 0.88, "Human-GEM v2.1.1 reactions\nMSigDB Hallmark gene sets", fontsize=5.8, va="center")

# L2
box(ax, 2.88, 5.6, 1.85, 2.6, "Per-cohort QC", ["log2 transform", "expression filter", "left-censored", "protein imputation", "metabolite name", "normalisation"])
box(ax, 2.88, 2.6, 1.85, 2.65, "Batch handling", ["within-cohort", "feature z-scoring", "(label-free)", "silhouette check", "patient-level", "sample matching"])
box(ax, 2.88, 0.75, 1.85, 1.5, "Leakage control", ["all supervised steps", "inside training folds"], fc="#fff5f5")

# L3
box(ax, 5.18, 5.0, 1.95, 3.2, "Pathway scores", ["163 pathways", "(113 metabolic,", "50 Hallmark)", "", "t_p, p_p on genes", "measured in both", "RNA and protein"])
for i, (lab, col) in enumerate([("g_p", C["G"]), ("t_p", C["T"]), ("p_p", C["P"]), ("m_p", C["M"])]):
    ax.add_patch(FancyBboxPatch((5.3 + i * 0.44, 4.35), 0.38, 0.38, boxstyle="round,pad=0.01", fc=col, ec="none"))
    ax.text(5.49 + i * 0.44, 4.54, lab, color="white", ha="center", va="center", fontsize=7, weight="bold")
network(ax, 6.15, 3.3, 1.2)
ax.text(6.15, 2.65, "gene-protein-metabolite\nlinks from shared reactions", ha="center", fontsize=6.2)
box(ax, 5.18, 0.45, 1.95, 1.8, "Mutation burden", ["mutated pathway genes", "/ pathway size"])

# L4
box(ax, 7.58, 5.0, 1.85, 3.2, "Transcript→protein", ["b_g = shrunk", "corr(T_g, P_g)", "(training folds)", "", "u_p = mean_g", "(P_g − b_g T_g)"], fc="#fdf0ec")
box(ax, 7.58, 1.6, 1.85, 3.0, "Protein→metabolite", ["c_p = shrunk", "corr(p_p, m_p)", "", "v_p = m_p − c_p p_p", "", "(four-layer core)"], fc="#f3eefa")
ax.text(8.5, 1.05, "Novelty 1: uncoupling\nas a feature space", ha="center", fontsize=6.6, weight="bold", color=C["P"])

# L5
box(ax, 9.88, 5.55, 1.95, 2.65, "Block ensembles", ["G, T, P, U, M, V", "L2-logistic, C by", "inner CV", "B bootstrap fits", "→ logit z_b, var σ²_b"])
box(ax, 9.88, 2.35, 1.95, 2.85, "Adaptive fusion", ["π_b = (σ²_b+τ)⁻¹", "(relative precision)", "missing block → π=0", "", "logit = β₀ + Σ α_b π_b z_b", "α_b ≥ 0 (OOF stacking)"], fc="#eef6f5")
ax.text(10.85, 1.75, "Novelty 2: sample-specific\nprecision weights,\nmissing-layer tolerant", ha="center", fontsize=6.6, weight="bold", color=C["T"])

# L6
box(ax, 12.28, 5.55, 2.05, 2.65, "MAP prioritisation", ["effect × bootstrap", "sign stability ×", "cross-layer", "concordance"], fc="#fdf6e3")
ax.text(13.3, 5.2, "Novelty 3", ha="center", fontsize=6.6, weight="bold", color="#b08900")
box(ax, 12.28, 1.75, 2.05, 3.15, "Validation", ["repeated nested CV", "INI↔EXP transfer", "TCGA external test", "atlas replication", "tumour vs normal", "synthetic ground truth", "ablation, sensitivity"], fc=C["val"])
box(ax, 12.28, 0.45, 2.05, 1.0, "Runtime profile", ["stage-wise timing"], fs=6.6)

# L7
outs = [("High-grade risk", "probability per patient", 7.55), ("Uncertainty", "per-patient σ, layers used", 6.3),
        ("Pathway programmes", "uncoupled metabolic and\nsignalling pathways", 5.05), ("Biomarker candidates", "replicated genes,\nmetabolites", 3.8),
        ("Prognostic stratification", "survival separation", 2.55), ("Testable hypotheses", "targets for wet-lab\nvalidation", 1.3)]
for t, sub, y in outs:
    ax.add_patch(FancyBboxPatch((14.78, y - 0.47), 1.95, 0.94, boxstyle="round,pad=0.02,rounding_size=0.06", fc=C["out"], ec="none"))
    ax.text(15.75, y + 0.17, t, color="white", ha="center", fontsize=7.4, weight="bold")
    ax.text(15.75, y - 0.17, sub, color="#e0e0e0", ha="center", fontsize=6.0, va="center")

# arrows
for _, _, _, col, y in inputs:
    arrow(ax, 2.37, y, 2.86, min(max(y, 2.9), 7.9), col=col)
arrow(ax, 2.37, 2.5, 2.86, 3.2, col=C["out"])
arrow(ax, 2.37, 0.62, 5.16, 0.62, col=C["K"], ls="--")
ax.text(3.8, 0.42, "pathway gene / metabolite sets", fontsize=6.2, ha="center", color=C["K"], style="italic")
arrow(ax, 6.15, 2.27, 6.15, 2.5, col=C["K"], ls="--")
arrow(ax, 4.75, 6.8, 5.16, 6.6)
arrow(ax, 4.75, 3.9, 5.16, 4.55)
arrow(ax, 7.15, 6.6, 7.56, 6.6, label="t_p, p_p")
arrow(ax, 7.15, 4.55, 7.56, 3.4, label="m_p")
arrow(ax, 7.15, 1.35, 9.86, 6.0, col=C["G"], rad=0.18)
ax.text(7.32, 1.62, "g_p", fontsize=6.5, color=C["G"], style="italic")
arrow(ax, 9.45, 6.6, 9.86, 6.9, label="u_p")
arrow(ax, 9.45, 3.1, 9.86, 6.2, label="v_p")
arrow(ax, 10.85, 5.53, 10.85, 5.22, label="z_b, σ²_b")
arrow(ax, 11.85, 6.9, 12.26, 6.9, label="block weights")
arrow(ax, 11.85, 3.8, 12.26, 3.6)
arrow(ax, 14.35, 6.9, 14.76, 5.05)
arrow(ax, 14.35, 6.9, 14.76, 3.8)
arrow(ax, 14.35, 4.6, 14.76, 7.55)
arrow(ax, 14.35, 4.3, 14.76, 6.3)
arrow(ax, 14.35, 3.0, 14.76, 2.55)
arrow(ax, 14.35, 2.4, 14.76, 1.3)
arrow(ax, 12.3, 1.9, 8.5, 1.65, col="#b08900", ls="--", rad=-0.25)
ax.text(10.85, 0.72, "feedback: hyper-parameters, pathway size", fontsize=6.3, ha="center", color="#b08900", style="italic")

ax.text(8.5, 0.25, "Arrows: dataflow; dashed: knowledge input or feedback. G genomics, T transcriptomics, P proteomics, M metabolomics, U transcript–protein uncoupling, V protein–metabolite uncoupling.",
        ha="center", fontsize=6.6, color="#444444")
fig.savefig(OUT / "fig1_architecture.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1_architecture.pdf", bbox_inches="tight")
print("saved")
