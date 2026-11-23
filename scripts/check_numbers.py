"""Cross-check key numbers quoted in the manuscript text against the result files."""
import re
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1]
M = R / "results/metrics"
text = " ".join(p.read_text() for p in (R / "manuscript/sections").glob("*.tex"))
e1 = pd.read_csv(M / "E1_main_cv_folds.csv").groupby("method").mean(numeric_only=True)
e2 = pd.read_csv(M / "E2_cross_cohort.csv").set_index(["train", "method"])
e3 = pd.read_csv(M / "E3_tcga_external.csv").set_index("method")
sv = pd.read_csv(M / "E3_tcga_survival.csv", index_col=0).iloc[:, 0]
ab = pd.read_csv(M / "E7_ablation_summary.csv").set_index("variant")
core = pd.read_csv(M / "E4_core_summary.csv").set_index("method")
checks = {
    "E1 LayerGap AUC": f"{e1.loc['LayerGap','AUC']:.3f}", "E1 late fusion AUC": f"{e1.loc['Late-fusion LR','AUC']:.3f}",
    "E1 TMC AUC": f"{e1.loc['TMC','AUC']:.3f}", "E1 LayerGap PR-AUC": f"{e1.loc['LayerGap','PR_AUC']:.3f}",
    "E1 LayerGap MCC": f"{e1.loc['LayerGap','MCC']:.3f}", "E1 LayerGap fit": f"{e1.loc['LayerGap','fit_s']:.2f}",
    "E2 INI->EXP": f"{e2.loc[('INI','LayerGap'),'AUC']:.3f}", "E2 EXP->INI": f"{e2.loc[('EXP','LayerGap'),'AUC']:.3f}",
    "E3 LayerGap": f"{e3.loc['LayerGap','AUC']:.3f}", "E3 late fusion": f"{e3.loc['Late-fusion LR','AUC']:.3f}",
    "E3 C-index": f"{e3.loc['LayerGap','OS_Cindex']:.3f}", "HR uni": f"{sv['uni_HR']:.2f}", "HR adj": f"{sv['multi_grade_HR']:.2f}",
    "Ablation U only": f"{ab.loc['U block only','AUC']:.3f}", "Ablation T only": f"{ab.loc['T block only','AUC']:.3f}",
    "Core LG full": f"{core.loc['LayerGap (G,T,P,U,M,V)','mean']:.3f}", "Core LG no M": f"{core.loc['LayerGap (G,T,P,U)','mean']:.3f}",
}
bad = 0
for k, v in checks.items():
    ok = v in text
    bad += not ok
    print(f"{'OK ' if ok else 'MISSING'}  {k}: {v}")
print("all found" if bad == 0 else f"{bad} values not found in text")
