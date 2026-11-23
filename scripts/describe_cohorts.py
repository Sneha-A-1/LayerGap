"""Table 1: cohort characteristics (computed from the harmonised data)."""
import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from src.data_access import load
d, kb = load()
c, t = d["cptac"], d["tcga"]
rows = []
def desc(name, s, g=None, layers=""):
    r = {"Cohort": name, "Layers": layers, "n": len(s), "High grade (G3/G4)": f"{int(s.high_grade.sum())} ({100*s.high_grade.mean():.1f}%)"}
    st = s.stage.astype(str)
    r["Stage III/IV"] = f"{int(st.str.contains('III|IV').sum())}"
    r["Deaths"] = int(np.nansum(s.os_event))
    r["Median follow-up (days)"] = f"{np.nanmedian(s.os_days):.0f}"
    if g is not None:
        gg = g.reindex(s.index)
        r["VHL mutated"] = f"{int(gg.mut_VHL.sum())} ({100*gg.mut_VHL.mean():.0f}%)"
        r["PBRM1 mutated"] = f"{int(gg.mut_PBRM1.sum())} ({100*gg.mut_PBRM1.mean():.0f}%)"
        r["BAP1 mutated"] = f"{int(gg.mut_BAP1.sum())} ({100*gg.mut_BAP1.mean():.0f}%)"
    return r
s = c["samples"]
rows.append(desc("CPTAC INI (discovery)", s[s.cohort=="INI"], c["genomic"], "G, T, P"))
rows.append(desc("CPTAC EXP (confirmatory)", s[s.cohort=="EXP"], c["genomic"], "G, T, P"))
rows.append(desc("CPTAC four-layer core", s.loc[c["core"]], c["genomic"], "G, T, P, M"))
tg = t["genomic"]
r = desc("TCGA-KIRC", t["samples"], None, "G, T"); 
gg = tg.reindex(t["samples"].index).dropna()
r.update({"VHL mutated": f"{int(gg.mut_VHL.sum())}/{len(gg)} ({100*gg.mut_VHL.mean():.0f}%)", "PBRM1 mutated": f"{int(gg.mut_PBRM1.sum())}/{len(gg)} ({100*gg.mut_PBRM1.mean():.0f}%)", "BAP1 mutated": f"{int(gg.mut_BAP1.sum())}/{len(gg)} ({100*gg.mut_BAP1.mean():.0f}%)"})
rows.append(r)
T = pd.DataFrame(rows); T.to_csv("results/tables/table1_cohorts.csv", index=False) if __import__("os").makedirs("results/tables", exist_ok=True) is None else None
print(T.T.to_string())
print({k:(len(v["met"]), int((v["tn"]=="Tumor").sum()), int((v["tn"]=="Normal").sum()), v["met"].shape[1], v["rna"].shape[1]) for k,v in d["atlas"].items()})
print("core grade", s.loc[c["core"]].grade.value_counts().to_dict(), "prot normals", len(c["prot_normal"]), "rna genes", c["rna"].shape[1], "prot", c["prot"].shape[1], "shared", len(set(c["rna"].columns)&set(c["prot"].columns)), "tcga genes", t["rna"].shape[1])
