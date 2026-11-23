"""Harmonise all cohorts into analysis-ready matrices.

Usage:
    python src/preprocessing/prepare_data.py --config experiments/configuration/paths.yaml

Writes data/processed/cohorts.pkl containing, per cohort, sample annotation and
gene/protein/metabolite matrices (samples x features), together with a QC log.
All normalisation steps are unsupervised (no outcome labels are used).
"""
import argparse
import glob
import gzip
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_knowledge import norm_met  # noqa: E402

NONSILENT = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site",
             "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation", "Translation_Start_Site"}
DRIVERS = ["VHL", "PBRM1", "SETD2", "BAP1", "KDM5C"]
QC = []


def log(msg):
    QC.append(msg)
    print(msg)


def zscore_within(df):
    """Feature-wise z-score within one cohort (location-scale batch adjustment, label-free)."""
    sd = df.std(ddof=0).replace(0, np.nan)
    return ((df - df.mean()) / sd).dropna(axis=1, how="all")


def collapse_genes(df):
    """samples x genes; average duplicated gene symbols."""
    return df.T.groupby(level=0).mean().T


def impute_left_censored(df, max_missing=0.5):
    """Drop features missing in > max_missing of samples; impute the rest with the feature
    minimum minus 1 log2 unit (missing-not-at-random assumption for DIA intensities)."""
    keep = df.isna().mean() <= max_missing
    df = df.loc[:, keep]
    return df.fillna(df.min() - 1.0), int((~keep).sum())


# --------------------------------------------------------------------------- CPTAC
def load_cptac(P):
    meta = pd.read_excel(P["table_s1"], sheet_name="Meta_table", header=2).drop_duplicates("CASE_ID").set_index("CASE_ID")
    s = pd.DataFrame(index=meta.index)
    s["cohort"] = meta["Cohort_label"]
    s["grade"] = meta["Tumor_grade"]
    s["high_grade"] = meta["Tumor_grade"].isin(["G3", "G4"]).astype(int)
    s["stage"] = meta["Tumor_stage"]
    s["os_days"] = meta["Overall survival, days"]
    s["os_event"] = meta["Survival status (1, dead; 0, alive)"]
    s["rfs_days"] = meta["Recurrence-free survival, days"]
    s["rfs_event"] = meta["Recurrence status (1, yes; 0, no)"]
    s["purity"] = meta["purity_CNVEX"]
    s["immune_subtype"] = meta["Immune_subtype"]
    s["methylation_subtype"] = meta["Methylation_subtype"]
    log(f"[CPTAC] Table S1: {len(s)} ccRCC cases; INI={int((s.cohort=='INI').sum())}, EXP={int((s.cohort=='EXP').sum())}")

    # genomic features shared by both sub-cohorts (Table S1)
    g = pd.DataFrame(index=meta.index)
    for d in DRIVERS:
        g[f"mut_{d}"] = meta[f"{d}_somatic_mutation"].notna().astype(float)
    g["gain_7"] = meta["gain_chr7"].astype(float)
    g["loss_9"] = meta["loss_chr9"].astype(float)
    g["loss_14q"] = meta["loss_chr14q"].astype(float)
    g["instability"] = meta["wgii"].astype(float)
    g["log_tmb"] = np.log1p(meta["Count_somatic_mutation_wxs"].astype(float))

    # gene-level non-silent mutation matrix (for pathway mutation burden)
    mut_rows = []
    ini = pd.read_csv(P["cptac_mut_ini"], low_memory=False)
    ini = ini[ini["Mutation"].isin(NONSILENT)]
    mut_rows += list(zip(ini["Patient_ID"], ini["Gene"]))
    idx = pd.read_csv(os.path.join(P["gdc_exp"], "file_index.tsv"), sep="\t")
    for r in idx[idx.data_type == "Masked Somatic Mutation"].itertuples():
        f = os.path.join(P["gdc_exp"], r.file_id, r.file_name)
        m = pd.read_csv(f, sep="\t", comment="#", usecols=["Hugo_Symbol", "Variant_Classification"], low_memory=False)
        m = m[m.Variant_Classification.isin(NONSILENT)]
        mut_rows += [(r.case, h) for h in m.Hugo_Symbol]
    mut = pd.crosstab(*zip(*mut_rows)).clip(upper=1).reindex(meta.index).fillna(0)
    log(f"[CPTAC] mutation matrix: {mut.shape[0]} cases x {mut.shape[1]} genes (non-silent)")

    # RNA: INI from WashU (FPKM), EXP from GDC STAR (TPM); log2(x+1); z within sub-cohort
    w = pd.read_csv(P["cptac_rna_ini"], header=[0, 1], index_col=0)
    w.columns = w.columns.get_level_values(0)
    w = w[~w.index.str.endswith(".N")]
    w = collapse_genes(np.log2(w.astype(float) + 1))
    exp_tpm = {}
    for r in idx[(idx.data_type == "Gene Expression Quantification") & idx.sample_type.str.startswith("Primary Tumor")].itertuples():
        f = os.path.join(P["gdc_exp"], r.file_id, r.file_name)
        d = pd.read_csv(f, sep="\t", comment="#", usecols=["gene_name", "gene_type", "tpm_unstranded"])
        d = d[d.gene_type == "protein_coding"].groupby("gene_name")["tpm_unstranded"].mean()
        exp_tpm.setdefault(r.case, []).append(d)
    e = pd.DataFrame({c: pd.concat(v, axis=1).mean(axis=1) for c, v in exp_tpm.items()}).T
    e = np.log2(e + 1)
    genes = sorted(set(w.columns) & set(e.columns))
    w, e = w[genes], e[genes]
    expressed = ((w > 1).mean() > 0.2) & ((e > 1).mean() > 0.2)
    rna_raw = pd.concat([w.loc[:, expressed], e.loc[:, expressed]])
    rna = pd.concat([zscore_within(w.loc[w.index.isin(s.index[s.cohort == "INI"]), expressed]),
                     zscore_within(e.loc[e.index.isin(s.index[s.cohort == "EXP"]), expressed])])
    log(f"[CPTAC] RNA: INI {int(rna.index.isin(s.index[s.cohort=='INI']).sum())}, EXP "
        f"{int(rna.index.isin(s.index[s.cohort=='EXP']).sum())} tumours; {rna.shape[1]} expressed protein-coding genes")

    # Protein: single DIA matrix for the whole cohort (Li et al. 2023)
    p = pd.read_csv(P["cptac_prot_dia"], sep="\t", index_col=0)
    p = p.loc[:, [c for c in p.columns if re.fullmatch(r"C3[LN]-\d+_[TN]", c)]]
    p = np.log2(p.T)
    prot_t = p[p.index.str.endswith("_T")].rename(index=lambda x: x[:-2])
    prot_n = p[p.index.str.endswith("_N")].rename(index=lambda x: x[:-2])
    prot_t = prot_t[prot_t.index.isin(s.index)]
    prot_t_raw, dropped = impute_left_censored(prot_t)
    prot_n = prot_n.loc[:, prot_n.columns.isin(prot_t_raw.columns)]
    prot_n = prot_n.fillna(prot_t_raw.min() - 1.0)
    prot = pd.concat([zscore_within(prot_t_raw[prot_t_raw.index.isin(s.index[s.cohort == c])]) for c in ["INI", "EXP"]])
    log(f"[CPTAC] Protein (DIA): {len(prot)} tumours x {prot.shape[1]} proteins (dropped {dropped} with >50% missing); "
        f"{len(prot_n)} adjacent normals")

    # Metabolites: Table S7 (50 tumours, 7 NATs), log2 abundances
    q = pd.read_excel(P["table_s7"], sheet_name="Quantified_metabolites", header=2, index_col=0)
    name_map = pd.DataFrame({"original": q.index, "normalised": [norm_met(m) for m in q.index]})
    q = q.groupby(name_map["normalised"].values).mean().T  # synonymous entries are averaged
    log(f"[CPTAC] metabolite names normalised: {len(name_map)} entries -> {q.shape[1]} unique metabolites")
    met_n = q[q.index.str.endswith("-N")].rename(index=lambda x: x[:-2])
    met_t = q[~q.index.str.endswith("-N")]
    met = zscore_within(met_t)
    log(f"[CPTAC] Metabolites: {len(met_t)} tumours, {len(met_n)} NATs x {q.shape[1]} metabolites")

    keep = s.index[s.index.isin(rna.index) & s.index.isin(prot.index)]
    log(f"[CPTAC] three-layer cohort (genomics+RNA+protein): {len(keep)} cases "
        f"(INI {int((s.loc[keep,'cohort']=='INI').sum())}, EXP {int((s.loc[keep,'cohort']=='EXP').sum())})")
    core = keep[keep.isin(met.index)]
    log(f"[CPTAC] four-layer core (+metabolomics): {len(core)} cases")
    return {"met_name_map": name_map, "samples": s.loc[keep], "genomic": g.loc[keep], "mut": mut.loc[keep], "rna": rna.loc[keep],
            "rna_raw": rna_raw.loc[rna_raw.index.isin(keep)], "prot": prot.loc[keep], "prot_raw": prot_t_raw.loc[keep],
            "prot_normal": prot_n, "met": met, "met_raw": met_t, "met_normal": met_n, "core": list(core)}


# --------------------------------------------------------------------------- TCGA
def load_tcga(P):
    T = P["tcga_dir"]
    pat = pd.read_csv(os.path.join(T, "data_clinical_patient.txt.gz"), sep="\t", comment="#").set_index("PATIENT_ID")
    smp = pd.read_csv(os.path.join(T, "data_clinical_sample.txt.gz"), sep="\t", comment="#")
    smp = smp[smp.SAMPLE_ID.str.endswith("-01")].set_index("SAMPLE_ID")
    s = pd.DataFrame(index=smp.index)
    s["patient"] = smp["PATIENT_ID"]
    s["grade"] = smp["GRADE"]
    s["high_grade"] = smp["GRADE"].map({"G1": 0, "G2": 0, "G3": 1, "G4": 1})
    s["stage"] = pat.reindex(s.patient)["AJCC_PATHOLOGIC_TUMOR_STAGE"].values
    s["os_days"] = pd.to_numeric(pat.reindex(s.patient)["OS_MONTHS"], errors="coerce").values * 30.4375
    s["os_event"] = pat.reindex(s.patient)["OS_STATUS"].astype(str).str.startswith("1").astype(float).values
    s["cohort"] = "TCGA"

    r = pd.read_csv(os.path.join(T, "data_mrna_seq_v2_rsem.txt.gz"), sep="\t").dropna(subset=["Hugo_Symbol"])
    r = r.drop(columns="Entrez_Gene_Id").groupby("Hugo_Symbol").mean().T
    r = np.log2(r.clip(lower=0) + 1)
    r = r.loc[r.index.isin(s.index), (r > 1).mean() > 0.2]
    rna = zscore_within(r)

    arm = pd.read_csv(os.path.join(T, "data_armlevel_cna.txt.gz"), sep="\t").set_index("NAME").drop(columns=["ENTITY_STABLE_ID", "DESCRIPTION"], errors="ignore").T
    mut = pd.read_csv(os.path.join(T, "data_mutations.txt.gz"), sep="\t", comment="#",
                      usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"], low_memory=False)
    sequenced = set(mut.Tumor_Sample_Barcode)
    mut = mut[mut.Variant_Classification.isin(NONSILENT)]
    mm = pd.crosstab(mut.Tumor_Sample_Barcode, mut.Hugo_Symbol).clip(upper=1).reindex(sorted(sequenced)).fillna(0)
    tmb = mut.groupby("Tumor_Sample_Barcode").size().reindex(sorted(sequenced)).fillna(0)
    g = pd.DataFrame(index=sorted(sequenced & set(arm.index)))
    for d in DRIVERS:
        g[f"mut_{d}"] = mm.reindex(g.index)[d].fillna(0) if d in mm else 0.0
    a = arm.reindex(g.index)
    g["gain_7"] = ((a["7p"] == "Gain") & (a["7q"] == "Gain")).astype(float)
    g["loss_9"] = ((a["9p"] == "Loss") | (a["9q"] == "Loss")).astype(float)
    g["loss_14q"] = (a["14q"] == "Loss").astype(float)
    g["instability"] = a.isin(["Gain", "Loss"]).mean(axis=1)
    g["log_tmb"] = np.log1p(tmb.reindex(g.index))
    g = g[g.index.isin(s.index)]
    keep = s.index[s.index.isin(rna.index) & s.high_grade.notna()]
    log(f"[TCGA-KIRC] {len(keep)} primary tumours with RNA and grade; {int(g.index.isin(keep).sum())} with exome-based genomics; "
        f"high grade {int(s.loc[keep,'high_grade'].sum())}; deaths {int(s.loc[keep,'os_event'].sum())}")
    return {"samples": s.loc[keep], "rna": rna.loc[keep], "genomic": g[g.index.isin(keep)],
            "mut": mm[mm.index.isin(keep)]}


# --------------------------------------------------------------------------- Metabolic atlas (CAMP)
def load_atlas(P):
    D = P["camp_dir"]
    mm = pd.read_csv(os.path.join(D, "MasterMapping_MetImmune_03_16_2022_release.csv"))
    mm = mm[mm.Dataset.isin(["ccRCC1", "ccRCC2", "ccRCC3", "ccRCC4"]) & (mm.Histology == "Clear Cell")]
    out = {}
    for ds, grp in mm.groupby("Dataset"):
        xl = os.path.join(D, "metabolomics_processed", grp.MetabFile.iloc[0])
        mets = []
        for sheet in grp.MetabFile_sheet.unique():
            mets.append(pd.read_excel(xl, sheet_name=sheet, index_col=0))
        met = pd.concat(mets, axis=1).T
        met = met.loc[met.index.isin(grp.MetabID)]
        met = met.T.groupby(met.columns.map(norm_met)).mean().T
        rf = os.path.join(D, "transcriptomics_processed", grp.RNAFile.iloc[0])
        if "Multiregions" in rf:  # use the verified complete copy of this file
            alt = glob.glob(os.path.join(D, "transcriptomics_processed", "*COMPLETE.csv"))
            rf = alt[0] if alt else rf
        if not os.path.exists(rf) and P.get("camp_extra_dir"):
            rf = os.path.join(P["camp_extra_dir"], os.path.basename(rf))
        if not os.path.exists(rf):
            log(f"[Atlas {ds}] RNA file {os.path.basename(rf)} not available; dataset not used")
            continue
        rna = pd.read_csv(rf, index_col=0).T
        rna = rna.T.groupby(level=0).mean().T
        id_map = dict(zip(grp.RNAID.astype(str), grp.MetabID.astype(str)))
        rna.index = rna.index.astype(str).map(lambda x: id_map.get(x, None))
        rna = rna[rna.index.notna()]
        both = sorted(set(met.index) & set(rna.index))
        tn = grp.set_index("MetabID").loc[both, "TN"]
        r = np.log2(rna.loc[both].clip(lower=0) + 1)
        r = r.loc[:, (r > 1).mean() > 0.2]
        out[ds] = {"met": zscore_within(met.loc[both]), "rna": zscore_within(r), "tn": tn}
        log(f"[Atlas {ds}] {len(both)} matched clear-cell samples (tumour {int((tn=='Tumor').sum())}, "
            f"normal {int((tn=='Normal').sum())}); {met.shape[1]} metabolites; {r.shape[1]} genes")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configuration/paths.yaml")
    ap.add_argument("--out", default="data/processed/cohorts.pkl")
    a = ap.parse_args()
    P = yaml.safe_load(open(a.config))
    data = {"cptac": load_cptac(P), "tcga": load_tcga(P), "atlas": load_atlas(P), "qc_log": QC}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(data, fh)
    os.replace(tmp, a.out)  # atomic replacement
    Path(a.out).with_suffix(".qc.txt").write_text("\n".join(QC) + "\n")


if __name__ == "__main__":
    main()
