# Data

All data used by the study are public, non-imaging molecular and clinical tables. Raw files are **not**
redistributed in this repository; the scripts and steps below obtain them. Place the downloads anywhere and
point `experiments/configuration/paths.yaml` to them.

| # | Resource | What is used | How to obtain | Access |
|---|----------|--------------|---------------|--------|
| 1 | CPTAC ccRCC annotation (Li et al., *Cancer Cell* 2023, doi:10.1016/j.ccell.2022.12.001) | Supplementary **Table S1** (`mmc2.xlsx`): 213 cases, grade, stage, survival, driver mutations, arm-level CNAs, wGII | Paper page → *Supplemental information*, or PubMed Central PMC9839644 | Open |
| 2 | CPTAC metabolome (same paper) | Supplementary **Table S7** (`mmc8.xlsx`): 183 metabolites, 50 tumours, 7 NATs (PDC000534) | As above | Open |
| 3 | CPTAC DIA proteome | `CPTAC_ccRCC_global_proteome_DIA.tsv` inside `Supplementary_Processed_Proteomic_Data_for_PMID-36563681.zip` | https://pdc.cancer.gov/pdc/study/PDC000411 → *Supplementary Data* | Open |
| 4 | CPTAC INI RNA and mutations | `transcriptomics` (WashU) and `somatic_mutation` (harmonized) tables of the `cptac` Python package v1.5.14 | `python scripts/download_data.py --out <dir> --only cptac` | Open |
| 5 | CPTAC EXP RNA and mutations | GDC CPTAC-3 open-access STAR gene counts (TPM) and masked somatic mutation MAFs for the 110 EXP cases | `python scripts/download_expanded_gdc.py --out <dir>` | Open (no token) |
| 6 | TCGA-KIRC PanCancer Atlas | `kirc_tcga_pan_can_atlas_2018` (RSEM RNA, mutations, arm-level CNA, clinical) | `python scripts/download_data.py --out <dir> --only tcga` (cBioPortal DataHub, `https://datahub.assets.cbioportal.org/kirc_tcga_pan_can_atlas_2018.tar.gz`) | Open |
| 7 | CAMP metabolic atlas (Benedetti et al., *Nat Metab* 2023) | `pancancer_metabolomics_v.0.3.4.tar.gz` (ccRCC1–4 metabolomics, matched RNA, `MasterMapping` file) | Zenodo DOI 10.5281/zenodo.7150252; `python scripts/download_data.py --out <dir> --only camp` | Open, CC-BY |
| 8 | Human-GEM v2.1.1 | `Human-GEM.yml`, `genes.tsv`, `metabolites.tsv`, `reactions.tsv` | `bash scripts/download_knowledge.sh data/raw/knowledge` | Open, CC-BY |
| 9 | MSigDB Hallmark v7.0 | `h.all.v7.0.symbols.gmt` | same script | Open |

## Expected layout of `paths.yaml`

```yaml
table_s1: /path/to/mmc2.xlsx
table_s7: /path/to/mmc8.xlsx
cptac_mut_ini: /path/to/cptac_ccrcc/somatic_mutation__harmonized.csv.gz
cptac_rna_ini: /path/to/cptac_ccrcc/transcriptomics__washu.csv.gz
cptac_prot_dia: /path/to/CPTAC_ccRCC_global_proteome_DIA.tsv
gdc_exp: /path/to/cptac_expanded_gdc          # contains file_index.tsv and <file_id>/<file> folders
tcga_dir: /path/to/kirc_tcga_pan_can_atlas_2018   # data_*.txt(.gz) files
camp_dir: /path/to/pancancer_metabolomics/data
camp_extra_dir: /optional/folder/with/extra/RNA/files
```

## Processed outputs

`python scripts/preprocess_data.py` writes `data/processed/cohorts.pkl` and a QC log
(`data/processed/cohorts.qc.txt`) with the sample counts of every cohort. `data/knowledge/knowledge.pkl` holds the
163 pathway definitions (113 Human-GEM subsystems, 50 Hallmark sets) and 26,372 enzyme–metabolite links.

## Integrity

`scripts/download_data.py` writes `MANIFEST.tsv` with the size and SHA-256 of every downloaded file. The CPTAC PDC
archives ship `.cksum` files with MD5 sums that can be checked with `md5sum -c`.
