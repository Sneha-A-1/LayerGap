# Pathway-constrained cross-layer uncoupling for multi-omics grading of clear cell renal cell carcinoma

Code, configuration and manuscript for the study *"LayerGap: pathway-constrained transcript–protein–metabolite
uncoupling with precision-weighted fusion stratifies clear cell renal cell carcinoma grade across independent
cohorts"* (Vaish, Singh, Kaur).

LayerGap integrates genomics (G), transcriptomics (T), proteomics (P) and metabolomics (M) through four components:

1. **Pathway-constrained layer scores** computed on genes measured at both RNA and protein level (163 pathways from
   Human-GEM v2.1.1 and MSigDB Hallmark).
2. **Cascade uncoupling features**: transcript→protein residuals `u_p` and protein→metabolite residuals `v_p`, with
   coupling coefficients estimated on training patients only.
3. **Precision-weighted adaptive fusion**: bootstrap ensembles per block, sample-specific precision weights, zero
   weight for unmeasured layers, per-patient uncertainty.
4. **Mechanism-aware pathway prioritisation (MAP)**: effect × bootstrap sign stability × cross-layer concordance.

## Repository layout

```
├── README.md, LICENSE, requirements.txt, environment.yml
├── data/                     README with download instructions (raw data are not redistributed)
├── src/
│   ├── preprocessing/        build_knowledge.py (Human-GEM + Hallmark), prepare_data.py (harmonisation)
│   ├── proposed_method/      layergap.py (PathwayEncoder, BlockModel, LayerGap)
│   ├── baselines/            baselines.py (EN, RF, XGBoost, late fusion, MB-PLS-DA, MOGONET, TMC)
│   ├── evaluation/           metrics.py (metrics, corrected resampled t-test, bootstrap tests, C-index)
│   └── data_access.py
├── experiments/
│   ├── configuration/paths.yaml
│   ├── run_experiments.py    E1 main CV, E2 cross-cohort, E3 TCGA, E4 four-layer core, E5 missing layers,
│   │                         E7 ablation, E8 sensitivity
│   ├── run_biology.py        E9 coupling, MAP stability, replication, metabolite layer, CAMP atlas
│   ├── run_synthetic.py      E6 synthetic stress tests with known ground truth
│   └── run_runtime.py        E10 runtime profile
├── scripts/                  download_data.py, download_expanded_gdc.py, download_knowledge.sh, preprocess_data.py,
│                             train.py, evaluate.py, run_baselines.py, run_ablation.py, generate_figures.py,
│                             fig_architecture.py, make_tables.py, describe_cohorts.py, build_bib.py, reproduce_all.sh
├── tests/                    unit tests (pytest)
├── results/                  metrics/ (all CSV outputs), tables/, logs/
```

## Installation

```bash
conda env create -f environment.yml && conda activate layergap
# or: python -m pip install -r requirements.txt
python -m pytest tests -q
```

Tested with Python 3.13.18 on Linux (2 CPU cores, 8 GB RAM); no GPU is needed.

## Reproducing the study

1. Obtain the data as described in `data/README.md` and edit `experiments/configuration/paths.yaml`.
2. Run everything:

```bash
bash scripts/reproduce_all.sh
```

or step by step:

```bash
bash scripts/download_knowledge.sh data/raw/knowledge
python scripts/preprocess_data.py                       # -> data/processed/cohorts.pkl (+ QC log)
python experiments/run_experiments.py --exp all         # E1-E5, E7, E8
python experiments/run_biology.py                       # E9
python experiments/run_synthetic.py                     # E6
python experiments/run_runtime.py                       # E10
python scripts/generate_figures.py && python scripts/make_tables.py
python scripts/train.py && python scripts/evaluate.py   # final model on all CPTAC, applied to TCGA
cd manuscript && latexmk -pdf main.tex
```

All random seeds are fixed (2026). Fold assignments and per-patient predictions are written next to every metric file
in `results/metrics/`, so every number in the manuscript can be traced to a CSV file.

## Using LayerGap on new data

```python
from src.proposed_method.layergap import LayerGap
model = LayerGap(pathways).fit(D, train_ids, y_train)      # D = {"G":..., "MUT":..., "T":..., "P":..., "M":...}
out = model.predict(D_new, new_ids, return_details=True)   # prob, uncertainty, block contributions, layers used
model.map_scores()                                         # ranked pathways
```

Matrices are pandas DataFrames (patients × features), standardised within cohort. A layer that is absent for a
patient is simply missing (NaN rows or an absent key); LayerGap gives that block zero weight.

## Citation and licence

Code: MIT licence. Data remain under the terms of their original providers (see `data/README.md`).
