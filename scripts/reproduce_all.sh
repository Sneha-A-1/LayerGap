#!/usr/bin/env bash
# Full reproduction after the raw data are in place (see data/README.md). Approximate time on 2 CPU cores: 4-5 h.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/download_knowledge.sh data/raw/knowledge
python scripts/preprocess_data.py
python -m pytest tests -q
for e in tcga cross ablation core missing sensitivity main; do python experiments/run_experiments.py --exp $e | tee results/logs/$e.log; done
python experiments/run_biology.py   | tee results/logs/biology.log
python experiments/run_synthetic.py | tee results/logs/synthetic.log
python experiments/run_runtime.py   | tee results/logs/runtime.log
python scripts/describe_cohorts.py
python scripts/fig_architecture.py manuscript/figures
python scripts/generate_figures.py
python scripts/make_tables.py
python scripts/train.py && python scripts/evaluate.py
