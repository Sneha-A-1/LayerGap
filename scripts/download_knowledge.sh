#!/usr/bin/env bash
# Human-GEM v2.1.1 model files and MSigDB Hallmark gene sets (v7.0 GMT as distributed with GSEApy tests)
set -euo pipefail
OUT=${1:-data/raw/knowledge}
mkdir -p "$OUT"
B=https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/v2.1.1/model
for f in Human-GEM.yml metabolites.tsv reactions.tsv genes.tsv; do curl -fsSL -o "$OUT/$f" "$B/$f"; done
curl -fsSL -o "$OUT/h.gmt" https://raw.githubusercontent.com/zqfang/GSEApy/master/tests/extdata/h.all.v7.0.symbols.gmt
echo "Knowledge files written to $OUT"
