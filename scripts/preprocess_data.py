"""Build the pathway knowledge base and the harmonised cohorts.

    python scripts/preprocess_data.py --knowledge-src data/raw/knowledge --config experiments/configuration/paths.yaml
"""
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--knowledge-src", default=str(ROOT / "data/raw/knowledge"))
ap.add_argument("--config", default=str(ROOT / "experiments/configuration/paths.yaml"))
a = ap.parse_args()
subprocess.run([sys.executable, str(ROOT / "src/preprocessing/build_knowledge.py"), a.knowledge_src,
                str(ROOT / "data/knowledge/knowledge.pkl")], check=True)
subprocess.run([sys.executable, str(ROOT / "src/preprocessing/prepare_data.py"), "--config", a.config,
                "--out", str(ROOT / "data/processed/cohorts.pkl")], check=True)
