"""Ablation (E7) and sensitivity (E8) analyses."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for e in ["ablation", "sensitivity"]:
    subprocess.run([sys.executable, str(ROOT / "experiments/run_experiments.py"), "--exp", e], check=True)
