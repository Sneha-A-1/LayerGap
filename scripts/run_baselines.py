"""Repeated cross-validation of LayerGap and all baselines (experiment E1)."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(ROOT / "experiments/run_experiments.py"), "--exp", "main"], check=True)
