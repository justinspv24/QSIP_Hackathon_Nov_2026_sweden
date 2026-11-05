"""Fetch the IBM job saved in results/hw_job_ibm.json and, if it is done, analyse it exactly like E5.

    python fetch_ibm_results.py              # check once; analyse if done
    python fetch_ibm_results.py --wait 1800  # keep polling up to 30 min

Writes results/e5_ibm_N4.json, results/e5_ibm_N16.json, results/e5_ibm_summary.md and re-renders figures + deck.
"""
import argparse, importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e5_ibm", HERE / "scripts" / "05b_ibm_hw_run.py")
e5_ibm = importlib.util.module_from_spec(spec); spec.loader.exec_module(e5_ibm)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--wait", type=int, default=0, help="seconds to keep polling")
    e5_ibm.collect(timeout=ap.parse_args().wait)
