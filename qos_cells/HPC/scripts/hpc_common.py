"""Shared helpers for HPC/scripts/e*.py: paths, logging, run metadata, peak RAM/VRAM tracking, JSON writer.

Every results JSON carries: schema_version 1, the full config (repo config.yaml + the experiment's own block),
git commit, hostname, hardware, wall time, peak RAM, peak VRAM, and `partial` (true until the run finishes).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HPC = Path(__file__).resolve().parents[1]          # qos_cells/HPC
ROOT = HPC.parent                                   # qos_cells
sys.path.insert(0, str(ROOT))
from qos_cells import load_config  # noqa: E402

RESULTS = HPC / "results"
FIGURES = HPC / "figures"
LOGS = HPC / "logs"
DATA = HPC / "data"                                  # gitignored
for _d in (RESULTS, FIGURES, LOGS, DATA):
    _d.mkdir(parents=True, exist_ok=True)


def log(tag, msg):
    print(f"[{tag} {dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def git_commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=ROOT) != 0
        return h + ("-dirty" if dirty else "")
    except Exception as e:  # pragma: no cover
        return f"unknown ({e})"


def hardware():
    hw = dict(hostname=socket.gethostname(), cpu_count=os.cpu_count(), platform=platform.platform(),
              python=sys.version.split()[0])
    try:
        hw["cpu_model"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
        hw["ram_gb"] = round(int([l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")][0]) / 2**20, 1)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                                       "--format=csv,noheader"], text=True)
        hw["gpus_on_node"] = [l.strip() for l in out.strip().splitlines()]
    except Exception:
        hw["gpus_on_node"] = []
    hw["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return hw


class Tracker:
    """Wall time, peak RSS (getrusage, exact) and peak GPU memory (CuPy pool, sampled every 0.2 s + on demand)."""

    def __init__(self):
        self.t0 = time.time()
        self.peak_vram = 0
        self._stop = False
        self._cp = None
        try:
            import cupy as cp
            self._cp = cp
            cp.cuda.Device(0).use()
            threading.Thread(target=self._poll, daemon=True).start()
        except Exception:
            pass

    def _poll(self):
        while not self._stop:
            self.sample()
            time.sleep(0.2)

    def sample(self):
        if self._cp is not None:
            try:
                b = self._cp.get_default_memory_pool().total_bytes()
                self.peak_vram = max(self.peak_vram, int(b))
            except Exception:
                pass

    def stats(self):
        self.sample()
        return dict(wall_seconds=round(time.time() - self.t0, 1),
                    peak_ram_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
                    peak_vram_gb=round(self.peak_vram / 2**30, 3))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(name: str, exp_config: dict, payload: dict, tracker: Tracker, partial: bool):
    cfg = load_config()
    cfg["experiment"] = exp_config
    body = dict(schema_version=1, experiment=name.replace(".json", ""),
                created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                partial=bool(partial), config=cfg, git_commit=git_commit(), hostname=socket.gethostname(),
                hardware=hardware(), resources=tracker.stats())
    body.update(payload)
    out = RESULTS / name
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(clean(body), indent=1))
    tmp.replace(out)                                   # atomic: a crash never leaves a half-written file
    return out


def load_result(name: str) -> dict:
    d = json.loads((RESULTS / name).read_text())
    assert d.get("schema_version") == 1, f"{name}: bad schema_version"
    return d
