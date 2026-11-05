"""results/*.json writer. Every file carries schema_version, a timestamp and the config it was produced with,
so slides and the dashboard can refuse a file they do not understand instead of showing a blank chart."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import ROOT, load_config

SCHEMA_VERSION = 1


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(name: str, payload: dict, config: dict | None = None) -> Path:
    out = ROOT / "results" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    body = dict(schema_version=SCHEMA_VERSION, experiment=name.replace(".json", ""),
                created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                config=config if config is not None else load_config())
    body.update(_clean(payload))
    out.write_text(json.dumps(body, indent=1))
    print(f"[results] wrote {out.relative_to(ROOT)} ({out.stat().st_size/1024:.1f} kB)")
    return out


def load_json(name: str) -> dict:
    p = ROOT / "results" / name
    d = json.loads(p.read_text())
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{name}: schema_version {d.get('schema_version')} != {SCHEMA_VERSION}")
    return d
