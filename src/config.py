from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    outputs = Path(cfg["paths"]["outputs_dir"])
    outputs.mkdir(parents=True, exist_ok=True)

    return cfg
