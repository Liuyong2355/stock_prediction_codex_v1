"""Load the canonical primary raw-alpha model (historical D0 F3 artifact)."""
from __future__ import annotations

import json
from pathlib import Path

import xgboost as xgb
import yaml

from .build_basic40 import sha256
from .phase_d0_e006_rank_view import resolve_rank_sources


def load_main_model(root: Path):
    root = Path(root)
    config = yaml.safe_load((root / "config/main_model.yaml").read_text(encoding="utf-8"))
    if config["model_id"] != "main_alpha_xgb_rank162" or config["historical_alias"] != "D0":
        raise ValueError("Primary model identity changed")
    path = root / config["artifact"]
    if sha256(path) != config["artifact_sha256"]:
        raise ValueError("Primary model artifact hash mismatch")
    manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    base_names = list(manifest["feature_names"])
    rank_config = yaml.safe_load((root / config["features"]["rank_view_config"]).read_text(encoding="utf-8"))
    _, new_names = resolve_rank_sources(rank_config, base_names)
    names = base_names + new_names
    if len(names) != config["features"]["count"]:
        raise ValueError("Primary feature count changed")
    model = xgb.XGBRegressor()
    model.load_model(path)
    booster = model.get_booster()
    if booster.feature_names != names or booster.num_boosted_rounds() != config["rounds"]:
        raise ValueError("Primary model feature order or round count changed")
    return model, names
