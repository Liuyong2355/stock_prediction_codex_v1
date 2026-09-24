"""Optional, failure-safe SwanLab tracking for aggregate experiment metadata."""
from __future__ import annotations

import importlib
import math
import os
import warnings
from pathlib import Path
from typing import Any, Mapping

import yaml


ALLOWED_MODES = {"disabled", "online", "local", "offline"}


def _safe_value(value: Any) -> Any:
    """Keep only small JSON-like metadata; reject file/data/model objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    raise TypeError(f"SwanLab metadata must be JSON-like, got {type(value).__name__}")


class SwanLabTracker:
    """Thin wrapper whose failures never interrupt model training."""

    def __init__(self, module=None, enabled=False):
        self._module = module
        self.enabled = enabled

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            payload = _safe_value(metrics)
            self._module.log(payload, step=step)
        except Exception as exc:  # tracking must never invalidate an experiment
            warnings.warn(f"SwanLab logging disabled after error: {exc}", RuntimeWarning, stacklevel=2)
            self.enabled = False

    def finish(self, state="success", error: str | None = None) -> None:
        if not self.enabled:
            return
        try:
            self._module.finish(state=state, error=error)
        except Exception as exc:
            warnings.warn(f"SwanLab finish failed: {exc}", RuntimeWarning, stacklevel=2)
        finally:
            self.enabled = False


def start_swanlab(root: Path, experiment_name: str, group: str, metadata: Mapping[str, Any],
                  description: str = "") -> SwanLabTracker:
    """Start tracking only when SWANLAB_MODE explicitly enables it."""
    root = Path(root)
    settings = yaml.safe_load((root / "config/swanlab.yaml").read_text(encoding="utf-8"))
    mode = os.getenv("STOCK_SWANLAB_MODE", settings["default_mode"]).strip().lower()
    if mode not in ALLOWED_MODES:
        warnings.warn(f"Invalid STOCK_SWANLAB_MODE={mode!r}; tracking disabled", RuntimeWarning, stacklevel=2)
        return SwanLabTracker()
    if mode == "disabled":
        return SwanLabTracker()
    try:
        swanlab = importlib.import_module("swanlab")
        kwargs = {
            "mode": mode,
            "project": os.getenv("STOCK_SWANLAB_PROJECT", settings["project"]),
            "name": experiment_name,
            "group": group,
            "public": bool(settings["public"]),
            "config": _safe_value(metadata),
            "log_dir": str(root / settings["log_dir"]),
        }
        if description:
            kwargs["description"] = description
        workspace = os.getenv("STOCK_SWANLAB_WORKSPACE", "").strip()
        if workspace:
            kwargs["workspace"] = workspace
        swanlab.init(**kwargs)
        return SwanLabTracker(swanlab, enabled=True)
    except Exception as exc:
        warnings.warn(f"SwanLab unavailable; training continues without tracking: {exc}",
                      RuntimeWarning, stacklevel=2)
        return SwanLabTracker()
