import sys
import types

import pytest
import yaml

from stock_prediction.swanlab_tracker import SwanLabTracker, _safe_value, start_swanlab


def write_config(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/swanlab.yaml").write_text(yaml.safe_dump({
        "project": "stock-prediction-codex", "default_mode": "disabled",
        "log_dir": "outputs/swanlab", "public": False,
    }), encoding="utf-8")


def test_disabled_mode_does_not_import_or_write(monkeypatch, tmp_path):
    write_config(tmp_path)
    monkeypatch.delenv("STOCK_SWANLAB_MODE", raising=False)
    tracker = start_swanlab(tmp_path, "smoke", "tests", {"score": 1.0})
    assert not tracker.enabled
    tracker.log({"score": 2.0})
    tracker.finish()
    assert not (tmp_path / "outputs").exists()


def test_online_mode_initializes_and_logs_only_explicit_payload(monkeypatch, tmp_path):
    write_config(tmp_path)
    calls = []
    fake = types.SimpleNamespace(
        init=lambda **kwargs: calls.append(("init", kwargs)),
        log=lambda data, step=None: calls.append(("log", data, step)),
        finish=lambda **kwargs: calls.append(("finish", kwargs)),
    )
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.setenv("STOCK_SWANLAB_MODE", "online")
    monkeypatch.setenv("STOCK_SWANLAB_WORKSPACE", "team")
    tracker = start_swanlab(tmp_path, "D4_F1", "D4", {"fold": "F1", "seed": 42})
    assert tracker.enabled
    tracker.log({"official/score": 0.24}, step=1)
    tracker.finish()
    assert calls[0][1]["project"] == "stock-prediction-codex"
    assert calls[0][1]["workspace"] == "team"
    assert calls[0][1]["public"] is False
    assert "description" not in calls[0][1]
    assert calls[1] == ("log", {"official/score": 0.24}, 1)
    assert calls[2] == ("finish", {"state": "success", "error": None})


def test_logging_failure_disables_tracker_without_raising():
    fake = types.SimpleNamespace(log=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")))
    tracker = SwanLabTracker(fake, enabled=True)
    with pytest.warns(RuntimeWarning, match="logging disabled"):
        tracker.log({"score": 0.2})
    assert not tracker.enabled


def test_safe_value_rejects_non_metadata_objects():
    assert _safe_value({"x": [1, 2.0], "bad": float("nan")}) == {"x": [1, 2.0], "bad": None}
    with pytest.raises(TypeError):
        _safe_value(object())
