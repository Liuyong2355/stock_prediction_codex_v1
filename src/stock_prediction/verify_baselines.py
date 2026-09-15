"""Read-only verification of saved predictions, scores and official models."""
import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .baselines import configs, save_json
from .build_basic40 import sha256
from .evaluator import evaluate
from .folds import split_fold


def verify(root):
    manifest = json.loads((root / "outputs/basic40_manifest.json").read_text(encoding="utf-8"))
    files = manifest["datasets"]["train"]["files"]
    keys = np.load(files["keys"]["path"], mmap_mode="r", allow_pickle=False)
    matrix = np.load(files["matrix"]["path"], mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "outputs/baselines/cache/labels.npy", mmap_mode="r", allow_pickle=False)
    folds, config = configs(root)
    checked = []
    for experiment in ["E000", "E001", "E002"]:
        for fold in folds["folds"]:
            folder = root / "outputs/baselines" / experiment / fold["id"]
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            for name, info in result["artifact_files"].items():
                assert sha256(folder / name) == info["sha256"], (folder, name)
            train_idx, valid_idx, split = split_fold(keys["trade_date"], labels, fold, folds["purge_rule"]["purge_trading_days"])
            assert split == result["split"]
            p = pd.read_csv(folder / "predictions.csv.gz", dtype={"ts_code": "string"}, float_precision="round_trip")
            assert len(p) == len(valid_idx)
            assert np.array_equal(p.ts_code.astype(str), keys["ts_code"][valid_idx])
            assert np.array_equal(p.trade_date, keys["trade_date"][valid_idx])
            assert np.array_equal(p.y_ret_1d, labels[valid_idx], equal_nan=True)
            assert np.array_equal(p.flag_limit_up, matrix[valid_idx, manifest["feature_names"].index("flag_limit_up")])
            reevaluated, daily = evaluate(p)
            assert reevaluated["coverage"] == result["coverage"]
            for metric, actual in reevaluated["metrics"].items():
                expected = result["metrics"][metric]
                assert (expected is None and np.isnan(actual)) or np.isclose(actual, expected, rtol=1e-12, atol=1e-14), metric
            saved_daily = pd.read_csv(folder / "daily_metrics.csv", float_precision="round_trip")
            np.testing.assert_allclose(daily.to_numpy(), saved_daily.to_numpy(), rtol=1e-12, atol=1e-14, equal_nan=True)
            model_info = None
            if experiment != "E000":
                artifact = joblib.load(folder / "model.joblib")  # Only project-generated, hash-verified models.
                model = artifact["model"]
                assert isinstance(model, Ridge if experiment == "E001" else lgb.LGBMRegressor)
                assert model.n_features_in_ == result["n_features"]
                for key, value in result["configured_params"].items():
                    assert model.get_params()[key] == value
                if experiment == "E001":
                    assert artifact["preprocessor"].metadata() == result["preprocessing"]
                    assert result["preprocessing"]["scaler_fit_rows"] == len(train_idx)
                    model_info = {"class": "sklearn.linear_model.Ridge", "solver_used": model.solver_}
                else:
                    assert model.booster_.current_iteration() == result["configured_params"]["n_estimators"]
                    model_info = {"class": "lightgbm.LGBMRegressor", "iterations": model.booster_.current_iteration()}
            else:
                assert np.array_equal(p.pred, matrix[valid_idx, manifest["feature_names"].index("ret_1")], equal_nan=True)
            checked.append({"experiment_id": experiment, "fold_id": fold["id"], "rows": len(p),
                            "artifact_hashes": "passed", "full_prediction_key_label_alignment": "passed",
                            "saved_prediction_metric_roundtrip": "passed", "daily_metrics_roundtrip": "passed",
                            "official_model": model_info})
            print(f"Verified saved {experiment}/{fold['id']}", flush=True)
    log = pd.read_csv(root / "outputs/experiment_log.csv")
    assert len(log) == 9 and not log.duplicated(["experiment_id", "fold_id"]).any()
    assert set(config["experiment_log"]["required_columns"]) <= set(log.columns)
    save_json(root / "outputs/baseline_verification.json", {"status": "passed", "evaluator_status": "provisional",
                                                            "official_evaluator_parity": "unavailable", "folds": checked})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    verify(args.root.resolve())
