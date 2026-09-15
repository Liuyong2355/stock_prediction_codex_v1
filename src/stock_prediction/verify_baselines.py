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
from .evaluator import evaluate, groups
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
            top_slots = missing_label_slots = 0
            for _, day in p.groupby("trade_date", sort=True):
                selected = groups(day.loc[np.isfinite(day.pred) & (day.flag_limit_up != 1)])
                if selected is not None:
                    top = selected[0]
                    top_slots += len(top)
                    missing_label_slots += int((~np.isfinite(top.y_ret_1d)).sum())
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
                            "turnover_top_slots": top_slots, "turnover_top_missing_label_slots": missing_label_slots,
                            "turnover_top_missing_label_fraction": missing_label_slots / top_slots if top_slots else None,
                            "official_model": model_info})
            print(f"Verified saved {experiment}/{fold['id']}", flush=True)
    log = pd.read_csv(root / "outputs/experiment_log.csv")
    assert len(log) == 9 and not log.duplicated(["experiment_id", "fold_id"]).any()
    assert set(config["experiment_log"]["required_columns"]) <= set(log.columns)
    save_json(root / "outputs/baseline_verification.json", {"status": "passed", "evaluator_status": "provisional",
                                                            "official_evaluator_parity": "different; see separate official_evaluator_comparison.json", "folds": checked})
    report = root / "outputs/baseline_report.md"
    text = report.read_text(encoding="utf-8").split("\n## 换手Top集合的标签覆盖诊断")[0]
    text += "\n## 换手Top集合的标签覆盖诊断\n\n换手Top集合按冻结规则不排除缺失标签，收益Top集合则排除。下表是各日换手Top记录中缺失标签的占比（按记录数加权），仅用于解释现有分数，未用于训练或修改评分。较低的该指标换手率不能直接解释为可交易组合的低换手。\n\n| 实验/折 | 换手Top中缺失标签占比 |\n|---|---:|\n"
    for item in checked:
        text += f"| {item['experiment_id']}/{item['fold_id']} | {item['turnover_top_missing_label_fraction']:.4%} |\n"
    text += "\n保存预测重新评分、逐日指标重算、完整键/标签对齐及文件指纹均通过。官方模型类型、Ridge训练预处理范围和LightGBM实际800轮已核验；结果见baseline_verification.json。\n"
    report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    verify(args.root.resolve())
