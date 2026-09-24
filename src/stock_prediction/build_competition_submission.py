"""Build the locked E006+C0b competition submission without reading test Y."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .phase_c0 import sha256, transform


def assert_same_keys(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    columns = ["ts_code", "trade_date"]
    if not actual[columns].equals(expected[columns]):
        raise AssertionError(f"{label} keys or row order changed")


def build(root: Path, verify_existing: bool = False) -> Path:
    output_dir = root / "submissions/E006_C0b_fixed"
    output_path = output_dir / "submission.csv"
    building_path = output_dir / "submission.building.csv"
    manifest_path = output_dir / "manifest.json"
    if verify_existing:
        if not output_path.is_file() or not manifest_path.is_file() or building_path.exists():
            raise FileNotFoundError("Complete frozen submission and manifest are required for verification")
    elif any(path.exists() for path in (output_path, building_path, manifest_path)):
        raise FileExistsError("Frozen submission already exists or an incomplete build needs review")

    raw_manifest = json.loads((root / "outputs/data_manifest.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    lock_path = root / "config/competition_submission_v1.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    c0b_path = root / "outputs/phase_c0b/phase_c0b_turnover_comparison.json"
    c0b = json.loads(c0b_path.read_text(encoding="utf-8"))
    params = c0b["selected_parameters"]["E006"]
    if (params["alpha"], params["exit_fraction"]) != (lock["alpha"], lock["exit_fraction"]) or lock["refit"]:
        raise AssertionError("Frozen E006 C0b parameters changed")

    x_path = root / "data/raw/测试集_X.csv"
    expected_x_hash = next(item["sha256_after"] for item in raw_manifest["files"] if Path(item["path"]).name == x_path.name)
    if sha256(x_path) != expected_x_hash:
        raise AssertionError("Organizer test X changed")
    matrix_path = root / "outputs/features/full147/test_full147.npy"
    keys_path = root / "outputs/features/basic40/test_keys.npy"
    files = feature_manifest["datasets"]["test"]["files"]
    for path, item in ((matrix_path, files["matrix"]), (keys_path, files["keys"])):
        if sha256(path) != item["sha256"]:
            raise AssertionError(f"Frozen test feature artifact changed: {path}")
    names = list(feature_manifest["feature_names"])
    if len(names) != 147:
        raise AssertionError("E006 must use exactly frozen Full147")
    matrix = np.load(matrix_path, mmap_mode="r")
    keys = np.load(keys_path, mmap_mode="r")
    x = pd.read_csv(x_path, usecols=["ts_code", "trade_date", "flag_limit_up"], dtype={"ts_code": "string"})
    if len(x) != 1_599_600 or len(keys) != len(x) or matrix.shape != (len(x), 147):
        raise AssertionError("Full competition test shape changed")
    if x[["ts_code", "trade_date"]].duplicated().any():
        raise AssertionError("Duplicate competition test keys")
    if not np.array_equal(x.ts_code.astype(str).to_numpy(), keys["ts_code"].astype(str)) or not np.array_equal(x.trade_date.to_numpy(), keys["trade_date"]):
        raise AssertionError("Raw test X and frozen features are not aligned")

    final_date = int(x.trade_date.max())
    final_mask = x.trade_date.eq(final_date).to_numpy()
    if final_date != 20260608 or final_mask.sum() != 4650:
        raise AssertionError("Expected final test date/count changed")
    previous = x.loc[~final_mask, ["ts_code", "trade_date"]].reset_index(drop=True)
    if len(previous) != 1_594_950 or int(previous.trade_date.max()) != 20260605:
        raise AssertionError("Frozen historical prediction coverage changed")

    old_raw_path = root / lock["raw_prefix"]
    old_policy_path = root / lock["policy_prefix"]
    for path, expected in ((old_raw_path, lock["raw_prefix_sha256"]),
                           (old_policy_path, lock["policy_prefix_sha256"])):
        if sha256(path) != expected:
            raise AssertionError(f"Frozen prediction fingerprint changed: {path}")
    old_raw = pd.read_csv(old_raw_path, dtype={"ts_code": "string"}, float_precision="round_trip")
    old_policy = pd.read_csv(old_policy_path, dtype={"ts_code": "string"}, float_precision="round_trip")
    assert_same_keys(old_raw, previous, "Frozen E006 raw")
    assert_same_keys(old_policy, previous, "Frozen E006 C0b")
    if not np.isfinite(old_raw.pred).all():
        raise AssertionError("Frozen raw prediction is nonfinite")

    model_path = root / lock["model"]
    if sha256(model_path) != lock["model_sha256"]:
        raise AssertionError("Frozen E006 model changed")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    if model.get_booster().num_boosted_rounds() != 800:
        raise AssertionError("Frozen E006 round count changed")
    if model.get_booster().feature_names != names:
        raise AssertionError("Frozen E006 feature order changed")
    last_rows = np.flatnonzero(final_mask)
    last_features = pd.DataFrame(np.asarray(matrix[last_rows]), columns=names, copy=False)
    final_pred = np.asarray(model.predict(last_features), dtype=float)
    if len(final_pred) != 4650 or not np.isfinite(final_pred).all():
        raise AssertionError("Invalid final-date raw predictions")

    full = x.copy()
    full["pred"] = np.nan
    full.loc[~final_mask, "pred"] = old_raw.pred.to_numpy()
    full.loc[final_mask, "pred"] = final_pred
    if not np.isfinite(full.pred).all():
        raise AssertionError("Full raw predictions are nonfinite")
    transformed, _ = transform(full, 0.3, 0.25)
    prefix = transformed.loc[~final_mask, ["ts_code", "trade_date", "pred"]].reset_index(drop=True)
    assert_same_keys(prefix, old_policy, "Frozen E006 C0b replay")
    if not np.array_equal(prefix.pred.to_numpy(), old_policy.pred.to_numpy()):
        raise AssertionError("New full-period postprocess changed previously frozen predictions")

    submission = transformed[["ts_code", "trade_date", "pred"]]
    if submission.columns.tolist() != ["ts_code", "trade_date", "pred"] or not np.isfinite(submission.pred).all():
        raise AssertionError("Invalid competition submission schema or scores")
    if verify_existing:
        reread = pd.read_csv(output_path, dtype={"ts_code": "string"}, float_precision="round_trip")
        assert_same_keys(reread, submission, "Existing competition submission")
        if not np.array_equal(reread.pred.to_numpy(), submission.pred.to_numpy()):
            raise AssertionError("Existing submission differs from locked no-label rebuild")
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(output_path) != recorded["submission_sha256"]:
            raise AssertionError("Existing submission fingerprint changed")
        if recorded.get("lock_config_sha256") != sha256(lock_path):
            raise AssertionError("Submission lock configuration changed")
        return output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(building_path, index=False, float_format="%.17g")
    reread = pd.read_csv(building_path, dtype={"ts_code": "string"}, float_precision="round_trip")
    assert_same_keys(reread, x, "Serialized competition submission")
    if len(reread) != len(x) or not np.isfinite(reread.pred).all() or reread[["ts_code", "trade_date"]].duplicated().any():
        raise AssertionError("Serialized competition submission failed validation")
    building_path.rename(output_path)
    report = {"strategy": "E006 F3 + frozen C0b", "retrained": False,
              "test_labels_read": False, "feature_count": 147, "alpha": 0.3,
              "exit_fraction": 0.25, "rows": len(submission),
              "dates": int(x.trade_date.nunique()), "stocks": int(x.ts_code.nunique()),
              "first_date": int(x.trade_date.min()), "last_date": final_date,
              "newly_predicted_final_date_rows": len(final_pred),
              "frozen_prefix_rows_verified": len(old_policy),
              "input_x_sha256": expected_x_hash,
              "feature_matrix_sha256": files["matrix"]["sha256"],
              "feature_keys_sha256": files["keys"]["sha256"],
              "model_sha256": sha256(model_path),
              "frozen_raw_prefix_sha256": sha256(old_raw_path),
              "frozen_policy_prefix_sha256": sha256(old_policy_path),
              "c0b_decision_sha256": sha256(c0b_path),
              "lock_config_sha256": sha256(lock_path),
              "submission_sha256": sha256(output_path),
              "git_head_before_build": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    print(build(Path.cwd(), verify_existing=args.verify_existing), flush=True)
