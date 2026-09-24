"""Score frozen F3 models once on the locally reconstructed 2025-2026 test period."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from .baselines import save_json
from .build_basic40 import sha256
from .compare_official import load_official
from .e003 import official_daily_and_diagnostic
from .phase_d0_e006_rank_view import materialize_augmented, resolve_rank_sources
from .ranking import predict_batches


MODELS = {
    "D0": "outputs/phase_d0_e006_rank_view/F3/model.ubj",
    "D3": "outputs/phase_d3_recent_2y/F3/model.ubj",
    "D4": "outputs/phase_d4_lightgbm_rank_regression/F3/model.txt",
}


def check_evaluation_inputs(root: Path, test_keys) -> tuple[pd.DataFrame, dict]:
    folder = root / "data/raw/evaluation"
    manifest = json.loads((folder / "label_manifest.json").read_text(encoding="utf-8"))
    expected_files = {
        root / manifest["source"]: manifest["source_sha256"],
        folder / "测试集_X.csv": manifest["x_sha256"],
        folder / "测试集_Y.csv": manifest["y_sha256"],
    }
    for path, digest in expected_files.items():
        if sha256(path) != digest:
            raise AssertionError(f"Evaluation input hash changed: {path}")
    if sha256(root / "reference/evaluate_official.py") != sha256(root / "evaluate.py"):
        raise AssertionError("Frozen official evaluator changed")
    x_keys = pd.read_csv(folder / "测试集_X.csv", usecols=["ts_code", "trade_date", "flag_limit_up"],
                         dtype={"ts_code": "string"})
    y_keys = pd.read_csv(folder / "测试集_Y.csv", usecols=["ts_code", "trade_date"],
                         dtype={"ts_code": "string"})
    count = int(manifest["rows"])
    if len(x_keys) != count or len(y_keys) != count:
        raise AssertionError("Evaluation X/Y row count differs from manifest")
    if not x_keys[["ts_code", "trade_date"]].equals(y_keys[["ts_code", "trade_date"]]):
        raise AssertionError("Evaluation X/Y key order differs")
    indices = np.flatnonzero(test_keys["trade_date"] != manifest["excluded_date"])
    if len(indices) != count:
        raise AssertionError("Unexpected number of nonfinal test rows")
    if not np.array_equal(x_keys["ts_code"].astype(str).to_numpy(), test_keys["ts_code"][indices].astype(str)):
        raise AssertionError("Evaluation/test-feature stock keys differ")
    if not np.array_equal(x_keys["trade_date"].to_numpy(), test_keys["trade_date"][indices]):
        raise AssertionError("Evaluation/test-feature dates differ")
    return x_keys, manifest


def load_model(name: str, path: Path):
    if name == "D4":
        # The native LightGBM file opener is not Unicode-safe on this Windows path.
        return lgb.Booster(model_str=path.read_text(encoding="utf-8"))
    model = xgb.XGBRegressor()
    model.load_model(path)
    if model.get_booster().num_boosted_rounds() != 800:
        raise AssertionError(f"{name} frozen round count changed")
    return model


def main() -> None:
    root = Path.cwd()
    out = root / "outputs/frozen_test_2025_2026"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "comparison.json"
    if result_path.exists():
        raise FileExistsError("Frozen test was already scored; preserve the first look")
    feature_manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    names = list(feature_manifest["feature_names"])
    feature_config = yaml.safe_load((root / "config/phase_d0_e006_rank_view.yaml").read_text(encoding="utf-8"))
    new_sources, new_names = resolve_rank_sources(feature_config, names)
    all_names = names + new_names
    matrix = np.load(root / "outputs/features/full147/test_full147.npy", mmap_mode="r")
    test_keys = np.load(root / "outputs/features/basic40/test_keys.npy", mmap_mode="r")
    x_keys, evaluation_manifest = check_evaluation_inputs(root, test_keys)
    if matrix.shape != (len(test_keys), 147) or len(all_names) != 162:
        raise AssertionError("Frozen feature dimensions changed")
    count = len(x_keys)
    indices = np.flatnonzero(test_keys["trade_date"] != evaluation_manifest["excluded_date"])
    model_paths = {name: root / relative for name, relative in MODELS.items()}
    model_hashes = {name: sha256(path) for name, path in model_paths.items()}
    with tempfile.TemporaryDirectory(prefix="frozen_test_features_") as temporary:
        augmented = materialize_augmented(
            matrix, test_keys, indices, [names.index(source) for source in new_sources],
            Path(temporary) / "test_162.npy")
        try:
            for name, path in model_paths.items():
                model = load_model(name, path)
                if name == "D4":
                    if model.num_trees() != 800:
                        raise AssertionError("D4 frozen round count changed")
                    pred = np.concatenate([
                        model.predict(pd.DataFrame(augmented[start:start + 100_000], columns=all_names, copy=False))
                        for start in range(0, count, 100_000)
                    ])
                else:
                    pred = predict_batches(model, augmented, np.arange(count), all_names)
                if not np.isfinite(pred).all():
                    raise AssertionError(f"{name} predictions contain NaN/Inf")
                submission = x_keys[["ts_code", "trade_date"]].copy()
                submission["pred"] = pred
                submission.to_csv(out / f"{name}_submission.csv", index=False, float_format="%.17g")
                print(f"Locked {name} frozen predictions: {len(pred)} rows", flush=True)
        finally:
            augmented._mmap.close()

    # Read reconstructed labels only after all frozen prediction files are locked.
    labels = pd.read_csv(root / "data/raw/evaluation/测试集_Y.csv", usecols=["y_ret_1d"])
    evaluator_path = root / "reference/evaluate_official.py"
    evaluator = load_official(root)
    models = {}
    for name in MODELS:
        submission_path = out / f"{name}_submission.csv"
        submission = pd.read_csv(submission_path, dtype={"ts_code": "string"}, float_precision="round_trip")
        frame = submission.copy()
        frame["y_ret_1d"] = labels["y_ret_1d"].to_numpy()
        frame["flag_limit_up"] = x_keys["flag_limit_up"].to_numpy()
        official = evaluator(str(submission_path), str(root / "data/raw/evaluation"))
        _, diagnostic, _, _ = official_daily_and_diagnostic(frame)
        models[name] = {"official": official, "diagnostic": diagnostic,
                        "submission": str(submission_path.relative_to(root)),
                        "submission_sha256": sha256(submission_path),
                        "model_sha256": model_hashes[name]}
        print(f"Scored {name}: {official['final_score']:.6f}", flush=True)
    result = {
        "evaluation": "retrospectively reconstructed 2025-01-02 to 2026-06-05 test labels",
        "first_look": True, "retrained_models": 0,
        "training_cutoff": "F3 model-specific training ends no later than 2023-12-28",
        "evaluation_manifest": evaluation_manifest,
        "evaluator_sha256": sha256(evaluator_path),
        "evaluation_folder_evaluator_sha256": sha256(root / "data/raw/evaluation/evaluate.py"),
        "feature_matrix_sha256_recorded": feature_manifest["datasets"]["test"]["files"]["matrix"]["sha256"],
        "models": models,
    }
    save_json(result_path, result)
    lines = ["# Frozen model test period, first look", "",
             "This is a retrospective local evaluation of labels reconstructed from the next test row's close. No model was retrained or selected with these labels.", "",
             "F3 models were frozen before this evaluation and trained through 2023 at the latest; 2024 was their validation year.", "",
             "| Model | Rank IC | Top10 annual excess | Official turnover | Official Score | Diagnostic turnover | Missing-label Top fraction |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in models.items():
        off, diag = item["official"], item["diagnostic"]
        lines.append(f"| {name} | {off['ic_mean']:.6f} | {off['annual_excess']:.6f} | {off['mean_turnover']:.6f} | {off['final_score']:.6f} | {diag['diagnostic_exclude_missing_y_turnover']:.6f} | {diag['missing_share_of_top']:.2%} |")
    lines += ["", "The table is a single test-period observation, not an extra training fold or a basis for tuning.", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def score_legacy_strategies() -> None:
    """One fixed comparison of the already selected E005/E006 C0b strategies."""
    from .phase_c0 import transform

    root = Path.cwd()
    out = root / "outputs/frozen_test_2025_2026"
    result_path = out / "legacy_strategies.json"
    if result_path.exists():
        raise FileExistsError("Frozen legacy strategies were already scored")
    if not (out / "comparison.json").exists():
        raise FileNotFoundError("Score frozen D0/D3/D4 before the legacy comparison")
    manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    names = list(manifest["feature_names"])
    matrix = np.load(root / "outputs/features/full147/test_full147.npy", mmap_mode="r")
    keys = np.load(root / "outputs/features/basic40/test_keys.npy", mmap_mode="r")
    x_keys, evaluation_manifest = check_evaluation_inputs(root, keys)
    indices = np.flatnonzero(keys["trade_date"] != evaluation_manifest["excluded_date"])
    c0b = json.loads((root / "outputs/phase_c0b/phase_c0b_turnover_comparison.json").read_text(encoding="utf-8"))
    evaluator = load_official(root)
    model_files = {
        "E005": root / "outputs/baselines/E005/F3/model.txt",
        "E006": root / "outputs/baselines/E006/F3/model.ubj",
    }
    prediction_files = {}
    for experiment, path in model_files.items():
        model = load_model("D4" if experiment == "E005" else "D0", path)
        pieces = []
        for start in range(0, len(indices), 100_000):
            rows = indices[start:start + 100_000]
            block = pd.DataFrame(np.asarray(matrix[rows]), columns=names, copy=False)
            pieces.append(model.predict(block))
        prediction = np.concatenate(pieces)
        if not np.isfinite(prediction).all():
            raise AssertionError(f"{experiment} raw prediction nonfinite")
        raw = x_keys[["ts_code", "trade_date", "flag_limit_up"]].copy()
        raw["pred"] = prediction
        raw_path = out / f"{experiment}_raw_submission.csv"
        raw[["ts_code", "trade_date", "pred"]].to_csv(raw_path, index=False, float_format="%.17g")
        parameter = c0b["selected_parameters"][experiment]
        if (parameter["alpha"], parameter["exit_fraction"]) != (0.3, 0.25):
            raise AssertionError("Frozen C0b selected parameters changed")
        transformed, _ = transform(raw, parameter["alpha"], parameter["exit_fraction"])
        selected_path = out / f"{experiment}_C0b_submission.csv"
        transformed[["ts_code", "trade_date", "pred"]].to_csv(
            selected_path, index=False, float_format="%.17g")
        prediction_files[experiment] = {"raw": raw_path, "C0b": selected_path}
        print(f"Locked {experiment} raw and C0b predictions", flush=True)

    labels = pd.read_csv(root / "data/raw/evaluation/测试集_Y.csv", usecols=["y_ret_1d"])
    results = {}
    for experiment, variants in prediction_files.items():
        results[experiment] = {}
        for variant, path in variants.items():
            frame = pd.read_csv(path, dtype={"ts_code": "string"}, float_precision="round_trip")
            frame["y_ret_1d"] = labels["y_ret_1d"].to_numpy()
            frame["flag_limit_up"] = x_keys["flag_limit_up"].to_numpy()
            official = evaluator(str(path), str(root / "data/raw/evaluation"))
            _, diagnostic, _, _ = official_daily_and_diagnostic(frame)
            results[experiment][variant] = {
                "official": official, "diagnostic": diagnostic,
                "submission_sha256": sha256(path), "model_sha256": sha256(model_files[experiment]),
            }
            print(f"Scored {experiment} {variant}: {official['final_score']:.6f}", flush=True)
    payload = {"scope": "previously frozen E005/E006 F3 models and C0b postprocess",
               "new_training": False, "parameter_search": False, "test_labels_used_for_choice": False,
               "evaluation_manifest": evaluation_manifest, "models": results}
    save_json(result_path, payload)
    lines = ["# Frozen E-series strategy check on the 2025-2026 test period", "",
             "The C0b parameters were fixed in F1/F2 before this period: alpha=0.30 and exit_fraction=0.25. The F3 model files were not retrained. These results are a one-time comparison and do not reopen the turnover grid.", "",
             "| Strategy | Rank IC | Top10 excess | Official turnover | Score | Diagnostic turnover | Missing-label Top |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    d0 = json.loads((out / "comparison.json").read_text(encoding="utf-8"))["models"]["D0"]
    for label, item in [("D0 raw", d0), *[
        (f"{experiment} {variant}", results[experiment][variant])
        for experiment in ("E005", "E006") for variant in ("raw", "C0b")
    ]]:
        official, diagnostic = item["official"], item["diagnostic"]
        lines.append(f"| {label} | {official['ic_mean']:.6f} | {official['annual_excess']:.6f} | {official['mean_turnover']:.6f} | {official['final_score']:.6f} | {diagnostic['diagnostic_exclude_missing_y_turnover']:.6f} | {diagnostic['missing_share_of_top']:.2%} |")
    lines += ["", "The local labels are reconstructed retrospectively from the next test row's close.", ""]
    (out / "legacy_strategies.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-strategies", action="store_true")
    args = parser.parse_args()
    score_legacy_strategies() if args.legacy_strategies else main()
