"""D5: diagnose residual alpha in the frozen D0 validation predictions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .baselines import save_json
from .build_basic40 import sha256
from .folds import split_fold
from .phase_d0_e006_rank_view import rank_block, resolve_rank_sources
from .swanlab_tracker import start_swanlab


REGIONS = ("full", "top20", "middle60")


def exact_column_spearman(values: np.ndarray, residual: np.ndarray, minimum_rows: int = 30) -> np.ndarray:
    """Exact pairwise-NaN Spearman for columns, grouping identical finite masks."""
    values = np.asarray(values, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    out = np.full(values.shape[1], np.nan, dtype=np.float64)
    groups: dict[bytes, list[int]] = defaultdict(list)
    masks: dict[bytes, np.ndarray] = {}
    residual_finite = np.isfinite(residual)
    for column in range(values.shape[1]):
        mask = residual_finite & np.isfinite(values[:, column])
        key = np.packbits(mask).tobytes()
        groups[key].append(column)
        masks[key] = mask
    for key, columns in groups.items():
        mask = masks[key]
        if int(mask.sum()) < minimum_rows:
            continue
        x = pd.DataFrame(values[np.ix_(mask, columns)]).rank(method="average").to_numpy(dtype=np.float64)
        y = pd.Series(residual[mask]).rank(method="average").to_numpy(dtype=np.float64)
        x -= x.mean(axis=0)
        y -= y.mean()
        denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y))
        valid = denominator > 0
        scores = np.full(len(columns), np.nan, dtype=np.float64)
        scores[valid] = (x[:, valid].T @ y) / denominator[valid]
        out[np.asarray(columns)] = scores
    return out


def residual_and_regions(y: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    finite = np.isfinite(y) & np.isfinite(prediction)
    y_rank = np.full(len(y), np.nan)
    prediction_rank = np.full(len(y), np.nan)
    if finite.any():
        y_rank[finite] = pd.Series(y[finite]).rank(method="average", pct=True).to_numpy()
        prediction_rank[finite] = pd.Series(prediction[finite]).rank(method="average", pct=True).to_numpy()
    residual = y_rank - prediction_rank
    return residual, prediction_rank, {
        "full": finite,
        "top20": finite & (prediction_rank > 0.8),
        "middle60": finite & (prediction_rank > 0.2) & (prediction_rank <= 0.8),
    }


def feature_metadata(root: Path, d0_config: dict, names: list[str]) -> pd.DataFrame:
    spec = yaml.safe_load((root / "config/features_v1.yaml").read_text(encoding="utf-8"))
    categories = {item["name"]: item["category"] for item in spec["features"]}
    _, new_names = resolve_rank_sources(d0_config, names)
    categories.update({name: "cross_section_rank" for name in new_names})
    all_names = names + new_names
    if set(all_names) != set(categories):
        missing = sorted(set(all_names) - set(categories))
        raise ValueError(f"Missing feature metadata: {missing}")
    return pd.DataFrame({"feature": all_names, "family": [categories[name] for name in all_names]})


def augmented_date_block(matrix, rows: np.ndarray, source_columns: list[int]) -> np.ndarray:
    raw = np.asarray(matrix[np.ix_(rows, np.arange(matrix.shape[1]))], dtype=np.float64)
    ranks = rank_block(raw[:, source_columns])
    return np.column_stack((raw, ranks))


def diagnose_fold(root: Path, fold_id: str, config: dict, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    names = list(manifest["feature_names"])
    d0_config = yaml.safe_load((root / "config/phase_d0_e006_rank_view.yaml").read_text(encoding="utf-8"))
    new_sources, _ = resolve_rank_sources(d0_config, names)
    source_columns = [names.index(name) for name in new_sources]
    matrix = np.load(root / "outputs/features/full147/train_full147.npy", mmap_mode="r")
    keys = np.load(root / "outputs/features/basic40/train_keys.npy", mmap_mode="r")
    labels = np.load(root / "outputs/baselines/cache/labels.npy", mmap_mode="r")
    folds = yaml.safe_load((root / "config/folds_v1.yaml").read_text(encoding="utf-8"))
    fold = next(item for item in folds["folds"] if item["id"] == fold_id)
    _, valid, split = split_fold(keys["trade_date"], labels, fold, folds["purge_rule"]["purge_trading_days"])
    prediction_path = root / config["prediction_path"].format(fold=fold_id)
    predictions = pd.read_csv(prediction_path, dtype={"ts_code": "string"}, float_precision="round_trip")
    if len(predictions) != len(valid):
        raise AssertionError("D0 validation row count changed")
    if not np.array_equal(predictions["trade_date"].to_numpy(), keys["trade_date"][valid]):
        raise AssertionError("D0 validation date/order changed")
    if not np.array_equal(predictions["ts_code"].astype(str).to_numpy(), keys["ts_code"][valid].astype(str)):
        raise AssertionError("D0 validation keys/order changed")
    saved_y = predictions["y_ret_1d"].to_numpy(dtype=np.float64)
    if not np.allclose(saved_y, labels[valid], equal_nan=True, rtol=0, atol=0):
        raise AssertionError("D0 saved validation labels changed")

    daily = {region: [] for region in REGIONS}
    daily_d0 = {region: [] for region in REGIONS}
    dates = predictions["trade_date"].to_numpy()
    unique_dates = np.unique(dates)
    for date in unique_dates:
        positions = np.flatnonzero(dates == date)
        rows = valid[positions]
        block = augmented_date_block(matrix, rows, source_columns)
        residual, prediction_rank, region_masks = residual_and_regions(
            saved_y[positions], predictions["pred"].to_numpy()[positions])
        for region in REGIONS:
            mask = region_masks[region]
            scores = exact_column_spearman(block[mask], residual[mask], config["minimum_daily_region_rows"])
            daily[region].append(scores)
            daily_d0[region].append(exact_column_spearman(
                block[mask], prediction_rank[mask], config["minimum_daily_region_rows"]))

    records = []
    for region in REGIONS:
        array = np.asarray(daily[region])
        finite_counts = np.sum(np.isfinite(array), axis=0)
        means = np.divide(np.nansum(array, axis=0), finite_counts,
                          out=np.full(array.shape[1], np.nan), where=finite_counts > 0)
        d0_array = np.asarray(daily_d0[region])
        d0_counts = np.sum(np.isfinite(d0_array), axis=0)
        d0_means = np.divide(np.nansum(d0_array, axis=0), d0_counts,
                             out=np.full(d0_array.shape[1], np.nan), where=d0_counts > 0)
        for index, row in metadata.iterrows():
            records.append({"fold": fold_id, "region": region, "feature": row.feature,
                            "family": row.family, "mean_daily_residual_ic": means[index],
                            "mean_daily_d0_prediction_ic": d0_means[index],
                            "finite_days": int(finite_counts[index]), "validation_days": len(unique_dates)})
    audit = {
        "fold": fold_id, "split": split, "prediction_path": str(prediction_path.relative_to(root)),
        "prediction_sha256": sha256(prediction_path), "validation_rows": len(valid),
        "finite_label_rows": int(np.isfinite(saved_y).sum()), "validation_days": len(unique_dates),
        "trained_models": 0,
    }
    return pd.DataFrame(records), audit


def build_summaries(feature_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = feature_long.pivot(index=["region", "feature", "family"], columns="fold",
                              values="mean_daily_residual_ic").reset_index()
    d0_wide = feature_long.pivot(index=["region", "feature", "family"], columns="fold",
                                 values="mean_daily_d0_prediction_ic").reset_index()
    for fold in ("F1", "F2", "F3"):
        if fold not in wide:
            wide[fold] = np.nan
    fold_values = wide[["F1", "F2", "F3"]].to_numpy(dtype=float)
    finite_counts = np.isfinite(fold_values).sum(axis=1)
    wide["mean_residual_ic"] = np.divide(
        np.nansum(fold_values, axis=1), finite_counts,
        out=np.full(len(wide), np.nan), where=finite_counts > 0)
    wide["median_residual_ic"] = [
        float(np.median(row[np.isfinite(row)])) if np.isfinite(row).any() else np.nan
        for row in fold_values
    ]
    signs = np.sign(fold_values)
    wide["same_direction_3of3"] = np.isfinite(fold_values).all(axis=1) & ((signs > 0).all(axis=1) | (signs < 0).all(axis=1))
    wide["at_least_2_folds_abs_ic_ge_0_01"] = (np.abs(fold_values) >= 0.01).sum(axis=1) >= 2
    wide["stable_meaningful"] = wide["same_direction_3of3"] & wide["at_least_2_folds_abs_ic_ge_0_01"]
    wide["abs_mean_residual_ic"] = wide["mean_residual_ic"].abs()
    wide["mean_d0_prediction_ic"] = d0_wide[["F1", "F2", "F3"]].mean(axis=1, skipna=True)

    family_rows = []
    for (region, family), group in wide.groupby(["region", "family"], sort=True):
        stable = group[group["stable_meaningful"]].sort_values("abs_mean_residual_ic", ascending=False)
        top = stable.head(5)["feature"].tolist()
        family_rows.append({
            "region": region, "family": family, "feature_count": len(group),
            "F1_mean_residual_ic": group["F1"].mean(),
            "F2_mean_residual_ic": group["F2"].mean(),
            "F3_mean_residual_ic": group["F3"].mean(),
            "F1_median_residual_ic": group["F1"].median(),
            "F2_median_residual_ic": group["F2"].median(),
            "F3_median_residual_ic": group["F3"].median(),
            "mean_residual_ic": group["mean_residual_ic"].mean(),
            "median_residual_ic": group["mean_residual_ic"].median(),
            "mean_absolute_residual_ic": group["abs_mean_residual_ic"].mean(),
            "mean_absolute_d0_prediction_ic": group["mean_d0_prediction_ic"].abs().mean(),
            "same_direction_feature_count": int(group["same_direction_3of3"].sum()),
            "stable_meaningful_feature_count": int(group["stable_meaningful"].sum()),
            "stable_features": ", ".join(top),
        })
    return wide.sort_values(["region", "family", "abs_mean_residual_ic"], ascending=[True, True, False]), pd.DataFrame(family_rows)


def write_report(root: Path, feature_summary: pd.DataFrame, family_summary: pd.DataFrame, audits: list[dict]) -> dict:
    out = root / "outputs/phase_d5_residual_alpha"
    out.mkdir(parents=True, exist_ok=True)
    feature_summary.to_csv(out / "feature_residual_ic.csv", index=False, float_format="%.9f")
    family_summary.to_csv(out / "family_summary.csv", index=False, float_format="%.9f")
    candidates = family_summary[family_summary["stable_meaningful_feature_count"] >= 2].copy()
    candidates = candidates.sort_values(["stable_meaningful_feature_count", "mean_absolute_residual_ic"], ascending=False)
    payload = {
        "phase": "D5", "diagnostic_only": True, "trained_models": 0,
        "residual_definition": "daily_rank(y_ret_1d)-daily_rank(D0_prediction)",
        "regions": {"full": "all finite labels", "top20": "D0 prediction rank > 0.8",
                    "middle60": "0.2 < D0 prediction rank <= 0.8"},
        "fold_audits": audits,
        "candidate_family_regions": candidates[["region", "family", "stable_meaningful_feature_count",
                                                   "mean_absolute_residual_ic", "stable_features"]].to_dict("records"),
    }
    save_json(out / "result.json", payload)
    lines = ["# D5 Residual Alpha Diagnostic", "",
             "Only frozen D0 validation predictions were used; no model was trained and no fusion was performed.", "",
             "Residual is the same-date finite-label average percentile rank of realized next-day return minus the corresponding rank of the D0 prediction.", "",
             "| Region | Family | F1 | F2 | F3 | Mean residual IC | Median residual IC | Mean abs(residual IC) | Mean abs(D0 IC) | Stable meaningful | Stable features |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, row in family_summary.sort_values(["region", "mean_absolute_residual_ic"], ascending=[True, False]).iterrows():
        lines.append(f"| {row.region} | {row.family} | {row.F1_mean_residual_ic:.6f} | {row.F2_mean_residual_ic:.6f} | {row.F3_mean_residual_ic:.6f} | {row.mean_residual_ic:.6f} | {row.median_residual_ic:.6f} | {row.mean_absolute_residual_ic:.6f} | {row.mean_absolute_d0_prediction_ic:.6f} | {int(row.stable_meaningful_feature_count)}/{int(row.feature_count)} | {row.stable_features or '—'} |")
    lines += ["", "A feature is marked stable/meaningful only when all three fold ICs have the same sign and at least two folds have absolute IC >= 0.01.",
              "Date-level market features are constant within a date and therefore intentionally remain undefined in a same-date cross-sectional diagnostic.", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    validation = ["# D5 Validation", "", "- Frozen D0 prediction artifacts: hash-checked and row-aligned to the official folds.",
                  "- Labels: exact equality checked against the frozen label cache.",
                  "- Feature construction: Full147 plus the frozen 15 D0 same-date average percentile-rank views.",
                  "- Residual and regions: reconstructed independently for each date using finite labels only.",
                  "- Training/fusion/postprocess: none.", ""]
    (out / "validation.md").write_text("\n".join(validation), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-swanlab", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    config_path = root / "config/phase_d5_residual_alpha.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    d0_config = yaml.safe_load((root / "config/phase_d0_e006_rank_view.yaml").read_text(encoding="utf-8"))
    metadata = feature_metadata(root, d0_config, list(manifest["feature_names"]))
    frames, audits = [], []
    for fold in config["folds"]:
        frame, audit = diagnose_fold(root, fold, config, metadata)
        frames.append(frame); audits.append(audit)
    feature_summary, family_summary = build_summaries(pd.concat(frames, ignore_index=True))
    payload = write_report(root, feature_summary, family_summary, audits)
    if not args.skip_swanlab:
        tracker = start_swanlab(root, "D5_residual_alpha_diagnostic", "D5",
                                {"trained_models": 0, "diagnostic_only": True})
        metrics = {f"{r.region}/{r.family}/mean_abs_ic": float(r.mean_absolute_residual_ic)
                   for r in family_summary.itertuples() if np.isfinite(r.mean_absolute_residual_ic)}
        tracker.log(metrics)
        tracker.finish()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
