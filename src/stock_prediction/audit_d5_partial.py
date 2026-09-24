"""Check D5 families against the mechanical rank-difference effect."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .folds import split_fold
from .phase_d5_residual_alpha import residual_and_regions


FAMILIES = ("volatility", "volume_price")
REGIONS = ("full", "top20", "middle60")


def daily_partial_spearman(values: np.ndarray, y: np.ndarray, prediction: np.ndarray,
                           minimum_rows: int = 30) -> np.ndarray:
    """Feature/realized-return rank correlation after linearly controlling D0 rank."""
    result = np.full((values.shape[1], 4), np.nan)
    groups: dict[bytes, list[int]] = defaultdict(list)
    masks: dict[bytes, np.ndarray] = {}
    target_ok = np.isfinite(y) & np.isfinite(prediction)
    for column in range(values.shape[1]):
        mask = target_ok & np.isfinite(values[:, column])
        key = np.packbits(mask).tobytes()
        groups[key].append(column)
        masks[key] = mask
    for key, columns in groups.items():
        mask = masks[key]
        if int(mask.sum()) < minimum_rows:
            continue
        x = pd.DataFrame(values[np.ix_(mask, columns)]).rank(method="average").to_numpy(dtype=float)
        yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
        pr = pd.Series(prediction[mask]).rank(method="average").to_numpy(dtype=float)
        x -= x.mean(axis=0)
        yr -= yr.mean()
        pr -= pr.mean()
        xn = np.sqrt(np.sum(x * x, axis=0))
        yn = np.sqrt(np.sum(yr * yr))
        pn = np.sqrt(np.sum(pr * pr))
        ok = (xn > 0) & (yn > 0) & (pn > 0)
        if not ok.any():
            continue
        xy = (x[:, ok].T @ yr) / (xn[ok] * yn)
        xp = (x[:, ok].T @ pr) / (xn[ok] * pn)
        yp = (yr @ pr) / (yn * pn)
        den = np.sqrt((1 - xp * xp) * (1 - yp * yp))
        partial = np.divide(xy - xp * yp, den, out=np.full(len(xy), np.nan), where=den > 0)
        result[np.asarray(columns)[ok]] = np.column_stack((xy, xp, np.full(len(xy), yp), partial))
    return result


def main() -> None:
    root = Path.cwd()
    spec = yaml.safe_load((root / "config/features_v1.yaml").read_text(encoding="utf-8"))
    family = {item["name"]: item["category"] for item in spec["features"]}
    manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    names = manifest["feature_names"]
    selected = [index for index, name in enumerate(names) if family[name] in FAMILIES]
    features = [names[index] for index in selected]
    matrix = np.load(root / "outputs/features/full147/train_full147.npy", mmap_mode="r")
    keys = np.load(root / "outputs/features/basic40/train_keys.npy", mmap_mode="r")
    labels = np.load(root / "outputs/baselines/cache/labels.npy", mmap_mode="r")
    folds = yaml.safe_load((root / "config/folds_v1.yaml").read_text(encoding="utf-8"))
    records = []
    for fold in folds["folds"]:
        fold_id = fold["id"]
        _, valid, _ = split_fold(keys["trade_date"], labels, fold, 1)
        prediction = pd.read_csv(root / "outputs/phase_d0_e006_rank_view" / fold_id / "predictions.csv.gz",
                                 usecols=["trade_date", "pred", "y_ret_1d"], float_precision="round_trip")
        dates = prediction["trade_date"].to_numpy()
        y = prediction["y_ret_1d"].to_numpy(dtype=float)
        p = prediction["pred"].to_numpy(dtype=float)
        if not np.array_equal(dates, keys["trade_date"][valid]):
            raise AssertionError("Frozen D0 prediction rows changed")
        daily = {region: [] for region in REGIONS}
        for date in np.unique(dates):
            positions = np.flatnonzero(dates == date)
            _, _, region_masks = residual_and_regions(y[positions], p[positions])
            for region in REGIONS:
                rows = positions[region_masks[region]]
                block = np.asarray(matrix[np.ix_(valid[rows], selected)], dtype=float)
                daily[region].append(daily_partial_spearman(block, y[rows], p[rows]))
        for region in REGIONS:
            array = np.asarray(daily[region])
            for column, name in enumerate(features):
                measures = array[:, column, :]
                counts = np.sum(np.isfinite(measures), axis=0)
                means = np.divide(np.nansum(measures, axis=0), counts,
                                  out=np.full(4, np.nan), where=counts > 0)
                records.append({"fold": fold_id, "region": region, "family": family[name], "feature": name,
                                "true_y_ic": means[0], "d0_prediction_ic": means[1],
                                "d0_true_y_ic": means[2], "partial_true_y_ic_given_d0": means[3],
                                "finite_days": int(counts[3])})
        print(f"Audited {fold_id}", flush=True)
    out = root / "outputs/phase_d5_residual_alpha"
    feature_frame = pd.DataFrame(records)
    feature_frame.to_csv(out / "partial_feature_audit.csv", index=False, float_format="%.9f")
    grouped = feature_frame.groupby(["region", "family", "fold"], sort=True)[
        ["true_y_ic", "d0_prediction_ic", "d0_true_y_ic", "partial_true_y_ic_given_d0"]].mean().reset_index()
    grouped.to_csv(out / "partial_family_audit.csv", index=False, float_format="%.9f")
    lines = ["# D5 rank-difference artifact check", "",
             "A feature can correlate with rank(y) - rank(D0) simply because it correlates with D0. This audit reports its direct true-return IC and its partial rank correlation with true return after controlling for D0 prediction rank, computed within each date and finite feature/label pair.", "",
             "| Region | Family | Fold | True-return IC | Feature-D0 IC | D0-return IC | Partial true-return IC given D0 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in grouped.itertuples():
        lines.append(f"| {row.region} | {row.family} | {row.fold} | {row.true_y_ic:.6f} | {row.d0_prediction_ic:.6f} | {row.d0_true_y_ic:.6f} | {row.partial_true_y_ic_given_d0:.6f} |")
    lines += ["", "Partial rank correlation is a diagnostic, not a tradable model result. It is not a training or validation score for a second model.", ""]
    (out / "partial_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
