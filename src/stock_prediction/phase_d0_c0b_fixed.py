"""Apply the already frozen C0b policy to saved D0 predictions, without fitting."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_official import load_official, replay
from .phase_c0 import diagnostics, sha256, transform

METRICS = ("ic_mean", "annual_excess", "mean_turnover", "final_score",
           "diagnostic_turnover", "missing_top_fraction")


def measures(frame: pd.DataFrame, evaluator, temporary: str) -> dict:
    official = replay(frame, temporary, evaluator)
    diagnostic = diagnostics(frame)
    return {**{key: float(official[key]) for key in METRICS[:4]},
            "diagnostic_turnover": diagnostic["diagnostic_turnover"],
            "missing_top_fraction": diagnostic["missing_top_fraction"]}


def summary(rows: list[dict]) -> dict:
    return {metric: {"mean": float(np.mean([r[metric] for r in rows])),
                     "worst": float((max if metric in {"mean_turnover", "diagnostic_turnover", "missing_top_fraction"} else min)(r[metric] for r in rows)),
                     "std": float(np.std([r[metric] for r in rows], ddof=0))}
            for metric in METRICS}


def main() -> None:
    root = Path.cwd()
    output = root / "outputs/phase_d0_c0b_fixed"
    if (output / "comparison.json").exists():
        raise FileExistsError("Fixed D0+C0b experiment was already scored")
    reference = root / "reference/evaluate_official.py"
    if reference.read_bytes() != (root / "evaluate.py").read_bytes():
        raise AssertionError("Official evaluator copies differ")
    c0b_path = root / "outputs/phase_c0b/phase_c0b_turnover_comparison.json"
    c0b = json.loads(c0b_path.read_text(encoding="utf-8"))
    selected = c0b["selected_parameters"]["E006"]
    if (selected["alpha"], selected["exit_fraction"]) != (0.3, 0.25):
        raise AssertionError("Frozen C0b policy changed")
    evaluator = load_official(root)
    records = []
    with tempfile.TemporaryDirectory(prefix="d0_c0b_fixed_") as temporary:
        for fold in ("F1", "F2", "F3"):
            path = root / "outputs/phase_d0_e006_rank_view" / fold / "predictions.csv.gz"
            frame = pd.read_csv(path, float_precision="round_trip")
            if frame.duplicated(["ts_code", "trade_date"]).any() or not np.isfinite(frame.pred).all():
                raise AssertionError(f"Invalid D0 predictions: {fold}")
            prior = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))
            raw = measures(frame, evaluator, temporary)
            for key in METRICS[:4]:
                if not np.isclose(raw[key], prior["official"][key], rtol=0, atol=1e-12):
                    raise AssertionError(f"D0 raw replay mismatch: {fold}/{key}")
            if not np.isclose(raw["diagnostic_turnover"], prior["diagnostic"]["diagnostic_exclude_missing_y_turnover"], rtol=0, atol=1e-12):
                raise AssertionError(f"D0 diagnostic replay mismatch: {fold}")
            transformed, _ = transform(frame, 0.3, 0.25)
            candidate = measures(transformed, evaluator, temporary)
            records.append({"fold": fold, "rows": len(frame), "input_sha256": sha256(path),
                            "raw": raw, "D0_C0b": candidate,
                            "delta": {key: candidate[key] - raw[key] for key in METRICS}})
            print(f"{fold}: raw={raw['final_score']:.6f}, D0+C0b={candidate['final_score']:.6f}", flush=True)
    baselines = {r["fold"]: r for r in c0b["comparison"] if r["experiment"] == "E006" and r["variant"] == "selected"}
    payload = {"experiment": "D0+C0b fixed transfer", "model_retrained": False,
               "alpha": 0.3, "exit_fraction": 0.25,
               "official_evaluator_sha256": sha256(reference), "c0b_source_sha256": sha256(c0b_path),
               "folds": records,
               "summary": {name: summary([r[name] for r in records]) for name in ("raw", "D0_C0b", "delta")},
               "E006_C0b": {fold: {key: float(row[key]) for key in METRICS} for fold, row in baselines.items()},
               "interpretation": "F1/F2 tuned C0b policy transfer; F3 also previously exposed. All folds exploratory, not independent holdouts."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# D0 + frozen C0b", "", payload["interpretation"], "", "No model retraining or parameter search. Official organizer evaluator replayed unchanged.", "",
             "| Fold | Strategy | IC | Top10 excess | Official turnover | Score | Diagnostic turnover | Missing-label Top |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        fold = record["fold"]
        for label, values in (("D0 raw", record["raw"]), ("D0+C0b", record["D0_C0b"]), ("E006+C0b", payload["E006_C0b"][fold])):
            lines.append(f"| {fold} | {label} | {values['ic_mean']:.6f} | {values['annual_excess']:.6f} | {values['mean_turnover']:.6f} | {values['final_score']:.6f} | {values['diagnostic_turnover']:.6f} | {values['missing_top_fraction']:.2%} |")
    lines += ["", "## Three-fold mean / worst / std", "", "| Strategy | Metric | Mean | Worst | Std |", "|---|---|---:|---:|---:|"]
    for label in ("raw", "D0_C0b", "delta"):
        for metric, values in payload["summary"][label].items():
            lines.append(f"| {label} | {metric} | {values['mean']:.6f} | {values['worst']:.6f} | {values['std']:.6f} |")
    e006_score = float(np.mean([payload["E006_C0b"][fold]["final_score"] for fold in ("F1", "F2", "F3")]))
    change = payload["summary"]["delta"]
    lines += ["", "## Interpretation", "",
              f"D0+C0b mean Score exceeds D0 raw by {change['final_score']['mean']:+.6f}, but exceeds E006+C0b ({e006_score:.6f}) by only {payload['summary']['D0_C0b']['final_score']['mean'] - e006_score:+.6f}. All three fold Score differences against E006+C0b are positive but below 0.003.",
              f"Score decomposition versus D0 raw: IC {0.4 * change['ic_mean']['mean']:+.6f}, Top10 excess {0.3 * change['annual_excess']['mean']:+.6f}, official turnover {-0.3 * change['mean_turnover']['mean']:+.6f}.",
              "The gain comes from lower turnover despite a substantial loss of IC and Top10 excess. Diagnostic turnover also falls; missing-label Top stays negligible. F2 Top10 excess is only 0.169931.",
              "The predeclared meaningful +0.01 Score margin versus E006+C0b is not met. Retain D0 raw as primary Alpha and E006+C0b as low-missing high-Score reference; do not promote D0+C0b or reopen turnover tuning."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def retrospective_test() -> None:
    """One fixed transfer check on already-viewed local test labels; never select parameters here."""
    root = Path.cwd()
    output = root / "outputs/phase_d0_c0b_fixed"
    result_path = output / "retrospective_test.json"
    if result_path.exists():
        raise FileExistsError("D0+C0b retrospective check already exists")
    if not (output / "comparison.json").exists():
        raise FileNotFoundError("Run fold comparison first")
    raw_path = root / "outputs/frozen_test_2025_2026/D0_submission.csv"
    prior = json.loads((root / "outputs/frozen_test_2025_2026/comparison.json").read_text(encoding="utf-8"))
    if sha256(raw_path) != prior["models"]["D0"]["submission_sha256"]:
        raise AssertionError("Frozen test D0 submission changed")
    x_path = root / "data/raw/evaluation/测试集_X.csv"
    y_path = root / "data/raw/evaluation/测试集_Y.csv"
    manifest = prior["evaluation_manifest"]
    if sha256(x_path) != manifest["x_sha256"] or sha256(y_path) != manifest["y_sha256"]:
        raise AssertionError("Retrospective evaluation files changed")
    raw = pd.read_csv(raw_path, float_precision="round_trip")
    x = pd.read_csv(x_path, usecols=["ts_code", "trade_date", "flag_limit_up"])
    y = pd.read_csv(y_path, usecols=["ts_code", "trade_date", "y_ret_1d"])
    for data in (x, y):
        if len(data) != len(raw) or not np.array_equal(data[["ts_code", "trade_date"]].to_numpy(), raw[["ts_code", "trade_date"]].to_numpy()):
            raise AssertionError("Retrospective keys do not align")
    frame = raw.copy()
    frame["flag_limit_up"] = x.flag_limit_up.to_numpy()
    frame["y_ret_1d"] = y.y_ret_1d.to_numpy()
    evaluator = load_official(root)
    with tempfile.TemporaryDirectory(prefix="d0_c0b_test_") as temporary:
        # Replay the original saved submission directly: reserializing its scores
        # can move a few floating-point ties and perturb Spearman by ~1e-7.
        original_official = evaluator(str(raw_path), str(root / "data/raw/evaluation"))
        control = {**{key: float(original_official[key]) for key in METRICS[:4]}, **diagnostics(frame)}
        control = {key: control[key] for key in METRICS}
        for key in METRICS[:4]:
            if not np.isclose(control[key], prior["models"]["D0"]["official"][key], atol=1e-12, rtol=0):
                raise AssertionError(f"Frozen test D0 replay mismatch: {key}")
        transformed, _ = transform(frame, 0.3, 0.25)
        candidate = measures(transformed, evaluator, temporary)
    legacy = json.loads((root / "outputs/frozen_test_2025_2026/legacy_strategies.json").read_text(encoding="utf-8"))
    payload = {"status": "retrospective corroboration only; labels already viewed", "retrained_models": 0,
               "parameters": {"alpha": 0.3, "exit_fraction": 0.25},
               "input_submission_sha256": sha256(raw_path), "evaluation_x_sha256": sha256(x_path),
               "evaluation_y_sha256": sha256(y_path), "official_evaluator_sha256": sha256(root / "reference/evaluate_official.py"),
               "D0_raw": control, "D0_C0b": candidate,
               "E006_C0b": {**{key: legacy["models"]["E006"]["C0b"]["official"][key] for key in METRICS[:4]},
                            "diagnostic_turnover": legacy["models"]["E006"]["C0b"]["diagnostic"]["diagnostic_exclude_missing_y_turnover"],
                            "missing_top_fraction": legacy["models"]["E006"]["C0b"]["diagnostic"]["missing_share_of_top"]}}
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# D0+C0b already-viewed 2025–2026 period", "", payload["status"], "",
             "| Strategy | IC | Top10 excess | Official turnover | Score | Diagnostic turnover | Missing-label Top |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, values in (("D0 raw", control), ("D0+C0b", candidate), ("E006+C0b", payload["E006_C0b"])):
        lines.append(f"| {label} | {values['ic_mean']:.6f} | {values['annual_excess']:.6f} | {values['mean_turnover']:.6f} | {values['final_score']:.6f} | {values['diagnostic_turnover']:.6f} | {values['missing_top_fraction']:.2%} |")
    result_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
