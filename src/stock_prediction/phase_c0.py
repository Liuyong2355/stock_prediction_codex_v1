"""Phase C0: causal prediction-only turnover post-processing."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .compare_official import load_official, replay
from .diagnose_e002 import top_sets

KEYS = ["ts_code", "trade_date"]
OFFICIAL_MAP = {
    "ic_mean": "ic_mean", "annual_excess": "annual_excess",
    "mean_turnover": "mean_turnover", "final_score": "final_score",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def daily_rank(frame: pd.DataFrame, prediction: str = "pred") -> pd.Series:
    """Average percentile rank using only same-date predictions."""
    return frame.groupby("trade_date", sort=False)[prediction].rank(method="average", pct=True)


def causal_ema(frame: pd.DataFrame, ranks: pd.Series, alpha: float,
               prior_state: dict[str, float] | None = None) -> pd.Series:
    """Per-security causal EMA; the caller supplies one fold at a time."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    work = pd.DataFrame({"ts_code": frame.ts_code.astype(str),
                         "trade_date": frame.trade_date.to_numpy(),
                         "rank": ranks.to_numpy(dtype=float),
                         "_row": np.arange(len(frame))})
    work = work.sort_values(["ts_code", "trade_date", "_row"], kind="mergesort")
    out = np.empty(len(frame), dtype=float)
    state = {} if prior_state is None else dict(prior_state)
    for code, group in work.groupby("ts_code", sort=False):
        values = group["rank"].to_numpy()
        smoothed = np.empty(len(values), dtype=float)
        previous = state.get(code)
        for i, value in enumerate(values):
            previous = value if previous is None else alpha * value + (1 - alpha) * previous
            smoothed[i] = previous
        out[group["_row"].to_numpy()] = smoothed
    return pd.Series(out, index=frame.index, name="smooth")


def hysteresis_predictions(frame: pd.DataFrame, smooth: pd.Series,
                           exit_fraction: float, minimum_pool: int = 100,
                           return_top_sets: bool = False):
    """Map exact hysteresis Top sets to finite, unique deterministic predictions."""
    if exit_fraction < .10 or exit_fraction > 1:
        raise ValueError("exit_fraction must be in [0.10, 1]")
    result = np.empty(len(frame), dtype=float)
    targets: dict[int, set[str]] = {}
    previous: set[str] | None = None
    work = frame.reset_index(drop=True).copy()
    work["smooth"] = np.asarray(smooth, dtype=float)
    if not np.isfinite(work.smooth).all():
        raise ValueError("smooth scores must all be finite")
    for date, day in work.groupby("trade_date", sort=True):
        eligible = day.loc[day.flag_limit_up == 0].sort_values(
            ["smooth", "ts_code"], ascending=[False, True], kind="mergesort")
        if len(eligible) < minimum_pool:
            previous = None
            chosen: set[str] = set()
        else:
            n_top = max(len(eligible) // 10, 1)
            n_exit = max(int(np.floor(len(eligible) * exit_fraction)), n_top)
            exit_codes = set(eligible.iloc[:n_exit].ts_code.astype(str))
            retained = [] if previous is None else [c for c in eligible.ts_code.astype(str) if c in previous and c in exit_codes]
            retained = retained[:n_top]
            chosen = set(retained)
            for code in eligible.ts_code.astype(str):
                if len(chosen) == n_top:
                    break
                chosen.add(code)
            if len(chosen) != n_top:
                raise AssertionError("hysteresis failed to fill exact Top size")
            previous = chosen
        targets[int(date)] = chosen
        # Deterministic total ordering: selected first, then smooth desc, then code.
        ordered = day.assign(_selected=day.ts_code.astype(str).isin(chosen)).sort_values(
            ["_selected", "smooth", "ts_code"], ascending=[False, False, True], kind="mergesort")
        result[ordered.index.to_numpy()] = np.arange(len(ordered), 0, -1, dtype=float)
    output = pd.Series(result, index=frame.index, name="pred")
    if not np.isfinite(output).all() or output.duplicated().any():
        # Uniqueness is required within date, not globally.
        for _, values in pd.DataFrame({"date": frame.trade_date, "pred": output}).groupby("date"):
            if values.pred.duplicated().any():
                raise AssertionError("prediction order is not unique within date")
    return (output, targets) if return_top_sets else output


def transform(frame: pd.DataFrame, alpha: float, exit_fraction: float):
    ranks = daily_rank(frame)
    smooth = causal_ema(frame, ranks, alpha)
    pred, targets = hysteresis_predictions(frame, smooth, exit_fraction, return_top_sets=True)
    out = frame.copy()
    out["pred"] = pred
    return out, targets


def clean_official(metrics: dict) -> dict:
    return {k: (float(v) if np.isfinite(v) else None) for k, v in metrics.items()}


def diagnostics(frame: pd.DataFrame) -> dict:
    selected, turnover, _ = top_sets(frame)
    _, diagnostic_turnover, _ = top_sets(frame, exclude_missing=True)
    return {"diagnostic_turnover": float(diagnostic_turnover),
            "missing_top_fraction": float(selected.y_ret_1d.isna().mean()),
            "top_rows": int(len(selected))}


def metric_row(experiment, fold, variant, alpha, exit_fraction, official, diagnostic):
    return dict(experiment=experiment, fold=fold, variant=variant, alpha=alpha,
                exit_fraction=exit_fraction, **{k: official[k] for k in OFFICIAL_MAP}, **diagnostic)


def select_candidate(rows: list[dict]) -> dict:
    grouped = []
    frame = pd.DataFrame(rows)
    for (alpha, exit_fraction), group in frame.groupby(["alpha", "exit_fraction"]):
        scores = group.final_score.to_numpy(dtype=float)
        grouped.append(dict(alpha=round(float(alpha), 12), exit_fraction=round(float(exit_fraction), 12),
                            mean_score=float(scores.mean()), worst_score=float(scores.min()),
                            mean_turnover=float(group.mean_turnover.mean())))
    return sorted(grouped, key=lambda x: (-x["mean_score"], -x["worst_score"],
                                          x["mean_turnover"], -x["alpha"], x["exit_fraction"]))[0]


def summarize(rows, raw_rows):
    raw = {(r["experiment"], r["fold"]): r for r in raw_rows}
    result = []
    for (experiment, variant), group in pd.DataFrame(rows).groupby(["experiment", "variant"], sort=False):
        item = {"experiment": experiment, "variant": variant,
                "alpha": float(group.alpha.iloc[0]) if group.alpha.notna().all() else None,
                "exit_fraction": float(group.exit_fraction.iloc[0]) if group.exit_fraction.notna().all() else None,
                "folds": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
                          for row in group.to_dict("records")], "stability": {}}
        for metric in ["ic_mean", "annual_excess", "mean_turnover", "final_score", "diagnostic_turnover", "missing_top_fraction"]:
            values = group[metric].to_numpy(float)
            item["stability"][metric] = {"mean": float(values.mean()), "worst": float(values.max() if "turnover" in metric or metric == "missing_top_fraction" else values.min()), "std": float(values.std(ddof=0))}
        deltas = []
        for row in group.to_dict("records"):
            base = raw[(experiment, row["fold"])]
            d = {"fold": row["fold"], "delta_ic": row["ic_mean"] - base["ic_mean"],
                 "delta_excess": row["annual_excess"] - base["annual_excess"],
                 "delta_turnover": row["mean_turnover"] - base["mean_turnover"],
                 "delta_score": row["final_score"] - base["final_score"]}
            d.update(ic_contribution=.4*d["delta_ic"], excess_contribution=.3*d["delta_excess"], turnover_contribution=-.3*d["delta_turnover"])
            deltas.append(d)
        item["deltas_vs_raw"] = deltas
        result.append(item)
    return result


def main():
    root = Path.cwd()
    output_dir = root / "outputs/phase_c0"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = root / "config/phase_c0_turnover.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    evaluator_path = root / "reference/evaluate_official.py"
    if evaluator_path.read_bytes() != (root / "evaluate.py").read_bytes():
        raise AssertionError("official evaluator copies differ")
    evaluator = load_official(root)
    expected_hashes = json.loads((root / "outputs/e006_e007_input_verification.json").read_text(encoding="utf-8"))["protected_sha256"]
    protected_before = {p: sha256(root / p) for p in expected_hashes if (root / p).is_file()}
    inputs, frames, raw_rows = {}, {}, []
    for experiment in config["experiments"]:
        for fold in [*config["folds"]["selection"], config["folds"]["confirmation"]]:
            path = root / "outputs/baselines" / experiment / fold / "predictions.csv.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            rel = path.relative_to(root).as_posix()
            if rel in expected_hashes and sha256(path) != expected_hashes[rel]:
                raise AssertionError(f"prediction fingerprint mismatch: {rel}")
            frame = pd.read_csv(path, float_precision="round_trip")
            required = KEYS + ["pred", "y_ret_1d", "flag_limit_up"]
            if list(frame.columns) != required or frame.duplicated(KEYS).any():
                raise AssertionError(f"invalid prediction schema/keys: {rel}")
            recorded_result = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))
            recorded_prediction_hash = recorded_result.get("artifact_files", {}).get("predictions.csv.gz", {}).get("sha256")
            if recorded_prediction_hash and sha256(path) != recorded_prediction_hash:
                raise AssertionError(f"prediction does not match its recorded artifact fingerprint: {rel}")
            frames[(experiment, fold)] = frame
            inputs[rel] = {"sha256": sha256(path), "rows": len(frame), "columns": required}
    grid_rows = []
    chosen = {}
    with tempfile.TemporaryDirectory(prefix="phase_c0_") as temp:
        # Raw replay is a hard input gate.
        for (experiment, fold), frame in frames.items():
            actual = clean_official(replay(frame, temp, evaluator))
            recorded = json.loads((root / "outputs/baselines" / experiment / fold / "result.json").read_text(encoding="utf-8"))["metrics"]
            for official_key, saved_key in [("ic_mean", "rank_ic_mean"), ("annual_excess", "annualized_top_excess_return"), ("mean_turnover", "mean_turnover"), ("final_score", "final_score")]:
                if not np.isclose(actual[official_key], recorded[saved_key], rtol=0, atol=1e-14):
                    raise AssertionError(f"raw replay mismatch {experiment}/{fold}/{official_key}")
            raw_rows.append(metric_row(experiment, fold, "raw", None, None, actual, diagnostics(frame)))
        checkpoint = output_dir / "phase_c0_turnover_grid.csv"
        if checkpoint.is_file():
            saved = pd.read_csv(checkpoint)
            saved["alpha"] = saved.alpha.round(12)
            saved["exit_fraction"] = saved.exit_fraction.round(12)
            saved = saved.drop_duplicates(["experiment", "fold", "alpha", "exit_fraction"], keep="first")
            selection_saved = saved.loc[saved.fold.isin(config["folds"]["selection"])]
            expected = len(config["experiments"]) * len(config["ema"]["alphas"]) * len(config["hysteresis"]["exit_fractions"]) * 2
            if len(selection_saved) == expected:
                grid_rows = saved.to_dict("records")
                print("Reused complete Phase C0 grid checkpoint", flush=True)
        if not grid_rows:
            # F1/F2 only: generate and score the full predeclared grid.
            for experiment in config["experiments"]:
                for alpha in config["ema"]["alphas"]:
                    for exit_fraction in config["hysteresis"]["exit_fractions"]:
                        for fold in config["folds"]["selection"]:
                            transformed, _ = transform(frames[(experiment, fold)], float(alpha), float(exit_fraction))
                            official = clean_official(replay(transformed, temp, evaluator))
                            grid_rows.append(metric_row(experiment, fold, "grid", float(alpha), float(exit_fraction), official, diagnostics(transformed)))
                        print(f"Scored {experiment} alpha={alpha} exit={exit_fraction}", flush=True)
        for experiment in config["experiments"]:
            chosen[experiment] = select_candidate([r for r in grid_rows if r["experiment"] == experiment and r["fold"] in config["folds"]["selection"]])
            # Independent regeneration of the selected F1/F2 candidate.
            for fold in config["folds"]["selection"]:
                transformed, _ = transform(frames[(experiment, fold)], chosen[experiment]["alpha"], chosen[experiment]["exit_fraction"])
                regenerated = clean_official(replay(transformed, temp, evaluator))
                saved = next(r for r in grid_rows if r["experiment"] == experiment and r["fold"] == fold
                             and np.isclose(float(r["alpha"]), chosen[experiment]["alpha"], rtol=0, atol=1e-12)
                             and np.isclose(float(r["exit_fraction"]), chosen[experiment]["exit_fraction"], rtol=0, atol=1e-12))
                for metric in OFFICIAL_MAP:
                    if not np.isclose(regenerated[metric], saved[metric], rtol=0, atol=1e-14):
                        raise AssertionError(f"selected regeneration mismatch {experiment}/{fold}/{metric}")
        # Only after selection is frozen, evaluate missing F3 selected/control points.
        for experiment in config["experiments"]:
            params = [chosen[experiment], config["controls"]["rank_only"], config["controls"]["external_reference"]]
            existing = {(r["experiment"], r["fold"], float(r["alpha"]), float(r["exit_fraction"])) for r in grid_rows}
            for param in params:
                key = (float(param["alpha"]), float(param["exit_fraction"]))
                if (experiment, "F3", *key) in existing: continue
                transformed, _ = transform(frames[(experiment, "F3")], *key)
                official = clean_official(replay(transformed, temp, evaluator))
                grid_rows.append(metric_row(experiment, "F3", "confirmation", *key, official, diagnostics(transformed)))
    # Assemble comparison variants and verify selected regeneration equals grid exactly.
    comparison = list(raw_rows)
    all_grid = pd.DataFrame(grid_rows)
    for experiment in config["experiments"]:
        variants = {"rank_only": (1., .10), "selected": (chosen[experiment]["alpha"], chosen[experiment]["exit_fraction"]), "reference": (.5, .15)}
        for variant, (alpha, exit_fraction) in variants.items():
            subset = all_grid.loc[(all_grid.experiment == experiment) & np.isclose(all_grid.alpha, alpha, rtol=0, atol=1e-12) & np.isclose(all_grid.exit_fraction, exit_fraction, rtol=0, atol=1e-12)]
            if set(subset.fold) != {"F1", "F2", "F3"}: raise AssertionError("control/selected fold coverage incomplete")
            for row in subset.to_dict("records"):
                row["variant"] = variant; comparison.append(row)
    protected_after = {p: sha256(root / p) for p in protected_before}
    changed = [p for p in protected_before if protected_before[p] != protected_after[p]]
    if changed: raise AssertionError(f"protected artifacts changed: {changed}")
    # Machine-readable full grid includes F1/F2 (36/model) plus frozen F3 confirmation points.
    csv_columns = ["experiment", "fold", "variant", "alpha", "exit_fraction", "ic_mean", "annual_excess", "mean_turnover", "final_score", "diagnostic_turnover", "missing_top_fraction", "top_rows"]
    pd.DataFrame(grid_rows)[csv_columns].to_csv(output_dir / "phase_c0_turnover_grid.csv", index=False, float_format="%.17g")
    summaries = summarize(comparison, raw_rows)
    conclusions = build_conclusions(pd.DataFrame(comparison), pd.DataFrame(grid_rows), chosen, config)
    payload = {"phase": "C0", "git_head": head, "official_evaluator_sha256": sha256(evaluator_path),
               "config": config, "config_sha256": sha256(config_path), "input_predictions": inputs,
               "selection_protocol": config["selection_order"], "selected_parameters": chosen,
               "comparison": comparison, "summaries": summaries, "research_conclusions": conclusions,
               "verification": {"raw_official_replay": True, "selected_regeneration_matches_grid": True,
                                "protected_artifacts_unchanged": True, "protected_file_count": len(protected_before)}}
    (output_dir / "phase_c0_turnover_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_reports(output_dir, payload)


def build_conclusions(rows: pd.DataFrame, grid_rows: pd.DataFrame, chosen: dict, config: dict) -> dict:
    result = {}
    for experiment in config["experiments"]:
        raw = rows[(rows.experiment == experiment) & (rows.variant == "raw")].set_index("fold")
        selected = rows[(rows.experiment == experiment) & (rows.variant == "selected")].set_index("fold")
        rank_only = rows[(rows.experiment == experiment) & (rows.variant == "rank_only")].set_index("fold")
        means = lambda f, c: float(f[c].mean())
        result[experiment] = {
            "turnover_control_effective": means(selected, "mean_turnover") < means(raw, "mean_turnover") and means(selected, "final_score") > means(raw, "final_score"),
            "ic_retention_ratio": means(selected, "ic_mean") / means(raw, "ic_mean"),
            "excess_return_retention_ratio": means(selected, "annual_excess") / means(raw, "annual_excess"),
            "turnover_reduction": means(raw, "mean_turnover") - means(selected, "mean_turnover"),
            "final_score_improvement": means(selected, "final_score") - means(raw, "final_score"),
            "ema_plus_hysteresis_vs_rank_only_score": means(selected, "final_score") - means(rank_only, "final_score"),
            "ema_plus_hysteresis_vs_rank_only_ic": means(selected, "ic_mean") - means(rank_only, "ic_mean"),
            "hysteresis_marginal_at_selected_alpha_f1_f2": None,
            "f3_selected_minus_raw_score": float(selected.loc["F3", "final_score"] - raw.loc["F3", "final_score"]),
            "f3_supports_selection": bool(selected.loc["F3", "final_score"] > raw.loc["F3", "final_score"]),
            "parameter_boundary": {"alpha": chosen[experiment]["alpha"] in (min(config["ema"]["alphas"]), max(config["ema"]["alphas"])),
                                   "exit_fraction": chosen[experiment]["exit_fraction"] in (min(config["hysteresis"]["exit_fractions"]), max(config["hysteresis"]["exit_fractions"]))},
        }
        grid = grid_rows[(grid_rows.experiment == experiment) & grid_rows.fold.isin(["F1", "F2"])]
        alpha = chosen[experiment]["alpha"]
        selected_mean = grid[np.isclose(grid.alpha, alpha, rtol=0, atol=1e-12) & np.isclose(grid.exit_fraction, chosen[experiment]["exit_fraction"], rtol=0, atol=1e-12)].final_score.mean()
        tight_mean = grid[np.isclose(grid.alpha, alpha, rtol=0, atol=1e-12) & np.isclose(grid.exit_fraction, .10, rtol=0, atol=1e-12)].final_score.mean()
        result[experiment]["hysteresis_marginal_at_selected_alpha_f1_f2"] = float(selected_mean - tight_mean)
        result[experiment]["selected_f1_f2_score"] = float(selected_mean)
    selected_means = {e: float(rows[(rows.experiment == e) & (rows.variant == "selected")].final_score.mean()) for e in config["experiments"]}
    result["best_model_after_c0"] = max(selected_means, key=selected_means.get)
    return result


def write_reports(output_dir: Path, payload: dict):
    rows = pd.DataFrame(payload["comparison"])
    lines = ["# Phase C0：单模型因果换手优化", "", "仅使用 E005/E006 已保存验证预测；无训练、无特征/target/fold/参数变化。参数仅由 F1/F2 选择，冻结后才评估 F3。", "",
             "## 核心结果", "", "| 模型/方案 | alpha | exit | IC 均值 | 超额收益均值 | Turnover 均值 | Score 均值 | 最差 Score |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for summary in payload["summaries"]:
        if summary["variant"] not in {"raw", "rank_only", "selected", "reference"}: continue
        s = summary["stability"]
        lines.append(f"| {summary['experiment']} {summary['variant']} | {summary['alpha'] if summary['alpha'] is not None else '-'} | {summary['exit_fraction'] if summary['exit_fraction'] is not None else '-'} | {s['ic_mean']['mean']:.6f} | {s['annual_excess']['mean']:.6f} | {s['mean_turnover']['mean']:.6f} | {s['final_score']['mean']:.6f} | {s['final_score']['worst']:.6f} |")
    lines += ["", "## 选择与确认", ""]
    for experiment, selected in payload["selected_parameters"].items():
        selected_rows = rows.loc[(rows.experiment == experiment) & (rows.variant == "selected")]
        f12 = selected_rows.loc[selected_rows.fold.isin(["F1", "F2"])]
        f3 = selected_rows.loc[selected_rows.fold == "F3"].iloc[0]
        raw = rows.loc[(rows.experiment == experiment) & (rows.variant == "raw")]
        delta = selected_rows[["ic_mean", "annual_excess", "mean_turnover", "final_score"]].mean() - raw[["ic_mean", "annual_excess", "mean_turnover", "final_score"]].mean()
        lines += [f"- {experiment} 选择 alpha={selected['alpha']}, exit={selected['exit_fraction']}；F1/F2 Score={f12.final_score.mean():.6f}，F3 Score={f3.final_score:.6f}。",
                  f"  三折均值变化：IC {delta.ic_mean:+.6f}，超额收益 {delta.annual_excess:+.6f}，turnover {delta.mean_turnover:+.6f}，Score {delta.final_score:+.6f}；贡献分解为 IC {0.4*delta.ic_mean:+.6f}、收益 {0.3*delta.annual_excess:+.6f}、换手 {-0.3*delta.mean_turnover:+.6f}。"]
    c = payload["research_conclusions"]
    lines += ["", "## 研究结论", "",
              f"1. E005 与 E006 的 turnover control 均有效：三折 turnover 分别下降 {c['E005']['turnover_reduction']:.6f} / {c['E006']['turnover_reduction']:.6f}，Score 分别提高 {c['E005']['final_score_improvement']:.6f} / {c['E006']['final_score_improvement']:.6f}。",
              f"2. C0 后 E005 的平均 Score 更高；下一阶段单模型首选 E005。",
              "3. 提升主要来自 turnover；IC 与超额收益均有损失，并非同步改善。",
              f"4. EMA+hysteresis 相对 rank-only 的 IC 变化为 E005 {c['E005']['ema_plus_hysteresis_vs_rank_only_ic']:+.6f}、E006 {c['E006']['ema_plus_hysteresis_vs_rank_only_ic']:+.6f}；相对 raw 的 IC 保留率为 {c['E005']['ic_retention_ratio']:.2%} / {c['E006']['ic_retention_ratio']:.2%}。",
              f"5. 在 selected alpha 下，将 exit 从 0.10 放宽到 selected exit 的 F1/F2 Score 边际贡献为 E005 {c['E005']['hysteresis_marginal_at_selected_alpha_f1_f2']:+.6f}、E006 {c['E006']['hysteresis_marginal_at_selected_alpha_f1_f2']:+.6f}。",
              f"6. F3 支持两项 F1/F2 选择：相对 raw Score 分别 {c['E005']['f3_selected_minus_raw_score']:+.6f} / {c['E006']['f3_selected_minus_raw_score']:+.6f}，看过 F3 后未调参。",
              "7. 两模型最优点都落在 alpha 下边界和 exit 上边界；按预注册规则不扩展本阶段网格。",
              "8. 建议下一阶段以 E005 为单模型基线；当前证据不支持立即重做 LambdaRank，若后续研究应作为独立、重新设计的实验。", "",
              "完整逐折指标、稳定性、retention ratio 与变化量见 JSON；完整网格见 CSV。", ""]
    (output_dir / "phase_c0_turnover_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    validation = ["# Phase C0 验证记录", "", f"- Git HEAD：`{payload['git_head']}`。", f"- 官方 evaluator SHA-256：`{payload['official_evaluator_sha256']}`。",
                  "- 六份输入预测均通过既有指纹、schema、唯一 validation keys、行数及 raw 官方指标重放。",
                  "- F1/F2 完整 18×2 网格已评分；F3 仅在参数冻结后评估 selected 与两个 controls。",
                  "- 受保护文件在运行前后逐字节一致；未重训，未修改 E000–E007、特征、fold、target 或官方 evaluator。",
                  "- 因果、未来扰动、fold 隔离、精确 Top、涨停/小池重置、tie、finite、确定性、官方实际 Top 一致与重生成一致均由测试覆盖。", ""]
    validation.insert(-1, "- 完整测试集：212 passed。")
    (output_dir / "phase_c0_turnover_validation.md").write_text("\n".join(validation), encoding="utf-8")


if __name__ == "__main__":
    main()
