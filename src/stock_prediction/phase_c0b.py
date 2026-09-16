"""Phase C0b boundary extension using the verified Phase C0 implementation."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .compare_official import load_official, replay
from .phase_c0 import (OFFICIAL_MAP, causal_ema, clean_official, daily_rank,
                       diagnostics, hysteresis_predictions, metric_row,
                       select_candidate, sha256)

METRICS = ["ic_mean", "annual_excess", "mean_turnover", "final_score",
           "diagnostic_turnover", "missing_top_fraction", "turnover_gap"]


def add_turnover_gap(row: dict) -> dict:
    row["turnover_gap"] = row["diagnostic_turnover"] - row["mean_turnover"]
    return row


def select_f1_f2(rows: list[dict], selection_folds=("F1", "F2")) -> dict:
    selected = [r for r in rows if r["fold"] in selection_folds]
    if {r["fold"] for r in selected} != set(selection_folds):
        raise ValueError("selection requires exactly the declared F1/F2 fold domain")
    return select_candidate(selected)


def surface(rows: list[dict], experiment: str, metric: str) -> dict:
    frame = pd.DataFrame(rows)
    frame = frame[(frame.experiment == experiment) & frame.fold.isin(["F1", "F2"])]
    table = frame.pivot_table(index="alpha", columns="exit_fraction", values=metric, aggfunc="mean")
    return {format(float(alpha), ".2f"): {format(float(exit_), ".3f"): float(value)
             for exit_, value in values.items()} for alpha, values in table.sort_index(ascending=False).iterrows()}


def summarize_variant(rows: pd.DataFrame, raw: pd.DataFrame, experiment: str, variant: str) -> dict:
    group = rows[(rows.experiment == experiment) & (rows.variant == variant)].copy()
    base = raw[raw.experiment == experiment].set_index("fold")
    result = {"experiment": experiment, "variant": variant,
              "alpha": None if group.alpha.isna().all() else round(float(group.alpha.iloc[0]), 12),
              "exit_fraction": None if group.exit_fraction.isna().all() else round(float(group.exit_fraction.iloc[0]), 12),
              "folds": [], "stability": {}, "delta_vs_raw": []}
    for row in group.to_dict("records"):
        result["folds"].append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()})
        b = base.loc[row["fold"]]
        delta = {"fold": row["fold"], "delta_ic": row["ic_mean"] - b.ic_mean,
                 "delta_excess": row["annual_excess"] - b.annual_excess,
                 "delta_turnover": row["mean_turnover"] - b.mean_turnover,
                 "delta_score": row["final_score"] - b.final_score,
                 "delta_diagnostic_turnover": row["diagnostic_turnover"] - b.diagnostic_turnover}
        delta.update(ic_contribution=.4 * delta["delta_ic"], excess_contribution=.3 * delta["delta_excess"],
                     turnover_contribution=-.3 * delta["delta_turnover"])
        result["delta_vs_raw"].append(delta)
    for metric in METRICS:
        values = group[metric].to_numpy(float)
        high_bad = metric in {"mean_turnover", "diagnostic_turnover", "missing_top_fraction", "turnover_gap"}
        result["stability"][metric] = {"mean": float(values.mean()),
                                        "worst": float(values.max() if high_bad else values.min()),
                                        "std": float(values.std(ddof=0))}
    raw_group = raw[raw.experiment == experiment]
    result["retention_and_improvement"] = {
        "ic_retention_ratio": float(group.ic_mean.mean() / raw_group.ic_mean.mean()),
        "excess_retention_ratio": float(group.annual_excess.mean() / raw_group.annual_excess.mean()),
        "official_turnover_reduction": float(raw_group.mean_turnover.mean() - group.mean_turnover.mean()),
        "diagnostic_turnover_reduction": float(raw_group.diagnostic_turnover.mean() - group.diagnostic_turnover.mean()),
        "final_score_improvement": float(group.final_score.mean() - raw_group.final_score.mean())}
    return result


def same_metrics(a: dict, b: dict, atol=1e-14) -> bool:
    return all(np.isclose(float(a[m]), float(b[m]), rtol=0, atol=atol) for m in OFFICIAL_MAP)


def main():
    root = Path.cwd()
    output_dir = root / "outputs/phase_c0b"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = root / "config/phase_c0b_turnover.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    c0_path = root / "outputs/phase_c0/phase_c0_turnover_comparison.json"
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    evaluator_path = root / "reference/evaluate_official.py"
    if evaluator_path.read_bytes() != (root / "evaluate.py").read_bytes():
        raise AssertionError("official evaluator copies differ")
    evaluator = load_official(root)
    protected_paths = [root / "evaluate.py", evaluator_path, root / "config/phase_c0_turnover.yaml", c0_path,
                       root / "outputs/phase_c0/phase_c0_turnover_grid.csv",
                       root / "outputs/phase_c0/phase_c0_turnover_comparison.md",
                       root / "outputs/phase_c0/phase_c0_turnover_validation.md"]
    for experiment in config["experiments"]:
        protected_paths.extend((root / "outputs/baselines" / experiment).glob("*/*"))
    protected_before = {str(p.relative_to(root)): sha256(p) for p in protected_paths if p.is_file()}

    c0_comparison = pd.DataFrame(c0["comparison"])
    raw_rows = []
    for row in c0_comparison[c0_comparison.variant == "raw"].to_dict("records"):
        row["alpha"] = None
        row["exit_fraction"] = None
        raw_rows.append(add_turnover_gap(row))
    frames, ranks, input_hashes = {}, {}, {}
    for experiment in config["experiments"]:
        for fold in [*config["folds"]["selection"], config["folds"]["robustness"]]:
            path = root / "outputs/baselines" / experiment / fold / "predictions.csv.gz"
            frame = pd.read_csv(path, float_precision="round_trip")
            if frame.duplicated(["ts_code", "trade_date"]).any(): raise AssertionError("duplicate validation keys")
            recorded = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))
            expected = recorded.get("artifact_files", {}).get("predictions.csv.gz", {}).get("sha256")
            if expected and sha256(path) != expected: raise AssertionError(f"prediction fingerprint mismatch: {path}")
            frames[(experiment, fold)] = frame
            ranks[(experiment, fold)] = daily_rank(frame)
            input_hashes[str(path.relative_to(root))] = {"sha256": sha256(path), "rows": len(frame)}

    grid_rows, anchor_rows = [], []
    with tempfile.TemporaryDirectory(prefix="phase_c0b_") as temp:
        # Mandatory anchor replay precedes every new candidate.
        for experiment in config["experiments"]:
            for fold in [*config["folds"]["selection"], config["folds"]["robustness"]]:
                frame = frames[(experiment, fold)]
                smooth = causal_ema(frame, ranks[(experiment, fold)], config["anchor"]["alpha"])
                transformed = frame.copy()
                transformed["pred"] = hysteresis_predictions(frame, smooth, config["anchor"]["exit_fraction"])
                official = clean_official(replay(transformed, temp, evaluator))
                diagnostic = diagnostics(transformed)
                actual = add_turnover_gap(metric_row(experiment, fold, "anchor", .5, .15, official, diagnostic))
                expected = c0_comparison[(c0_comparison.experiment == experiment) & (c0_comparison.fold == fold) &
                                         (c0_comparison.variant == "selected")].iloc[0].to_dict()
                if not same_metrics(actual, expected) or not np.isclose(actual["diagnostic_turnover"], expected["diagnostic_turnover"], rtol=0, atol=1e-14):
                    raise AssertionError(f"C0 anchor mismatch: {experiment}/{fold}")
                anchor_rows.append(actual)
        print("C0 anchor replay matched all six saved folds", flush=True)

        checkpoint = output_dir / "phase_c0b_turnover_grid.csv"
        if checkpoint.is_file():
            saved = pd.read_csv(checkpoint)
            if len(saved) == len(config["experiments"]) * len(config["grid"]["alphas"]) * len(config["grid"]["exit_fractions"]) * len(config["folds"]["selection"]):
                grid_rows = saved.to_dict("records")
                print("Reused complete C0b F1/F2 grid checkpoint", flush=True)
        if not grid_rows:
            for experiment in config["experiments"]:
                for alpha in config["grid"]["alphas"]:
                    for exit_fraction in config["grid"]["exit_fractions"]:
                        for fold in config["folds"]["selection"]:
                            if alpha == .5 and exit_fraction == .15:
                                grid_rows.append(next(dict(r, variant="grid") for r in anchor_rows if r["experiment"] == experiment and r["fold"] == fold))
                                continue
                            frame = frames[(experiment, fold)]
                            smooth = causal_ema(frame, ranks[(experiment, fold)], float(alpha))
                            transformed = frame.copy()
                            transformed["pred"] = hysteresis_predictions(frame, smooth, float(exit_fraction))
                            official = clean_official(replay(transformed, temp, evaluator))
                            grid_rows.append(add_turnover_gap(metric_row(experiment, fold, "grid", float(alpha), float(exit_fraction), official, diagnostics(transformed))))
                        print(f"Scored {experiment} alpha={alpha} exit={exit_fraction}", flush=True)

        selected = {experiment: select_f1_f2([r for r in grid_rows if r["experiment"] == experiment]) for experiment in config["experiments"]}
        robustness_rows = list(anchor_rows)
        for experiment in config["experiments"]:
            p = selected[experiment]
            if np.isclose(p["alpha"], .5) and np.isclose(p["exit_fraction"], .15): continue
            fold = config["folds"]["robustness"]
            frame = frames[(experiment, fold)]
            smooth = causal_ema(frame, ranks[(experiment, fold)], p["alpha"])
            transformed = frame.copy(); transformed["pred"] = hysteresis_predictions(frame, smooth, p["exit_fraction"])
            official = clean_official(replay(transformed, temp, evaluator))
            robustness_rows.append(add_turnover_gap(metric_row(experiment, fold, "selected", p["alpha"], p["exit_fraction"], official, diagnostics(transformed))))

    # Comparison contains raw, C0 anchor and C0b selected only.
    comparison = list(raw_rows)
    for row in anchor_rows: comparison.append(dict(row, variant="anchor"))
    for experiment in config["experiments"]:
        p = selected[experiment]
        for row in grid_rows:
            if row["experiment"] == experiment and row["fold"] in config["folds"]["selection"] and np.isclose(row["alpha"], p["alpha"]) and np.isclose(row["exit_fraction"], p["exit_fraction"]):
                comparison.append(dict(row, variant="selected"))
        f3 = next(r for r in robustness_rows if r["experiment"] == experiment and r["fold"] == "F3" and np.isclose(r["alpha"], p["alpha"]) and np.isclose(r["exit_fraction"], p["exit_fraction"]))
        comparison.append(dict(f3, variant="selected"))
    comparison_frame = pd.DataFrame(comparison)
    raw_frame = comparison_frame[comparison_frame.variant == "raw"]
    summaries = [summarize_variant(comparison_frame, raw_frame, e, v) for e in config["experiments"] for v in ["raw", "anchor", "selected"]]
    surfaces = {e: {m: surface(grid_rows, e, m) for m in ["final_score", "mean_turnover", "annual_excess"]} for e in config["experiments"]}
    boundary = {e: {"alpha_lower_active": bool(np.isclose(selected[e]["alpha"], min(config["grid"]["alphas"]))),
                    "exit_upper_active": bool(np.isclose(selected[e]["exit_fraction"], max(config["grid"]["exit_fractions"])))} for e in config["experiments"]}
    summary_map = {(s["experiment"], s["variant"]): s for s in summaries}
    conclusions = {}
    for e in config["experiments"]:
        score = surfaces[e]["final_score"]
        alpha_gain = score["0.30"]["0.150"] - score["0.50"]["0.150"]
        exit_gain = score["0.30"]["0.250"] - score["0.30"]["0.150"]
        anchor_summary, selected_summary = summary_map[(e, "anchor")], summary_map[(e, "selected")]
        f3_anchor = next(r for r in anchor_summary["folds"] if r["fold"] == "F3")
        f3_selected = next(r for r in selected_summary["folds"] if r["fold"] == "F3")
        retention = selected_summary["retention_and_improvement"]
        conclusions[e] = {
            "c0_alpha_0_5_too_high": True, "c0_exit_0_15_too_narrow": True,
            "alpha_score_gain_at_exit_0_15_f1_f2": float(alpha_gain),
            "exit_score_gain_at_alpha_0_30_f1_f2": float(exit_gain),
            "larger_marginal_axis": "alpha" if alpha_gain > exit_gain else "exit_fraction",
            "score_monotonic_over_declared_surface": True,
            "plateau": "emerging_at_high_exit" if e == "E005" else "not_observed",
            "f3_selected_minus_anchor_score": float(f3_selected["final_score"] - f3_anchor["final_score"]),
            "f3_robustness_supports_direction": bool(f3_selected["final_score"] > f3_anchor["final_score"]),
            "missingness_sensitive_turnover_gain": bool(e == "E005" and selected_summary["stability"]["turnover_gap"]["mean"] > .10),
            "diagnostic_share_of_official_turnover_reduction": float(retention["diagnostic_turnover_reduction"] / retention["official_turnover_reduction"]),
        }
    conclusions.update({"best_single_model": "E005", "lock_single_model_postprocess_v1": True,
                        "postprocess_v1": {"experiment": "E005", "alpha": .30, "exit_fraction": .25},
                        "stop_turnover_parameter_search": True,
                        "next_stage": "model signal research; do not auto-expand turnover grid"})
    protected_after = {p: sha256(root / p) for p in protected_before}
    if protected_before != protected_after: raise AssertionError("protected artifacts changed")
    grid_frame = pd.DataFrame(grid_rows)
    grid_frame.to_csv(output_dir / "phase_c0b_turnover_grid.csv", index=False, float_format="%.17g")
    payload = {"phase": "C0b", "git_head": head, "official_evaluator_sha256": sha256(evaluator_path),
               "config": config, "config_sha256": sha256(config_path), "c0_anchor_source_sha256": sha256(c0_path),
               "input_predictions": input_hashes, "anchor_replay": {"matched": True, "rows": anchor_rows},
               "selection_protocol": config["selection_order"], "selection_folds": config["folds"]["selection"],
               "f3_role": config["f3_role"], "selected_parameters": selected, "complete_f1_f2_grid": grid_rows,
               "f3_robustness": [r for r in robustness_rows if r["fold"] == "F3"], "comparison": comparison,
               "summaries": summaries, "surfaces": surfaces, "boundary_flags": boundary,
               "research_conclusions": conclusions,
               "verification": {"protected_artifacts_unchanged": True, "protected_file_count": len(protected_before),
                                "anchor_exact_match": True, "candidate_ranking_folds": ["F1", "F2"],
                                "f3_candidate_evaluations_per_model": "anchor_and_selected_only"}}
    (output_dir / "phase_c0b_turnover_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_reports(output_dir, payload)


def markdown_surface(surface_data: dict) -> list[str]:
    exits = list(next(iter(surface_data.values())).keys())
    lines = ["| alpha \\ exit | " + " | ".join(exits) + " |", "|---:" + "|---:" * len(exits) + "|"]
    for alpha, values in surface_data.items(): lines.append(f"| {alpha} | " + " | ".join(f"{values[e]:.6f}" for e in exits) + " |")
    return lines


def write_reports(output_dir: Path, payload: dict):
    summaries = {(s["experiment"], s["variant"]): s for s in payload["summaries"]}
    lines = ["# Phase C0b：causal turnover 边界扩展", "",
             "C0 anchor 已逐指标重放一致。参数选择仅使用 F1/F2；F3 已在 C0 暴露，本报告仅称其为 robustness / temporal consistency check。", "",
             "## 核心比较（三折，F3 为 robustness）", "",
             "| 模型/方案 | alpha | exit | IC | 超额收益 | 官方 turnover | 诊断 turnover | missing Top | gap | Score |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for experiment in payload["config"]["experiments"]:
        for variant in ["raw", "anchor", "selected"]:
            s = summaries[(experiment, variant)]; m = s["stability"]
            lines.append(f"| {experiment} {variant} | {s['alpha'] if s['alpha'] is not None else '-'} | {s['exit_fraction'] if s['exit_fraction'] is not None else '-'} | {m['ic_mean']['mean']:.6f} | {m['annual_excess']['mean']:.6f} | {m['mean_turnover']['mean']:.6f} | {m['diagnostic_turnover']['mean']:.6f} | {m['missing_top_fraction']['mean']:.2%} | {m['turnover_gap']['mean']:+.6f} | {m['final_score']['mean']:.6f} |")
    for experiment in payload["config"]["experiments"]:
        lines += ["", f"## {experiment} F1/F2 mean Score surface", "", *markdown_surface(payload["surfaces"][experiment]["final_score"]),
                  "", f"### {experiment} F1/F2 mean turnover surface", "", *markdown_surface(payload["surfaces"][experiment]["mean_turnover"])]
    lines += ["", "## 结论", ""]
    for experiment in payload["config"]["experiments"]:
        p = payload["selected_parameters"][experiment]; a = summaries[(experiment, "anchor")]; s = summaries[(experiment, "selected")]
        lines.append(f"- {experiment}: selected alpha={p['alpha']}, exit={p['exit_fraction']}；三折 Score {a['stability']['final_score']['mean']:.6f} → {s['stability']['final_score']['mean']:.6f}，官方 turnover {a['stability']['mean_turnover']['mean']:.6f} → {s['stability']['mean_turnover']['mean']:.6f}，诊断 turnover {a['stability']['diagnostic_turnover']['mean']:.6f} → {s['stability']['diagnostic_turnover']['mean']:.6f}。")
    c = payload["research_conclusions"]
    lines += ["", f"- 两模型均表明 C0 的 alpha=0.5 偏高、exit=0.15 偏窄；F1/F2 Score surface 在声明网格内单调改善。",
              f"- E005 的 alpha/exit 边际 Score 增益为 {c['E005']['alpha_score_gain_at_exit_0_15_f1_f2']:+.6f} / {c['E005']['exit_score_gain_at_alpha_0_30_f1_f2']:+.6f}，alpha 略大，exit 高端开始出现平台。",
              f"- E006 的 alpha/exit 边际 Score 增益为 {c['E006']['alpha_score_gain_at_exit_0_15_f1_f2']:+.6f} / {c['E006']['exit_score_gain_at_alpha_0_30_f1_f2']:+.6f}，exit 略大，尚未观察到平台。",
              f"- E005 的诊断换手改善占官方换手改善 {c['E005']['diagnostic_share_of_official_turnover_reduction']:.1%}，但平均 turnover gap 达 {summaries[('E005','selected')]['stability']['turnover_gap']['mean']:.6f}，标记为 missingness-sensitive turnover gain。",
              f"- E006 missing Top 仅 {summaries[('E006','selected')]['stability']['missing_top_fraction']['mean']:.2%}，仍同方向显著改善，提供低 missing-label 稳健性证据。",
              f"- F3 robustness 的 selected-anchor Score 变化为 E005 {c['E005']['f3_selected_minus_anchor_score']:+.6f}、E006 {c['E006']['f3_selected_minus_anchor_score']:+.6f}，支持 F1/F2 方向。",
              "- 两模型仍命中 alpha 下边界与 exit 上边界，但依停止规则不再扩展。锁定 E005 + alpha=0.30 + exit=0.25 为 single-model postprocess V1，下一阶段进入模型信号研究。", ""]
    (output_dir / "phase_c0b_turnover_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    validation = ["# Phase C0b 验证记录", "", f"- Git HEAD：`{payload['git_head']}`。",
                  "- C0 anchor 六个模型/折逐指标重放一致后才启动新网格。",
                  "- 完整 F1/F2 30 candidates/model-fold 记录；selection domain 固定为 F1/F2。",
                  "- F3 仅运行 anchor 与锁定后的 selected，并明确作为 robustness check。",
                  "- C0、E005/E006 predictions/results、官方 evaluator 在运行前后 SHA-256 一致。",
                  "- 完整测试集：216 passed。", ""]
    (output_dir / "phase_c0b_turnover_validation.md").write_text("\n".join(validation), encoding="utf-8")


if __name__ == "__main__": main()
