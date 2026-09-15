"""Replay saved validation predictions through the unmodified organizer script."""
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_official(root):
    path = root / "reference/evaluate_official.py"
    spec = importlib.util.spec_from_file_location("organizer_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def replay(frame, directory, evaluator):
    directory = Path(directory)
    keys = ["ts_code", "trade_date"]
    frame[keys + ["pred"]].to_csv(directory / "submission.csv", index=False, float_format="%.17g")
    frame[keys + ["y_ret_1d"]].to_csv(directory / "测试集_Y.csv", index=False, float_format="%.17g")
    frame[keys + ["flag_limit_up"]].to_csv(directory / "测试集_X.csv", index=False)
    return evaluator(str(directory / "submission.csv"), str(directory))


def main():
    root = Path.cwd()
    source = root / "reference/evaluate_official.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source.read_bytes() == (root / "evaluate.py").read_bytes()
    evaluator = load_official(root)
    mapping = dict(ic_mean="rank_ic_mean", ic_std="rank_ic_std", icir="icir",
                   ic_positive_ratio="ic_positive_ratio", annual_excess="annualized_top_excess_return",
                   top1_annual_ret="top1_annualized_absolute_return", mean_turnover="mean_turnover", final_score="final_score")
    records = []
    with tempfile.TemporaryDirectory(prefix="official_replay_") as directory:
        for experiment in ["E000", "E001", "E002"]:
            for fold in ["F1", "F2", "F3"]:
                folder = root / "outputs/baselines" / experiment / fold
                frame = pd.read_csv(folder / "predictions.csv.gz", float_precision="round_trip")
                original = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                actual = replay(frame, directory, evaluator)
                clean = {k: float(v) if np.isfinite(v) else None for k, v in actual.items()}
                delta = {k: clean[k] - original["metrics"][v] if clean[k] is not None and original["metrics"][v] is not None else None for k, v in mapping.items()}
                records.append(dict(experiment=experiment, fold=fold, rows=len(frame), official=clean,
                                    provisional=original["metrics"], official_minus_provisional=delta))
                print(f"Replayed {experiment}/{fold}: {clean['final_score']}", flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    result = dict(status="completed", provenance="User confirmed unmodified organizer script",
                  script_sha256=digest, scope="Saved validation predictions only; no retraining; not hidden-test results",
                  nonfinite_encoding="JSON null means undefined/nonfinite, never zero", folds=records)
    (root / "outputs/official_evaluator_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 官方脚本对照复算", "", "用户确认主办方原始脚本；原样归档。仅复算已保存验证预测，未重训、未替换 provisional 结果。不是隐藏测试集比赛成绩。", "", f"SHA-256: `{digest}`", "", "| 实验/折 | Provisional Score | 官方脚本 Score |", "|---|---:|---:|"]
    for r in records:
        score = r['official']['final_score']
        lines.append(f"| {r['experiment']}/{r['fold']} | {r['provisional']['final_score']:.9f} | {'undefined (NaN)' if score is None else f'{score:.9f}'} |")
    lines += ["", "差异详见 evaluator_arrival_review.md；逐项指标及差值详见同名 JSON。官方 ddof=1 与用户冻结 ddof=0 冲突，待审查后再做版本化规则调整。E000 不填充缺失预测；原脚本 NaN 传播导致的未定义值保留为 JSON null。"]
    (root / "outputs/official_evaluator_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
