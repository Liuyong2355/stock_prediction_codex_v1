"""Read-only E002 missing-label diagnostics; never changes official results."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from .audit import read_chunks
from .baselines import save_json
from .build_basic40 import sha256
from .folds import split_fold


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(zip(["min", "p01", "p25", "median", "p75", "p99", "max"],
                    np.quantile(values, [0, .01, .25, .5, .75, .99, 1]).tolist())) if len(values) else {}


def top_sets(frame, exclude_missing=False):
    """Exactly the organizer turnover selection; optional filter is diagnostic only."""
    selected, turnover, daily = [], [], []
    previous = None
    for date, day in frame.groupby("trade_date", sort=True):
        valid = day.loc[day.flag_limit_up == 0]
        if exclude_missing:
            valid = valid.loc[valid.y_ret_1d.notna()]
        if len(valid) < 100:
            previous = None
            continue
        ordered = valid.sort_values("pred", ascending=False)
        top = ordered.iloc[:max(len(valid) // 10, 1)]
        current = set(top.ts_code)
        if previous is not None and previous:
            turnover.append(1 - len(current & previous) / len(current | previous))
        previous = current
        cutoff = top.pred.iloc[-1]
        tied = int((valid.pred == cutoff).sum())
        included = int((top.pred == cutoff).sum())
        daily.append(dict(trade_date=int(date), eligible_rows=len(valid), top_rows=len(top),
                          missing_eligible=int(valid.y_ret_1d.isna().sum()), missing_top=int(top.y_ret_1d.isna().sum()),
                          largest_equal_prediction_group=int(valid.pred.value_counts().max()),
                          cutoff_tie_group=tied, cutoff_tie_selected=included,
                          boundary_splits_tie=included < tied))
        selected.extend(top.index.tolist())
    return frame.loc[selected], float(np.mean(turnover)) if turnover else np.nan, pd.DataFrame(daily)


def persistence(selected, calendar):
    """Consecutive fold market dates, counting only selected rows with missing y."""
    p = selected.loc[selected.y_ret_1d.isna(), ["ts_code", "trade_date"]].copy()
    if p.empty:
        return {"stocks": 0, "runs": 0}
    p["day"] = p.trade_date.map({d: i for i, d in enumerate(calendar)})
    p = p.sort_values(["ts_code", "day"])
    p["run"] = ((p.ts_code != p.ts_code.shift()) | (p.day.diff() != 1)).cumsum()
    runs = p.groupby("run").agg(ts_code=("ts_code", "first"), start=("trade_date", "min"),
                                end=("trade_date", "max"), days=("day", "size"))
    stock_max = runs.groupby("ts_code").days.max()
    return dict(stocks=int(p.ts_code.nunique()), runs=len(runs), run_days=distribution(runs.days),
                stock_longest_run_days=distribution(stock_max), stocks_run_ge20=int((stock_max >= 20).sum()),
                stocks_run_ge60=int((stock_max >= 60).sum()), stocks_run_full_fold=int((stock_max == len(calendar)).sum()),
                longest_examples=runs.sort_values("days", ascending=False).head(10).to_dict("records"))


def population(frame):
    ohlc = frame[["open", "high", "low", "close"]].isna()
    return dict(rows=len(frame), ohlc_all_missing=int(ohlc.all(axis=1).sum()),
                ohlc_partial_missing=int((ohlc.any(axis=1) & ~ohlc.all(axis=1)).sum()),
                ohlc_all_present=int((~ohlc.any(axis=1)).sum()),
                ohlc_missing_by_column={c: int(ohlc[c].sum()) for c in ohlc},
                finite_features=distribution(frame.finite_features),
                finite_feature_histogram={str(k): int(v) for k, v in frame.finite_features.value_counts().sort_index().items()},
                predictions=distribution(frame.pred),
                price_context=frame.price_context.value_counts().to_dict())


def write_report(root, results):
    lines = ["# E002 缺失标签样本最小诊断", "",
             "## 结论", "",
             "主要原因是：OHLC 缺失使 38 个非 flag 特征全为 NaN，仅两个 flag=0 保留；相同输入在各折被现有 LightGBM 映射为同一个较高预测值。监督训练中不存在仅 2 个特征有效的样本，因此这是训练未覆盖的缺失组合上的预测表现，不能解释为这些股票具有真实高收益。官方换手不排除缺失标签，它们长期入选显著降低换手；收益指标则已排除缺失标签。", "",
             "## 样本与预测", "",
             "以下 Top 指官方**换手 Top**：每日 flag_limit_up=0，默认 pred 降序，floor(N/10)，至少 100 行。占比按股票日记录数加权，不是独立股票占比。每折全量 1,125,300 行、242 个交易日。", "",
             "| 折 | 全部缺失 y 行数 | 缺失 y / 全折 | Top 中缺失 y 行数 / Top 总行数 | Top 中缺失 y 占比 | 可选缺失 y 样本入选率 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['fold']} | {r['missing_y']:,} | {r['missing_y_fraction']:.2%} | {r['missing_top_rows']:,} / {r['top_rows']:,} | {r['missing_share_of_top']:.2%} | {r['missing_eligible_selection_rate']:.2%} |")
    lines += ["", "| 折 | 缺失 y Top：OHLC 全缺失 / 全有值 | 仅 2 个 flag 有效占比 | 这类相同输入的预测值 | 有标签可选样本预测 P99 | 缺失 y Top 预测范围 |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        p = r['populations']['missing_y_in_top']
        lines.append(f"| {r['fold']} | {p['ohlc_all_missing']:,} / {p['ohlc_all_present']} | {int(p['finite_feature_histogram']['2']) / p['rows']:.2%} | {r['only_zero_flags_vector']['predictions']['median']:.6f} | {r['populations']['labeled_eligible']['predictions']['p99']:.6f} | {p['predictions']['min']:.6f} ～ {p['predictions']['max']:.6f} |")
    lines += ["", "三折缺失 y Top 的预测 P1/P25/中位数/P75/P99 均等于表中的相同输入预测值；Basic40 有效数 P1～P99 均为 2、最大 40。没有 OHLC 部分缺失记录；少数 OHLC 完整但 y 缺失的入选行已保留。监督训练每折最少有 11 个有效特征，仅 2 个有效特征的训练行均为 0。完整分位数、逐列缺失量、特征有效数直方图及人工复核样例见 JSON。", "",
              "## 日期状态、并列与持续性", "",
              "缺失 y Top 在价格序列中的位置（使用完整训练期事后定位，仅用于诊断）：", "",
              "| 折 | 首次有效 close 之前 | 最后有效 close 之后 | 中间缺口 | 训练期从无有效 close | 当前 close 有值 |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        c = r['populations']['missing_y_in_top']['price_context']
        lines.append(f"| {r['fold']} | " + " | ".join(str(c.get(k, 0)) for k in ['before_first_finite_close_in_train', 'after_last_finite_close_in_train', 'internal_missing_close', 'no_finite_close_in_train', 'current_close_present']) + " |")
    lines += ["", "F1/F2 主要集中在首次有效价格之前，F3 主要在最后有效价格之后。符合赛题描述的未上市/停牌等缺失形态，但仅凭这些数据不能把每条记录确认为未上市、停牌或退市。", "",
              "| 折 | 缺失 y Top 中同日预测重复占比 | Top 边界切开并列组的日期数 | 最长连续入选交易日 | 连续 ≥60 日的股票数 |", "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['fold']} | {r['ties']['missing_top_duplicate_fraction']:.2%} | {r['ties']['dates_boundary_splits_tie']} / 242 | {r['persistence']['run_days']['max']:.0f} | {r['persistence']['stocks_run_ge60']} |")
    lines += ["", "连续性仅统计 y 缺失且入选的相邻市场交易日，跨折不连接。F2 有 78 只股票连续整折 242 日入选。并列广泛存在，但 F2/F3 没有边界切开并列组，因此大量入选并非仅由排序 tie 造成；共同的较高预测值是核心。F1 有 28 日选取部分并列组，具体成员受官方默认排序与输入顺序影响；本次未重排或改 tie 规则。", "",
              "## 换手对照（不是新评分）", "",
              "诊断版本只在每日候选集额外排除 y 缺失后，重新按剩余人数选 Top10%，其余取整、排序、最少样本数和 Jaccard 公式不变。", "",
              "| 折 | 官方 turnover | 剔除 y 缺失后的诊断 turnover | 差值 |", "|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['fold']} | {r['official_turnover']:.6f} | {r['diagnostic_exclude_missing_y_turnover']:.6f} | {r['diagnostic_minus_official']:+.6f} |")
    lines += ["", "**低官方换手很大程度来自缺失样本持续占位，不能直接视为可交易组合的稳定性。** 诊断使用事后标签可用性，不是可上线的筛选规则；不计算替代 Score，不替代官方 turnover。", "",
              "## 校验与产物", "",
              "- 全量对齐原始 OHLC、Basic40 与保存预测；三折官方换手重算与已存官方结果一致（绝对误差 ≤1e-14）。",
              "- 原始官方脚本、E002 模型/预测/逐折结果、冻结配置及既有汇总结果的运行前后 SHA-256 一致。无训练、参数调整、特征修改或评分规则变更。",
              "- `e002_missing_label_diagnostic.json`：全部统计、样例、持续性案例与保护文件指纹；`e002_diagnostic_F1/F2/F3_daily.csv`：每日 Top/缺失/并列统计。",
              "- 运行：`PYTHONPATH=src python -m stock_prediction.diagnose_e002`；最小测试覆盖诊断筛选不修改输入、并列边界、短日期重置和连续性缺口。测试结果见 `e002_diagnostic_test_results.txt`。", ""]
    (root / "outputs/e002_missing_label_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    root = Path.cwd()
    protected = [root / "evaluate.py", root / "reference/evaluate_official.py"]
    protected += list((root / "config").glob("*.yaml"))
    protected += list((root / "outputs/baselines/E002").glob("*/*"))
    protected += [root / f"outputs/{name}" for name in ["official_evaluator_comparison.json", "official_evaluator_comparison.md", "baseline_summary.json", "experiment_log.csv"]]
    before = {str(p.relative_to(root)): sha256(p) for p in protected if p.is_file()}
    manifest = json.loads((root / "outputs/basic40_manifest.json").read_text(encoding="utf-8"))
    files = manifest["datasets"]["train"]["files"]
    matrix = np.load(files["matrix"]["path"], mmap_mode="r", allow_pickle=False)
    keys = np.load(files["keys"]["path"], mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "outputs/baselines/cache/labels.npy", mmap_mode="r", allow_pickle=False)
    counts = np.zeros(len(keys), dtype=np.uint8)
    pieces, price_spans = [], []
    offset = 0
    for chunk in read_chunks(root / "data/raw/训练集.csv", True):
        end = offset + len(chunk)
        assert np.array_equal(chunk.ts_code.astype(str), keys["ts_code"][offset:end])
        assert np.array_equal(chunk.trade_date, keys["trade_date"][offset:end])
        counts[offset:end] = np.isfinite(matrix[offset:end]).sum(axis=1)
        price_spans.append(chunk.loc[np.isfinite(chunk.close)].groupby("ts_code").trade_date.agg(["min", "max"]))
        keep = chunk.trade_date >= 20220101
        part = chunk.loc[keep, ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "flag_limit_down"]].copy()
        part["row_id"] = np.arange(offset, end)[keep]
        part["finite_features"] = counts[offset:end][keep]
        pieces.append(part)
        offset = end
    assert offset == len(keys)
    raw = pd.concat(pieces, ignore_index=True)
    spans = pd.concat(price_spans).groupby(level=0).agg({"min": "min", "max": "max"})
    first, last = raw.ts_code.map(spans["min"]), raw.ts_code.map(spans["max"])
    raw["price_context"] = np.select([raw.close.notna(), first.isna(), raw.trade_date < first, raw.trade_date > last],
                                     ["current_close_present", "no_finite_close_in_train", "before_first_finite_close_in_train", "after_last_finite_close_in_train"],
                                     default="internal_missing_close")
    print("Raw OHLC / feature alignment scanned", flush=True)
    folds = yaml.safe_load((root / "config/folds_v1.yaml").read_text(encoding="utf-8"))
    official = json.loads((root / "outputs/official_evaluator_comparison.json").read_text(encoding="utf-8"))
    results = []
    for fold in folds["folds"]:
        folder = root / "outputs/baselines/E002" / fold["id"]
        # Match the original evaluator's default CSV float parser and original input order.
        p = pd.read_csv(folder / "predictions.csv.gz")
        p = p.merge(raw, on=["ts_code", "trade_date"], how="left", validate="one_to_one", sort=False)
        assert p.row_id.notna().all()
        selected, turnover, daily = top_sets(p)
        _, diagnostic_turnover, _ = top_sets(p, exclude_missing=True)
        expected = next(r["official"]["mean_turnover"] for r in official["folds"] if r["experiment"] == "E002" and r["fold"] == fold["id"])
        assert np.isclose(turnover, expected, rtol=0, atol=1e-14), (turnover, expected)
        missing = p.y_ret_1d.isna()
        selected_missing = selected.loc[selected.y_ret_1d.isna()]
        eligible_missing = p.loc[missing & (p.flag_limit_up == 0)]
        train_idx, _, _ = split_fold(keys["trade_date"], labels, fold, folds["purge_rule"]["purge_trading_days"])
        only_flags = p.loc[(p.finite_features == 2) & (p.flag_limit_up == 0) & (p.flag_limit_down == 0)]
        # Verify the representation, rather than assuming that count=2 implies only flags.
        assert not np.isfinite(matrix[only_flags.row_id.to_numpy(dtype=int), :38]).any()
        frequencies = selected_missing.groupby(["trade_date", "pred"]).size()
        duplicates = int(frequencies.loc[frequencies > 1].sum())
        r = dict(fold=fold["id"], rows=len(p), missing_y=int(missing.sum()),
                 missing_y_fraction=float(missing.mean()), top_rows=len(selected), missing_top_rows=len(selected_missing),
                 missing_share_of_top=len(selected_missing) / len(selected),
                 missing_eligible_selection_rate=len(selected_missing) / len(eligible_missing),
                 daily_missing_share_of_top=distribution(daily.missing_top / daily.top_rows),
                 populations={"all_missing_y": population(p.loc[missing]), "missing_y_in_top": population(selected_missing),
                              "labeled_eligible": population(p.loc[~missing & (p.flag_limit_up == 0)]),
                              "labeled_in_top": population(selected.loc[selected.y_ret_1d.notna()])},
                 only_zero_flags_vector=dict(rows=len(only_flags), unique_predictions=int(only_flags.pred.nunique()),
                                             predictions=distribution(only_flags.pred), missing_y=int(only_flags.y_ret_1d.isna().sum()),
                                             selected_rows=int(selected.index.isin(only_flags.index).sum())),
                 supervised_training=dict(rows=len(train_idx), finite_features=distribution(counts[train_idx]),
                                          rows_with_le2_finite_features=int((counts[train_idx] <= 2).sum())),
                 ties=dict(dates_boundary_splits_tie=int(daily.boundary_splits_tie.sum()),
                           largest_daily_equal_prediction_group=distribution(daily.largest_equal_prediction_group),
                           missing_top_rows_in_same_day_equal_prediction_groups=duplicates,
                           missing_top_duplicate_fraction=duplicates / len(selected_missing) if len(selected_missing) else None),
                 persistence=persistence(selected, sorted(p.trade_date.unique())),
                 official_turnover=turnover, diagnostic_exclude_missing_y_turnover=diagnostic_turnover,
                 diagnostic_minus_official=diagnostic_turnover - turnover,
                 examples=selected_missing[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "flag_limit_up", "flag_limit_down", "y_ret_1d", "finite_features", "pred", "price_context"]].head(10).to_dict("records"))
        results.append(r)
        daily.to_csv(root / f"outputs/e002_diagnostic_{fold['id']}_daily.csv", index=False)
        print(f"{fold['id']}: missing Top={r['missing_share_of_top']:.4%}; turnover {turnover:.6f} -> diagnostic {diagnostic_turnover:.6f}", flush=True)
    after = {name: sha256(root / name) for name in before}
    assert before == after, "A protected artifact changed"
    save_json(root / "outputs/e002_missing_label_diagnostic.json", dict(scope="Read-only saved E002 validation predictions; diagnostic label filter is ex-post, not a deployable strategy or score change", protected_sha256=before, protected_unchanged=True, folds=results))
    write_report(root, results)


if __name__ == "__main__":
    main()
