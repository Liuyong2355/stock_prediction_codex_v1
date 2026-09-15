"""Read-only full CSV audit. Diagnostics never feed features, labels or models."""
from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["ts_code", "trade_date"]
PRICES = ["open", "high", "low", "close"]
NUMERIC = PRICES + ["vol", "amount", "flag_limit_up", "flag_limit_down"]
TARGET = "y_ret_1d"
ATOL, RTOL = 1e-10, 1e-8


def fingerprint(path: Path) -> dict:
    """Hash all bytes and validate UTF-8 incrementally (not sample detection)."""
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
    ascii_only, size = True, 0
    with path.open("rb") as stream:
        prefix = stream.read(3)
        stream.seek(0)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            decoder.decode(block)
            ascii_only &= block.isascii()
            size += len(block)
    decoder.decode(b"", final=True)
    return {"path": str(path.resolve()), "size_bytes": size,
            "sha256": digest.hexdigest(), "utf8_validated_all_bytes": True,
            "encoding": "UTF-8 BOM" if prefix == codecs.BOM_UTF8 else
            ("ASCII-only; compatible with UTF-8; original producer encoding indeterminate"
             if ascii_only else "UTF-8 (no BOM)")}


def read_chunks(path: Path, training: bool, chunksize: int = 200_000):
    expected = KEYS + NUMERIC + ([TARGET] if training else [])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if header != expected:
        raise ValueError(f"Unexpected schema in {path}: {header}; expected {expected}")
    dtype = {"ts_code": "string", "trade_date": "int64"}
    dtype.update({name: "float64" for name in expected if name not in KEYS})
    # Explicit missing tokens; numeric conversion errors fail loudly, never coerce.
    return pd.read_csv(path, encoding="utf-8-sig", dtype=dtype,
                       keep_default_na=False, na_values=["", "NaN", "nan", "NA", "null", "NULL"],
                       chunksize=chunksize)


def examples(frame: pd.DataFrame, mask, columns=None, limit=5) -> list:
    subset = frame.loc[mask, columns or list(frame.columns)].head(limit)
    return json.loads(subset.to_json(orient="records", double_precision=15))


def condition_stats(frame: pd.DataFrame) -> dict:
    result = {}
    for name, columns, violation in [
        ("high_below_open_or_close", ["high", "open", "close"],
         frame.high < np.maximum(frame.open, frame.close)),
        ("low_above_open_or_close", ["low", "open", "close"],
         frame.low > np.minimum(frame.open, frame.close)),
        ("high_below_low", ["high", "low"], frame.high < frame.low),
    ]:
        eligible = np.isfinite(frame[columns]).all(axis=1)
        bad = eligible & violation
        result[name] = {"eligible": int(eligible.sum()), "count": int(bad.sum()),
                        "samples": examples(frame, bad, KEYS + PRICES)}
    return result


def panel_stats(panel: pd.DataFrame) -> dict:
    """Calendar gaps use the union of dates in this panel, without inserting rows."""
    p = panel.sort_values(KEYS, kind="stable")
    calendar = np.sort(p.trade_date.unique())
    g = p.groupby("ts_code", observed=True, sort=False)
    counts = g.trade_date.nunique()
    duplicate = p.duplicated(KEYS, keep=False)
    pos = pd.Series(np.searchsorted(calendar, p.trade_date), index=p.index)
    gaps = pos.groupby(p.ts_code, observed=True).diff() - 1
    gap_mask = gaps > 0
    first_pos = pos.groupby(p.ts_code, observed=True).min()
    last_pos = pos.groupby(p.ts_code, observed=True).max()
    missing = len(calendar) * len(counts) - int(counts.sum())
    all_nan = p.price_all_nan
    any_nan = p.price_any_nan
    finite_close = np.isfinite(p.close)
    first_finite_date = p.trade_date.where(finite_close).groupby(p.ts_code, observed=True).transform("min")
    last_finite_date = p.trade_date.where(finite_close).groupby(p.ts_code, observed=True).transform("max")
    between = (p.trade_date > first_finite_date) & (p.trade_date < last_finite_date)
    output = {
        "rows": len(p), "stocks": len(counts), "trading_dates": len(calendar),
        "date_min": int(calendar[0]), "date_max": int(calendar[-1]),
        "missing_key_rows": int(p[KEYS].isna().any(axis=1).sum()),
        "duplicate_extra_rows": int(p.duplicated(KEYS).sum()),
        "duplicate_key_groups": int(p.loc[duplicate, KEYS].drop_duplicates().shape[0]),
        "duplicate_participating_rows": int(duplicate.sum()),
        "duplicate_samples": examples(p, duplicate, KEYS + ["close"]),
        "observations_per_stock_distribution": {str(int(k)): int(v) for k, v in counts.value_counts().sort_index().items()},
        "observations_per_stock_quantiles": {str(k): float(v) for k, v in counts.quantile([0, .25, .5, .75, 1]).items()},
        "observations_per_stock": {str(k): int(v) for k, v in counts.items()},
        "market_grid_absent_rows": int(missing),
        "internal_market_gap_intervals": int(gap_mask.sum()),
        "internal_market_missing_days": int(gaps[gap_mask].sum()),
        "stocks_with_internal_market_gaps": int(p.loc[gap_mask, "ts_code"].nunique()),
        "max_internal_missing_market_days": int(gaps[gap_mask].max()) if gap_mask.any() else 0,
        "leading_absent_market_days": int(first_pos.sum()),
        "trailing_absent_market_days": int((len(calendar) - 1 - last_pos).sum()),
        "gap_samples": examples(p.assign(missing_market_days=gaps), gap_mask,
                                KEYS + ["missing_market_days"]),
        "rows_all_ohlc_nan": int(all_nan.sum()),
        "rows_any_ohlc_nan": int(any_nan.sum()),
        "all_ohlc_nan_before_first_finite_close": int((all_nan & (p.trade_date < first_finite_date)).sum()),
        "all_ohlc_nan_between_finite_closes": int((all_nan & between).sum()),
        "all_ohlc_nan_after_last_finite_close": int((all_nan & (p.trade_date > last_finite_date)).sum()),
        "all_ohlc_nan_in_stocks_without_finite_close": int((all_nan & first_finite_date.isna()).sum()),
        "stocks_without_finite_close": int(first_finite_date.isna().groupby(p.ts_code, observed=True).all().sum()),
        "missing_price_samples": examples(p, all_nan, KEYS + ["close"]),
    }
    if TARGET in p:
        missing_y = p[TARGET].isna()
        output["missing_label_patterns"] = {
            "label_nan_and_all_ohlc_nan": int((missing_y & all_nan).sum()),
            "label_nan_and_all_ohlc_present": int((missing_y & ~any_nan).sum()),
            "label_present_and_any_ohlc_nan": int((~missing_y & any_nan).sum()),
        }
    return output


def label_check(panel: pd.DataFrame, market_calendar=None) -> dict:
    """Diagnostic only; compare finite labels to immediate next observed close."""
    if panel.duplicated(KEYS).any() or panel[KEYS].isna().any(axis=1).any():
        return {"status": "blocked: ambiguous or missing keys"}
    p = panel.sort_values(KEYS, kind="stable")
    next_values = p.groupby("ts_code", observed=True, sort=False)[["close", "trade_date"]].shift(-1)
    dates = np.asarray(market_calendar if market_calendar is not None else sorted(p.trade_date.unique()))
    position = np.searchsorted(dates, p.trade_date)
    expected_date = np.full(len(p), np.nan)
    has_market_next = position + 1 < len(dates)
    expected_date[has_market_next] = dates[position[has_market_next] + 1]
    adjacent = next_values.trade_date.to_numpy() == expected_date
    finite_y = np.isfinite(p[TARGET])
    # Test labels are absent by design, not missing training observations.
    label_scope = p["label_scope"] if "label_scope" in p else pd.Series(True, index=p.index)
    valid = finite_y & np.isfinite(p.close) & (p.close != 0) & np.isfinite(next_values.close)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        calculated = next_values.close / p.close - 1
    valid &= np.isfinite(calculated)
    diff = np.abs(p[TARGET] - calculated)
    mismatch = valid & ~np.isclose(p[TARGET], calculated, atol=ATOL, rtol=RTOL)
    gap_comparable = valid & ~adjacent
    return {
        "status": "full statistical comparison; no label changes",
        "label_scope_rows": int(label_scope.sum()),
        "atol": ATOL, "rtol": RTOL, "finite_labels": int(finite_y.sum()),
        "comparable_next_observation_pairs": int(valid.sum()),
        "matching_pairs": int(valid.sum() - mismatch.sum()),
        "mismatching_pairs": int(mismatch.sum()),
        "max_absolute_error": float(diff[valid].max()) if valid.any() else None,
        "absolute_error_quantiles": {str(k): float(v) for k, v in diff[valid].quantile([.5,.95,.99,1]).items()} if valid.any() else {},
        "finite_labels_not_comparable": int((finite_y & ~valid).sum()),
        "finite_label_with_missing_current_close": int((finite_y & ~np.isfinite(p.close)).sum()),
        "finite_label_with_missing_next_close": int((finite_y & ~np.isfinite(next_values.close)).sum()),
        "pairs_also_next_market_day": int((valid & adjacent).sum()),
        "comparable_pairs_crossing_market_gap": int(gap_comparable.sum()),
        "matching_pairs_crossing_market_gap": int((gap_comparable & ~mismatch).sum()),
        "nan_label_despite_finite_current_next_close": int((label_scope & p[TARGET].isna() & np.isfinite(p.close) &
                                                            (p.close != 0) & np.isfinite(next_values.close)).sum()),
        "mismatch_samples": examples(p.assign(next_close=next_values.close, next_date=next_values.trade_date,
                                               calculated=calculated, absolute_error=diff), mismatch,
                                     KEYS + ["close", TARGET, "next_close", "next_date", "calculated", "absolute_error"]),
        "uncomparable_samples": examples(p.assign(next_close=next_values.close), finite_y & ~valid,
                                          KEYS + ["close", TARGET, "next_close"]),
        "interpretation": "If no comparable market-gap pairs exist, next observed row and next market day cannot be distinguished empirically on valid labels. Missing close values are never skipped or filled.",
    }


def audit_file(path: Path, training: bool, chunksize=200_000):
    result = fingerprint(path)
    stats, checks, flags, pieces = {}, {}, {}, []
    patterns, pattern_samples = Counter(), {}
    rows, invalid_dates, invalid_codes = 0, 0, 0
    last_key, sorted_input = None, True
    for chunk in read_chunks(path, training, chunksize):
        rows += len(chunk)
        result["columns"] = list(chunk.columns)
        result["parsed_dtypes"] = {k: str(v) for k, v in chunk.dtypes.items()}
        parsed_dates = pd.to_datetime(chunk.trade_date.astype(str), format="%Y%m%d", errors="coerce")
        invalid_dates += int(parsed_dates.isna().sum())
        invalid_codes += int((~chunk.ts_code.str.fullmatch(r"\d{6}\.[A-Z]+", na=False)).sum())
        index = pd.MultiIndex.from_frame(chunk[KEYS])
        sorted_input &= index.is_monotonic_increasing and (last_key is None or last_key <= index[0])
        last_key = index[-1]
        for col in chunk:
            a = chunk[col]
            nan = int(a.isna().sum())
            inf = int(np.isinf(a).sum()) if col != "ts_code" else 0
            item = stats.setdefault(col, Counter())
            item.update({"nan": nan, "inf": inf})
            if col != "ts_code":
                finite = a[np.isfinite(a)]
                item["negative"] += int((finite < 0).sum())
                item["zero"] += int((finite == 0).sum())
                if len(finite):
                    item["finite_min"] = min(item.get("finite_min", float("inf")), float(finite.min()))
                    item["finite_max"] = max(item.get("finite_max", -float("inf")), float(finite.max()))
        for name, values in condition_stats(chunk).items():
            total = checks.setdefault(name, {"eligible": 0, "count": 0, "samples": []})
            total["eligible"] += values["eligible"]
            total["count"] += values["count"]
            total["samples"] = (total["samples"] + values["samples"])[:5]
        for name in ["flag_limit_up", "flag_limit_down"]:
            total = flags.setdefault(name, Counter())
            total.update({str(k): int(v) for k, v in chunk[name].value_counts(dropna=False).items()})
        price_nan = chunk[PRICES].isna()
        all_nan = price_nan.all(axis=1)
        all_finite = np.isfinite(chunk[PRICES]).all(axis=1)
        for name, mask in {
            "all_ohlc_nan_with_finite_vol": all_nan & np.isfinite(chunk.vol),
            "all_ohlc_nan_with_finite_amount": all_nan & np.isfinite(chunk.amount),
            "finite_ohlc_with_zero_vol_and_amount": all_finite & (chunk.vol == 0) & (chunk.amount == 0),
            "finite_ohlc_with_missing_vol_or_amount": all_finite & (chunk.vol.isna() | chunk.amount.isna()),
        }.items():
            patterns[name] += int(mask.sum())
            pattern_samples[name] = (pattern_samples.get(name, []) + examples(chunk, mask))[:5]
        piece = chunk[KEYS + ["close"] + ([TARGET] if training else [])].copy()
        piece["label_scope"] = training
        piece["price_all_nan"] = price_nan.all(axis=1)
        piece["price_any_nan"] = price_nan.any(axis=1)
        pieces.append(piece)
    if rows == 0:
        raise ValueError(f"Empty data file: {path}")
    panel = pd.concat(pieces, ignore_index=True)
    del pieces
    panel["ts_code"] = panel.ts_code.astype("category")
    for item in stats.values():
        item["nan_ratio"] = item["nan"] / rows
        item["inf_ratio"] = item["inf"] / rows
    result.update({"rows": rows, "column_statistics": stats, "ohlc_checks": checks,
                   "flags": {name: {"value_counts": values,
                                    "abnormal_count": sum(n for value, n in values.items() if value not in ["0.0", "1.0"])}
                             for name, values in flags.items()},
                   "invalid_date_rows": invalid_dates, "invalid_code_rows": invalid_codes,
                   "input_sorted_by_stock_date": bool(sorted_input),
                   "observation_patterns": dict(patterns), "observation_pattern_samples": pattern_samples,
                   "price_nonpositive_counts": {name: stats[name]["negative"] + stats[name]["zero"] for name in PRICES},
                   "volume_amount_anomalies": {name: {k: stats[name][k] for k in ["negative", "zero", "nan", "inf", "finite_min", "finite_max"]}
                                                for name in ["vol", "amount"]}})
    result["panel"] = panel_stats(panel)
    if training:
        result["label_missing"] = {"count": stats[TARGET]["nan"], "ratio": stats[TARGET]["nan_ratio"]}
        result["label_check_internal"] = label_check(panel)
    return result, panel


def render_report(summary: dict) -> str:
    lines = ["# 数据全量审计报告", "", f"生成时间：{summary['generated_at']}",
             f"data_version：`{summary['data_version']}`", "",
             "仅工程初始化与只读全量审计；未生成特征、修改标签、填充数据、训练或运行 E000–E007。",
             "数值 dtype 为读取后的显式类型；CSV本身没有类型元数据。编码结论来自全文件校验。",
             "所有统计覆盖全部行；标签公式仅用于诊断，不写回数据。详细逐股计数与样例见 JSON。", ""]
    for section in ["confirmed_facts", "warnings", "unresolved_questions", "blockers"]:
        lines += [f"## {section}", ""]
        lines += [f"- {x}" for x in summary[section]] or ["- 无。"]
        lines += [""]
    for name, data in summary["datasets"].items():
        lines += [f"## {name}", "", f"路径：`{data['path']}`", f"文件大小：{data['size_bytes']:,} bytes",
                  f"编码：{data['encoding']}", f"SHA-256：`{data['sha256']}`", "",
                  "| 字段 | dtype | NaN | NaN比例 | inf | inf比例 |", "|---|---|---:|---:|---:|---:|"]
        for col, stat in data["column_statistics"].items():
            lines.append(f"| {col} | {data['parsed_dtypes'][col]} | {stat['nan']:,} | {stat['nan_ratio']:.8%} | {stat['inf']:,} | {stat['inf_ratio']:.8%} |")
        panel = {k: v for k, v in data["panel"].items() if k != "observations_per_stock"}
        for title, payload in [("规模、唯一性、观测分布、缺行与缺失形态", panel),
                               ("涨跌停标志", data["flags"]), ("OHLC合法性", data["ohlc_checks"]),
                               ("非正价格", data["price_nonpositive_counts"]),
                               ("其他观测形态", {"counts": data["observation_patterns"], "samples": data["observation_pattern_samples"]}),
                               ("成交量额：负数、零、缺失、无穷及有限范围", data["volume_amount_anomalies"])]:
            lines += ["", f"### {title}", "", "```json", json.dumps(payload, ensure_ascii=False, indent=2), "```"]
        if "label_check_internal" in data:
            lines += ["", "### 训练集内部标签核验", "", "```json", json.dumps(data["label_check_internal"],ensure_ascii=False,indent=2), "```"]
    lines += ["", "## 训练/测试交叉核验", "", "```json", json.dumps(summary["cross_dataset"],ensure_ascii=False,indent=2), "```",
              "", "## 方法与限制", "",
              "- 市场交易日取两份文件实际观测日期的并集；不能证明全市场共同缺失的日期不存在。",
              "- 缺行按股票×观测市场日期网格比较；不补行。停牌/未上市归因不能仅由量价缺失确定。",
              "- OHLC不等式仅在所需价格均为有限值时核验；不使用容差掩盖原始违法值。",
              "- vol/amount异常统计包括负数、NaN、inf；零单独列示。没有赛题阈值，不把大额值自动判为错误。",
              f"- 标签容差 atol={ATOL}, rtol={RTOL}；只比较当前close非零且标签及相邻close有限的行。",
              "- 跨训练/测试边界close仅用于只读数据诊断，不参与训练。没有测试标签。",
              "- 官方评分器缺失是官方评分一致性的阻碍，不阻止独立审计或Basic40实现。", ""]
    return "\n".join(lines)


def run(root: Path, chunksize=200_000) -> dict:
    results, panels = {}, {}
    for name, filename, training in [("train", "训练集.csv", True), ("test", "测试集_X.csv", False)]:
        print(f"Auditing {name}: every row", flush=True)
        results[name], panels[name] = audit_file(root / "data/raw" / filename, training, chunksize)
        print(f"Completed {name}: {results[name]['rows']:,} rows", flush=True)
    train, test = panels["train"], panels["test"]
    train_codes, test_codes = set(train.ts_code), set(test.ts_code)
    date_overlap = sorted(set(train.trade_date) & set(test.trade_date))
    combined = pd.concat([train, test], ignore_index=True)
    combined["ts_code"] = combined.ts_code.astype("category")
    cross = {"stock_sets_equal": train_codes == test_codes,
             "train_only_stocks": sorted(train_codes - test_codes), "test_only_stocks": sorted(test_codes - train_codes),
             "overlapping_dates": [int(d) for d in date_overlap],
             "train_max_before_test_min": bool(train.trade_date.max() < test.trade_date.min()),
             "overlapping_keys": int(train[KEYS].merge(test[KEYS], on=KEYS).shape[0]) if date_overlap else 0,
             "train_last_date": int(train.trade_date.max()), "test_first_date": int(test.trade_date.min()),
             "label_check_with_test_boundary_close": label_check(combined)}
    data_version = "sha256:" + hashlib.sha256("\n".join(f"{name}:{results[name]['sha256']}" for name in sorted(results)).encode()).hexdigest()
    facts, warnings, questions, blockers = [], [], [], []
    for name, data in results.items():
        p = data["panel"]
        facts.append(f"{name}: {p['rows']:,}行，{p['stocks']}只股票，{p['trading_dates']}个观测市场日期，{p['date_min']}—{p['date_max']}；重复多余行{p['duplicate_extra_rows']}。")
        facts.append(f"{name}: 股票×日期网格缺行{p['market_grid_absent_rows']:,}；内部市场日期缺口{p['internal_market_gap_intervals']:,}段；OHLC全缺失行{p['rows_all_ohlc_nan']:,}。")
        expected = (7_900_350, 4650, 20180102, 20241231) if name == "train" else (1_599_600, 4650, 20250102, 20260608)
        actual = (p["rows"], p["stocks"], p["date_min"], p["date_max"])
        if actual != expected:
            warnings.append(f"{name}规模/日期与COMPETITION_SPEC不一致：实际{actual}，规范{expected}；未修改规范。")
        if p["duplicate_extra_rows"] or p["missing_key_rows"] or data["invalid_date_rows"]:
            blockers.append(f"{name}存在重复/缺失键或非法日期，必须先明确处理才能可靠生成时序特征。")
        if sum(v["count"] for v in data["ohlc_checks"].values()):
            warnings.append(f"{name}有OHLC不等式异常，详见计数和样例；不自动修复。")
        if sum(data["price_nonpositive_counts"].values()):
            warnings.append(f"{name}有非正价格，需按已冻结的无效输入规则处理派生值，原值保持不变。")
        if any(v["inf"] for v in data["column_statistics"].values()):
            warnings.append(f"{name}存在inf，详见逐列统计。")
        if any(v["abnormal_count"] for v in data["flags"].values()):
            warnings.append(f"{name}涨跌停标志存在非0/1值或缺失。")
        if any(v["negative"] for v in data["volume_amount_anomalies"].values()):
            warnings.append(f"{name}成交量或成交额存在负数。")
    missing = results["train"]["label_missing"]
    facts.append(f"训练标签缺失{missing['count']:,}行，占{missing['ratio']:.6%}。")
    for name, data in results.items():
        patterns = data["observation_patterns"]
        if patterns["finite_ohlc_with_zero_vol_and_amount"]:
            warnings.append(f"{name}有{patterns['finite_ohlc_with_zero_vol_and_amount']:,}行OHLC有限但vol/amount均为0；保留原值，不能自动认定其业务原因。")
        if patterns["all_ohlc_nan_with_finite_vol"]:
            warnings.append(f"{name}有{patterns['all_ohlc_nan_with_finite_vol']:,}行OHLC全NaN但vol有限，详见样例。")
    facts.append(f"训练/测试股票集合一致：{cross['stock_sets_equal']}；重叠日期{len(date_overlap)}；重叠键{cross['overlapping_keys']}。")
    check = cross["label_check_with_test_boundary_close"]
    if "mismatching_pairs" in check:
        facts.append(f"含边界close的标签核验：可比较{check['comparable_next_observation_pairs']:,}对，一致{check['matching_pairs']:,}对，不一致{check['mismatching_pairs']:,}对。")
        if check["mismatching_pairs"]:
            warnings.append("有限标签与下一观测行收益存在不一致，见误差统计和样例；未修改任何标签。")
        if check["finite_labels_not_comparable"]:
            warnings.append(f"有{check['finite_labels_not_comparable']:,}个有限标签无法用相邻close安全复算；需区分缺失价格与文件尾边界。")
    if date_overlap or not cross["train_max_before_test_min"]:
        blockers.append("训练/测试时间边界异常，需要先确认数据版本。")
    warnings.append("价格/量额缺失和零成交仅是观测事实，不能直接认定为停牌或未上市。")
    questions += ["缺失价格区段分别属于停牌、未上市还是其他数据处理？仅凭现有字段无法逐段定性。",
                  "观测日期并集是否完整覆盖交易所日历？禁止引入外部数据或自行补齐。",
                  "官方evaluate.py仍未提供，评分边界无法完成官方一致性验证；任何自实现评分器只能标为provisional。"]
    if check.get("comparable_pairs_crossing_market_gap", 0) == 0:
        questions.append("有效标签核验中下一观测行与下一观测市场日重合，数据无法区分两种生成算法在缺行情况下的行为；特征lag仍严格执行用户冻结的观测行规则。")
    basic40_blockers = list(blockers)
    blockers.append("官方评分一致性验证：缺少reference/evaluate_official.py；仅阻碍官方评分验证，不阻碍数据审计或Basic40实现。")
    summary = {"schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
               "data_version": data_version, "python": platform.python_version(),
               "pandas": pd.__version__, "numpy": np.__version__, "scope": "initialization_and_full_read_only_audit",
               "confirmed_facts": facts, "warnings": warnings, "unresolved_questions": questions,
               "blockers": blockers, "basic40_ready": not basic40_blockers,
               "basic40_blockers": basic40_blockers,
               "datasets": results, "cross_dataset": cross,
               "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "config_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "config").glob("*.yaml"))}}
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    (output / "data_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "data_audit_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    run(args.root.resolve(), args.chunksize)


if __name__ == "__main__":
    main()
