import hashlib
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from stock_prediction.audit import (KEYS, NUMERIC, TARGET, audit_file, condition_stats,
                                    fingerprint, label_check, panel_stats, read_chunks, run)


def raw_frame():
    return pd.DataFrame([
        ["000001.SZ", 20240102, 10, 11, 9, 10, 100, 1000, 0, 0, .1],
        ["000001.SZ", 20240103, 11, 12, 10, 11, 100, 1100, 1, 0, np.nan],
        ["000002.SZ", 20240102, np.nan, np.nan, np.nan, np.nan, 0, 0, 0, 0, np.nan],
        ["000002.SZ", 20240103, 20, 19, 21, 20, -1, np.inf, 2, 0, np.nan],
    ], columns=KEYS + NUMERIC + [TARGET])


def panel(frame):
    out = frame[KEYS + ["close", TARGET]].copy()
    out["price_all_nan"] = frame[["open", "high", "low", "close"]].isna().all(axis=1)
    out["price_any_nan"] = frame[["open", "high", "low", "close"]].isna().any(axis=1)
    return out


def test_reader_preserves_code_nan_and_bytes(tmp_path):
    path = tmp_path / "训练集.csv"
    raw_frame().to_csv(path, index=False)
    before = path.read_bytes()
    result, p = audit_file(path, True, chunksize=2)
    assert result["rows"] == 4
    assert p.ts_code.iloc[0] == "000001.SZ"
    assert result["label_missing"]["count"] == 3
    assert result["column_statistics"]["amount"]["inf"] == 1
    assert result["volume_amount_anomalies"]["vol"]["negative"] == 1
    assert result["flags"]["flag_limit_up"]["abnormal_count"] == 1
    assert result["sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    assert result["panel"]["market_grid_absent_rows"] == 0
    assert result["panel"]["rows_all_ohlc_nan"] == 1


def test_duplicate_cross_chunk_boundary(tmp_path):
    data = raw_frame().iloc[[0, 0, 1]].copy()
    path = tmp_path / "duplicate.csv"
    data.to_csv(path, index=False)
    result, _ = audit_file(path, True, chunksize=1)
    assert result["panel"]["duplicate_extra_rows"] == 1
    assert result["panel"]["duplicate_key_groups"] == 1
    assert result["label_check_internal"]["status"].startswith("blocked")


def test_market_gap_is_not_price_nan_and_next_row_is_not_next_day():
    p = panel(raw_frame())
    p.loc[1, "trade_date"] = 20240104
    p.loc[1, "close"] = 11
    s = panel_stats(p)
    assert s["internal_market_gap_intervals"] == 1
    assert s["internal_market_missing_days"] == 1
    assert s["rows_all_ohlc_nan"] == 1
    check = label_check(p)
    assert check["matching_pairs_crossing_market_gap"] == 1
    assert check["pairs_also_next_market_day"] == 0


def test_label_check_no_stock_crossing_no_skip_missing_and_no_mutation():
    p = panel(raw_frame())
    p.loc[1, TARGET] = 10  # last stock row cannot use next stock's close
    p.loc[2, TARGET] = 0
    original = p.copy(deep=True)
    result = label_check(p)
    assert result["matching_pairs"] == 1
    assert result["finite_labels_not_comparable"] == 2
    pd.testing.assert_frame_equal(p, original)
    p.loc[0, TARGET] = .2
    assert label_check(p)["mismatching_pairs"] == 1


def test_boundary_uses_test_close_but_not_absent_test_labels():
    p = pd.DataFrame({"ts_code": ["A"] * 3, "trade_date": [20241231, 20250102, 20250103],
                      "close": [10., 11., 12.], TARGET: [.1, np.nan, np.nan],
                      "label_scope": [True, False, False]})
    result = label_check(p)
    assert result["label_scope_rows"] == 1
    assert result["matching_pairs"] == 1
    assert result["nan_label_despite_finite_current_next_close"] == 0


def test_never_skip_missing_next_close():
    p = pd.DataFrame({"ts_code": ["A"] * 3, "trade_date": [20240102, 20240103, 20240104],
                      "close": [10., np.nan, 11.], TARGET: [.1, np.nan, np.nan]})
    result = label_check(p)
    assert result["comparable_next_observation_pairs"] == 0
    assert result["finite_label_with_missing_next_close"] == 1


def test_end_to_end_report_and_data_version(tmp_path):
    import json
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    frame = raw_frame().iloc[:2].copy()
    frame.to_csv(raw / "训练集.csv", index=False)
    test = frame.drop(columns=[TARGET]).copy()
    test.trade_date = [20250102, 20250103]
    test.to_csv(raw / "测试集_X.csv", index=False)
    summary = run(tmp_path, chunksize=1)
    saved = json.loads((tmp_path / "outputs/data_audit_summary.json").read_text(encoding="utf-8"))
    assert saved["data_version"] == summary["data_version"]
    assert saved["cross_dataset"]["overlapping_keys"] == 0
    report = (tmp_path / "outputs/data_audit_report.md").read_text(encoding="utf-8")
    for section in ["confirmed_facts", "warnings", "unresolved_questions", "blockers"]:
        assert f"## {section}" in report


def test_ohlc_eligible_denominators():
    checks = condition_stats(raw_frame())
    for check in checks.values():
        assert check["eligible"] == 3
        assert check["count"] == 1


def test_reject_schema_and_invalid_numeric(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("ts_code,trade_date\n000001.SZ,20240102\n")
    with pytest.raises(ValueError, match="schema"):
        read_chunks(path, True)
    frame = raw_frame().astype({"close": "object"})
    frame.loc[0, "close"] = "invalid"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        list(read_chunks(path, True))


def test_encoding_checks_entire_file(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"a" * (1024 * 1024 + 1) + b"\xff")
    with pytest.raises(UnicodeDecodeError):
        fingerprint(path)


def test_frozen_contract_integrity():
    root = Path(__file__).resolve().parents[1]
    features = yaml.safe_load((root / "config/features_v1.yaml").read_text(encoding="utf-8"))
    for name, count in [("Basic40", 40), ("Full147", 147)]:
        assert len(features["feature_sets"][name]) == count
        assert len(set(features["feature_sets"][name])) == count
    assert set(features["feature_sets"]["Basic40"]) <= set(features["feature_sets"]["Full147"])
    assert features["global_conventions"]["rolling_min_periods"] == "window"
    experiments = yaml.safe_load((root / "config/experiments_v1.yaml").read_text(encoding="utf-8"))
    assert experiments["evaluator_status"] == "provisional"
    assert experiments["rank_ic_std_ddof"] == 0
    assert experiments["seed"] == 42
