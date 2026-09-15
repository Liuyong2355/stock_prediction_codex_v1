from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_prediction.basic40 import (KEYS, SOURCES, compute_basic40, load_contract, stock_batches)
from stock_prediction.build_basic40 import audit_matrix, build

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract(ROOT / "config/features_v1.yaml")
NAMES = CONTRACT["feature_sets"]["Basic40"]


def raw_stock(n=85, code="000001.SZ", start="2023-09-01"):
    rng = np.random.default_rng(42)
    close = 20 * np.cumprod(1 + rng.normal(0, .015, n))
    opened = close * (1 + rng.normal(0, .01, n))
    return pd.DataFrame({"ts_code": code,
                         "trade_date": pd.bdate_range(start, periods=n).strftime("%Y%m%d").astype(int),
                         "open": opened, "high": np.maximum(opened, close) + .4,
                         "low": np.minimum(opened, close) - .3, "close": close,
                         "vol": np.arange(n, dtype=float) + 100,
                         "amount": (np.arange(n, dtype=float) + 100) * 21,
                         "flag_limit_up": 0., "flag_limit_down": 0.})


def scalar_reference(raw, t, name):
    """Independent direct-window NumPy reference: no pandas shift/rolling."""
    eps = 1e-12
    def v(col, pos=t):
        return float(raw.iloc[pos][col]) if pos >= 0 else np.nan
    def window(col, w):
        return raw.iloc[max(0, t - w + 1):t + 1][col].to_numpy(float)
    o, h, l, c = (v(k) for k in ["open", "high", "low", "close"])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if name.startswith("ret_"):
            w = int(name.split("_")[1])
            return c / v("close", t - w) - 1
        if name.startswith("gap_"):
            return v(name[4:]) / v("close", t - 1) - 1
        if name.startswith("madev_"):
            w = int(name.split("_")[1]); a = window("close", w)
            return c / np.mean(a) - 1 if len(a) == w and np.isfinite(a).all() else np.nan
        if name.startswith("stdret_"):
            w = int(name.split("_")[1]); a = window("close", w + 1)
            r = a[1:] / a[:-1] - 1
            return np.std(r, ddof=0) if len(r) == w and np.isfinite(r).all() else np.nan
        if name.startswith(("volratio_", "amtratio_")):
            col = "vol" if name.startswith("volratio_") else "amount"
            w = int(name.split("_")[1]); a = window(col, w)
            return v(col) / (np.mean(a) + eps) if len(a) == w and np.isfinite(a).all() else np.nan
        if name.startswith("flag_"):
            return v(name)
        return {"kmid": (c-o)/(o+eps), "klen": (h-l)/(o+eps),
                "kmid2": (c-o)/(h-l+eps), "kup": (h-max(o,c))/(o+eps),
                "kup2": (h-max(o,c))/(h-l+eps), "klow": (min(o,c)-l)/(o+eps),
                "klow2": (min(o,c)-l)/(h-l+eps), "ksft": (2*c-h-l)/(o+eps),
                "ksft2": (2*c-h-l)/(h-l+eps)}[name]


@pytest.mark.parametrize("name", NAMES)
def test_every_feature_matches_independent_observed_window_reference(name):
    raw = raw_stock()
    raw.loc[8, ["open", "high", "low", "close", "vol", "amount"]] = np.nan
    features = compute_basic40(raw, CONTRACT)
    expected = [scalar_reference(raw, t, name) for t in range(len(raw))]
    np.testing.assert_allclose(features[name], expected, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_alignment_full_window_first_valid_and_order():
    raw = raw_stock()
    result = compute_basic40(raw.sample(frac=1, random_state=12), CONTRACT)
    assert list(result) == KEYS + NAMES
    pd.testing.assert_frame_equal(result[KEYS], raw[KEYS])
    for name in NAMES:
        if name.startswith(("ret_", "stdret_")):
            first = int(name.split("_")[1])
        elif name.startswith(("madev_", "volratio_", "amtratio_")):
            first = int(name.split("_")[1]) - 1
        elif name.startswith("gap_"):
            first = 1
        else:
            first = 0
        assert result[name].first_valid_index() == first


def test_nan_positions_are_not_filled_or_skipped():
    raw = raw_stock(15)
    raw.loc[4, SOURCES[:6]] = np.nan
    result = compute_basic40(raw, CONTRACT)
    assert result.loc[4:5, "ret_1"].isna().all()
    assert result.loc[4:8, "madev_5"].isna().all()
    assert np.isfinite(result.loc[9, "madev_5"])
    assert result.loc[4:9, "stdret_5"].isna().all()
    assert np.isfinite(result.loc[10, "stdret_5"])
    # A long return uses only the two endpoints, not every intermediate close.
    assert result.loc[5, "ret_5"] == pytest.approx(raw.close.iloc[5] / raw.close.iloc[0] - 1)
    assert result.loc[4, "flag_limit_up"] == 0


def test_observed_row_lag_with_absent_market_day():
    raw = raw_stock(10).drop(index=4).reset_index(drop=True)
    result = compute_basic40(raw, CONTRACT)
    assert result.loc[4, "ret_1"] == pytest.approx(raw.close.iloc[4] / raw.close.iloc[3] - 1)
    assert len(result) == 9


def test_no_lookahead_or_label_dependency():
    raw = raw_stock()
    before = compute_basic40(raw.iloc[:65], CONTRACT)
    changed = raw.copy()
    changed.loc[65:, SOURCES[:6]] *= 100
    changed["y_ret_1d"] = np.arange(len(changed)) * 999
    after = compute_basic40(changed, CONTRACT)
    pd.testing.assert_frame_equal(before, after.iloc[:65].reset_index(drop=True))
    assert "y_ret_1d" not in after


def test_cross_section_invariance_and_stock_isolation():
    first = raw_stock()
    second = raw_stock(code="999999.SZ")
    second.loc[:, SOURCES[:6]] *= 1000
    second.loc[10, "close"] = np.nan
    both = compute_basic40(pd.concat([second, first]).sample(frac=1, random_state=5), CONTRACT)
    pd.testing.assert_frame_equal(both.iloc[:len(first)].reset_index(drop=True), compute_basic40(first, CONTRACT))
    pd.testing.assert_frame_equal(both.iloc[len(first):].reset_index(drop=True), compute_basic40(second, CONTRACT))
    assert all(f["groupby"] == "ts_code" for f in CONTRACT["features"] if f["name"] in NAMES)


def test_test_period_inherits_history_without_affecting_train():
    raw = raw_stock()
    joined = compute_basic40(raw, CONTRACT)
    train = compute_basic40(raw.iloc[:65], CONTRACT)
    pd.testing.assert_frame_equal(joined.iloc[:65], train)
    assert joined.loc[65, "ret_60"] == pytest.approx(raw.close.iloc[65] / raw.close.iloc[5] - 1)
    assert np.isfinite(joined.loc[65, "stdret_60"])
    assert compute_basic40(raw.iloc[65:], CONTRACT).ret_60.isna().all()


def test_flat_prices_zero_volume_and_dual_flags_preserved():
    raw = raw_stock(65)
    raw.loc[:, ["open", "high", "low", "close"]] = 10.
    raw.loc[:, ["vol", "amount"]] = 0.
    raw.loc[10, ["flag_limit_up", "flag_limit_down"]] = 1
    original = raw.copy(deep=True)
    result = compute_basic40(raw, CONTRACT)
    assert result.loc[10, "flag_limit_up"] == result.loc[10, "flag_limit_down"] == 1
    assert (result.loc[4:, "volratio_5"] == 0).all()
    assert (result.loc[5:, "stdret_5"] == 0).all()
    assert (result[["kmid2", "kup2", "klow2", "ksft2"]] == 0).all(axis=None)
    pd.testing.assert_frame_equal(raw, original)


def test_infinite_inputs_cannot_become_finite_features():
    raw = raw_stock(15)
    raw.loc[4, ["close", "vol"]] = np.inf
    result = compute_basic40(raw, CONTRACT)
    assert not np.isinf(result[NAMES]).any(axis=None)
    assert result.loc[4:5, "ret_1"].isna().all()
    assert result.loc[4:8, "volratio_5"].isna().all()


def test_stock_batches_span_chunk_boundaries_and_reject_bad_keys():
    raw = pd.concat([raw_stock(9), raw_stock(7, code="000002.SZ")], ignore_index=True)
    batches = list(stock_batches([raw.iloc[:3], raw.iloc[3:11], raw.iloc[11:]]))
    assert [len(b[1]) for b in batches] == [9, 7]
    pd.testing.assert_frame_equal(pd.concat([b[1] for b in batches], ignore_index=True), raw)
    with pytest.raises(ValueError):
        list(stock_batches([raw.iloc[:3], raw.iloc[2:]]))
    with pytest.raises(ValueError):
        compute_basic40(pd.concat([raw, raw.iloc[:1]]), CONTRACT)


def test_persisted_audit_quantiles_ignore_nan_and_show_keys(tmp_path):
    matrix = np.full((4, 40), np.nan)
    matrix[:, 0] = [1, 2, np.nan, 9]
    keys = np.array([("A", i) for i in range(4)], dtype=[("ts_code", "U16"), ("trade_date", "i8")])
    np.save(tmp_path / "features.npy", matrix)
    np.save(tmp_path / "keys.npy", keys)
    stats = audit_matrix(tmp_path / "features.npy", tmp_path / "keys.npy", NAMES)["features"][NAMES[0]]
    assert stats["finite_count"] == 3 and stats["nan_ratio"] == .25
    assert stats["quantiles"]["p50"] == 2
    assert stats["extreme_examples"][1]["trade_date"] == 3


def test_full_pipeline_small_files(tmp_path):
    import json
    import shutil
    from stock_prediction.audit import run
    for directory in ["config", "docs", "data/raw"]:
        (tmp_path / directory).mkdir(parents=True)
    shutil.copyfile(ROOT / "config/features_v1.yaml", tmp_path / "config/features_v1.yaml")
    shutil.copyfile(ROOT / "docs/FEATURE_SPEC_V1.md", tmp_path / "docs/FEATURE_SPEC_V1.md")
    raw = raw_stock()
    train = raw.iloc[:65].copy()
    train["y_ret_1d"] = np.nan  # All raw rows must still get features.
    train.to_csv(tmp_path / "data/raw/训练集.csv", index=False)
    raw.iloc[65:].to_csv(tmp_path / "data/raw/测试集_X.csv", index=False)
    run(tmp_path, chunksize=13)
    result = build(tmp_path, chunksize=7)
    expected = compute_basic40(raw, CONTRACT)
    for split, lo, hi in [("train", 0, 65), ("test", 65, 85)]:
        actual = np.load(tmp_path / f"outputs/features/basic40/{split}_basic40.npy", allow_pickle=False)
        np.testing.assert_allclose(actual, expected.iloc[lo:hi][NAMES], equal_nan=True)
    assert result["validation"]["raw_sha256_unchanged"]
    saved = json.loads((tmp_path / "outputs/basic40_manifest.json").read_text(encoding="utf-8"))
    assert saved["feature_names"] == NAMES
    assert (tmp_path / "outputs/basic40_audit_report.md").exists()
