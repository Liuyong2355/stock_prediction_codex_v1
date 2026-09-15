"""Frozen Basic40 only: observed-row, backward-looking stock features."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

KEYS = ["ts_code", "trade_date"]
SOURCES = ["open", "high", "low", "close", "vol", "amount", "flag_limit_up", "flag_limit_down"]
RETURN_WINDOWS = (1, 2, 3, 5, 10, 20, 30, 60)
MEAN_WINDOWS = (3, 5, 10, 20, 30, 60)
RATIO_WINDOWS = (5, 10, 20, 60)
EPSILON = 1e-12


def load_contract(path: Path) -> dict:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = contract["feature_sets"]["Basic40"]
    definitions = {f["name"]: f for f in contract["features"]}
    if len(names) != 40 or len(set(names)) != 40:
        raise ValueError("Basic40 must contain exactly 40 unique names")
    if any(definitions[n]["groupby"] != "ts_code" for n in names):
        raise ValueError("This implementation supports per-stock Basic40 only")
    if contract["global_conventions"]["epsilon"] != EPSILON:
        raise ValueError("Frozen epsilon differs from implementation")
    return contract


def _one_stock(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    # Work on a copy; nonfinite inputs cannot become finite derived values.
    x = frame[SOURCES].astype("float64").replace([np.inf, -np.inf], np.nan)
    o, h, l, c = (x[name] for name in ["open", "high", "low", "close"])
    values = {}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for w in RETURN_WINDOWS:
            values[f"ret_{w}"] = c / c.shift(w) - 1
        span = h - l
        values.update({
            "kmid": (c - o) / (o + EPSILON),
            "klen": span / (o + EPSILON),
            "kmid2": (c - o) / (span + EPSILON),
            "kup": (h - np.maximum(o, c)) / (o + EPSILON),
            "kup2": (h - np.maximum(o, c)) / (span + EPSILON),
            "klow": (np.minimum(o, c) - l) / (o + EPSILON),
            "klow2": (np.minimum(o, c) - l) / (span + EPSILON),
            "ksft": (2 * c - h - l) / (o + EPSILON),
            "ksft2": (2 * c - h - l) / (span + EPSILON),
        })
        for w in MEAN_WINDOWS:
            values[f"madev_{w}"] = c / c.rolling(w, min_periods=w).mean() - 1
        ret = values["ret_1"].replace([np.inf, -np.inf], np.nan)
        for w in RATIO_WINDOWS:
            values[f"stdret_{w}"] = ret.rolling(w, min_periods=w).std(ddof=0)
            for prefix, column in [("volratio", "vol"), ("amtratio", "amount")]:
                v = x[column]
                values[f"{prefix}_{w}"] = v / (v.rolling(w, min_periods=w).mean() + EPSILON)
        for column in ["open", "high", "low"]:
            values[f"gap_{column}"] = x[column] / c.shift(1) - 1
    for column in ["flag_limit_up", "flag_limit_down"]:
        values[column] = x[column]
    if set(values) != set(names):
        raise ValueError("Implementation names differ from frozen Basic40 membership")
    return pd.DataFrame(values, index=frame.index)[names].replace([np.inf, -np.inf], np.nan)


def compute_basic40(raw: pd.DataFrame, contract: dict) -> pd.DataFrame:
    """Return sorted keys + 40 float64 features, without labels or raw prices.

    Absent dates are never inserted. A missing row value still occupies its lag
    position. Every rolling window includes the current observed row and needs
    all w finite inputs. Input order and labels cannot influence the output.
    """
    names = contract["feature_sets"]["Basic40"]
    frame = raw[KEYS + SOURCES].copy()
    if frame[KEYS].isna().any(axis=None) or frame.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate stock/date keys")
    frame = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
    parts = []
    for _, group in frame.groupby("ts_code", sort=False, observed=True):
        features = _one_stock(group, names)
        parts.append(pd.concat([group[KEYS], features], axis=1))
    if not parts:
        return pd.DataFrame(columns=KEYS + names)
    return pd.concat(parts, ignore_index=True)


def stock_batches(chunks):
    """Yield complete stock histories across CSV chunks; reject unsorted input."""
    carry = None
    previous_key = None
    for chunk in chunks:
        chunk = chunk[KEYS + SOURCES]
        if chunk.empty:
            continue
        index = pd.MultiIndex.from_frame(chunk[KEYS])
        if not index.is_monotonic_increasing or (previous_key is not None and previous_key >= index[0]):
            raise ValueError("Streaming input must be strictly sorted by stock/date")
        if index.has_duplicates:
            raise ValueError("Duplicate stock/date keys in streaming input")
        previous_key = index[-1]
        joined = pd.concat([carry, chunk], ignore_index=True) if carry is not None else chunk
        last_code = joined.ts_code.iloc[-1]
        last = joined.ts_code == last_code
        complete = joined.loc[~last]
        for code, group in complete.groupby("ts_code", sort=False, observed=True):
            yield str(code), group.reset_index(drop=True)
        carry = joined.loc[last].copy()
    if carry is not None:
        yield str(carry.ts_code.iloc[0]), carry.reset_index(drop=True)
