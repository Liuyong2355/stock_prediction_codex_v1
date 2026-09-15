# 数据全量审计报告

生成时间：2026-09-15T06:41:29.514610+00:00
data_version：`sha256:6f27670167a0b305ebde0d303a272a0a3215def354c15cc316d287f1444540e2`

仅工程初始化与只读全量审计；未生成特征、修改标签、填充数据、训练或运行 E000–E007。
数值 dtype 为读取后的显式类型；CSV本身没有类型元数据。编码结论来自全文件校验。
所有统计覆盖全部行；标签公式仅用于诊断，不写回数据。详细逐股计数与样例见 JSON。

## confirmed_facts

- train: 7,900,350行，4650只股票，1699个观测市场日期，20180102—20241231；重复多余行0。
- train: 股票×日期网格缺行0；内部市场日期缺口0段；OHLC全缺失行1,139,980。
- test: 1,599,600行，4650只股票，344个观测市场日期，20250102—20260608；重复多余行0。
- test: 股票×日期网格缺行0；内部市场日期缺口0段；OHLC全缺失行44,844。
- 训练标签缺失1,141,819行，占14.452765%。
- 训练/测试股票集合一致：True；重叠日期0；重叠键0。
- 含边界close的标签核验：可比较6,758,531对，一致6,758,531对，不一致0对。

## warnings

- train有38,468行OHLC有限但vol/amount均为0；保留原值，不能自动认定其业务原因。
- train有40行OHLC全NaN但vol有限，详见样例。
- test有13行OHLC全NaN但vol有限，详见样例。
- 价格/量额缺失和零成交仅是观测事实，不能直接认定为停牌或未上市。

## unresolved_questions

- 缺失价格区段分别属于停牌、未上市还是其他数据处理？仅凭现有字段无法逐段定性。
- 观测日期并集是否完整覆盖交易所日历？禁止引入外部数据或自行补齐。
- 官方evaluate.py仍未提供，评分边界无法完成官方一致性验证；任何自实现评分器只能标为provisional。
- 有效标签核验中下一观测行与下一观测市场日重合，数据无法区分两种生成算法在缺行情况下的行为；特征lag仍严格执行用户冻结的观测行规则。

## blockers

- 官方评分一致性验证：缺少reference/evaluate_official.py；仅阻碍官方评分验证，不阻碍数据审计或Basic40实现。

## train

路径：`D:\Virtual C_Drive\金融建模\stock_prediction_codex_v1\data\raw\训练集.csv`
文件大小：946,832,276 bytes
编码：ASCII-only; compatible with UTF-8; original producer encoding indeterminate
SHA-256：`e9d23e87f7e1f9439f28f7dd951d129ea49eaacda5ad650da8576a830a6deac9`

| 字段 | dtype | NaN | NaN比例 | inf | inf比例 |
|---|---|---:|---:|---:|---:|
| ts_code | string | 0 | 0.00000000% | 0 | 0.00000000% |
| trade_date | int64 | 0 | 0.00000000% | 0 | 0.00000000% |
| open | float64 | 1,139,980 | 14.42948730% | 0 | 0.00000000% |
| high | float64 | 1,139,980 | 14.42948730% | 0 | 0.00000000% |
| low | float64 | 1,139,980 | 14.42948730% | 0 | 0.00000000% |
| close | float64 | 1,139,980 | 14.42948730% | 0 | 0.00000000% |
| vol | float64 | 1,139,940 | 14.42898099% | 0 | 0.00000000% |
| amount | float64 | 1,139,940 | 14.42898099% | 0 | 0.00000000% |
| flag_limit_up | float64 | 0 | 0.00000000% | 0 | 0.00000000% |
| flag_limit_down | float64 | 0 | 0.00000000% | 0 | 0.00000000% |
| y_ret_1d | float64 | 1,141,819 | 14.45276475% | 0 | 0.00000000% |

### 规模、唯一性、观测分布、缺行与缺失形态

```json
{
  "rows": 7900350,
  "stocks": 4650,
  "trading_dates": 1699,
  "date_min": 20180102,
  "date_max": 20241231,
  "missing_key_rows": 0,
  "duplicate_extra_rows": 0,
  "duplicate_key_groups": 0,
  "duplicate_participating_rows": 0,
  "duplicate_samples": [],
  "observations_per_stock_distribution": {
    "1699": 4650
  },
  "observations_per_stock_quantiles": {
    "0.0": 1699.0,
    "0.25": 1699.0,
    "0.5": 1699.0,
    "0.75": 1699.0,
    "1.0": 1699.0
  },
  "market_grid_absent_rows": 0,
  "internal_market_gap_intervals": 0,
  "internal_market_missing_days": 0,
  "stocks_with_internal_market_gaps": 0,
  "max_internal_missing_market_days": 0,
  "leading_absent_market_days": 0,
  "trailing_absent_market_days": 0,
  "gap_samples": [],
  "rows_all_ohlc_nan": 1139980,
  "rows_any_ohlc_nan": 1139980,
  "all_ohlc_nan_before_first_finite_close": 1082406,
  "all_ohlc_nan_between_finite_closes": 8694,
  "all_ohlc_nan_after_last_finite_close": 25094,
  "all_ohlc_nan_in_stocks_without_finite_close": 23786,
  "stocks_without_finite_close": 14,
  "missing_price_samples": [
    {
      "ts_code": "000004.SZ",
      "trade_date": 20220505,
      "close": null
    },
    {
      "ts_code": "000004.SZ",
      "trade_date": 20230627,
      "close": null
    },
    {
      "ts_code": "000005.SZ",
      "trade_date": 20210430,
      "close": null
    },
    {
      "ts_code": "000005.SZ",
      "trade_date": 20240306,
      "close": null
    },
    {
      "ts_code": "000005.SZ",
      "trade_date": 20240307,
      "close": null
    }
  ],
  "missing_label_patterns": {
    "label_nan_and_all_ohlc_nan": 1139980,
    "label_nan_and_all_ohlc_present": 1839,
    "label_present_and_any_ohlc_nan": 0
  }
}
```

### 涨跌停标志

```json
{
  "flag_limit_up": {
    "value_counts": {
      "0.0": 7786130,
      "1.0": 114220
    },
    "abnormal_count": 0
  },
  "flag_limit_down": {
    "value_counts": {
      "0.0": 7853990,
      "1.0": 46360
    },
    "abnormal_count": 0
  }
}
```

### OHLC合法性

```json
{
  "high_below_open_or_close": {
    "eligible": 6760370,
    "count": 0,
    "samples": []
  },
  "low_above_open_or_close": {
    "eligible": 6760370,
    "count": 0,
    "samples": []
  },
  "high_below_low": {
    "eligible": 6760370,
    "count": 0,
    "samples": []
  }
}
```

### 非正价格

```json
{
  "open": 0,
  "high": 0,
  "low": 0,
  "close": 0
}
```

### 其他观测形态

```json
{
  "counts": {
    "all_ohlc_nan_with_finite_vol": 40,
    "all_ohlc_nan_with_finite_amount": 40,
    "finite_ohlc_with_zero_vol_and_amount": 38468,
    "finite_ohlc_with_missing_vol_or_amount": 0
  },
  "samples": {
    "all_ohlc_nan_with_finite_vol": [
      {
        "ts_code": "001277.SZ",
        "trade_date": 20240903,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 13149352.0,
        "amount": 614645453.29,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      },
      {
        "ts_code": "001279.SZ",
        "trade_date": 20241011,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 31915473.0,
        "amount": 4708534662.04,
        "flag_limit_up": 1.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      },
      {
        "ts_code": "001389.SZ",
        "trade_date": 20240402,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 28991027.0,
        "amount": 1522338078.36,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      },
      {
        "ts_code": "001391.SZ",
        "trade_date": 20241230,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 613161688.0,
        "amount": 6210293647.37,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      },
      {
        "ts_code": "300784.SZ",
        "trade_date": 20240607,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 11708760.0,
        "amount": 1600613087.11,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      }
    ],
    "all_ohlc_nan_with_finite_amount": [
      {
        "ts_code": "001277.SZ",
        "trade_date": 20240903,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 13149352.0,
        "amount": 614645453.29,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      },
      {
        "ts_code": "001279.SZ",
        "trade_date": 20241011,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 31915473.0,
        "amount": 4708534662.04,
        "flag_limit_up": 1.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      },
      {
        "ts_code": "001389.SZ",
        "trade_date": 20240402,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 28991027.0,
        "amount": 1522338078.36,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      },
      {
        "ts_code": "001391.SZ",
        "trade_date": 20241230,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 613161688.0,
        "amount": 6210293647.37,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      },
      {
        "ts_code": "300784.SZ",
        "trade_date": 20240607,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 11708760.0,
        "amount": 1600613087.11,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": null
      }
    ],
    "finite_ohlc_with_zero_vol_and_amount": [
      {
        "ts_code": "000004.SZ",
        "trade_date": 20181008,
        "open": 86.08335702518336,
        "high": 87.8268209328068,
        "low": 85.70197153505993,
        "close": 86.62818527463327,
        "vol": 0.0,
        "amount": 0.0,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": 0.0
      },
      {
        "ts_code": "000004.SZ",
        "trade_date": 20181009,
        "open": 86.08335702518336,
        "high": 87.8268209328068,
        "low": 85.70197153505993,
        "close": 86.62818527463327,
        "vol": 0.0,
        "amount": 0.0,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": 0.0
      },
      {
        "ts_code": "000004.SZ",
        "trade_date": 20181010,
        "open": 86.08335702518336,
        "high": 87.8268209328068,
        "low": 85.70197153505993,
        "close": 86.62818527463327,
        "vol": 0.0,
        "amount": 0.0,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": 0.0
      },
      {
        "ts_code": "000004.SZ",
        "trade_date": 20181011,
        "open": 86.08335702518336,
        "high": 87.8268209328068,
        "low": 85.70197153505993,
        "close": 86.62818527463327,
        "vol": 0.0,
        "amount": 0.0,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": 0.0
      },
      {
        "ts_code": "000004.SZ",
        "trade_date": 20181012,
        "open": 86.08335702518336,
        "high": 87.8268209328068,
        "low": 85.70197153505993,
        "close": 86.62818527463327,
        "vol": 0.0,
        "amount": 0.0,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0,
        "y_ret_1d": 0.0
      }
    ],
    "finite_ohlc_with_missing_vol_or_amount": []
  }
}
```

### 成交量额：负数、零、缺失、无穷及有限范围

```json
{
  "vol": {
    "negative": 0,
    "zero": 38468,
    "nan": 1139940,
    "inf": 0,
    "finite_min": 0.0,
    "finite_max": 5616134841.0
  },
  "amount": {
    "negative": 0,
    "zero": 38468,
    "nan": 1139940,
    "inf": 0,
    "finite_min": 0.0,
    "finite_max": 90037647343.97
  }
}
```

### 训练集内部标签核验

```json
{
  "status": "full statistical comparison; no label changes",
  "label_scope_rows": 7900350,
  "atol": 1e-10,
  "rtol": 1e-08,
  "finite_labels": 6758531,
  "comparable_next_observation_pairs": 6754006,
  "matching_pairs": 6754006,
  "mismatching_pairs": 0,
  "max_absolute_error": 7.563394355258879e-16,
  "absolute_error_quantiles": {
    "0.5": 5.724587470723463e-17,
    "0.95": 2.5066754227864863e-16,
    "0.99": 3.122502256758253e-16,
    "1.0": 7.563394355258879e-16
  },
  "finite_labels_not_comparable": 4525,
  "finite_label_with_missing_current_close": 0,
  "finite_label_with_missing_next_close": 4525,
  "pairs_also_next_market_day": 6754006,
  "comparable_pairs_crossing_market_gap": 0,
  "matching_pairs_crossing_market_gap": 0,
  "nan_label_despite_finite_current_next_close": 0,
  "mismatch_samples": [],
  "uncomparable_samples": [
    {
      "ts_code": "000001.SZ",
      "trade_date": 20241231,
      "close": 2036.695331268,
      "y_ret_1d": -0.023076923076923,
      "next_close": null
    },
    {
      "ts_code": "000002.SZ",
      "trade_date": 20241231,
      "close": 1403.0856288306,
      "y_ret_1d": -0.020661157024793,
      "next_close": null
    },
    {
      "ts_code": "000004.SZ",
      "trade_date": 20241231,
      "close": 75.4046611928,
      "y_ret_1d": 0.024566473988439,
      "next_close": null
    },
    {
      "ts_code": "000006.SZ",
      "trade_date": 20241231,
      "close": 301.5953653248,
      "y_ret_1d": -0.009562841530054,
      "next_close": null
    },
    {
      "ts_code": "000007.SZ",
      "trade_date": 20241231,
      "close": 45.8892261757,
      "y_ret_1d": -0.001422475106686,
      "next_close": null
    }
  ],
  "interpretation": "If no comparable market-gap pairs exist, next observed row and next market day cannot be distinguished empirically on valid labels. Missing close values are never skipped or filled."
}
```
## test

路径：`D:\Virtual C_Drive\金融建模\stock_prediction_codex_v1\data\raw\测试集_X.csv`
文件大小：168,825,396 bytes
编码：ASCII-only; compatible with UTF-8; original producer encoding indeterminate
SHA-256：`dceffbf68bbab07a254bc6caa4d8a1e33de8de9c15489d936c5d15020dbfa90f`

| 字段 | dtype | NaN | NaN比例 | inf | inf比例 |
|---|---|---:|---:|---:|---:|
| ts_code | string | 0 | 0.00000000% | 0 | 0.00000000% |
| trade_date | int64 | 0 | 0.00000000% | 0 | 0.00000000% |
| open | float64 | 44,844 | 2.80345086% | 0 | 0.00000000% |
| high | float64 | 44,844 | 2.80345086% | 0 | 0.00000000% |
| low | float64 | 44,844 | 2.80345086% | 0 | 0.00000000% |
| close | float64 | 44,844 | 2.80345086% | 0 | 0.00000000% |
| vol | float64 | 44,831 | 2.80263816% | 0 | 0.00000000% |
| amount | float64 | 44,831 | 2.80263816% | 0 | 0.00000000% |
| flag_limit_up | float64 | 0 | 0.00000000% | 0 | 0.00000000% |
| flag_limit_down | float64 | 0 | 0.00000000% | 0 | 0.00000000% |

### 规模、唯一性、观测分布、缺行与缺失形态

```json
{
  "rows": 1599600,
  "stocks": 4650,
  "trading_dates": 344,
  "date_min": 20250102,
  "date_max": 20260608,
  "missing_key_rows": 0,
  "duplicate_extra_rows": 0,
  "duplicate_key_groups": 0,
  "duplicate_participating_rows": 0,
  "duplicate_samples": [],
  "observations_per_stock_distribution": {
    "344": 4650
  },
  "observations_per_stock_quantiles": {
    "0.0": 344.0,
    "0.25": 344.0,
    "0.5": 344.0,
    "0.75": 344.0,
    "1.0": 344.0
  },
  "market_grid_absent_rows": 0,
  "internal_market_gap_intervals": 0,
  "internal_market_missing_days": 0,
  "stocks_with_internal_market_gaps": 0,
  "max_internal_missing_market_days": 0,
  "leading_absent_market_days": 0,
  "trailing_absent_market_days": 0,
  "gap_samples": [],
  "rows_all_ohlc_nan": 44844,
  "rows_any_ohlc_nan": 44844,
  "all_ohlc_nan_before_first_finite_close": 440,
  "all_ohlc_nan_between_finite_closes": 3501,
  "all_ohlc_nan_after_last_finite_close": 7879,
  "all_ohlc_nan_in_stocks_without_finite_close": 33024,
  "stocks_without_finite_close": 96,
  "missing_price_samples": [
    {
      "ts_code": "000004.SZ",
      "trade_date": 20250429,
      "close": null
    },
    {
      "ts_code": "000004.SZ",
      "trade_date": 20260428,
      "close": null
    },
    {
      "ts_code": "000004.SZ",
      "trade_date": 20260429,
      "close": null
    },
    {
      "ts_code": "000004.SZ",
      "trade_date": 20260430,
      "close": null
    },
    {
      "ts_code": "000004.SZ",
      "trade_date": 20260506,
      "close": null
    }
  ]
}
```

### 涨跌停标志

```json
{
  "flag_limit_up": {
    "value_counts": {
      "0.0": 1573408,
      "1.0": 26192
    },
    "abnormal_count": 0
  },
  "flag_limit_down": {
    "value_counts": {
      "0.0": 1589885,
      "1.0": 9715
    },
    "abnormal_count": 0
  }
}
```

### OHLC合法性

```json
{
  "high_below_open_or_close": {
    "eligible": 1554756,
    "count": 0,
    "samples": []
  },
  "low_above_open_or_close": {
    "eligible": 1554756,
    "count": 0,
    "samples": []
  },
  "high_below_low": {
    "eligible": 1554756,
    "count": 0,
    "samples": []
  }
}
```

### 非正价格

```json
{
  "open": 0,
  "high": 0,
  "low": 0,
  "close": 0
}
```

### 其他观测形态

```json
{
  "counts": {
    "all_ohlc_nan_with_finite_vol": 13,
    "all_ohlc_nan_with_finite_amount": 13,
    "finite_ohlc_with_zero_vol_and_amount": 0,
    "finite_ohlc_with_missing_vol_or_amount": 0
  },
  "samples": {
    "all_ohlc_nan_with_finite_vol": [
      {
        "ts_code": "001356.SZ",
        "trade_date": 20250123,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 97678702.0,
        "amount": 2400688304.3,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0
      },
      {
        "ts_code": "001395.SZ",
        "trade_date": 20250127,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 14770560.0,
        "amount": 859765177.93,
        "flag_limit_up": 1.0,
        "flag_limit_down": 0.0
      },
      {
        "ts_code": "301173.SZ",
        "trade_date": 20250303,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 13517054.0,
        "amount": 1030621561.84,
        "flag_limit_up": 1.0,
        "flag_limit_down": 0.0
      },
      {
        "ts_code": "301275.SZ",
        "trade_date": 20250311,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 24383121.0,
        "amount": 1686728860.81,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0
      },
      {
        "ts_code": "301458.SZ",
        "trade_date": 20250110,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 46378010.0,
        "amount": 1812436016.83,
        "flag_limit_up": 1.0,
        "flag_limit_down": 0.0
      }
    ],
    "all_ohlc_nan_with_finite_amount": [
      {
        "ts_code": "001356.SZ",
        "trade_date": 20250123,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 97678702.0,
        "amount": 2400688304.3,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0
      },
      {
        "ts_code": "001395.SZ",
        "trade_date": 20250127,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 14770560.0,
        "amount": 859765177.93,
        "flag_limit_up": 1.0,
        "flag_limit_down": 0.0
      },
      {
        "ts_code": "301173.SZ",
        "trade_date": 20250303,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 13517054.0,
        "amount": 1030621561.84,
        "flag_limit_up": 1.0,
        "flag_limit_down": 0.0
      },
      {
        "ts_code": "301275.SZ",
        "trade_date": 20250311,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 24383121.0,
        "amount": 1686728860.81,
        "flag_limit_up": 0.0,
        "flag_limit_down": 0.0
      },
      {
        "ts_code": "301458.SZ",
        "trade_date": 20250110,
        "open": null,
        "high": null,
        "low": null,
        "close": null,
        "vol": 46378010.0,
        "amount": 1812436016.83,
        "flag_limit_up": 1.0,
        "flag_limit_down": 0.0
      }
    ],
    "finite_ohlc_with_zero_vol_and_amount": [],
    "finite_ohlc_with_missing_vol_or_amount": []
  }
}
```

### 成交量额：负数、零、缺失、无穷及有限范围

```json
{
  "vol": {
    "negative": 0,
    "zero": 0,
    "nan": 44831,
    "inf": 0,
    "finite_min": 5900.0,
    "finite_max": 6011059197.0
  },
  "amount": {
    "negative": 0,
    "zero": 0,
    "nan": 44831,
    "inf": 0,
    "finite_min": 59059.0,
    "finite_max": 58324832701.61
  }
}
```

## 训练/测试交叉核验

```json
{
  "stock_sets_equal": true,
  "train_only_stocks": [],
  "test_only_stocks": [],
  "overlapping_dates": [],
  "train_max_before_test_min": true,
  "overlapping_keys": 0,
  "train_last_date": 20241231,
  "test_first_date": 20250102,
  "label_check_with_test_boundary_close": {
    "status": "full statistical comparison; no label changes",
    "label_scope_rows": 7900350,
    "atol": 1e-10,
    "rtol": 1e-08,
    "finite_labels": 6758531,
    "comparable_next_observation_pairs": 6758531,
    "matching_pairs": 6758531,
    "mismatching_pairs": 0,
    "max_absolute_error": 7.563394355258879e-16,
    "absolute_error_quantiles": {
      "0.5": 5.724587470723463e-17,
      "0.95": 2.5066754227864863e-16,
      "0.99": 3.122502256758253e-16,
      "1.0": 7.563394355258879e-16
    },
    "finite_labels_not_comparable": 0,
    "finite_label_with_missing_current_close": 0,
    "finite_label_with_missing_next_close": 0,
    "pairs_also_next_market_day": 6758531,
    "comparable_pairs_crossing_market_gap": 0,
    "matching_pairs_crossing_market_gap": 0,
    "nan_label_despite_finite_current_next_close": 0,
    "mismatch_samples": [],
    "uncomparable_samples": [],
    "interpretation": "If no comparable market-gap pairs exist, next observed row and next market day cannot be distinguished empirically on valid labels. Missing close values are never skipped or filled."
  }
}
```

## 方法与限制

- 市场交易日取两份文件实际观测日期的并集；不能证明全市场共同缺失的日期不存在。
- 缺行按股票×观测市场日期网格比较；不补行。停牌/未上市归因不能仅由量价缺失确定。
- OHLC不等式仅在所需价格均为有限值时核验；不使用容差掩盖原始违法值。
- vol/amount异常统计包括负数、NaN、inf；零单独列示。没有赛题阈值，不把大额值自动判为错误。
- 标签容差 atol=1e-10, rtol=1e-08；只比较当前close非零且标签及相邻close有限的行。
- 跨训练/测试边界close仅用于只读数据诊断，不参与训练。没有测试标签。
- 官方评分器缺失是官方评分一致性的阻碍，不阻止独立审计或Basic40实现。
