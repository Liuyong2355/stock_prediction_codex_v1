# Basic40 full-data verification

- Full pytest suite: 66 passed.
- Both raw SHA-256 values and feature contract files unchanged.
- Persisted matrices: train 7,900,350 x 40; test 1,599,600 x 40; no inf.
- Original flags preserved, including 4 simultaneous-up/down train records and 0 test records.
- Full-row output is independent of missing labels.

## Independent return-extreme rechecks

The 8 min/max endpoints below were retrieved independently from the unchanged raw CSV files after generation. Observed-row offsets are applied within stock, across the train/test boundary where necessary. No value was clipped or rewritten.

| Split | Feature | Kind | Stock | Date | Lag date | Close | Lag close | Saved value | Recomputed |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| train | ret_1 | min | 301551.SZ | 20241009 | 20241008 | 80.99 | 228.01 | -0.644796280864874 | -0.644796280864874 |
| train | ret_1 | max | 301551.SZ | 20240930 | 20240927 | 181 | 50.64 | 2.57424960505529 | 2.57424960505529 |
| train | ret_60 | min | 600766.SH | 20240614 | 20240314 | 1.2713354868 | 59.266669017 | -0.978548895899054 | -0.978548895899054 |
| train | ret_60 | max | 605358.SH | 20201214 | 20200911 | 132.610000610352 | 7.07999992370605 | 17.730226276745 | 17.730226276745 |
| test | ret_1 | min | 300108.SZ | 20250417 | 20250416 | 2.8292581474 | 3.5673254902 | -0.206896551724138 | -0.206896551724138 |
| test | ret_1 | max | 302132.SZ | 20250219 | 20250218 | 393.797197993 | 64.02 | 5.15115898145892 | 5.15115898145892 |
| test | ret_60 | min | 605081.SH | 20260608 | 20260310 | 0.602485876 | 15.4537627194 | -0.961013645224172 | -0.961013645224172 |
| test | ret_60 | max | 002969.SZ | 20260212 | 20251118 | 36.3825289989 | 4.2168429837 | 7.62790697674419 | 7.62790697674419 |
