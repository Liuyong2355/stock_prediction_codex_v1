# D5 Residual Alpha Diagnostic

**Interpretation correction (2026-09-24):** The positive IC against `rank(y)-rank(D0)` is substantially affected by subtracting D0 rank. It does not establish positive independent alpha. The [conditional audit](partial_audit.md) finds small, consistently negative true-return partial IC for volatility and volume-price in the full and middle sections; Top20 is weak and unstable. Do not use the earlier positive residual IC as a standalone second-model success criterion.

Only frozen D0 validation predictions were used; no model was trained and no fusion was performed.

Residual is the same-date finite-label average percentile rank of realized next-day return minus the corresponding rank of the D0 prediction.

| Region | Family | F1 | F2 | F3 | Mean residual IC | Median residual IC | Mean abs(residual IC) | Mean abs(D0 IC) | Stable meaningful | Stable features |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full | volatility | 0.154963 | 0.213157 | 0.154361 | 0.174160 | 0.181213 | 0.174160 | 0.291656 | 20/20 | gkvol_5, parkinson_5, gkvol_10, atr_5, parkinson_10 |
| full | volume_price | 0.156477 | 0.113613 | 0.151944 | 0.140678 | 0.141588 | 0.140678 | 0.231145 | 4/4 | corr_ret_dlogamt_20, corr_ret_dlogamt_5, corr_ret_dlogvol_5, corr_ret_dlogvol_20 |
| full | kbar | 0.092607 | 0.119338 | 0.096620 | 0.102855 | 0.138782 | 0.132256 | 0.207419 | 12/12 | klen, klow, ksft2, ksft, kmid |
| full | amount | 0.127652 | 0.116220 | 0.126057 | 0.123310 | 0.127609 | 0.123310 | 0.205491 | 11/11 | amtratio_60, logamt_std_20, logamt_std_5, amtchg_20, amtratio_20 |
| full | relative | 0.116429 | 0.087052 | 0.113414 | 0.105631 | 0.131768 | 0.123147 | 0.203861 | 8/8 | relstd_20, relret_1, relret_5, relamtratio_20, relvolratio_20 |
| full | cross_section_rank | 0.118682 | 0.086468 | 0.120023 | 0.108391 | 0.122496 | 0.119791 | 0.200111 | 26/31 | klen__csr_d0, klow__csr_d0, csr_stdret_20, stdret_60__csr_d0, volratio_60__csr_d0 |
| full | volume | 0.124192 | 0.117195 | 0.117395 | 0.119594 | 0.122492 | 0.119594 | 0.198901 | 11/11 | volratio_60, logvol_std_20, volchg_20, logvol_std_5, volratio_20 |
| full | return | 0.098207 | 0.058647 | 0.101342 | 0.086065 | 0.101889 | 0.097668 | 0.160204 | 8/10 | ret_1, ret_2, ret_5, ret_3, ret_20 |
| full | trend | 0.103856 | 0.033523 | 0.114267 | 0.083882 | 0.087974 | 0.091616 | 0.153070 | 20/31 | distmin_5, distmin_20, distmin_60, emadev_5, madev_3 |
| full | liquidity | -0.081561 | -0.031493 | -0.132268 | -0.081774 | -0.082072 | 0.081774 | 0.140930 | 4/4 | illiq_1, illiq_5, illiq_20, illiq_60 |
| full | limit | 0.024997 | 0.016828 | 0.022717 | 0.021514 | 0.041154 | 0.062783 | 0.115351 | 10/10 | days_since_limit_up, limit_up_count_60, limit_up_count_20, limit_down_count_60, days_since_limit_down |
| full | market | nan | nan | nan | nan | nan | nan | nan | 0/10 | — |
| middle60 | volatility | 0.064030 | 0.085228 | 0.059772 | 0.069677 | 0.074360 | 0.069677 | 0.184465 | 19/20 | gkvol_5, gkvol_10, parkinson_5, gkvol_20, parkinson_10 |
| middle60 | kbar | 0.036703 | 0.047203 | 0.049728 | 0.044545 | 0.062443 | 0.054211 | 0.122331 | 12/12 | ksft, ksft2, klen, klow, kmid |
| middle60 | volume_price | 0.058882 | 0.032911 | 0.052701 | 0.048165 | 0.048645 | 0.048165 | 0.125205 | 4/4 | corr_ret_dlogamt_5, corr_ret_dlogvol_5, corr_ret_dlogamt_20, corr_ret_dlogvol_20 |
| middle60 | relative | 0.044532 | 0.030439 | 0.047197 | 0.040723 | 0.038903 | 0.045976 | 0.115665 | 6/8 | relstd_20, relret_1, relret_5, relamtratio_20, relvolratio_20 |
| middle60 | cross_section_rank | 0.044354 | 0.026482 | 0.048955 | 0.039930 | 0.039669 | 0.043638 | 0.111348 | 21/31 | klen__csr_d0, klow__csr_d0, csr_stdret_20, kmid__csr_d0, stdret_60__csr_d0 |
| middle60 | amount | 0.044250 | 0.037004 | 0.038784 | 0.040013 | 0.039413 | 0.040013 | 0.094970 | 11/11 | logamt_std_20, logamt_std_5, amtratio_60, amtchg_20, logamt_std_60 |
| middle60 | volume | 0.042694 | 0.037723 | 0.034905 | 0.038441 | 0.038394 | 0.038441 | 0.089985 | 11/11 | logvol_std_20, volratio_60, logvol_std_5, volchg_20, logvol_std_60 |
| middle60 | return | 0.036830 | 0.016329 | 0.048371 | 0.033843 | 0.035142 | 0.038035 | 0.093185 | 6/10 | ret_1, ret_2, ret_5, ret_3, ret_60 |
| middle60 | trend | 0.040328 | 0.003553 | 0.054014 | 0.032631 | 0.034886 | 0.035072 | 0.087904 | 13/31 | distmin_5, distmin_20, madev_3, distmin_60, emadev_5 |
| middle60 | liquidity | -0.029050 | 0.013251 | -0.057585 | -0.024462 | -0.024492 | 0.024462 | 0.095642 | 0/4 | — |
| middle60 | limit | 0.009813 | 0.006268 | 0.008152 | 0.008078 | 0.011168 | 0.020984 | 0.052072 | 7/10 | days_since_limit_up, limit_up_count_60, limit_up_count_20, limit_down_count_60, days_since_limit_down |
| middle60 | market | nan | nan | nan | nan | nan | nan | nan | 0/10 | — |
| top20 | limit | 0.015217 | 0.005332 | 0.018101 | 0.012883 | 0.014570 | 0.021076 | 0.137451 | 7/10 | flag_limit_up, limit_up_count_5, limit_up_count_20, limit_down_count_20, limit_down_count_60 |
| top20 | kbar | -0.001414 | 0.005899 | 0.013777 | 0.006087 | 0.009243 | 0.020453 | 0.079300 | 9/12 | ksft, kup, ksft2, kup2, kmid |
| top20 | relative | 0.008910 | -0.003563 | 0.012451 | 0.005933 | 0.003538 | 0.013940 | 0.070712 | 4/8 | relret_1, relret_5, relvolratio_20, relamtratio_20 |
| top20 | return | 0.013822 | 0.005075 | 0.022521 | 0.013806 | 0.006688 | 0.013806 | 0.057488 | 4/10 | ret_1, ret_2, ret_3, ret_5 |
| top20 | volume | -0.001174 | -0.012766 | -0.007299 | -0.007080 | -0.012791 | 0.012887 | 0.054151 | 6/11 | volratio_60, volratio_20, logvol_std_60, volchg_20, volratio_5 |
| top20 | cross_section_rank | 0.005147 | -0.007962 | 0.011263 | 0.002816 | 0.001062 | 0.012447 | 0.068430 | 12/31 | kup__csr_d0, csr_ret_1, csr_madev_5, ret_2__csr_d0, csr_ret_3 |
| top20 | liquidity | 0.016579 | 0.029700 | -0.011689 | 0.011530 | 0.011885 | 0.011530 | 0.078572 | 0/4 | — |
| top20 | trend | 0.012179 | -0.003763 | 0.021538 | 0.009985 | 0.005005 | 0.011355 | 0.063857 | 8/31 | distmax_5, madev_3, madev_5, emadev_5, rsv_5 |
| top20 | amount | -0.000016 | -0.012382 | -0.005612 | -0.006004 | -0.010406 | 0.011128 | 0.059095 | 4/11 | amtratio_60, amtratio_20, amtchg_20, amtratio_10 |
| top20 | volatility | -0.016740 | -0.026661 | 0.022096 | -0.007102 | -0.006197 | 0.007208 | 0.074704 | 1/20 | downvol_5 |
| top20 | volume_price | 0.007265 | -0.002991 | 0.003497 | 0.002590 | 0.002522 | 0.002590 | 0.009468 | 0/4 | — |
| top20 | market | nan | nan | nan | nan | nan | nan | nan | 0/10 | — |

A feature is marked stable/meaningful only when all three fold ICs have the same sign and at least two folds have absolute IC >= 0.01.
Date-level market features are constant within a date and therefore intentionally remain undefined in a same-date cross-sectional diagnostic.
