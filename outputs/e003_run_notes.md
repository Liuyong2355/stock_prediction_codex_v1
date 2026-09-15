# E003 execution notes

Before any E003 fold completed, numerical review identified that mean subtraction can leave tiny nonzero residuals for exactly constant decimal-valued windows. Exact max==min checks were added for correlation/OLS variance; this implements the frozen zero-variance rule without changing formulas or parameters. Twelve decimal-constant regression tests pass.

The initial E003/F1 process was stopped before any result, predictions, model export or experiment-log row existed. Only its materialized scratch matrix existed. It is archived under outputs/baselines/cache/e003_interrupted_zero_variance_fix. The first Full147 build and reports are archived under outputs/features/full147_before_zero_variance_fix. They are not final results.

Full147 is regenerated from raw data and re-audited before restarting the identical frozen E003 run. No score-based model selection or parameter search occurred. All completed E002 files remain unchanged.
