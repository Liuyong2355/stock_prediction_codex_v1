# Research Decisions

This file records frozen design decisions. Do not rewrite history; append new decisions when the plan changes.

## D001 — Use an agent-first repository contract
**Status:** Accepted  
**Decision:** `AGENTS.md` is the entry point; detailed rules live in versioned Markdown and YAML.  
**Reason:** Codex should receive a compact map plus machine-readable implementation contracts, not a monolithic document.

## D002 — Organizer evaluator has highest metric precedence
**Status:** Accepted  
**Decision:** The organizer-provided `evaluate.py`, once obtained, is the source of truth for metric implementation details.  
**Reason:** The PDF does not fully specify ties, group sizes and several edge cases.

## D003 — Use time-based expanding-window validation
**Status:** Accepted  
**Decision:** V1 uses F1=2022, F2=2023, F3=2024 validation with expanding historical training. Random split is prohibited for official research runs.  
**Reason:** The held-out competition period is later in time, so validation must emulate forward prediction.

## D004 — Purge one final training trading date
**Status:** Accepted  
**Decision:** Remove the final distinct training date before each fold fit.  
**Reason:** `y_ret_1d(t)` consumes `close(t+1)` and would otherwise cross the train/validation boundary.

## D005 — `ts_code` is not a V1 model feature
**Status:** Accepted  
**Decision:** Keep `ts_code` as an identifier/grouping key only.  
**Reason:** V1 prioritizes transferable state signals and avoids identity memorization. Revisit only through a controlled later experiment.

## D006 — Raw absolute OHLC values are not V1 model features
**Status:** Accepted  
**Decision:** Use relative/normalized transformations instead of raw price levels.  
**Reason:** Absolute price level is not naturally comparable across stocks and time; V1 focuses on scale-free signals.

## D007 — Implement Basic40 before Full147
**Status:** Accepted  
**Decision:** Baseline correctness must be established on Basic40 and E000-E002 before Full147 is implemented.  
**Reason:** This isolates pipeline bugs from feature-complexity bugs.

## D008 — Full147 combines time-series and same-day cross-sectional state
**Status:** Accepted  
**Decision:** Same-day market aggregates, relative features and percentile ranks are allowed.  
**Reason:** At end of date t, all date-t source fields are observable while the target is t+1 return.

## D009 — Compare raw, rank and market-relative training targets
**Status:** Accepted  
**Decision:** V1 includes separate target experiments instead of assuming raw-return regression is optimal.  
**Reason:** The official evaluation is heavily rank-oriented while Top1 excess return also depends on tail selection.

## D010 — Final score, not RMSE, governs research decisions
**Status:** Accepted  
**Decision:** Out-of-time official score and stability across folds are primary.  
**Reason:** The organizer score explicitly combines Rank IC, Top excess return and turnover.

## D011 — Feature-family evidence requires ablation and cross-fold stability
**Status:** Accepted  
**Decision:** LightGBM feature importance alone cannot justify keeping or removing a family.  
**Reason:** Financial predictors are correlated and unstable; out-of-time contribution is the relevant evidence.

## D012 — No hyperparameter search before pipeline acceptance
**Status:** Accepted  
**Decision:** E000-E002 use frozen baseline settings.  
**Reason:** Tuning an unverified pipeline can optimize bugs or leakage.

## D013 — Freeze observed-row and finite-value semantics
**Status:** Accepted, explicitly authorized by user on 2026-09-15.
**Version:** Feature/protocol V1.0.1; YAML structural schema remains 1.0.
**Decision:** Per-stock lag/rolling count ascending observed rows. Ordinary rolling requires the full window of finite values; correlation requires w finite pairs; EMA min_periods=window. Cross-sectional statistics use finite same-date values only; breadth denominator counts finite ret_1. Zero-variance dependent-variable windows yield NaN OLS R-squared. Calendar row insertion, forward fill and backfill are prohibited.
**Reason:** Resolve implementation ambiguity without inferring missing-day semantics or modifying raw observations.

## D014 — Freeze evaluation and training edge cases
**Status:** Accepted, explicitly authorized by user on 2026-09-15.
**Decision:** IC standard deviation uses ddof=0. E007 uses average percentile rank among finite daily labels and min(floor(rank_pct*10),9); tied labels share relevance. Ridge drops all-NaN supervised-training columns per fold and logs names; median/scaler fit only training data. Every internal evaluator remains provisional while official evaluate.py is missing.
**Reason:** Deterministic implementations; no silent changes to frozen folds, feature membership, baseline parameters or experiment order.

## D015 — Initialize audit-only engineering stage
**Status:** Accepted, explicitly authorized by user on 2026-09-15.
**Decision:** Initialize Git and Python src/tests/outputs structure; archive unchanged CSV bytes from data/ into data/raw/ with before/after SHA-256 provenance. Perform full read-only audit and minimal tests only. No E000-E007, model training, Full147 implementation, label replacement or automatic specification repair based on audit findings.

## D016 — Implement Basic40 without changing frozen definitions
**Status:** Accepted, explicitly authorized in the user's Basic40 B1 request.
**Decision:** Implement only the frozen 40 features, causal train/test generation, tests and descriptive feature audit. Preserve V1.0.1 definitions and all rows/flags. The user's B1 name denotes the Basic40 implementation step, not authorization for the protocol's Full147 phase. No E000-E007, model training or tuning.
**Implementation:** Store float64 NumPy matrices and aligned keys locally, track metadata/fingerprints and reports in Git, and stop after the reviewable commit. Basic40 has no EMA or cross-sectional feature; verify same-day stock independence without adding Full147 functionality.

## D017 — B2 frozen baseline execution with provisional scoring
**Status:** Authorized in the user's B2 request; official scoring edge cases remain unresolved.
**Decision:** Implement frozen expanding folds/purge and run only E000–E002. E001 uses official sklearn Ridge and train-only sklearn preprocessing; E002 uses official lightgbm with the existing frozen params. No custom training algorithms, Full147, alternative targets, tuning or ensembles.
**Provisional implementation:** Record PDF edge conventions separately in PROVISIONAL_EVALUATOR.md and each result. This does not amend the competition rules or claim organizer parity. Preserve validation rows and label-independent turnover eligibility. Stop after results, tests and Git submission for review.

## D018 — Late official evaluator receipt and preserved B2 results (2026-09-15)

User confirmed the root evaluate.py is the unmodified organizer script. Archive identical bytes at reference/evaluate_official.py; replay all nine saved validation predictions and deterministic edge cases without retraining or replacing provisional results. Official ddof=1 differs from user-frozen ddof=0; minimum sample thresholds, Top group rounding, ties and NaN handling also differ. Comparison is recorded separately. No frozen YAML, feature, fold, target or metric definition is changed in this delivery; resolve conflicts explicitly during review before changing versions.

## D019 — Authorized Full147 and E003 only

The user authorized the frozen Full147 implementation and E003 (Full147 + official LightGBM + raw target), with unchanged folds, purge, parameters and predictions. Use the unmodified organizer script for primary validation metrics and compare against the existing E002 official replay. Preserve missing-label Top effects and report ex-post label-filtered turnover only as a diagnostic. Official IC standard deviation (ddof=1) and the user-frozen auxiliary ddof=0 are stored separately; no frozen configuration or formula is amended. The official script does not return Top-Bottom spread; the separately labeled auxiliary statistic uses floor-sized end groups and does not enter Score. No E004–E007, tuning, ensembles or feature ablations are authorized in this stage.

## D020 — Authorized E004/E005 target comparison only

The user authorized continuation from eb9660a for E004 (rank) and E005 (relative),
holding every E003 condition except the frozen target constant. Reuse the existing
Full147 artifacts and shared E003 training/evaluation path. Construct targets only
after the original fold purge and finite-label selection. Keep raw validation
labels and all validation predictions, with primary scores returned directly by
the unmodified organizer evaluator. Retain missing-label diagnostic turnover and
Top missing share. Verify independent target construction, training-only scope,
official replay, model prediction reproduction and protected artifact hashes.
No frozen YAML or feature/target formula changes; no E006/E007, tuning, ensembles
or turnover postprocessing. Commit the code, reports and tests, then await review.

## D021 — Authorized E006/E007 frozen model comparison only

Continue from 24e979d using existing Full147, folds, purge and all frozen YAML.
E006 uses official XGBoost regression with the exact E004 rank labels and row order.
E007 uses official LightGBM Ranker with finite daily average-rank decile relevance
and stable contiguous date groups after purge. XGBoost 3.4.1 is a new pinned library
dependency selected before experiment scores; existing dependencies stay unchanged.
Use the unmodified organizer script for every primary metric. Preserve all previous
artifacts and report missing-label diagnostics independently. Compare E000–E007 and
recommend Phase C candidates, but do not execute tuning, feature ablation, ensembles
or turnover postprocessing. Commit results, code and tests, then await review.

Observed frozen results: E006 mean official Score 0.233857, above E004 in all three
folds; E007 mean Score 0.027029, with negative IC and excess return in all three
folds. The comparison report recommends E002/E005/E006 as Phase C candidates and
retains the other useful baselines as controls. These are recommendations for
review; no Phase C work is executed. See `outputs/e006_e007_comparison.md` and
`outputs/e006_e007_validation.md` for evidence and reproduction limits.

## D019 — Phase C0 uses prediction-only causal turnover control
**Status:** Accepted for the C0 experiment.  
**Decision:** Keep E005/E006 models and saved fold predictions frozen. Within each
validation fold, apply same-date average percentile rank, per-stock causal EMA
initialized from that fold's first observation, and exact-size Top hysteresis.
Select alpha/exit_fraction independently per model using F1/F2 only, in the
predeclared score/worst-score/turnover/alpha/exit order; use F3 only once for
confirmation after parameters are frozen.  
**Reason:** Isolate whether causal prediction post-processing can improve the
official turnover/Score trade-off without retraining or using confirmation data
for parameter choice.

## D020 — C0b performs one bounded extension and then stops turnover search
**Status:** Accepted for the C0b experiment.  
**Decision:** Reuse the verified C0 implementation and expand once to alpha
`[0.50, 0.40, 0.30]` and exit fraction `[0.15, 0.175, 0.20, 0.225, 0.25]`.
Select independently for E005/E006 using F1/F2 only. Because F3 was observed in
C0, use it only as a post-selection robustness/temporal-consistency check. Stop
after this grid even if a selected parameter remains on the boundary.  
**Reason:** Test whether C0's active boundaries hid a nearby stable region while
limiting repeated adaptation to the same validation years.

## D021 — C1 tests only raw daily-rank fusion under locked postprocessing
**Status:** Accepted for the C1 experiment.  
**Decision:** Blend same-date percentile ranks of saved E005/E006 raw predictions
at five predeclared E005 weights. Apply the locked C0b postprocess exactly once
after blending (`alpha=0.30`, `exit_fraction=0.25`). Select weight using F1/F2
only; use F3 only for post-selection robustness.  
**Reason:** Measure whether the two frozen models contain complementary ordering
signal without reopening model training or turnover parameter search.

## D022 — C2 redesigns LambdaRank with an explicit bounded protocol
**Status:** Accepted for the C2 experiment.  
**Decision:** Compare exactly two Full147 LambdaRank candidates using linear
0–9 gains, truncation 500, normalization, 100 rounds and corrected average-rank
deciles. R1 uses raw features; R2 ranks every non-market feature cross-sectionally
using feature metadata while preserving date-level market features. Select on
F1/F2 and use F3 only for robustness. Apply the locked C0b postprocess unchanged.
**Reason:** Isolate E007 protocol design from input representation without broad
tuning, feature selection, fusion or renewed turnover search.

## D023 — Validate C2 representation stability on an earlier 2020 fold
**Status:** Accepted for the early-stability experiment.  
**Decision:** Train only on 2018–2019 with the existing one-trading-day purge and
validate raw R1/R2 predictions on 2020. Reuse the complete C2 LambdaRank protocol
unchanged; fixed C0b postprocessing is secondary only. Do not use 2021, which is
reserved for a later one-time confirmation.
**Reason:** Test whether the all-non-market cross-sectional rank representation
generalizes to an earlier regime without feature research, tuning, or fusion.

## D024 — First R1 family ablation uses only three metadata categories
**Status:** Accepted for the first family-ablation round.  
**Decision:** Compare Full147 with removal of exactly one of `cross_section_rank`,
`relative`, or `market`, using frozen C2-R1 and raw predictions on 2019 and 2020.
A family is retained when deletion lowers Score in both years, becomes a deletion
candidate when deletion raises Score in both, and otherwise remains as unstable.
**Reason:** Attribute broad cross-sectional representation value without single-
feature selection, tuning, additional families, or use of reserved 2021 data.

## D025 — Retire completed low-information-gain routes
**Status:** Accepted on 2026-09-22.  
**Decision:** Freeze historical E007, C0/C0b turnover search, C1 E005/E006 fusion,
C2-R2 replacement view, early-stability checks and the completed C2 family
ablation/interaction branch. Preserve their results as evidence, but remove their
configs and dedicated tests from the default workflow. Archive all uncommitted
research files before removal. Keep corrected LambdaRank only as a conditional
second experiment after the E006 feature-view test succeeds.
**Reason:** Concentrate research capacity on structural alpha improvements and
avoid reopening exhausted grids, weak ensembles or low-upside follow-up tests.

## D026 — Accept mapped core cross-sectional ranks as a structural representation gain
**Status:** Accepted on 2026-09-22.  
**Decision:** Add the 15 previously missing same-date percentile-rank views from
the explicit teammate-core mapping while retaining all Full147 raw features. In
E006, mean Rank IC, annual excess and Official Score improve by `0.002408`,
`0.020199` and `0.006269`, respectively, with all three folds improving and mean
missing-label Top fraction at `0.02%`. The same view in corrected LambdaRank
improves Rank IC by `0.002273` in all three folds, but excess and Score improve
in only one fold. Treat this as a robust ordering representation gain, not as
stable proof of a LambdaRank tail-return or final-score gain. Do not expand the
rank allowlist or add RSI/MACD/ATR/Bollinger in response to these results.
**Reason:** The single-variable test confirms the teammate feature-view idea on
clean E006 and independently on LambdaRank IC, while the mixed LambdaRank tail
result supplies a clear stopping boundary against further opportunistic search.
