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

## D027 — Stop Top10-stretched continuous rank target
**Status:** Accepted on 2026-09-22.  
**Decision:** Reject D2-A and stop this continuous target-stretch route. Relative
to the D0 Candidate, stretching only the daily-label top decile by the fixed
coefficient 10 reduces mean Rank IC from `0.120835` to `0.080186`, Top10% annual
excess from `0.477742` to `0.398918`, and Official Score from `0.240126` to
`0.201676`. All three folds regress on IC, excess and Score; F2 Score falls by
`0.074236`. The small mean turnover benefit contributes only `+0.001457` to
Score and cannot offset the IC (`-0.016260`) and excess (`-0.023647`) losses.
Missing-label Top occupancy also rises from `0.02%` to `0.72%`, while official
and diagnostic turnover move consistently, so there is no clean hidden tail
gain masked by the turnover convention. Do not tune the threshold/coefficient
or append another continuous target formula in response.
**Reason:** The fixed single-variable test fails every predeclared success check,
including the 2/3-fold excess and Score rules, F2 non-degradation, stable IC and
the `0.250` mean-Score threshold. The evidence is broad deterioration rather
than a near miss.

## D028 — Reject Top10 binary target and stop dedicated Tail-target research
**Status:** Accepted on 2026-09-22.  
**Decision:** Reject D2-B and formally stop the dedicated Tail-target route. With
the D0 feature view and all other controls fixed, an XGBoost binary classifier
for same-date finite-label `rank_pct > 0.9` produces mean Rank IC `-0.066144`,
Top10% annual excess `-0.403016`, and Official Score `-0.027615`. Top10% excess
is negative and below D0 in all three folds, including F2 delta `-0.860885`.
Although D2-B is very different from D0 (mean daily prediction Spearman
`-0.390155`, Top10 Jaccard `0.043645`, overlap `0.079474`), this is consistently
anti-predictive rather than useful complementary alpha. Mean missing-label Top
occupancy rises from `0.02%` to `15.65%`; the apparent turnover reduction is
therefore not a clean portfolio-stability gain and does not rescue the negative
finite-label excess. Do not modify the Top10 threshold, add class weights, tune
the classifier, create another classification target, or fuse D2-B with D0.
**Reason:** D2-B fails every predeclared tail-signal criterion. The three-fold
direction is consistently adverse, F2 deteriorates materially, and large signal
difference without positive realized tail return is not independent tail alpha.

## D029 — Retain Recent-2Y as a second Alpha candidate; stop window-length search
**Status:** Accepted on 2026-09-23.  
**Decision:** Do not replace D0 with D3 Recent-2Y. Shortening training history to
the two calendar years before validation reduces mean Rank IC from `0.120835`
to `0.104569`, Top10% annual excess from `0.477742` to `0.419696`, and Official
Score from `0.240126` to `0.216125`; IC, excess, and Score are below D0 in all
three folds, including F2 Score delta `-0.014736`. However, D3 remains a valid
second Alpha candidate because every fold retains positive IC, excess, and
Score, while mean daily prediction Spearman versus D0 is `0.786118` and mean
Top10 overlap is only `0.489518` (`0.334588` Jaccard). Missing-label Top
occupancy remains negligible at `0.03%`, and official/diagnostic turnover show
no anomalous advantage. Stop all further window-length experiments (including
1Y, 3Y, 4Y, 18-month, grids, and time decay). Preserve D3 only for a separately
authorized future complementarity/fusion decision; do not fuse in D3.
**Reason:** Recent history alone is consistently weaker than expanding history,
so there is no evidence for replacing the main model or tuning the window. Its
stable standalone alpha and materially different Top10 selections nevertheless
satisfy the predeclared second-Alpha route without relying on missing labels or
turnover artifacts.

## D030 — Reject plain LightGBM regression as a clean second-model candidate
**Status:** Accepted on 2026-09-23.  
**Decision:** Keep D0 XGBoost as the main model and stop ordinary LightGBM rank
regression without tuning. D4 produces stable positive alpha: mean Rank IC
`0.120923`, Top10% excess `0.450875`, and Score `0.236052`. Score exceeds D0
only in F2 and trails by `0.004074` on average. Missing-label Top occupancy
averages `7.49%` and reaches `19.53%` in F2, so its lower official turnover and
part of its selection difference are not clean. Prediction Spearman versus D0
averages `0.787081`, Top10 Jaccard `0.552922`, overlap `0.701200`, and each
model's exclusive Top10 fraction `29.88%`. Do not promote D4, tune it, fuse it,
or prioritize a third generic model-family experiment.
**Reason:** Frozen LightGBM preserves broad ranking alpha, but it is neither a
stronger main model nor a clean differentiated portfolio signal. The
missing-label anomaly violates the predeclared second-model criterion.

## D031 — D5 residual diagnostic requires a mechanical-effect correction
**Status:** Corrected on 2026-09-24 after a conditional audit.  
**Decision:** D5 correctly computed `rank(y_ret_1d)-rank(D0 prediction)` from the
frozen OOF predictions and trained no model. Its large positive volatility and
volume-price IC cannot be interpreted as independent positive alpha: subtracting
D0 prediction rank mechanically rewards any feature negatively correlated with
D0. A follow-up read-only audit measured direct true-return IC and same-date
partial rank correlation with true return controlling D0 prediction. Volatility
partial IC is `-0.023417/-0.030414/-0.014442` in F1/F2/F3 for the full section;
volume-price is `-0.011025/-0.018337/-0.014619`. Middle60 is also consistently
negative, while Top20 is weak or unstable. These are small possible inverse
correction signals, not evidence that a residual model will improve Score.
Suspend the proposed residual-model training until a leakage-safe training design
and validation criterion are frozen; do not train a residual target on in-sample
D0 predictions or use the newly visible test labels to choose a correction.
**Reason:** Direct and conditional IC reverse the interpretation of the original
positive rank-difference IC. The corrected finding is narrower and does not
justify immediate second-model promotion.

## D032 — First local test-period evaluation of frozen F3 models
**Status:** Accepted on 2026-09-24.  
**Decision:** Score the previously frozen D0, D3 and D4 F3 models once on the
locally reconstructed 2025-01-02 to 2026-06-05 test labels (343 dates,
1,594,950 rows). Use the unchanged archived organizer evaluator, not the
evaluation-folder copy with an optional logger-import modification. No model is
retrained and all predictions are saved before labels are read for scoring.
Official Scores are D0 `0.238108`, D3 `0.208391`, D4 `0.232123`; Rank IC is
`0.129418/0.110341/0.127397`, and annual Top10 excess is
`0.461515/0.379274/0.436533`, respectively. D0 remains strongest of the three.
The test labels are retrospective returns reconstructed from the next test row's
close and are reserved as a one-time holdout observation, not a tuning fold.
**Reason:** This checks temporal generalization of existing artifacts on one
later period without reusing its labels for model development.

## D033 — Distinguish the primary Alpha model from high-Score strategies
**Status:** Accepted on 2026-09-24.  
**Decision:** Name the unchanged D0 F3 artifact `main_alpha_xgb_rank162` and
retain it as the primary raw Alpha model. This does not claim that D0 has the
highest Official Score of all historical strategies. F1/F2/F3 are validation
folds, not another model family. In those folds, E005+C0b Score is `0.371482`,
E006+C0b is `0.358599`, and corrected C2-R1 with postprocess is `0.399444`,
all above D0 raw `0.240126`. E005 and C2 scores have severe missing-label Top
occupancy (about `39%` and `56%`), whereas E006+C0b is about `0.02%`. A fixed,
one-time comparison on reconstructed 2025–2026 labels gives E005+C0b `0.369300`
and E006+C0b `0.366424`, versus D0 raw `0.238108`; E005's tiny test lead does
not erase its earlier missing-label failure mode. Preserve E006+C0b as a frozen
low-missing high-Score reference, but do not relabel it as the 162-feature main
model or reopen turnover tuning. See `docs/MODEL_AUDIT_2026-09-24.md`.
**Reason:** Model signal strength and full-strategy Score answer different
questions. The new naming must not hide better recorded E-series Scores or
mistake missing-label-driven turnover gains for clean Alpha.

## D034 — Archive completed work and identify obsolete scratch
**Status:** Archive accepted on 2026-09-24; physical deletion pending.  
**Decision:** Keep the canonical F3 model binary in Git with a hash-checked
identity contract. Preserve stopped-route reports at their original paths and
index them under `archive/completed_research_2026-09-24/`; replace the active
README/handoff with concise current guidance while archiving prior versions.
The unfinished E003 training scratch and obsolete Full147 `.npy` copies from
before the zero-variance fix are verified cleanup targets, while their small
audit records remain protected. The automated deletion was rejected by the
execution policy, so the three `.npy` files still exist and require manual
cleanup. Keep current raw data, current feature matrices, labels, validation
artifacts and test submissions locally. Exclude the user-supplied evaluation
X/Y and transfer archive from Git.
**Reason:** The identified three files occupy about 14.3 GiB and are
superseded, but repository documentation must not claim a cleanup that did
not occur.

## D035 — Fixed C0b transfer to D0 improves Score but does not justify promotion
**Status:** Accepted on 2026-09-24.  
**Decision:** Apply only the previously frozen C0b policy (causal daily-rank
EMA alpha `0.30`, Top10 hysteresis exit fraction `0.25`) to saved D0
predictions, with no model fitting or parameter search. Raw D0 replay exactly
matched its three official fold scores. D0+C0b mean Rank IC is `0.099770`,
Top10 annual excess `0.278879`, official turnover `0.210498`, diagnostic
turnover `0.258329`, missing-label Top `0.02%`, and Official Score `0.360422`.
Score improves over D0 raw by `0.120296`, mainly from turnover, but exceeds
the already frozen low-missing E006+C0b reference (`0.358599`) by only
`0.001823`, well below the predeclared `0.01` practical margin. F2 Top10
excess is weak (`0.169931`). A fixed retrospective check on the already-viewed
2025–2026 local labels scores `0.368088` versus E006+C0b `0.366424`, likewise
too small to change the decision. Preserve D0 raw as primary Alpha and E006+C0b
as high-Score reference. Record D0+C0b as a completed transfer check; do not
promote it or reopen turnover tuning. See
`outputs/phase_d0_c0b_fixed/comparison.md` and `retrospective_test.md`.
**Reason:** Official and diagnostic turnover both fall, while missing-label
Top remains negligible, so the Score gain is real under the evaluator.
However, IC and Top10 excess fall materially, the incremental advantage over
the existing clean high-Score strategy is tiny, and F1–F3 plus 2025–2026 are
already-exposed research periods rather than fresh confirmation.

## D036 — Enforce competition-only data and close the current validation gate
**Status:** Accepted on 2026-09-24.  
**Decision:** The organizer PDF section 6(1) explicitly restricts feature
engineering to fields in the supplied training/test datasets and prohibits
future information; section 6(2) reserves test Y for post-competition scoring.
Therefore do not use announcements, analyst forecasts, or other external data
as competition features. Correct the earlier exploratory suggestion to pursue
such sources. The supplied test X ends `2026-06-08`; the local retrospectively
reconstructed Y ends `2026-06-05` and has already been viewed. There is no
unseen later labeled panel in the workspace. Do not use this reconstructed Y
for another competition model or parameter choice. Freeze D035 as completed
evidence, without promoting D0+C0b. If a fixed competition submission is the
goal, define a separate deployment/refit protocol using only eligible original
training labels and supplied X; if prospective live research is the goal,
acquire genuinely later data and keep that evaluation separate from official
competition-Score claims. See `docs/NEXT_VALIDATION_GATE.md`.
**Reason:** The organizer's explicit data boundary overrides the prior plan's
external-data branch. A future live period cannot serve as an independent
official Score for the already fixed competition test dates, and the local
reconstructed test labels are no longer a blind holdout.
