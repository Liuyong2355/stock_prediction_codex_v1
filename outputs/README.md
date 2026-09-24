# Output artifact index

This directory is organized here by research stage. Historical Phase A/B files
remain at their original top-level paths because scripts, stored fingerprints and
reproduction checks treat those paths as part of the frozen artifact contract.
New stages should use their own subdirectory, beginning with `phase_c0/`.

## Phase A0 — raw-data audit

- Human report: [data_audit_report.md](data_audit_report.md)
- Machine summary: [data_audit_summary.json](data_audit_summary.json)
- Source manifest: [data_manifest.json](data_manifest.json)
- Environment: [environment-lock.txt](environment-lock.txt)

## Phase A1 — Basic40 features

- Human report: [basic40_audit_report.md](basic40_audit_report.md)
- Machine summary: [basic40_audit_summary.json](basic40_audit_summary.json)
- Manifest: [basic40_manifest.json](basic40_manifest.json)
- Validation: [basic40_validation.md](basic40_validation.md)
- Local matrices: `features/basic40/`

## Phase A2 — E000–E002 baselines and official evaluator

- Baseline report: [baseline_report.md](baseline_report.md)
- Baseline summary: [baseline_summary.json](baseline_summary.json)
- Experiment ledger: [experiment_log.csv](experiment_log.csv)
- Official replay: [official_evaluator_comparison.md](official_evaluator_comparison.md)
- Evaluator arrival review: [evaluator_arrival_review.md](evaluator_arrival_review.md)
- Missing-label diagnosis: [e002_missing_label_diagnostic.md](e002_missing_label_diagnostic.md)
- Local fold artifacts: `baselines/E000/`, `baselines/E001/`, `baselines/E002/`

## Phase B0 — Full147 and E003

- Feature audit: [full147_audit_report.md](full147_audit_report.md)
- Feature manifest: [full147_manifest.json](full147_manifest.json)
- E002/E003 comparison: [e002_e003_comparison.md](e002_e003_comparison.md)
- Local matrices and folds: `features/full147/`, `baselines/E003/`

## Phase B1 — E004/E005 target comparison

- Results: [e003_e004_e005_comparison.md](e003_e004_e005_comparison.md)
- Full-precision results: [e003_e004_e005_comparison.json](e003_e004_e005_comparison.json)
- Validation: [e004_e005_validation.md](e004_e005_validation.md)
- Local fold artifacts: `baselines/E004/`, `baselines/E005/`

## Phase B2 — E006/E007 model comparison

- Results: [e006_e007_comparison.md](e006_e007_comparison.md)
- Full-precision results: [e006_e007_comparison.json](e006_e007_comparison.json)
- Validation: [e006_e007_validation.md](e006_e007_validation.md)
- Local fold artifacts: `baselines/E006/`, `baselines/E007/`

## Frozen evidence — Phase C0 causal turnover optimization

- Stage directory: [phase_c0](phase_c0/)
- Results: [phase_c0_turnover_comparison.md](phase_c0/phase_c0_turnover_comparison.md)
- Full-precision results: [phase_c0_turnover_comparison.json](phase_c0/phase_c0_turnover_comparison.json)
- Complete grid: [phase_c0_turnover_grid.csv](phase_c0/phase_c0_turnover_grid.csv)
- Validation: [phase_c0_turnover_validation.md](phase_c0/phase_c0_turnover_validation.md)

## Frozen evidence — Phase C0b bounded turnover extension

- Stage directory: [phase_c0b](phase_c0b/)
- Results: [phase_c0b_turnover_comparison.md](phase_c0b/phase_c0b_turnover_comparison.md)
- Full-precision results: [phase_c0b_turnover_comparison.json](phase_c0b/phase_c0b_turnover_comparison.json)
- Complete F1/F2 grid: [phase_c0b_turnover_grid.csv](phase_c0b/phase_c0b_turnover_grid.csv)
- Validation: [phase_c0b_turnover_validation.md](phase_c0b/phase_c0b_turnover_validation.md)

## Frozen evidence — Phase C1 E005/E006 daily-rank fusion

- Stage directory: [phase_c1](phase_c1/)
- Complementarity: [phase_c1_complementarity.md](phase_c1/phase_c1_complementarity.md)
- Results: [phase_c1_comparison.md](phase_c1/phase_c1_comparison.md)
- Full-precision results: [phase_c1_comparison.json](phase_c1/phase_c1_comparison.json)
- Fusion grid: [phase_c1_fusion_grid.csv](phase_c1/phase_c1_fusion_grid.csv)
- Validation: [phase_c1_validation.md](phase_c1/phase_c1_validation.md)

## Phase C2 — LambdaRank redesign

- Stage directory: [phase_c2](phase_c2/)
- E007 audit: [phase_c2_e007_audit.md](phase_c2/phase_c2_e007_audit.md)
- Results: [phase_c2_comparison.md](phase_c2/phase_c2_comparison.md)
- Full-precision results: [phase_c2_comparison.json](phase_c2/phase_c2_comparison.json)
- Validation: [phase_c2_validation.md](phase_c2/phase_c2_validation.md)

The uncommitted C2 early-stability and family-ablation/interaction branches were
retired on 2026-09-22 and preserved under
`archive/retired_research_2026-09-22/uncommitted/`.

## Phase D0 — E006 mapped core rank view

- Results: [comparison.md](phase_d0_e006_rank_view/comparison.md)
- Full precision: [comparison.json](phase_d0_e006_rank_view/comparison.json)

## Phase D1 — corrected LambdaRank mapped core rank view

- Results: [comparison.md](phase_d1_lambdarank_rank_view/comparison.md)
- Full precision: [comparison.json](phase_d1_lambdarank_rank_view/comparison.json)

## Phase D2-A — Top10-stretched continuous rank target

- Results: [comparison.md](phase_d2a_top10_stretched_rank/comparison.md)
- Full precision: [comparison.json](phase_d2a_top10_stretched_rank/comparison.json)
- Validation: [validation.md](phase_d2a_top10_stretched_rank/validation.md)
- Per-fold results: `phase_d2a_top10_stretched_rank/F1/`, `F2/`, `F3/`

## Phase D2-B — Top10 binary target

- Results: [comparison.md](phase_d2b_top10_binary/comparison.md)
- Full precision: [comparison.json](phase_d2b_top10_binary/comparison.json)
- Validation: [validation.md](phase_d2b_top10_binary/validation.md)
- Per-fold results and daily complementarity: `phase_d2b_top10_binary/F1/`,
  `F2/`, `F3/`

## Phase D3 — Recent-2Y training window

- Results: [comparison.md](phase_d3_recent_2y/comparison.md)
- Full precision: [comparison.json](phase_d3_recent_2y/comparison.json)
- Validation: [validation.md](phase_d3_recent_2y/validation.md)
- Per-fold results and daily complementarity: `phase_d3_recent_2y/F1/`, `F2/`,
  `F3/`

## Phase D4 — LightGBM Rank Regression

- Results: [comparison.md](phase_d4_lightgbm_rank_regression/comparison.md)
- Full precision: [comparison.json](phase_d4_lightgbm_rank_regression/comparison.json)
- Validation: [validation.md](phase_d4_lightgbm_rank_regression/validation.md)
- Per-fold results and daily complementarity: `phase_d4_lightgbm_rank_regression/F1/`,
  `F2/`, `F3/`

## Naming convention for future stages

Use `outputs/phase_<id>/` and keep the minimum standard set:

- `<stage>_comparison.md` for the readable conclusion;
- `<stage>_comparison.json` for full-precision provenance and metrics;
- `<stage>_grid.csv` when a parameter grid exists;
- `<stage>_validation.md` for executed checks.
# D5 residual alpha diagnostic

`phase_d5_residual_alpha/` contains the diagnostic-only analysis of frozen D0
validation predictions against all 162 frozen features. `comparison.md` is the
family report, `family_summary.csv` and `feature_residual_ic.csv` contain the
auditable aggregates, `result.json` records fold hashes and row checks, and
`validation.md` documents the no-training/no-fusion boundary.

`phase_d5_residual_alpha/partial_audit.md` corrects the interpretation of the
rank-difference IC by reporting true-return and D0-controlled partial IC.
`frozen_test_2025_2026/` contains the first local test-period evaluation of
frozen D0/D3/D4 F3 models; large row-level submissions remain local and ignored.

`frozen_test_2025_2026/legacy_strategies.md` records the fixed E005/E006 raw
and C0b test-period comparison. The canonical main model is described in
`docs/MAIN_MODEL.md`; the broader E/F comparison is in
`docs/MODEL_AUDIT_2026-09-24.md`. Completed Phase C/D directories remain here
as historical evidence so their result hashes and paths are not broken.
