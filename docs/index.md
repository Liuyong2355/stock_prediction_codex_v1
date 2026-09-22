# Documentation Index

This repository is designed for Codex-driven implementation. `AGENTS.md` is the entry point; this file is the map.

## Source documents

- `../reference/赛题五-更新.pdf` — organizer task statement.
- `../reference/README.md` — status of organizer evaluation code.

## Specifications

- `COMPETITION_SPEC.md` — task, schema, submission and metrics reproduced from the organizer PDF.
- `FEATURE_SPEC_V1.md` — human-readable Feature Set V1 definitions and implementation invariants.
- `EXPERIMENT_PROTOCOL_V1.md` — validation, targets, baseline models, experiments and logging rules.
- `DECISIONS.md` — frozen design decisions and rationale.

## Machine-readable contracts

- `../config/features_v1.yaml` — canonical feature names, formulas, metadata and Basic40/Full147 membership.
- `../config/folds_v1.yaml` — canonical expanding-window folds and purge rule.
- `../config/experiments_v1.yaml` — canonical targets, baseline model settings, E000-E007 and log schema.

## What to read for a task

### Implement or modify features
Read `FEATURE_SPEC_V1.md` and `../config/features_v1.yaml`.

### Implement data splitting
Read `EXPERIMENT_PROTOCOL_V1.md` and `../config/folds_v1.yaml`.

### Implement models or run experiments
Read `EXPERIMENT_PROTOCOL_V1.md` and `../config/experiments_v1.yaml`.

### Implement metrics
Read `COMPETITION_SPEC.md`, the organizer PDF, and organizer `evaluate.py` when it becomes available.

### Change a frozen design
Read `DECISIONS.md`, record the new decision, update the canonical YAML/spec, and bump versions if material.

## Audit outputs

- `../outputs/data_audit_report.md` — full raw-panel audit with facts/warnings/questions/blockers.
- `../outputs/data_audit_summary.json` — machine-readable full audit.
- `../outputs/data_manifest.json` — raw-file migration provenance and stable SHA-256.
- `../README.md` — environment and audit commands.

## Basic40 implementation outputs

- `BASIC40_IMPLEMENTATION.md` — causal implementation, scope, storage and validation.
- `../outputs/basic40_audit_report.md` — full feature distributions and manual-review examples.
- `../outputs/basic40_audit_summary.json` — machine-readable feature audit.
- `../outputs/basic40_manifest.json` — local feature/key artifacts and fingerprints.
- `../outputs/basic40_test_results.txt` — pytest results for the Basic40 stage.
- `../outputs/basic40_validation.md` — full-data checks and independent return-extreme rechecks.

## B2 baseline implementation and results

- `PROVISIONAL_EVALUATOR.md` — preserved rules and explicitly provisional unresolved edges.
- `../outputs/baseline_report.md` — E000/E001/E002 scores, fold counts and limitations.
- `../outputs/baseline_summary.json` — complete per-fold metrics and provenance.
- `../outputs/experiment_log.csv` — required experiment log.
- `../outputs/baselines/` — fold result.json, daily metrics, local prediction/model files and fingerprints.

- `../outputs/official_evaluator_comparison.md` — original organizer script replay; preserved provisional comparison.
- `../outputs/evaluator_arrival_review.md` — confirmed provenance and semantic differences.

## Full147 / E003

- `FULL147_IMPLEMENTATION.md` — frozen implementation and reproducible commands.
- `../outputs/full147_audit_report.md` — full train/test feature audit.
- `../outputs/full147_manifest.json` — column order, local files and fingerprints.
- `../outputs/e002_e003_comparison.md` — paired official metrics and missing-label diagnostics.
- `../outputs/baselines/E003/` — fold predictions, model, official metrics and diagnostics.

## E004 / E005 target comparison

- `TARGET_COMPARISON.md` — training-only targets, strict E003 controls and reproduction commands.
- `../outputs/e003_e004_e005_comparison.md` — three-fold official results, stability and Score decomposition.
- `../outputs/e003_e004_e005_comparison.json` — full-precision results and independent reproduction checks.
- `../outputs/e004_e005_validation.md` — executed tests, provenance and validation limits.
- `../outputs/baselines/E004/`, `../outputs/baselines/E005/` — models/predictions locally; metrics and hashes in Git.

## E006 / E007 model comparison

- `MODEL_COMPARISON.md` — strict E004 input control, ranking groups/relevance and reproduction.
- `../outputs/e006_e007_comparison.md` — all-model official scores, diagnostics and Phase C recommendations.
- `../outputs/e006_e007_comparison.json` — full-precision results and independent verification.
- `../outputs/e006_e007_validation.md` — tests, provenance and reproduction evidence.

## Frozen historical evidence

- `../outputs/phase_c0/phase_c0_turnover_comparison.md` — frozen E005/E006 causal turnover evidence; do not extend the grid.
- `../outputs/phase_c0/phase_c0_turnover_comparison.json` — full-precision selection, metrics and verification.
- `../outputs/phase_c0/phase_c0_turnover_grid.csv` — complete F1/F2 parameter grid plus frozen F3 confirmations.
- `../outputs/phase_c0/phase_c0_turnover_validation.md` — executed tests and protected-artifact checks.

## Phase C0b boundary extension

- `../outputs/phase_c0b/phase_c0b_turnover_comparison.md` — bounded alpha/exit extension and V1 lock decision.
- `../outputs/phase_c0b/phase_c0b_turnover_comparison.json` — complete grid, surfaces, robustness and provenance.
- `../outputs/phase_c0b/phase_c0b_turnover_grid.csv` — complete F1/F2 grid.
- `../outputs/phase_c0b/phase_c0b_turnover_validation.md` — anchor and protected-artifact verification.

## Phase C1 rank fusion

- `../outputs/phase_c1/phase_c1_complementarity.md` — rank, Top overlap and group-return diagnostics.
- `../outputs/phase_c1/phase_c1_comparison.md` — fixed-postprocess fusion results.
- `../outputs/phase_c1/phase_c1_comparison.json` — full diagnostics, metrics and provenance.
- `../outputs/phase_c1/phase_c1_fusion_grid.csv` — five-weight fold grid.
- `../outputs/phase_c1/phase_c1_validation.md` — endpoint and protected-artifact verification.

## Phase C2 LambdaRank redesign

- `../outputs/phase_c2/phase_c2_e007_audit.md` — historical E007 protocol audit.
- `../outputs/phase_c2/phase_c2_comparison.md` — R1/R2 raw and fixed-postprocess comparison.
- `../outputs/phase_c2/phase_c2_comparison.json` — full training, metrics, robustness and provenance.
- `../outputs/phase_c2/phase_c2_validation.md` — executed protocol and protection checks.

## Retired research archive

- `../archive/retired_research_2026-09-22/README.md` — stopped routes, preserved
  uncommitted research, and recovery instructions.

## Phase D representation experiments

- `../outputs/phase_d0_e006_rank_view/comparison.md` — E006 Full147 versus
  Full147 plus 15 mapped core percentile-rank features.
- `../outputs/phase_d1_lambdarank_rank_view/comparison.md` — the same feature
  view under corrected LambdaRank, raw predictions only.
