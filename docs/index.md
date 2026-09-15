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
