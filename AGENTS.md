# AGENTS.md

## Project

This repository implements the A-share next-day stock-return prediction competition described in
`reference/赛题五-更新.pdf`.

The project objective is to maximize the official competition score while preserving strict temporal
causality and reproducibility.

## Read first

Always read `docs/index.md` before making non-trivial changes.

For feature work, read:
- `docs/FEATURE_SPEC_V1.md`
- `config/features_v1.yaml`

For split/model/experiment work, read:
- `docs/EXPERIMENT_PROTOCOL_V1.md`
- `config/folds_v1.yaml`
- `config/experiments_v1.yaml`

For task semantics and official metrics, read:
- `docs/COMPETITION_SPEC.md`
- `reference/赛题五-更新.pdf`

For previous research decisions, read:
- `docs/DECISIONS.md`

## Non-negotiable rules

1. Never use future information.
2. Every lag/rolling per-stock feature must be calculated independently within `ts_code` and in ascending `trade_date`.
3. Same-day cross-sectional features may use only information observable on that same `trade_date`.
4. Do not random-split the panel for official experiments. Use `config/folds_v1.yaml`.
5. Apply the one-trading-day purge specified in `config/folds_v1.yaml`.
6. Do not silently change a feature formula, feature-set membership, fold, target, metric, or experiment definition.
7. If a specification must change, update the relevant config/spec, add a record to `docs/DECISIONS.md`, and bump the version when the change is material.
8. Training rows with missing `y_ret_1d` must not be used for supervised fitting or label-dependent evaluation.
9. `ts_code` is a key, not a V1 model feature.
10. Raw absolute OHLC values are source columns, not V1 model features.
11. All experiments must be reproducible, seeded where applicable, and logged.
12. Do not call an internally reimplemented evaluator "official" until it is verified against the organizer-provided `evaluate.py`.

## Source-of-truth precedence

When sources disagree, use this order:

1. Organizer-provided `evaluate.py` for metric implementation details.
2. Organizer competition PDF for task/data rules.
3. Versioned YAML under `config/`.
4. Versioned Markdown specifications under `docs/`.
5. Implementation code.

If `evaluate.py` is not present, implement the PDF rules provisionally and clearly mark unresolved edge cases.

## Required validation

After changing feature code:
- verify feature names/counts against `config/features_v1.yaml`;
- verify causality/no cross-stock rolling contamination;
- verify Basic40 = 40 and Full147 = 147.

After changing split code:
- verify chronological separation and one-trading-day purge.

After changing evaluation code:
- compare against organizer `evaluate.py` when available.

After changing model/experiment code:
- run at least the smallest relevant smoke test before launching a full experiment.

## Development sequence

Do not start with Full147.

First implement and validate:
1. raw data audit;
2. Basic40;
3. temporal folds;
4. provisional/official evaluator;
5. E000, E001, E002.

Only after the baseline pipeline is trustworthy, implement Full147 and E003-E007.
