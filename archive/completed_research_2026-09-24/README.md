# Completed research archive — 2026-09-24

This index marks completed research as historical evidence. The original
result/config/code paths remain in place so recorded hashes, cross-references
and reproduction commands continue to work. They are outside the default
main-model workflow.

Snapshots of the prior top-level README and research handoff are retained as
`README_before_cleanup.md` and `NEXT_RESEARCH_HANDOFF_before_cleanup.md`.

| Route | Status | Evidence |
|---|---|---|
| D2-A stretched target | Stopped: all three folds worse | `outputs/phase_d2a_top10_stretched_rank/` |
| D2-B Top10 classifier | Stopped: negative tail alpha | `outputs/phase_d2b_top10_binary/` |
| D3 Recent-2Y | Auxiliary candidate only; window search closed | `outputs/phase_d3_recent_2y/` |
| D4 LightGBM rank regression | Not promoted; validation missing-label anomaly | `outputs/phase_d4_lightgbm_rank_regression/` |
| D5 residual diagnostic | Corrected: rank subtraction created a mechanical effect | `outputs/phase_d5_residual_alpha/` |
| C0/C0b turnover controls | Frozen historical strategies; grid closed | `outputs/phase_c0/`, `outputs/phase_c0b/` |
| C1 fusion and C2 LambdaRank | Historical, not active | `outputs/phase_c1/`, `outputs/phase_c2/` |

The old pre-zero-variance Full147 matrices and interrupted E003 scratch matrix
were never final results. Their small audit records remain; the obsolete `.npy`
copies are verified cleanup targets, but deletion was blocked by the execution
policy and is pending. Their exact paths are in
`docs/MODEL_AUDIT_2026-09-24.md`.

For current model identity and the E/F-fold comparison, read
`docs/MODEL_AUDIT_2026-09-24.md`. Frozen decisions remain in
`docs/DECISIONS.md`; model/test result JSON and Markdown remain at their
original paths. Large local prediction files and non-primary historical models
are not part of the Git push.
