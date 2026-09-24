# Current handoff — 2026-09-24

Primary raw alpha model: `main_alpha_xgb_rank162` (historical D0). Its checked
F3 binary is `models/main_alpha_xgb_rank162/model.ubj`; the canonical feature
and hash contract is `config/main_model.yaml`. It was trained only through
2023-12-28 after the one-day purge. It has not been refit on 2024–2026 data.

The full model comparison is `docs/MODEL_AUDIT_2026-09-24.md`. F1/F2/F3 are
validation years, not an additional model series. D0 has the strongest clean
raw broad-ranking signal among the checked models; it is not the highest
Official Score strategy. The frozen E006+C0b postprocess is the low-missing
high-score reference, while E005+C0b has a slightly higher test Score but
historically severe missing-label Top occupancy. Do not combine raw and
postprocessed scores as though they measured the same setup.

The one-time fixed C0b transfer to saved D0 predictions is complete (D035):
mean Score `0.360422`, only `+0.001823` over E006+C0b, with lower IC and
Top10 excess than D0 raw. The already-viewed 2025–2026 period gives `0.368088`.
This does not meet the predeclared meaningful-margin gate; do not promote it
or reopen turnover tuning. See `../outputs/phase_d0_c0b_fixed/comparison.md`.

The next validation gate is `NEXT_VALIDATION_GATE.md` (D036). The organizer PDF
restricts competition feature engineering to the supplied train/test fields;
external announcements or analyst data are out of scope. The current workspace
has no unviewed later labels. The reconstructed 2025–2026 Y must not be used
to select another competition strategy. No new model experiment has begun.

D2-A/B target routes, D3 window search, ordinary D4 LightGBM tuning, C0/C0b
turnover grids, C1 fusion and C2 LambdaRank expansion remain closed. D3 may be
kept only as an auxiliary candidate. D5's original positive rank-difference
IC was partly mechanical; its conditional audit does not justify a residual
model without a leakage-safe training design. The 2025–2026 local test period
has been viewed once; reserve it from further model/parameter selection.

Historical reports, old README/handoff snapshots and stopped-route statuses
are indexed in `archive/completed_research_2026-09-24/README.md`. Decisions
are recorded in `docs/DECISIONS.md`. The default next step is maintenance and
deployment verification of the existing main model, not a new research grid.
