# Late-arriving evaluate.py: confirmed organizer source and executed comparison

The project-root evaluate.py appeared during B2 execution, after the implementation and baseline runs had started. It is separate from the expected reference/evaluate_official.py location. The user confirmed that this is the unmodified organizer script. An identical copy is archived at reference/evaluate_official.py. Executed comparison results are in official_evaluator_comparison.md/json; the provisional implementation is not equivalent.

## Observed source differences

| Detail | Current frozen/provisional run | Newly observed evaluate.py |
|---|---|---|
| Auxiliary IC std | User-frozen ddof=0 | ddof=1 |
| Minimum IC rows | 2 finite pred/label pairs, nonconstant | 30 rows after dropping missing labels |
| Top size | First group from array_split into 10, remainder to earlier groups | max(eligible_count // 10, 1) |
| Minimum return/turnover rows | 10 | 100 |
| Prediction NaN/inf | Explicit finite-pred filter per metric, coverage reported | No explicit finite-pred filter |
| Prediction ties | Stable pred descending, stock code ascending | pandas sort_values(pred) default; no stock-code tie key |
| Undefined IC aggregation | Mean of finite daily ICs | np.mean(ic_list), propagating NaNs |
| IC positive fraction | Defined IC dates | All appended IC entries, NaN > 0 becomes false |
| Zero/undefined IC std | ICIR NaN | ICIR 0 when std > 0 is false |
| Auxiliary long-short spread | Calculated and logged | Not returned by this script |

Both implementations use the same 40/30/30 score expression and arithmetic annualization by 252. The newly observed script also does not explicitly remove missing labels from turnover.

## Handling in this delivery

- Preserve all B2 trained models, predictions and scores as explicitly provisional; they are not organizer-verified scores.
- No frozen YAML or competition/experiment specification was changed, and no retraining or automatic metric replacement followed this discovery.
- Provenance confirmed. Deterministic edge tests and all nine saved validation predictions are compared using the original function; the user-frozen ddof conflict remains pending review before versioned rule changes.
- Executed E000 replay produces undefined aggregate IC/Score because the official script propagates NaN predictions. Missing predictions are preserved, not filled.

This late arrival does not invalidate the completed model/library, purge, preprocessing-leakage and prediction-persistence checks. It prevents describing the provisional scores as verified against the newly observed script.
