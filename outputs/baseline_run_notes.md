# B2 execution notes

- E000 and E001 completed all three folds using implementation commit 8fe4454.
- The first E002/F1 attempt completed its frozen 800-round fit, but LightGBM's native save_model call could not write model.txt at the current Unicode Windows path. That attempt produced no scored fold result or experiment-log row.
- Export was changed to the official Booster.model_to_string interface followed by Python UTF-8 file writing. A Unicode-path serialization/reload prediction-equivalence check was added to the existing official-library smoke test.
- The unfinished E002/F1 fold was rerun with identical frozen parameters. Completed E000/E001 results were preserved. This was an export-failure recovery, not hyperparameter search or model selection.
- Each completed result records its actual implementation Git commit and resolved parameters. No config/spec change accompanied the export fix.
