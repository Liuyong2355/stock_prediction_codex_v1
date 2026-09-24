# Next validation gate (2026-09-24)

## Competition boundary

The organizer PDF, `reference/赛题五-更新.pdf`, section 6(1), says feature
engineering may use **only fields in the supplied training/test datasets** and
must not use future information. Section 6(2) says test Y is unavailable to
participants until after the competition for scoring. Therefore company
announcements, analyst forecasts, and any other external data are **not** an
eligible feature source for this competition. A separate post-competition
research project would require a separate data and evaluation contract.

## Local data-readiness result

- The supplied test X ends on `2026-06-08`; its previous observed date is
  `2026-06-05`.
- The local `data/raw/evaluation` Y was reconstructed retrospectively from the
  next test row's close. It covers `2025-01-02` through `2026-06-05` and is
  already exposed to research. It is **not** an organizer-provided hidden Y.
- The workspace contains no later price panel or unviewed labeled period.
  `2026-06-08` alone has no next observed close with which to calculate its
  `y_ret_1d`.
- Dates after the supplied test range could validate a live research strategy
  if acquired later, but would not themselves produce the official competition
  Score for the fixed 2025-2026 submission set.

## Operational gate

1. Freeze D0 raw, E006+C0b, and the completed D0+C0b transfer evidence.
   D0+C0b is not promoted: its three-fold Score edge over E006+C0b is only
   `0.001823` and the local test edge `0.001664` is retrospective.
2. Do not use reconstructed test Y for model, feature, parameter, or
   postprocess selection. Do not add external features to a competition model.
3. If the aim is the fixed competition submission, the next authorized work
   should be a **separate, predeclared deployment decision**: choose a frozen
   eligible strategy, define whether and how to refit on the original labeled
   training data, verify complete test-key coverage and causal feature
   computation, and produce the submission without using reconstructed Y.
4. If the aim is live post-competition research, first obtain genuinely later
   point-in-time X and subsequent labels, freeze predictions before labels are
   observed, and keep those results outside competition-Score claims.

No new model or strategy experiment is authorized by this gate. The PDF rule
supersedes earlier exploratory suggestions about external information sources.
