# SwanLab experiment tracking

SwanLab is optional observability. It must never change training data, model
parameters, targets, validation, selection, post-processing, or official
evaluation. Tracking failures must not invalidate a completed experiment.

## Installation and login

The research dependency pins `swanlab==0.10.1`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\swanlab.exe login
```

Enter the API key only in the interactive login prompt. Never add it to the
repository, YAML, command history, experiment metadata, or chat messages.

## Modes

Tracking is disabled by default. Explicitly enable it for a shell session:

```powershell
$env:STOCK_SWANLAB_MODE = "online"
$env:STOCK_SWANLAB_PROJECT = "stock-prediction-codex"
# Optional for a team rather than the logged-in personal workspace:
$env:STOCK_SWANLAB_WORKSPACE = "your-team-workspace"
```

Unset the variables or use `disabled` to turn tracking off. SwanLab 0.10.1
requires the separate `swanlab[dashboard]` extra for `local` mode; that optional
dashboard is not installed by this project because the selected deployment is
cloud tracking.

## Training-code contract

Future experiment runners should create one run per fold:

```python
from stock_prediction.swanlab_tracker import start_swanlab

tracker = start_swanlab(
    root,
    experiment_name=f"{experiment_id}_{fold_id}",
    group=experiment_id,
    metadata={
        "experiment": experiment_id,
        "fold": fold_id,
        "seed": seed,
        "model_params": model_params,
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
    },
)

tracker.log({
    "official/rank_ic": official["ic_mean"],
    "official/top10_excess": official["annual_excess"],
    "official/turnover": official["mean_turnover"],
    "official/score": official["final_score"],
    "diagnostic/turnover": diagnostic["diagnostic_exclude_missing_y_turnover"],
    "diagnostic/missing_top_fraction": diagnostic["missing_share_of_top"],
})
tracker.finish()
```

Call `finish(state="crashed", error=str(exc))` when a runner catches a training
exception. SwanLab logging must remain outside score computation.

## Upload boundary

Allowed: aggregate metrics, fold/date ranges, row counts, durations, seed,
model parameters, config/evaluator hashes, Git commit, and research decisions.

Forbidden: raw market data, feature matrices, stock codes, row-level
predictions, labels, trained models, credentials, or API keys. Do not call
artifact-upload APIs for these files.
