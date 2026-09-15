"""Expanding-window folds; purge raw candidate dates before filtering labels."""
import numpy as np


def date_int(value):
    return int(str(value).replace("-", ""))


def split_fold(dates, labels, fold, purge_days=1):
    dates, labels = np.asarray(dates), np.asarray(labels)
    if len(dates) != len(labels) or purge_days != 1:
        raise ValueError("V1 requires aligned dates/labels and exactly one purge trading date")
    candidate = (dates >= date_int(fold["train_start"])) & (dates <= date_int(fold["train_end"]))
    valid = (dates >= date_int(fold["valid_start"])) & (dates <= date_int(fold["valid_end"]))
    candidate_dates = np.unique(dates[candidate])
    if len(candidate_dates) <= purge_days or not valid.any():
        raise ValueError("Empty or insufficient chronological fold")
    if candidate_dates[-1] >= dates[valid].min():
        raise ValueError("Candidate training dates overlap validation")
    purge_dates = candidate_dates[-purge_days:]
    purged = candidate & np.isin(dates, purge_dates)
    after_purge = candidate & ~purged
    supervised = after_purge & np.isfinite(labels)
    if not supervised.any():
        raise ValueError("No finite training labels after purge")
    train_idx, valid_idx = np.flatnonzero(supervised), np.flatnonzero(valid)
    metadata = {
        "fold_id": fold["id"], "configured_periods": dict(fold),
        "candidate_train_rows": int(candidate.sum()), "candidate_train_dates": len(candidate_dates),
        "candidate_train_start": int(candidate_dates[0]), "candidate_train_end": int(candidate_dates[-1]),
        "purge_dates": [int(d) for d in purge_dates], "purge_rows": int(purged.sum()),
        "purge_finite_label_rows": int((purged & np.isfinite(labels)).sum()),
        "train_rows_after_purge_before_label_filter": int(after_purge.sum()),
        "excluded_nonfinite_train_label_rows": int((after_purge & ~np.isfinite(labels)).sum()),
        "train_rows": len(train_idx), "train_start": int(dates[train_idx].min()), "train_end": int(dates[train_idx].max()),
        "valid_rows": len(valid_idx), "valid_finite_label_rows": int(np.isfinite(labels[valid_idx]).sum()),
        "valid_start": int(dates[valid_idx].min()), "valid_end": int(dates[valid_idx].max()),
        "train_dates_after_purge": len(candidate_dates) - purge_days, "valid_dates": int(len(np.unique(dates[valid]))),
        "feature_history": "Full causal Basic40 reused; purge applies to supervised fitting, not historical feature rows.",
    }
    return train_idx, valid_idx, metadata
