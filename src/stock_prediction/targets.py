"""Frozen V1 training labels; callers pass only purged supervised training rows."""
import hashlib

import numpy as np
import pandas as pd


TARGETS = {'E003': 'raw', 'E004': 'rank', 'E005': 'relative'}


def training_target(dates, labels, target):
    dates, labels = np.asarray(dates), np.asarray(labels, dtype=np.float64)
    if dates.ndim != 1 or labels.ndim != 1 or len(dates) != len(labels):
        raise ValueError('Training dates and labels must be aligned vectors')
    if target not in TARGETS.values():
        raise ValueError('Only raw, rank and relative targets are authorized')
    if not len(labels) or not np.isfinite(labels).all():
        raise ValueError('Pass finite supervised labels after fold purge/filtering')
    values = pd.Series(labels)
    if target == 'rank':
        result = values.groupby(dates, sort=False).rank(method='average', pct=True) - .5
    elif target == 'relative':
        result = values - values.groupby(dates, sort=False).transform('median')
    else:
        result = values.copy()
    return result.to_numpy(copy=True)


def array_sha256(values):
    """Hash an ordered numeric vector, with dtype/shape included."""
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(f'{values.dtype.str}:{values.shape}:'.encode('ascii'))
    digest.update(memoryview(values).cast('B'))
    return digest.hexdigest()
