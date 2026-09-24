import numpy as np

from stock_prediction.phase_d3_recent_2y import recent_training_indices


def test_recent_training_indices_selects_only_configured_dates():
    dates = np.array([20191231, 20200102, 20211230, 20211231, 20220103])
    train = np.array([0, 1, 2, 3])
    actual = recent_training_indices(train, dates, {"train_start": 20200101, "train_end": 20211231})
    np.testing.assert_array_equal(actual, [1, 2, 3])


def test_recent_window_cannot_reintroduce_purged_or_validation_rows():
    dates = np.array([20210104, 20221229, 20221230, 20230103])
    already_purged_train = np.array([0, 1])
    actual = recent_training_indices(already_purged_train, dates, {"train_start": 20210101, "train_end": 20221231})
    np.testing.assert_array_equal(actual, [0, 1])
