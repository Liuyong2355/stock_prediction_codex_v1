import numpy as np

from stock_prediction.phase_d0_e006_rank_view import materialize_augmented, rank_block, resolve_rank_sources


def test_rank_block_is_columnwise_finite_only_with_average_ties():
    values = np.array([[1.0, np.nan], [1.0, 30.0], [3.0, 10.0]])
    ranked = rank_block(values)
    np.testing.assert_allclose(ranked[:, 0], [0.5, 0.5, 1.0])
    assert np.isnan(ranked[0, 1])
    np.testing.assert_allclose(ranked[1:, 1], [1.0, 0.5])


def test_resolve_rank_sources_adds_only_not_already_represented():
    config = {
        "source_feature_mapping": ["a", "b", "c"],
        "existing_rank_dependencies": {"a": "csr_a"},
        "new_feature_suffix": "__rank",
    }
    sources, names = resolve_rank_sources(config, ["a", "b", "c", "csr_a"])
    assert sources == ["b", "c"]
    assert names == ["b__rank", "c__rank"]


def test_materialize_augmented_ranks_within_date_and_restores_input_order(tmp_path):
    matrix = np.array([[10., 1.], [30., 3.], [20., 2.], [40., 4.]])
    keys = np.array([(2,), (1,), (2,), (1,)], dtype=[("trade_date", "i8")])
    out = materialize_augmented(matrix, keys, np.array([0, 1, 2, 3]), [0], tmp_path / "view.npy")
    np.testing.assert_array_equal(out[:, :2], matrix)
    np.testing.assert_allclose(out[:, 2], [.5, .5, 1., 1.])
    out._mmap.close()
