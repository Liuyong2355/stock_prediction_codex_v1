import pytest

from stock_prediction.phase_c0b import add_turnover_gap, select_f1_f2, surface


def rows():
    result = []
    for alpha in [.5, .4, .3]:
        for exit_fraction in [.15, .175]:
            for fold in ["F1", "F2"]:
                result.append({"experiment": "E005", "fold": fold, "alpha": alpha,
                               "exit_fraction": exit_fraction, "final_score": 1-alpha+exit_fraction,
                               "mean_turnover": alpha-exit_fraction})
    return result


def test_c0b_range_and_selection_use_only_f1_f2():
    data = rows() + [{"experiment": "E005", "fold": "F3", "alpha": .5,
                      "exit_fraction": .15, "final_score": 999., "mean_turnover": 0.}]
    chosen = select_f1_f2(data)
    assert chosen["alpha"] == .3 and chosen["exit_fraction"] == .175
    assert select_f1_f2(rows()) == chosen


def test_selection_rejects_missing_declared_fold():
    with pytest.raises(ValueError): select_f1_f2([r for r in rows() if r["fold"] == "F1"])


def test_turnover_gap():
    row = add_turnover_gap({"diagnostic_turnover": .6, "mean_turnover": .4})
    assert row["turnover_gap"] == pytest.approx(.2)


def test_surface_matches_grid_means():
    data = rows()
    table = surface(data, "E005", "final_score")
    assert table["0.50"]["0.150"] == pytest.approx(.65)
    changed = data + [{"experiment": "E005", "fold": "F3", "alpha": .5,
                       "exit_fraction": .15, "final_score": 100., "mean_turnover": 0.}]
    assert surface(changed, "E005", "final_score") == table
