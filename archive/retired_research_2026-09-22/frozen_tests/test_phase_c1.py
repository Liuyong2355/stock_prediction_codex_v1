import numpy as np
import pytest

from stock_prediction.phase_c1 import blend_daily_ranks, select_weight


def test_blend_and_endpoints():
    a=np.array([.2,.8]); b=np.array([.6,.4])
    np.testing.assert_array_equal(blend_daily_ranks(a,b,1.),a)
    np.testing.assert_array_equal(blend_daily_ranks(a,b,0.),b)
    np.testing.assert_allclose(blend_daily_ranks(a,b,.25),.25*a+.75*b)


def test_invalid_weight():
    with pytest.raises(ValueError): blend_daily_ranks([1],[2],1.1)


def test_selection_excludes_f3_and_tie_breaks_to_half():
    rows=[]
    for w in [0.,.5,1.]:
        for fold in ["F1","F2"]:
            rows.append({"fold":fold,"weight_e005":w,"final_score":1.,"mean_turnover":.2})
    rows.append({"fold":"F3","weight_e005":0.,"final_score":999.,"mean_turnover":0.})
    assert select_weight(rows)["weight_e005"]==.5


def test_selection_requires_both_folds():
    with pytest.raises(ValueError): select_weight([{"fold":"F1","weight_e005":.5,"final_score":1.,"mean_turnover":.2}])
