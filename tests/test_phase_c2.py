import numpy as np
import pandas as pd
import yaml

from stock_prediction.phase_c2 import relevance_and_groups, ranked_feature_indices, rank_matrix_groups_inplace, select_candidate


def test_relevance_range_direction_ties_and_groups():
    dates=np.array([2,1,1,2,1,2]); labels=np.array([3.,1.,1.,1.,3.,2.])
    order,y,groups,group_dates=relevance_and_groups(dates,labels)
    assert y.min()>=0 and y.max()<=9 and set(y)<=set(range(10))
    assert groups.sum()==len(labels)
    np.testing.assert_array_equal(np.repeat(group_dates,groups),dates[order])
    for date in group_dates:
        idx=np.flatnonzero(dates[order]==date); values=labels[order][idx]
        assert np.all(np.diff(y[idx][np.argsort(values)])>=0)
    assert y[dates[order]==1][0]==y[dates[order]==1][1]


def test_cross_section_rank_is_group_local_and_preserves_market():
    a=np.array([[1.,10.],[3.,10.],[100.,20.],[200.,20.]])
    rank_matrix_groups_inplace(a,[2,2],[0])
    np.testing.assert_allclose(a[:,0],[.5,1.,.5,1.]); np.testing.assert_array_equal(a[:,1],[10.,10.,20.,20.])


def test_metadata_rule_not_name_guessing():
    config={"features":[{"name":"odd_market_name","category":"market"},{"name":"market_looking","category":"price"}]}
    ranked,preserved=ranked_feature_indices(config,["odd_market_name","market_looking"])
    assert ranked==[1] and preserved==[0]


def test_f3_excluded_from_selection():
    rows=[]
    for candidate,score in [("R1",1.),("R2",.5)]:
        for fold in ["F1","F2"]: rows.append({"candidate":candidate,"fold":fold,"variant":"postprocessed","final_score":score,"mean_turnover":.2})
    rows.append({"candidate":"R2","fold":"F3","variant":"postprocessed","final_score":999.,"mean_turnover":0.})
    assert select_candidate(rows)["candidate"]=="R1"


def test_frozen_c2_parameters():
    c=yaml.safe_load(open("config/phase_c2_lambdarank.yaml",encoding="utf-8"))
    assert c["ranker"]["label_gain"]==list(range(10)); assert c["ranker"]["lambdarank_truncation_level"]==500
    assert c["ranker"]["lambdarank_norm"] is True and c["ranker"]["n_estimators"]==100
    assert c["postprocess"]=={"alpha":.3,"exit_fraction":.25}
