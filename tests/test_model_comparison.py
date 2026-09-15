import numpy as np

from stock_prediction.compare_models import independent_plan,summary_rows,paired_differences
from stock_prediction.ranking import training_plan


def test_independent_full_plan_oracle_and_group_counts():
    rng=np.random.default_rng(72)
    dates=rng.integers(1,32,size=6000); labels=rng.integers(-50,50,size=6000)/1000
    for experiment in ['E006','E007']:
        actual=training_plan(dates,labels,experiment)
        expected=independent_plan(dates,labels,experiment)
        for a,b in zip(actual,expected): np.testing.assert_array_equal(a,b)


def test_summary_preserves_undefined_fold_instead_of_selective_averaging():
    rows=[]
    for experiment in ['E000','E006']:
        for fold,value in [('F1',.1),('F2',.2),('F3',.3)]:
            rows.append(dict(experiment=experiment,fold=fold,target='rank',model='fixture',
                             ic_mean=value,annual_excess=value,mean_turnover=value,final_score=value,
                             diagnostic_turnover=value,missing_top_fraction=value))
    rows[0]['final_score']=None
    summary=summary_rows(rows)
    assert summary[0]['final_score']['mean'] is None
    assert summary[0]['final_score']['finite_folds']==2
    assert summary[1]['final_score']['worst_fold']=='F1'
    assert summary[1]['mean_turnover']['worst_fold']=='F3'
    assert np.isclose(summary[1]['final_score']['std_ddof0'],np.std([.1,.2,.3]))


def test_pair_decompositions_include_all_three_folds_and_means():
    rows=[]
    for i,experiment in enumerate(['E004','E005','E006','E007']):
        for fold in ['F1','F2','F3']:
            ic=.1+.01*i; excess=.5+.02*i; turnover=.8-.03*i
            rows.append(dict(experiment=experiment,fold=fold,ic_mean=ic,annual_excess=excess,
                             mean_turnover=turnover,final_score=.4*ic+.3*excess+.3*(1-turnover)))
    result=paired_differences(rows)
    assert len(result)==20
    for row in result:
        assert np.isclose(row['delta_score'],row['ic_contribution']+row['excess_contribution']+row['turnover_contribution'])
