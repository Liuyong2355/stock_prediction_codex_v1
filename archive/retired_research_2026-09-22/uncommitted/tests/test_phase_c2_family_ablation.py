import yaml

from stock_prediction.phase_c2_family_ablation import classify_family, feature_set_indices, assert_scope


def test_metadata_driven_exact_family_removal():
    cfg={"features":[{"name":"misleading_market","category":"relative"},{"name":"csr_x","category":"market"},{"name":"z","category":"trend"}]}
    kept,removed=feature_set_indices(cfg,["misleading_market","csr_x","z"],"market")
    assert kept==[0,2] and removed==[1]


def test_frozen_two_year_decision_rule():
    assert classify_family([{"delta_score":-.1},{"delta_score":-.2}])=="retain"
    assert classify_family([{"delta_score":.1},{"delta_score":.2}])=="deletion_candidate"
    assert classify_family([{"delta_score":.1},{"delta_score":-.2}])=="unstable_retain_for_now"


def test_scope_excludes_2021_and_copies_c2_ranker():
    config=yaml.safe_load(open("config/phase_c2_family_ablation.yaml",encoding="utf-8")); c2=yaml.safe_load(open("config/phase_c2_lambdarank.yaml",encoding="utf-8"))
    assert assert_scope(config,c2)==c2["ranker"]
    assert max(int(f["valid_end"][:4]) for f in config["folds"])==2020
