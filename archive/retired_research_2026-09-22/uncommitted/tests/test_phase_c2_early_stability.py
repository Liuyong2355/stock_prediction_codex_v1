import yaml

from stock_prediction.phase_c2_early_stability import assert_frozen_protocol


def test_early_fold_excludes_2021_and_is_raw_primary():
    c=yaml.safe_load(open("config/phase_c2_early_stability.yaml",encoding="utf-8"))
    assert c["fold"]=={"id":"ES2020","train_start":"2018-01-02","train_end":"2019-12-31","valid_start":"2020-01-01","valid_end":"2020-12-31"}
    assert c["primary_variant"]=="raw" and c["constraints"]["year_2021_unused"] is True


def test_protocol_is_exact_c2_copy():
    early=yaml.safe_load(open("config/phase_c2_early_stability.yaml",encoding="utf-8"))
    c2=yaml.safe_load(open("config/phase_c2_lambdarank.yaml",encoding="utf-8"))
    assert_frozen_protocol(early,c2)
