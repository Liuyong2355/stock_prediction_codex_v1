from pathlib import Path

from stock_prediction.main_model import load_main_model


def test_canonical_main_model_is_frozen_d0_f3():
    root = Path(__file__).resolve().parents[1]
    model, names = load_main_model(root)
    assert len(names) == 162
    assert model.get_booster().feature_names == names
