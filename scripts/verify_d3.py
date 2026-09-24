"""Independently replay D3 official metrics and D0 complementarity."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from stock_prediction.compare_official import load_official, replay
from stock_prediction.phase_d2b_top10_binary import complementarity


def main():
    root = Path.cwd()
    evaluator = load_official(root)
    for fold in ("F1", "F2", "F3"):
        folder = root / "outputs/phase_d3_recent_2y" / fold
        saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(folder / "predictions.csv.gz", float_precision="round_trip")
        d0 = pd.read_csv(root / "outputs/phase_d0_e006_rank_view" / fold / "predictions.csv.gz",
                         float_precision="round_trip")
        with tempfile.TemporaryDirectory(prefix="d3_verify_") as temp:
            actual = replay(predictions, temp, evaluator)
        comp, _ = complementarity(predictions, d0)
        for metric in ("ic_mean", "annual_excess", "mean_turnover", "final_score"):
            np.testing.assert_allclose(actual[metric], saved["official"][metric], rtol=0, atol=1e-14)
        for metric in ("prediction_spearman", "top10_jaccard", "top10_overlap"):
            np.testing.assert_allclose(comp[metric]["mean"], saved["complementarity"][metric]["mean"],
                                       rtol=0, atol=1e-14)
        print(f"D3_REPLAY_OK {fold}")


if __name__ == "__main__":
    main()
