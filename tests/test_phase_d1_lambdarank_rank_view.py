from stock_prediction.phase_d1_lambdarank_rank_view import baseline_rows


def test_corrected_lambdarank_baseline_has_three_raw_folds():
    rows = baseline_rows(__import__("pathlib").Path.cwd())
    assert set(rows) == {"F1", "F2", "F3"}
    assert all(row["variant"] == "raw" and row["candidate"] == "C2-R1" for row in rows.values())
