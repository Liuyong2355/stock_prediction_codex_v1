"""C2-R1 interaction ablation for market, relative, and cross-section-rank families."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .compare_official import load_official
from .folds import split_fold
from .phase_c0 import sha256
from .phase_c2 import evaluate_prediction, make_model, relevance_and_groups
from .phase_c2_early_stability import validation_control
from .phase_c2_family_ablation import materialize_subset


FAMILIES = ("market", "relative", "cross_section_rank")
METRICS = ("ic_mean", "annual_excess", "mean_turnover", "final_score")


def feature_sets_from_metadata(feature_config, names, specs, family_counts):
    metadata = {item["name"]: item for item in feature_config["features"]}
    if set(names) != set(metadata):
        raise ValueError("Full147 manifest and feature metadata differ")
    actual = {family: [name for name in names if metadata[name]["category"] == family] for family in FAMILIES}
    if {key: len(value) for key, value in actual.items()} != family_counts:
        raise ValueError("metadata family counts differ from frozen counts")
    resolved = {}
    for spec in specs:
        categories = tuple(spec["remove_categories"])
        if not categories or set(categories) - set(FAMILIES):
            raise ValueError("feature set outside frozen interaction scope")
        removed = [name for name in names if metadata[name]["category"] in categories]
        kept = [index for index, name in enumerate(names) if name not in set(removed)]
        if len(kept) != spec["expected_count"]:
            raise ValueError(f"{spec['id']} expected {spec['expected_count']} features, got {len(kept)}")
        resolved[spec["id"]] = {"kept": kept, "removed_features": removed,
                                "remove_categories": list(categories), "feature_count": len(kept)}
    if [resolved[item["id"]]["feature_count"] for item in specs] != [137, 129, 121, 113]:
        raise ValueError("frozen feature dimensions are not 137/129/121/113")
    return resolved, actual


def assert_scope(config, c2):
    assert config["primary_variant"] == "raw"
    assert config["purge_trading_days"] == 1
    assert config["constraints"] == {"year_2021_unused": True, "tuning": "prohibited",
                                      "postprocess_search": "prohibited", "other_family_experiments": "prohibited"}
    assert config["selection_rule"]["order"] == ["mean_score_desc", "worst_fold_score_desc", "feature_count_asc"]
    assert c2["ranker"]["n_estimators"] == 100 and c2["ranker"]["random_state"] == 42
    assert c2["ranker"]["objective"] == "lambdarank"
    return c2["ranker"]


def validate_anchor(root, config, ranker, resolved):
    source = root / config["anchor"]["source"]
    payload = json.loads(source.read_text(encoding="utf-8"))
    source_script = root / "src/stock_prediction/phase_c2_family_ablation.py"
    source_config = root / "config/phase_c2_family_ablation.yaml"
    checks = {
        "git_head_matches": payload["git_head"] == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_script_sha256_matches": payload["source_sha256"] == sha256(source_script),
        "source_config_sha256_matches": payload["config_sha256"] == sha256(source_config),
        "ranker_matches": payload["ranker"] == ranker,
        "folds_match": payload["config"]["folds"] == config["folds"],
        "purge_matches": payload["config"]["purge_trading_days"] == config["purge_trading_days"] == 1,
        "primary_raw_matches": payload["config"]["primary_variant"] == config["primary_variant"] == "raw",
        "official_evaluation_attested": payload["verification"]["primary_raw_only"] is True,
        "year_2021_unused": payload["verification"]["year_2021_unused"] is True,
    }
    anchor_meta = [item for item in payload["training_metadata"] if item["feature_set"] == config["anchor"]["feature_set"]]
    checks["two_anchor_folds_present"] = [item["fold"] for item in anchor_meta] == [item["id"] for item in config["folds"]]
    checks["feature_lists_match"] = all(item["feature_count"] == 137 and
                                        item["removed_features"] == resolved["minus_market"]["removed_features"]
                                        for item in anchor_meta)
    checks["one_day_purge_verified"] = all(len(item["split"]["purge_dates"]) == 1 and
                                           item["split"]["train_end"] < item["split"]["valid_start"]
                                           for item in anchor_meta)
    if not all(checks.values()):
        raise RuntimeError(f"anchor reuse validation failed: {[key for key, value in checks.items() if not value]}")
    rows = [dict(item) for item in payload["results"] if item["feature_set"] == config["anchor"]["feature_set"]]
    if len(rows) != 2:
        raise RuntimeError("anchor must contain exactly two fold results")
    return rows, anchor_meta, {"source_file": str(source.relative_to(root)), "source_sha256": sha256(source),
                               "validation_checks": checks}


def run_one(root, spec, fold, ranker, manifest, evaluator, temp):
    names = manifest["feature_names"]
    matrix = np.load(manifest["datasets"]["train"]["files"]["matrix"]["path"], mmap_mode="r", allow_pickle=False)
    keys = np.load(manifest["datasets"]["train"]["files"]["keys"]["path"], mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "outputs/baselines/cache/labels.npy", mmap_mode="r", allow_pickle=False)
    train, valid, split = split_fold(keys["trade_date"], labels, fold, 1)
    if np.any(keys["trade_date"][train] >= 20210101) or np.any(keys["trade_date"][valid] >= 20210101):
        raise RuntimeError("2021 data entered an interaction fold")
    order, fit_y, groups, _ = relevance_and_groups(keys["trade_date"][train], labels[train])
    fit = train[order]
    path = Path(temp) / f"{spec['id']}_{fold['id']}.npy"
    x = materialize_subset(matrix, fit, spec["kept"], path)
    selected_names = [names[index] for index in spec["kept"]]
    model = make_model(ranker)
    started = time.perf_counter()
    model.fit(pd.DataFrame(x, columns=selected_names, copy=False), fit_y, group=groups)
    seconds = time.perf_counter() - started
    x._mmap.close(); del x; path.unlink()
    pred = np.empty(len(valid), dtype=float)
    for start in range(0, len(valid), 100_000):
        idx = valid[start:start + 100_000]
        block = matrix[np.ix_(idx, spec["kept"])]
        pred[start:start + len(idx)] = model.predict(pd.DataFrame(block, columns=selected_names, copy=False))
    official, diagnostic, _ = evaluate_prediction(root, pred, validation_control(keys, labels, matrix, valid, names), evaluator, temp)
    row = {"feature_set": spec["id"], "fold": fold["id"],
           **{key: official[key] for key in METRICS}, **diagnostic}
    metadata = {"feature_set": spec["id"], "fold": fold["id"], "split": split,
                "feature_count": spec["feature_count"], "removed_feature_count": len(spec["removed_features"]),
                "removed_features": spec["removed_features"], "group_sum": int(groups.sum()), "groups": len(groups),
                "iterations": model.booster_.current_iteration(), "train_seconds": seconds}
    return row, metadata


def add_deltas(rows):
    anchor = {row["fold"]: row for row in rows if row["feature_set"] == "minus_market"}
    output = []
    for row in rows:
        base = anchor[row["fold"]]
        output.append({**row, "delta_ic": row["ic_mean"] - base["ic_mean"],
                       "delta_excess": row["annual_excess"] - base["annual_excess"],
                       "delta_turnover": row["mean_turnover"] - base["mean_turnover"],
                       "delta_score": row["final_score"] - base["final_score"]})
    return output


def select_feature_set(rows, feature_counts):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["feature_set"], []).append(row)
    summaries = []
    for name, values in grouped.items():
        eligible = name == "minus_market" or (len(values) == 2 and all(item["delta_score"] >= 0 for item in values))
        summaries.append({"feature_set": name, "feature_count": feature_counts[name], "eligible": eligible,
                          "mean_score": float(np.mean([item["final_score"] for item in values])),
                          "worst_fold_score": min(item["final_score"] for item in values)})
    eligible = [item for item in summaries if item["eligible"]]
    selected = max(eligible, key=lambda item: (item["mean_score"], item["worst_fold_score"], -item["feature_count"]))
    return summaries, selected["feature_set"]


def tree_sha256(path):
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def write_report(out, payload):
    lines = ["# C2 R1 feature-family interaction", "", "Raw prediction only; 2021 is unused. The 137-feature anchor is reused after strict validation.", "",
             "| Feature set | Fold | IC | Excess | Turnover | Score | Diagnostic turnover | Missing Top | ΔIC | Δexcess | Δturnover | ΔScore |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["results"]:
        lines.append(f"| {row['feature_set']} | {row['fold']} | {row['ic_mean']:.6f} | {row['annual_excess']:.6f} | {row['mean_turnover']:.6f} | {row['final_score']:.6f} | {row['diagnostic_turnover']:.6f} | {row['missing_top_fraction']:.2%} | {row['delta_ic']:+.6f} | {row['delta_excess']:+.6f} | {row['delta_turnover']:+.6f} | {row['delta_score']:+.6f} |")
    lines += ["", f"Selected by the frozen rule: `{payload['selected_feature_set']}`.", ""]
    (out / "phase_c2_family_interaction_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    root = Path.cwd()
    config_path = root / "config/phase_c2_family_interaction.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    c2 = yaml.safe_load((root / "config/phase_c2_lambdarank.yaml").read_text(encoding="utf-8"))
    ranker = assert_scope(config, c2)
    feature_config = yaml.safe_load((root / "config/features_v1.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((root / "outputs/full147_manifest.json").read_text(encoding="utf-8"))
    resolved, families = feature_sets_from_metadata(feature_config, manifest["feature_names"], config["feature_sets"], config["family_counts"])
    anchor_rows, anchor_metadata, anchor_reuse = validate_anchor(root, config, ranker, resolved)
    protected = [root / "outputs/phase_c2", root / "outputs/phase_c2_early_stability", root / "outputs/phase_c2_family_ablation"]
    before = {str(path.relative_to(root)): tree_sha256(path) for path in protected}
    evaluator = load_official(root); rows = anchor_rows; metadata = anchor_metadata
    with tempfile.TemporaryDirectory(prefix="c2_interaction_") as temp:
        for fold in config["folds"]:
            for item in config["feature_sets"][1:]:
                spec = {**item, **resolved[item["id"]]}
                row, meta = run_one(root, spec, fold, ranker, manifest, evaluator, temp)
                rows.append(row); metadata.append(meta)
                print(f"DONE {item['id']}/{fold['id']}", flush=True)
    rows = add_deltas(rows)
    summaries, selected = select_feature_set(rows, {key: value["feature_count"] for key, value in resolved.items()})
    after = {str(path.relative_to(root)): tree_sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("protected historical artifacts changed")
    out = root / "outputs/phase_c2_family_interaction"; out.mkdir(parents=True, exist_ok=False)
    payload = {"phase": config["phase"], "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "source_sha256": sha256(Path(__file__)), "config_sha256": sha256(config_path), "config": config, "ranker": ranker,
               "family_features": families, "feature_sets": {key: {k: v for k, v in value.items() if k != "kept"} for key, value in resolved.items()},
               "anchor_reuse": anchor_reuse, "results": rows, "training_metadata": metadata,
               "selection_summary": summaries, "selected_feature_set": selected,
               "verification": {"year_2021_unused": True, "primary_raw_only": True, "anchor_retrained": False,
                                "protected_artifacts_unchanged": True, "deterministic_seed": ranker["random_state"]}}
    (out / "phase_c2_family_interaction_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_report(out, payload)


if __name__ == "__main__":
    main()
