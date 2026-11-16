"""Refresh resumable per-seed and mean/sample-SD benchmark CSVs."""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results" / "benchmarks"
MODEL_KEYS = ["mlp", "cnn1d", "resnet1d", "gru", "lstm", "bilstm", "cnn_gru", "cnn_lstm", "tcn", "tiny_transformer"]
METRICS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "macro_f1", "mcc_multiclass", "roc_auc_ovr_macro", "pr_auc_ovr_macro_trapezoidal"]
BASE_FIELDS = ["dataset", "model_key", "model", "seed", "best_epoch", "epochs_completed", "selected_validation_macro_f1", "train_seconds", "validation_seconds", "fit_seconds", "inference_seconds", "inference_ms_per_record", "parameter_count", *METRICS]


def atomic_csv(path: Path, fields: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def completed_results() -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob("*/*/seed_*/result.json")):
        with path.open(encoding="utf-8") as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: (r["dataset"], r["model_key"], r["seed"]))
    return rows


def refresh_aggregates() -> list[dict]:
    rows = completed_results()
    seed_rows = [{k: r.get(k, "") for k in BASE_FIELDS} for r in rows]
    atomic_csv(RESULTS / "per_seed_results.csv", BASE_FIELDS, seed_rows)
    for dataset in ("uav", "unsw"):
        atomic_csv(RESULTS / f"{dataset}_results.csv", BASE_FIELDS,
                   [r for r in seed_rows if r["dataset"] == dataset])
    atomic_csv(RESULTS / "overall_results.csv", BASE_FIELDS, seed_rows)
    model_dir = RESULTS / "individual_model_results"
    for model_key in MODEL_KEYS:
        atomic_csv(model_dir / f"{model_key}.csv", BASE_FIELDS,
                   [r for r in seed_rows if r["model_key"] == model_key])

    summary_fields = ["dataset", "model_key", "model", "n_seeds"]
    for metric in [*METRICS, "train_seconds", "validation_seconds", "fit_seconds", "inference_seconds", "inference_ms_per_record"]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_sample_std"])
    summaries = []
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["model_key"]), []).append(r)
    for (dataset, model_key), group in sorted(groups.items()):
        s = {"dataset": dataset, "model_key": model_key, "model": group[0]["model"], "n_seeds": len(group)}
        for metric in [*METRICS, "train_seconds", "validation_seconds", "fit_seconds", "inference_seconds", "inference_ms_per_record"]:
            vals = [float(x[metric]) for x in group if x.get(metric) is not None]
            s[f"{metric}_mean"] = statistics.mean(vals) if vals else ""
            s[f"{metric}_sample_std"] = statistics.stdev(vals) if len(vals) > 1 else ""
        summaries.append(s)
    atomic_csv(RESULTS / "mean_std_by_model_dataset.csv", summary_fields, summaries)
    return rows


if __name__ == "__main__":
    print(f"Refreshed aggregate files from {len(refresh_aggregates())} completed runs.")
