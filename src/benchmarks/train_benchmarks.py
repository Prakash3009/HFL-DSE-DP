"""Resumable CPU training for the approved 60-run multiclass E1 matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from src.benchmarks.aggregate_results import refresh_aggregates
from src.benchmarks.evaluation.metrics import evaluate_multiclass
from src.benchmarks.models import MODEL_REGISTRY, build_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "experiments" / "results" / "benchmarks"
CONFIG_PATH = ROOT / "experiments" / "configs" / "e1_multiclass_benchmark.json"
SEEDS = (42, 123, 2026)
BATCH_SIZE = 256
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 5
THREADS = 6
DEVICE = torch.device("cpu")


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(dataset: str, split: str):
    path = DATA / dataset / f"{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        x = np.array(data["X"], dtype=np.float32, copy=True)
        labels = np.array(data["original_labels"], dtype=str, copy=True)
        # The legacy binary `y` array is deliberately not read or modified.
    return x, labels


def get_dataset_metadata(dataset: str, train_labels: np.ndarray):
    names_path = DATA / dataset / "feature_names.json"
    with names_path.open(encoding="utf-8") as f:
        feature_order = json.load(f)
    classes = sorted(np.unique(train_labels).tolist())
    if not classes or len(classes) != (5 if dataset == "uav" else 10):
        raise ValueError(f"Unexpected {dataset} training classes: {classes}")
    mapping = {name: i for i, name in enumerate(classes)}
    metadata_dir = RESULTS / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = metadata_dir / f"{dataset}_class_mapping.json"
    features_path = metadata_dir / f"{dataset}_feature_order.json"
    mapping_data = {
        "dataset": dataset,
        "class_ids": mapping,
        "encoding_basis": "sorted original class strings from training split only",
        "legacy_binary_y_modified": False,
    }
    feature_data = {"dataset": dataset, "feature_count": len(feature_order), "feature_order": feature_order}
    for path, obj in ((mapping_path, mapping_data), (features_path, feature_data)):
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != obj:
                raise RuntimeError(f"Refusing to replace mismatching metadata: {path}")
        else:
            atomic_json(path, obj)
    return classes, mapping, feature_order


def encode_labels(labels: np.ndarray, mapping: dict[str, int], split: str) -> np.ndarray:
    unknown = sorted(set(labels.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"{split} labels not present in training-derived mapping: {unknown}")
    return np.fromiter((mapping[x] for x in labels), dtype=np.int64, count=len(labels))


def seed_hash(dataset: str, model_key: str, seed: int) -> str:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = {
        "config": config, "dataset": dataset, "model_key": model_key, "seed": seed,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "threads": THREADS,
    }
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def atomic_history(path: Path, history: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = ["epoch", "train_loss", "validation_macro_f1", "train_seconds", "validation_seconds"]
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    os.replace(tmp, path)


def validation_macro_f1(model, loader, y_true, n_classes: int) -> float:
    model.eval()
    logits_parts = []
    with torch.inference_mode():
        for x, _ in loader:
            logits_parts.append(model(x.to(DEVICE)).cpu())
    pred = torch.cat(logits_parts).argmax(1).numpy()
    return float(f1_score(y_true, pred, labels=np.arange(n_classes), average="macro", zero_division=0))


def save_confusion(path: Path, cm: np.ndarray, class_names: list[str]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual_class_id", "actual_class_name", *class_names])
        for i, row in enumerate(cm):
            writer.writerow([i, class_names[i], *[int(x) for x in row]])
    os.replace(tmp, path)


def save_per_class(path: Path, rows: list[dict]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = ["class_id", "class_name", "support", "precision", "recall", "f1", "roc_auc_ovr", "pr_auc_ovr_trapezoidal"]
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run_one(dataset: str, model_key: str, seed: int) -> dict | None:
    display_name = MODEL_REGISTRY[model_key][0]
    run_dir = RESULTS / dataset / model_key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = seed_hash(dataset, model_key, seed)
    result_path = run_dir / "result.json"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config_hash or not best_path.exists():
            raise RuntimeError(f"Completed run has mismatching/incomplete artifacts; refusing to rerun: {run_dir}")
        print(f"SKIP completed {dataset}/{display_name}/seed={seed}", flush=True)
        return None

    print(f"START dataset={dataset} model={display_name} seed={seed}", flush=True)
    set_seed(seed)
    train_x, train_labels = load_split(dataset, "train")
    val_x, val_labels = load_split(dataset, "validation")
    class_names, class_to_id, feature_order = get_dataset_metadata(dataset, train_labels)
    n_classes = len(class_names)
    if train_x.shape[1] != len(feature_order):
        raise ValueError(f"Feature count mismatch in {dataset}: {train_x.shape[1]} vs {len(feature_order)}")
    train_y = encode_labels(train_labels, class_to_id, "train")
    val_y = encode_labels(val_labels, class_to_id, "validation")
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    loader_generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(model_key, train_x.shape[1], n_classes).to(DEVICE)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_score = -1.0
    best_epoch = 0
    stale = 0
    start_epoch = 1
    train_seconds = 0.0
    validation_seconds = 0.0
    stop_after_epoch = False

    if last_path.exists():
        ckpt = torch.load(last_path, map_location=DEVICE, weights_only=False)
        if ckpt.get("config_hash") != config_hash:
            raise RuntimeError(f"Checkpoint config mismatch; refusing to resume: {last_path}")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt["history"]
        best_score = ckpt["best_validation_macro_f1"]
        best_epoch = ckpt["best_epoch"]
        stale = ckpt["stale_epochs"]
        train_seconds = ckpt["train_seconds"]
        validation_seconds = ckpt["validation_seconds"]
        start_epoch = ckpt["next_epoch"]
        stop_after_epoch = ckpt.get("stop_after_epoch", False)
        loader_generator.set_state(ckpt["loader_generator_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        np.random.set_state(ckpt["numpy_rng_state"])
        random.setstate(ckpt["python_rng_state"])
        print(f"RESUME at epoch {start_epoch}; best_epoch={best_epoch}; best_val_macro_f1={best_score:.5f}", flush=True)

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if stop_after_epoch:
            break
        model.train()
        total_loss = 0.0
        total_seen = 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_seen += len(y)
        epoch_train_seconds = time.perf_counter() - t0
        train_seconds += epoch_train_seconds

        t1 = time.perf_counter()
        val_macro = validation_macro_f1(model, val_loader, val_y, n_classes)
        epoch_validation_seconds = time.perf_counter() - t1
        validation_seconds += epoch_validation_seconds
        train_loss = total_loss / max(1, total_seen)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "validation_macro_f1": val_macro,
            "train_seconds": epoch_train_seconds,
            "validation_seconds": epoch_validation_seconds,
        })
        improved = val_macro > best_score + 1e-12
        if improved:
            best_score, best_epoch, stale = val_macro, epoch, 0
            atomic_torch_save(best_path, {
                "model_state": model.state_dict(), "dataset": dataset,
                "model_key": model_key, "model": display_name, "seed": seed,
                "input_dim": len(feature_order), "n_classes": n_classes,
                "class_names": class_names, "best_epoch": best_epoch,
                "validation_macro_f1": best_score, "config_hash": config_hash,
            })
        else:
            stale += 1
        stop_after_epoch = stale >= PATIENCE or epoch == MAX_EPOCHS
        atomic_history(run_dir / "history.csv", history)
        atomic_torch_save(last_path, {
            "config_hash": config_hash, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "history": history,
            "best_validation_macro_f1": best_score, "best_epoch": best_epoch,
            "stale_epochs": stale, "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "next_epoch": epoch + 1, "stop_after_epoch": stop_after_epoch,
            "loader_generator_state": loader_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        })
        print(
            f"PROGRESS dataset={dataset} model={display_name} seed={seed} "
            f"epoch={epoch}/{MAX_EPOCHS} train_loss={train_loss:.5f} "
            f"val_macro_f1={val_macro:.5f} best={best_score:.5f}@{best_epoch} "
            f"train_s={epoch_train_seconds:.1f} val_s={epoch_validation_seconds:.1f} "
            f"stale={stale}/{PATIENCE}", flush=True,
        )
        if stale >= PATIENCE:
            break

    if not best_path.exists():
        raise RuntimeError(f"No validation checkpoint was produced: {run_dir}")
    best = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model_state"])
    model.eval()

    # Test features and labels are loaded only after the checkpoint/model was selected.
    test_x, test_labels = load_split(dataset, "test")
    test_y = encode_labels(test_labels, class_to_id, "test")
    test_loader = DataLoader(TensorDataset(torch.from_numpy(test_x)), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    probability_parts = []
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for (x,) in test_loader:
            logits = model(x.to(DEVICE))
            probability_parts.append(torch.softmax(logits, dim=1).cpu())
    inference_seconds = time.perf_counter() - inference_started
    probabilities = torch.cat(probability_parts).numpy()
    metrics, cm, per_class, predictions = evaluate_multiclass(test_y, probabilities, class_names)

    np.savez_compressed(
        run_dir / "test_predictions.npz",
        y_true=test_y.astype(np.int64), y_pred=predictions.astype(np.int64),
        probabilities=probabilities.astype(np.float32), class_names=np.asarray(class_names),
    )
    save_confusion(run_dir / "confusion_matrix.csv", cm, class_names)
    save_per_class(run_dir / "per_class_metrics.csv", per_class)
    epochs_completed = len(history)
    result = {
        "dataset": dataset, "model_key": model_key, "model": display_name,
        "seed": seed, "config_hash": config_hash,
        "n_classes": n_classes, "class_names": class_names,
        "n_features": len(feature_order), "n_train": len(train_y),
        "n_validation": len(val_y), "n_test": len(test_y),
        "best_epoch": int(best_epoch), "epochs_completed": epochs_completed,
        "selected_validation_macro_f1": float(best_score),
        "train_seconds": float(train_seconds),
        "validation_seconds": float(validation_seconds),
        "fit_seconds": float(train_seconds + validation_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_ms_per_record": float(1000 * inference_seconds / len(test_y)),
        "parameter_count": int(parameter_count),
        **metrics,
    }
    atomic_json(result_path, result)
    refresh_aggregates()
    print(
        f"COMPLETE dataset={dataset} model={display_name} seed={seed} "
        f"best_epoch={best_epoch} fit_s={train_seconds+validation_seconds:.1f} "
        f"infer_s={inference_seconds:.2f} macro_f1={metrics['macro_f1']:.5f} "
        f"saved={run_dir}", flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["uav", "unsw", "all"], required=True)
    parser.add_argument("--model", choices=[*MODEL_REGISTRY, "all"], required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS + (0,), default=0, help="0 runs all approved seeds")
    parser.add_argument("--all", action="store_true", help="Run every requested combination; use only after controlled batches")
    args = parser.parse_args()
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    datasets = ["uav", "unsw"] if args.dataset == "all" else [args.dataset]
    models = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    seeds = list(SEEDS) if args.seed == 0 else [args.seed]
    if not args.all and (len(datasets) * len(models) * len(seeds) != 1):
        parser.error("A multi-run selection requires --all; execute small controlled batches explicitly")
    for dataset in datasets:
        for model_key in models:
            for seed in seeds:
                run_one(dataset, model_key, seed)


if __name__ == "__main__":
    main()
