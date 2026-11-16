from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "experiments" / "results"
REPORTS_DIR = ROOT / "reports"


@st.cache_data
def load_dataset(domain: str, split: str):
    if domain not in {"uav", "unsw"}:
        raise ValueError(f"Unknown domain: {domain}")

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")

    path = DATA_DIR / domain / f"{split}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {
            "X": data["X"],
            "y": data["y"],
            "original_labels": data["original_labels"],
        }


@st.cache_data
def load_feature_names(domain: str):
    path = DATA_DIR / domain / "feature_names.json"

    with path.open() as file:
        return json.load(file)


@st.cache_data
def load_csv(relative_path: str):
    path = ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    return pd.read_csv(path)


@st.cache_data
def get_dataset_summary(domain: str):
    summary = {}

    for split in ["train", "validation", "test"]:
        data = load_dataset(domain, split)

        summary[split] = {
            "records": int(data["X"].shape[0]),
            "features": int(data["X"].shape[1]),
            "normal": int(np.sum(data["y"] == 0)),
            "attacks": int(np.sum(data["y"] == 1)),
            "categories": pd.Series(data["original_labels"]).value_counts().to_dict(),
        }

    return summary


def get_project_counts():
    return {
        "domains": 2,
        "clients_per_domain": 5,
        "total_clients": 10,
        "federated_rounds": 2,
        "local_epochs": 1,
    }
