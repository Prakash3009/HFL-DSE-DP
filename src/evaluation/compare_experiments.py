
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. EXPERIMENT FILES
# --------------------------------------------------

EXPERIMENTS = {
    "Non-private HFL": {
        "metrics": "hfl_test_metrics_noniid_selected_thresholds.csv",
        "privacy": None,
        "noise": None,
    },
    "DP-HFL (noise=0.5)": {
    "metrics": "hfl_dp_test_metrics_noise_0.5.csv",
    "privacy": "hfl_dp_final_privacy_prv_noise_0.5.csv",
    "noise": 0.5,
},

"DP-HFL (noise=1.0)": {
    "metrics": "hfl_dp_test_metrics_selected_thresholds.csv",
    "privacy": "hfl_dp_final_privacy_prv_noise_1.0.csv",
    "noise": 1.0,
},

"DP-HFL (noise=1.5)": {
    "metrics": "hfl_dp_test_metrics_noise_1.5.csv",
    "privacy": "hfl_dp_final_privacy_prv_noise_1.5.csv",
    "noise": 1.5,
},
}


# --------------------------------------------------
# 3. LOAD AND NORMALIZE TEST METRICS
# --------------------------------------------------

all_metrics = []

for experiment_name, config in EXPERIMENTS.items():

    metrics_path = RESULTS / config["metrics"]

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    df = pd.read_csv(metrics_path)

    # Normalize domain names
    df["domain"] = df["domain"].str.lower()

    # Add experiment metadata
    df["experiment"] = experiment_name
    df["noise_multiplier"] = config["noise"]

    # Ensure expected columns exist
    required = [
        "domain", "threshold", "accuracy", "precision",
        "attack_recall", "macro_f1", "mcc", "fpr",
        "pr_auc", "roc_auc"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{experiment_name}: missing columns {missing}"
        )

    all_metrics.append(df)


comparison = pd.concat(all_metrics, ignore_index=True)


# --------------------------------------------------
# 4. LOAD PRIVACY BUDGETS
# --------------------------------------------------

privacy_records = []

for experiment_name, config in EXPERIMENTS.items():

    if config["privacy"] is None:
        continue

    privacy_path = RESULTS / config["privacy"]

    if not privacy_path.exists():
        raise FileNotFoundError(privacy_path)

    privacy_df = pd.read_csv(privacy_path)

    for domain in ["uav", "unsw"]:

        domain_df = privacy_df[
            privacy_df["domain"].str.lower() == domain
        ]

        if domain_df.empty:
            raise ValueError(
                f"No privacy records for {experiment_name}, {domain}"
            )

        privacy_records.append({
            "experiment": experiment_name,
            "domain": domain,
            "max_client_epsilon": domain_df["epsilon"].max(),
            "mean_client_epsilon": domain_df["epsilon"].mean(),
            "delta": domain_df["delta"].max(),
        })


privacy_summary = pd.DataFrame(privacy_records)


# --------------------------------------------------
# 5. MERGE METRICS AND PRIVACY
# --------------------------------------------------

comparison = comparison.merge(
    privacy_summary,
    on=["experiment", "domain"],
    how="left"
)

comparison["noise_multiplier"] = comparison[
    "noise_multiplier"
].astype("Float64")


# --------------------------------------------------
# 6. SAVE COMPARISON TABLES
# --------------------------------------------------

comparison_path = TABLES / "final_experiment_comparison_prv.csv"

privacy_path = TABLES / "final_privacy_comparison_prv.csv"

comparison.to_csv(comparison_path, index=False)
privacy_summary.to_csv(privacy_path, index=False)

print("\nFINAL EXPERIMENT COMPARISON")
print(
    comparison[
        [
            "experiment", "domain", "threshold",
            "accuracy", "macro_f1", "mcc",
            "attack_recall", "fpr", "pr_auc",
            "roc_auc", "max_client_epsilon"
        ]
    ].to_string(index=False)
)

print("\nPRIVACY SUMMARY")
print(privacy_summary.to_string(index=False))


# --------------------------------------------------
# 7. PRIVACY-UTILITY PLOT
# --------------------------------------------------

for domain in ["uav", "unsw"]:

    domain_df = comparison[
        (comparison["domain"] == domain)
        & comparison["max_client_epsilon"].notna()
    ].sort_values("max_client_epsilon")

    plt.figure(figsize=(8, 5))

    plt.plot(
        domain_df["max_client_epsilon"],
        domain_df["macro_f1"],
        marker="o",
        linewidth=2
    )

    for _, row in domain_df.iterrows():
        plt.annotate(
            f"Noise={row['noise_multiplier']}",
            (row["max_client_epsilon"], row["macro_f1"]),
            xytext=(5, 7),
            textcoords="offset points"
        )

    plt.xlabel("Maximum Client Epsilon (ε)")
    plt.ylabel("Test Macro-F1")
    plt.title(f"Privacy-Utility Trade-off: {domain.upper()}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURES / f"privacy_utility_{domain}_prv.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 8. MACRO-F1 COMPARISON
# --------------------------------------------------

for domain in ["uav", "unsw"]:

    domain_df = comparison[
        comparison["domain"] == domain
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        domain_df["experiment"],
        domain_df["macro_f1"]
    )

    plt.ylabel("Test Macro-F1")
    plt.xlabel("Experiment")
    plt.title(f"Macro-F1 Comparison: {domain.upper()}")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
       FIGURES / f"macro_f1_comparison_{domain}_prv.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 9. FALSE-POSITIVE RATE COMPARISON
# --------------------------------------------------

for domain in ["uav", "unsw"]:

    domain_df = comparison[
        comparison["domain"] == domain
    ]

    plt.figure(figsize=(9, 5))

    plt.bar(
        domain_df["experiment"],
        domain_df["fpr"]
    )

    plt.ylabel("False Positive Rate")
    plt.xlabel("Experiment")
    plt.title(f"False Positive Rate Comparison: {domain.upper()}")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
       FIGURES / f"fpr_comparison_{domain}_prv.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 10. COMPLETION
# --------------------------------------------------

print("\nCOMPARISON COMPLETED SUCCESSFULLY")

print(f"Comparison CSV: {comparison_path}")
print(f"Privacy CSV:    {privacy_path}")
print(f"Figures folder: {FIGURES}")
