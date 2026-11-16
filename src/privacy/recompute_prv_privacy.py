
import math
from pathlib import Path

import numpy as np
import pandas as pd
from opacus.accountants import PRVAccountant


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(".")
CLIENTS_DIR = PROJECT_ROOT / "data/processed/clients/noniid"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"

BATCH_SIZE = 256
DELTA = 1e-5

NOISE_MULTIPLIERS = [0.5, 1.0, 1.5]
DOMAINS = ["uav", "unsw"]
CLIENT_IDS = range(1, 6)


# ============================================================
# RECONSTRUCT CLIENT SAMPLING RATES
# ============================================================

def get_client_sampling_rate(domain, client_id):

    path = (
        CLIENTS_DIR
        / domain
        / f"client_{client_id}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path)
    n_samples = len(data["X"])

    original_loader_batches = math.ceil(
        n_samples / BATCH_SIZE
    )

    sample_rate = 1 / original_loader_batches

    return n_samples, original_loader_batches, sample_rate


# ============================================================
# RECONSTRUCT PRV ACCOUNTING
# ============================================================

def recompute_experiment(noise_multiplier):

    noise_label = str(noise_multiplier)

    source_file = (
        RESULTS_DIR
        / f"hfl_dp_privacy_history_noise_{noise_label}.csv"
    )

    if not source_file.exists():
        raise FileNotFoundError(source_file)

    history_df = pd.read_csv(source_file)

    required_columns = {
        "round",
        "domain",
        "client_id",
        "steps_this_round",
        "noise_multiplier",
        "delta"
    }

    missing = required_columns - set(history_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {source_file}: {missing}"
        )

    # Validate the experiment configuration.
    if not np.allclose(
        history_df["noise_multiplier"],
        noise_multiplier
    ):
        raise ValueError(
            f"Noise multiplier mismatch in {source_file}"
        )

    if not np.allclose(history_df["delta"], DELTA):
        raise ValueError(
            f"Delta mismatch in {source_file}"
        )

    if len(history_df) != 20:
        raise ValueError(
            f"Expected 20 history rows, found {len(history_df)}"
        )

    # One persistent accountant for each client.
    accountants = {
        (domain, client_id): PRVAccountant()
        for domain in DOMAINS
        for client_id in CLIENT_IDS
    }

    reconstructed_history = []
    final_privacy = []

    # Process clients and rounds in chronological order.
    history_df = history_df.sort_values(
        ["domain", "client_id", "round"]
    )

    for _, row in history_df.iterrows():

        domain = row["domain"]
        client_id = int(row["client_id"])
        round_number = int(row["round"])

        steps = int(row["steps_this_round"])

        key = (domain, client_id)

        if key not in accountants:
            raise ValueError(f"Unexpected client: {key}")

        n_samples, loader_batches, sample_rate = (
            get_client_sampling_rate(domain, client_id)
        )

        accountant = accountants[key]

        # Add this round's DP-SGD events.
        for _ in range(steps):
            accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate
            )

        cumulative_epsilon = accountant.get_epsilon(
            delta=DELTA
        )

        reconstructed_history.append({
            "round": round_number,
            "domain": domain,
            "client_id": client_id,
            "n_samples": n_samples,
            "loader_batches": loader_batches,
            "sample_rate": sample_rate,
            "steps_this_round": steps,
            "cumulative_steps": sum(
                int(r["steps_this_round"])
                for r in reconstructed_history
                if r["domain"] == domain
                and r["client_id"] == client_id
            ) + steps,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": float(row["max_grad_norm"]),
            "delta": DELTA,
            "previous_epsilon": float(row["epsilon"]),
            "prv_cumulative_epsilon": cumulative_epsilon
        })

    # Final cumulative epsilon for every client.
    for (domain, client_id), accountant in accountants.items():

        epsilon = accountant.get_epsilon(delta=DELTA)

        n_samples, loader_batches, sample_rate = (
            get_client_sampling_rate(domain, client_id)
        )

        final_privacy.append({
            "domain": domain,
            "client_id": client_id,
            "n_samples": n_samples,
            "sample_rate": sample_rate,
            "epsilon": epsilon,
            "delta": DELTA,
            "noise_multiplier": noise_multiplier
        })

    history_output = pd.DataFrame(reconstructed_history)
    privacy_output = pd.DataFrame(final_privacy)

    # Save to NEW files. Original results remain untouched.
    history_path = (
        RESULTS_DIR
        / f"hfl_dp_privacy_history_prv_noise_{noise_label}.csv"
    )

    privacy_path = (
        RESULTS_DIR
        / f"hfl_dp_final_privacy_prv_noise_{noise_label}.csv"
    )

    history_output.to_csv(history_path, index=False)
    privacy_output.to_csv(privacy_path, index=False)

    maximum_epsilon = privacy_output["epsilon"].max()

    print("\n" + "=" * 65)
    print(f"PRV RECONSTRUCTION: NOISE {noise_multiplier}")
    print("=" * 65)

    print(
        privacy_output[
            ["domain", "client_id", "sample_rate", "epsilon"]
        ].to_string(index=False)
    )

    print(f"\nMaximum client epsilon: {maximum_epsilon:.6f}")

    print("\nSaved:")
    print(history_path)
    print(privacy_path)

    return privacy_output


# ============================================================
# MAIN
# ============================================================

def main():

    print("PRV PRIVACY RECONSTRUCTION")
    print("Partition: noniid")
    print("Batch size:", BATCH_SIZE)
    print("Delta:", DELTA)

    all_results = []

    for noise_multiplier in NOISE_MULTIPLIERS:

        result = recompute_experiment(noise_multiplier)

        all_results.append(result)

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    summary = (
        combined
        .groupby("noise_multiplier")
        .agg(
            max_client_epsilon=("epsilon", "max"),
            mean_client_epsilon=("epsilon", "mean")
        )
        .reset_index()
    )

    summary_path = (
        RESULTS_DIR / "hfl_dp_prv_privacy_comparison.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL PRV PRIVACY COMPARISON")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nSummary saved to:", summary_path)


if __name__ == "__main__":
    main()
