from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"

st.set_page_config(
    page_title="Federated Training | HFL-DSE-DP",
    page_icon="🧠",
    layout="wide",
)

st.title("Federated Training")
st.caption(
    "Explore recorded validation histories from the federated-learning experiments."
)

st.info(
    "These charts display saved experiment results. "
    "They do not represent a live training run."
)

# ---------------------------------------------------------
# Experiment selection
# ---------------------------------------------------------

left, right = st.columns(2)

with left:
    training_type = st.selectbox(
        "Training configuration",
        ["Non-private", "Differentially private (DP)"],
    )

with right:
    partition = st.selectbox(
        "Client partition",
        ["Non-IID", "IID"],
        disabled=(training_type == "Differentially private (DP)"),
        help="The available DP histories correspond to the non-IID experiment.",
    )

noise = None

if training_type == "Differentially private (DP)":
    noise = st.selectbox(
        "DP noise multiplier",
        [0.5, 1.0, 1.5],
        format_func=lambda x: f"{x:.1f}",
    )

domain_choice = st.radio(
    "Domain",
    ["Both domains", "UAV-IDS", "UNSW-NB15"],
    horizontal=True,
)

# ---------------------------------------------------------
# Select the correct real CSV
# ---------------------------------------------------------

if training_type == "Differentially private (DP)":
    filename = f"hfl_dp_training_history_noise_{noise:.1f}.csv"
    experiment_label = f"DP-SGD | Noise multiplier {noise:.1f}"
else:
    suffix = "iid" if partition == "IID" else "noniid"
    filename = f"hfl_training_history_{suffix}.csv"
    experiment_label = f"Non-private | {partition}"

history_path = RESULTS_DIR / filename

if not history_path.exists():
    st.error(f"Training history file not found: {history_path}")
    st.stop()

df = pd.read_csv(history_path)

required_columns = {
    "round",
    "domain",
    "validation_loss",
    "validation_accuracy",
}

if not required_columns.issubset(df.columns):
    st.error("The selected CSV does not have the expected training-history columns.")
    st.stop()

if domain_choice == "UAV-IDS":
    df = df[df["domain"] == "uav"].copy()
elif domain_choice == "UNSW-NB15":
    df = df[df["domain"] == "unsw"].copy()
else:
    df = df.copy()

if df.empty:
    st.warning("No records are available for this selection.")
    st.stop()

df["Domain"] = df["domain"].map({
    "uav": "UAV-IDS",
    "unsw": "UNSW-NB15",
})

df = df.sort_values(["domain", "round"])

# ---------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------

st.subheader("Experiment configuration")
st.write(f"**Selected experiment:** {experiment_label}")
st.write(f"**Source file:** `{filename}`")

round_count = int(df["round"].nunique())
domain_count = int(df["domain"].nunique())

latest = df.sort_values("round").groupby("domain").tail(1)

a, b, c = st.columns(3)
a.metric("Federated rounds recorded", round_count)
b.metric("Domains displayed", domain_count)
c.metric(
    "Latest mean validation accuracy",
    f"{latest['validation_accuracy'].mean() * 100:.2f}%",
)

# ---------------------------------------------------------
# Validation accuracy
# ---------------------------------------------------------

st.subheader("Validation accuracy")

accuracy_fig = px.line(
    df,
    x="round",
    y="validation_accuracy",
    color="Domain",
    markers=True,
    title="Validation accuracy across federated rounds",
)

accuracy_fig.update_layout(
    xaxis_title="Federated round",
    yaxis_title="Validation accuracy",
    yaxis_tickformat=".0%",
    xaxis=dict(tickmode="linear", dtick=1),
    legend_title="Domain",
)

st.plotly_chart(accuracy_fig, use_container_width=True)

# ---------------------------------------------------------
# Validation loss
# ---------------------------------------------------------

st.subheader("Validation loss")

loss_fig = px.line(
    df,
    x="round",
    y="validation_loss",
    color="Domain",
    markers=True,
    title="Validation loss across federated rounds",
)

loss_fig.update_layout(
    xaxis_title="Federated round",
    yaxis_title="Validation loss",
    xaxis=dict(tickmode="linear", dtick=1),
    legend_title="Domain",
)

st.plotly_chart(loss_fig, use_container_width=True)

# ---------------------------------------------------------
# Actual recorded values
# ---------------------------------------------------------

st.subheader("Recorded training history")

display_df = df[
    ["round", "Domain", "validation_loss", "validation_accuracy"]
].rename(columns={
    "round": "Federated round",
    "validation_loss": "Validation loss",
    "validation_accuracy": "Validation accuracy",
})

display_df["Validation accuracy"] = (
    display_df["Validation accuracy"] * 100
).round(3)

display_df["Validation loss"] = display_df["Validation loss"].round(6)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------
# Interpretation
# ---------------------------------------------------------

st.subheader("How to interpret these results")

st.markdown(
    """
    - **Validation accuracy:** proportion of correctly classified validation records.
    - **Validation loss:** model prediction loss on the validation data.
    - **Federated round:** one global aggregation cycle after local client training.
    - **DP noise multiplier:** controls the Gaussian noise scale applied during
      differentially private gradient updates.
    """
)

if training_type == "Differentially private (DP)":
    st.warning(
        "DP results must be interpreted alongside the privacy-accounting results. "
        "The historical training used secure_mode=False, so do not treat these "
        "charts alone as proof of a formally secured end-to-end privacy guarantee."
    )

st.caption(
    "Source: experiments/results/" + filename
)
