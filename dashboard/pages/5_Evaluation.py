from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"

st.set_page_config(
    page_title="Model Evaluation | HFL-DSE-DP",
    page_icon="📊",
    layout="wide",
)

st.title("Model Evaluation")
st.caption("Test-set evaluation of federated intrusion-detection models.")

st.warning(
    "These are historical test-set results. Thresholds were selected using "
    "validation data, but test results have already been evaluated during the "
    "research experiments. Treat this as a retrospective evaluation, not a "
    "new untouched test."
)

# ---------------------------------------------------------
# Experiment definitions: use explicit, verified files
# ---------------------------------------------------------

EXPERIMENTS = {
    "Non-private — IID": "hfl_test_metrics_iid_selected_thresholds.csv",
    "Non-private — Non-IID": "hfl_test_metrics_noniid_selected_thresholds.csv",
    "DP — Noise 0.5": "hfl_dp_test_metrics_noise_0.5.csv",
    "DP — Noise 1.0": "hfl_dp_test_metrics_selected_thresholds.csv",
    "DP — Noise 1.5": "hfl_dp_test_metrics_noise_1.5.csv",
}

@st.cache_data
def load_metrics(filename):
    path = RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "domain", "threshold", "accuracy", "precision",
        "attack_recall", "macro_f1", "mcc", "fpr",
        "pr_auc", "roc_auc", "tn", "fp", "fn", "tp",
    }

    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected columns in {filename}")

    df["domain"] = df["domain"].astype(str).str.lower()
    return df


@st.cache_data
def load_all_experiments():
    frames = []

    for label, filename in EXPERIMENTS.items():
        frame = load_metrics(filename).copy()
        frame["Experiment"] = label
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------
# User selections
# ---------------------------------------------------------

experiment = st.selectbox(
    "Select experiment",
    list(EXPERIMENTS.keys()),
)

domain_choice = st.radio(
    "Select dataset",
    ["UAV-IDS", "UNSW-NB15"],
    horizontal=True,
)

domain = "uav" if domain_choice == "UAV-IDS" else "unsw"

try:
    selected_df = load_metrics(EXPERIMENTS[experiment])
    all_df = load_all_experiments()
except Exception as exc:
    st.error(f"Could not load evaluation results: {exc}")
    st.stop()

selected = selected_df[selected_df["domain"] == domain]

if selected.empty:
    st.error("No evaluation record exists for this experiment and domain.")
    st.stop()

row = selected.iloc[0]

# ---------------------------------------------------------
# Main metrics
# ---------------------------------------------------------

st.subheader(f"{experiment} | {domain_choice}")

a, b, c, d = st.columns(4)

a.metric("Accuracy", f"{row['accuracy']:.2%}")
b.metric("Precision", f"{row['precision']:.2%}")
c.metric("Attack recall", f"{row['attack_recall']:.2%}")
d.metric("Macro-F1", f"{row['macro_f1']:.4f}")

a, b, c, d = st.columns(4)

a.metric("MCC", f"{row['mcc']:.4f}")
b.metric("False positive rate", f"{row['fpr']:.2%}")
c.metric("ROC-AUC", f"{row['roc_auc']:.4f}")
d.metric("PR-AUC", f"{row['pr_auc']:.4f}")

st.write(
    f"**Decision threshold:** {row['threshold']:.2f}  \n"
    f"**Source CSV:** `{EXPERIMENTS[experiment]}`"
)

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

st.subheader("Confusion matrix")

tn = int(row["tn"])
fp = int(row["fp"])
fn = int(row["fn"])
tp = int(row["tp"])

matrix = [
    [tn, fp],
    [fn, tp],
]

fig = go.Figure(
    data=go.Heatmap(
        z=matrix,
        x=["Predicted Normal", "Predicted Attack"],
        y=["Actual Normal", "Actual Attack"],
        text=[[f"{v:,}" for v in line] for line in matrix],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
    )
)

fig.update_layout(
    title="Test-set confusion matrix",
    xaxis_title="Predicted class",
    yaxis_title="Actual class",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"TN: {tn:,} | FP: {fp:,} | FN: {fn:,} | TP: {tp:,}"
)

# ---------------------------------------------------------
# Experiment comparison
# ---------------------------------------------------------

st.divider()
st.subheader("Compare all federated experiments")

comparison = all_df[all_df["domain"] == domain].copy()

comparison["Domain"] = domain_choice

comparison_metrics = [
    "accuracy",
    "precision",
    "attack_recall",
    "macro_f1",
    "mcc",
    "fpr",
]

comparison_table = comparison[
    ["Experiment", "threshold"] + comparison_metrics
].copy()

comparison_table = comparison_table.rename(columns={
    "threshold": "Threshold",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "attack_recall": "Attack recall",
    "macro_f1": "Macro-F1",
    "mcc": "MCC",
    "fpr": "FPR",
})

st.dataframe(
    comparison_table.round(4),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------
# Macro-F1 and FPR comparison
# ---------------------------------------------------------

left, right = st.columns(2)

with left:
    fig = px.bar(
        comparison,
        x="Experiment",
        y="macro_f1",
        title="Macro-F1 comparison",
        text="macro_f1",
    )
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )
    fig.update_layout(
        xaxis_title="Experiment",
        yaxis_title="Macro-F1",
        xaxis_tickangle=-30,
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        comparison,
        x="Experiment",
        y="fpr",
        title="False positive rate comparison",
        text="fpr",
    )
    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
    )
    fig.update_layout(
        xaxis_title="Experiment",
        yaxis_title="False positive rate",
        xaxis_tickangle=-30,
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Privacy–utility comparison
# ---------------------------------------------------------

st.subheader("Privacy–utility comparison")

dp_comparison = comparison[
    comparison["Experiment"].str.startswith("DP")
].copy()

if not dp_comparison.empty:
    dp_comparison["Noise multiplier"] = (
        dp_comparison["Experiment"].str.extract(r"([\d.]+)$")[0].astype(float)
    )

    fig = px.scatter(
        dp_comparison,
        x="macro_f1",
        y="fpr",
        color="Experiment",
        size="attack_recall",
        hover_data=["threshold", "accuracy", "mcc"],
        title="DP model utility: Macro-F1 versus false positive rate",
        labels={
            "macro_f1": "Macro-F1",
            "fpr": "False positive rate",
            "attack_recall": "Attack recall",
        },
    )

    fig.update_layout(
        xaxis_title="Macro-F1 (higher is better)",
        yaxis_title="False positive rate (lower is better)",
        yaxis_tickformat=".0%",
    )

    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "The privacy accountant's epsilon values are displayed on the separate "
    "Privacy page. This chart compares model utility, not formal privacy guarantees."
)

# ---------------------------------------------------------
# Interpretation
# ---------------------------------------------------------

st.divider()
st.subheader("Evaluation interpretation")

if domain == "uav":
    st.markdown(
        """
        UAV-IDS results should be examined using Macro-F1, attack recall,
        and false-positive rate together. A model can detect attacks effectively
        while still generating unnecessary alerts on normal traffic.
        """
    )
else:
    st.markdown(
        """
        UNSW-NB15 results require particular attention to false positives.
        High attack recall alone does not mean the detector is operationally
        effective if many normal records are incorrectly flagged as attacks.
        """
    )

st.warning(
    "The HFL model performs binary normal-versus-attack classification. "
    "The separate 10-architecture multiclass benchmark is not included here."
)

st.caption(
    "All values are loaded from the existing experiments/results CSV files. "
    "No metrics are simulated or recalculated."
)
