import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "experiments/results/benchmarks"

st.set_page_config(
    page_title="Research Benchmark | HFL-DSE-DP",
    page_icon="🏆",
    layout="wide",
)

st.title("Deep Learning Architecture Benchmark")
st.caption(
    "Independent non-federated benchmark: 10 architectures, "
    "2 datasets, and 3 random seeds per architecture-dataset combination."
)


@st.cache_data
def load_summary():
    return pd.read_csv(BENCH / "mean_std_by_model_dataset.csv")


@st.cache_data
def load_runs():
    return pd.read_csv(BENCH / "overall_results.csv")


summary = load_summary()
runs = load_runs()

if len(runs) != 60 or summary["n_seeds"].min() != 3:
    st.warning(
        "The benchmark files do not match the expected 60-run, "
        "three-seed structure. Review the source result files."
    )

dataset_options = {
    "UAV-IDS": "uav",
    "UNSW-NB15": "unsw",
}

dataset_label = st.selectbox("Select benchmark dataset", list(dataset_options))
dataset = dataset_options[dataset_label]

data = summary[summary["dataset"] == dataset].copy()
run_data = runs[runs["dataset"] == dataset].copy()

st.divider()

c1, c2, c3 = st.columns(3)
c1.metric("Architectures", data["model"].nunique())
c2.metric("Completed runs", len(run_data))
c3.metric("Random seeds", int(data["n_seeds"].min()))

st.info(
    "This benchmark is separate from the binary federated intrusion-detection "
    "experiments. It evaluates multiclass architectures independently."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Rankings",
    "Metric comparison",
    "Efficiency",
    "Individual runs",
])

with tab1:
    st.header("Architecture rankings")

    metric_options = {
        "Macro-F1": ("macro_f1_mean", "macro_f1_sample_std"),
        "Accuracy": ("accuracy_mean", "accuracy_sample_std"),
        "Weighted F1": ("f1_weighted_mean", "f1_weighted_sample_std"),
        "Multiclass MCC": ("mcc_multiclass_mean", "mcc_multiclass_sample_std"),
        "ROC-AUC (OvR macro)": (
            "roc_auc_ovr_macro_mean",
            "roc_auc_ovr_macro_sample_std",
        ),
        "PR-AUC (OvR macro)": (
            "pr_auc_ovr_macro_trapezoidal_mean",
            "pr_auc_ovr_macro_trapezoidal_sample_std",
        ),
    }

    selected_metric = st.selectbox("Ranking metric", list(metric_options))
    mean_col, std_col = metric_options[selected_metric]

    ranking = data[
        ["model", "n_seeds", mean_col, std_col]
    ].sort_values(mean_col, ascending=False).reset_index(drop=True)

    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    ranking = ranking.rename(columns={
        mean_col: "Mean",
        std_col: "Sample SD",
    })

    st.dataframe(
        ranking.style.format({"Mean": "{:.4f}", "Sample SD": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        ranking,
        x="model",
        y="Mean",
        error_y="Sample SD",
        title=f"{selected_metric}: mean ± sample SD across seeds",
        labels={"model": "Architecture", "Mean": selected_metric},
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Metric comparison")

    metric_columns = {
        "Accuracy": "accuracy_mean",
        "Weighted Precision": "precision_weighted_mean",
        "Weighted Recall": "recall_weighted_mean",
        "Weighted F1": "f1_weighted_mean",
        "Macro-F1": "macro_f1_mean",
        "MCC": "mcc_multiclass_mean",
        "ROC-AUC": "roc_auc_ovr_macro_mean",
        "PR-AUC": "pr_auc_ovr_macro_trapezoidal_mean",
    }

    selected_metrics = st.multiselect(
        "Metrics to compare",
        list(metric_columns),
        default=["Macro-F1", "MCC", "ROC-AUC"],
    )

    if selected_metrics:
        plot_df = data[["model"] + [
            metric_columns[m] for m in selected_metrics
        ]].melt(
            id_vars="model",
            var_name="metric",
            value_name="score",
        )

        reverse = {v: k for k, v in metric_columns.items()}
        plot_df["metric"] = plot_df["metric"].map(reverse)

        fig = px.bar(
            plot_df,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            title="Mean performance across random seeds",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one metric.")

with tab3:
    st.header("Computational efficiency")

    efficiency_metric = st.selectbox(
        "Efficiency metric",
        [
            "fit_seconds_mean",
            "inference_ms_per_record_mean",
            "parameter_count",
        ],
        format_func=lambda x: {
            "fit_seconds_mean": "Mean fit time (seconds)",
            "inference_ms_per_record_mean": "Mean inference latency (ms/record)",
            "parameter_count": "Parameter count",
        }[x],
    )

    efficiency_labels = {
        "fit_seconds_mean": "Mean fit time (seconds)",
        "inference_ms_per_record_mean": "Inference latency (ms/record)",
        "parameter_count": "Parameter count",
    }

    fig = px.bar(
        data.sort_values(efficiency_metric),
        x="model",
        y=efficiency_metric,
        title=efficiency_labels[efficiency_metric],
        labels={
            "model": "Architecture",
            efficiency_metric: efficiency_labels[efficiency_metric],
        },
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Efficiency and performance table")

    efficiency_table = data[
        [
            "model",
            "macro_f1_mean",
            "macro_f1_sample_std",
            "fit_seconds_mean",
            "inference_ms_per_record_mean",
        ]
    ].sort_values("macro_f1_mean", ascending=False)

    st.dataframe(efficiency_table, use_container_width=True, hide_index=True)

with tab4:
    st.header("Per-seed benchmark results")

    available_seeds = sorted(run_data["seed"].unique().tolist())
    selected_seed = st.selectbox(
        "Random seed",
        ["All seeds"] + available_seeds,
    )

    displayed = run_data.copy()

    if selected_seed != "All seeds":
        displayed = displayed[displayed["seed"] == selected_seed]

    st.dataframe(displayed, use_container_width=True, hide_index=True)

    st.download_button(
        "Download displayed run results",
        data=displayed.to_csv(index=False).encode("utf-8"),
        file_name=f"{dataset}_benchmark_runs.csv",
        mime="text/csv",
    )

st.divider()

st.subheader("Download complete benchmark summary")

summary_download = data.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download mean and standard deviation results",
    data=summary_download,
    file_name=f"{dataset}_benchmark_mean_std.csv",
    mime="text/csv",
)

st.caption(
    "Reported values are means and sample standard deviations across three seeds. "
    "The benchmark treats feature vectors as model inputs; it does not establish "
    "that the underlying records are temporal sequences."
)
