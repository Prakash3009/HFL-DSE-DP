import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.services.data_loader import (
    get_dataset_summary,
    load_dataset,
    load_feature_names,
)

st.set_page_config(
    page_title="Datasets | HFL-DSE-DP",
    page_icon="📊",
    layout="wide",
)

st.title("Dataset Explorer")
st.write(
    "Explore the real processed UAV-IDS and UNSW-NB15 datasets "
    "used by the federated binary-classification experiments."
)

domain_choice = st.selectbox(
    "Select dataset",
    options=["UAV-IDS", "UNSW-NB15"],
)

domain = "uav" if domain_choice == "UAV-IDS" else "unsw"

summary = get_dataset_summary(domain)

st.header("Dataset split summary")

summary_rows = []

for split, values in summary.items():
    summary_rows.append({
        "Split": split.capitalize(),
        "Records": values["records"],
        "Features": values["features"],
        "Normal": values["normal"],
        "Attacks": values["attacks"],
    })

st.dataframe(
    pd.DataFrame(summary_rows),
    hide_index=True,
    use_container_width=True,
)

selected_split = st.selectbox(
    "Inspect label distribution",
    options=["train", "validation", "test"],
)

data = load_dataset(domain, selected_split)

col1, col2, col3 = st.columns(3)
col1.metric("Records", f'{data["X"].shape[0]:,}')
col2.metric("Features", data["X"].shape[1])
col3.metric("Attack proportion", f'{data["y"].mean() * 100:.2f}%')

st.divider()

st.subheader("Binary classification distribution")

binary_counts = pd.DataFrame({
    "Class": ["Normal", "Attack"],
    "Records": [
        int((data["y"] == 0).sum()),
        int((data["y"] == 1).sum()),
    ],
})

fig_binary = px.bar(
    binary_counts,
    x="Class",
    y="Records",
    color="Class",
    text="Records",
    title=f"{domain_choice} — {selected_split.capitalize()} split",
)
fig_binary.update_layout(showlegend=False)

st.plotly_chart(fig_binary, use_container_width=True)

st.subheader("Original attack-category distribution")

category_counts = (
    pd.Series(data["original_labels"])
    .value_counts()
    .rename_axis("Category")
    .reset_index(name="Records")
)

fig_categories = px.bar(
    category_counts,
    x="Category",
    y="Records",
    text="Records",
    title="Original dataset labels",
)
fig_categories.update_layout(xaxis_tickangle=-35)

st.plotly_chart(fig_categories, use_container_width=True)

st.divider()

st.subheader("Feature information")

feature_names = load_feature_names(domain)

st.write(f"**Model input dimensions:** {len(feature_names)}")

with st.expander("View model feature names"):
    st.dataframe(
        pd.DataFrame({"Feature": feature_names}),
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Preview processed records"):
    preview_count = st.slider(
        "Number of records",
        min_value=5,
        max_value=50,
        value=10,
    )

    preview = pd.DataFrame(
        data["X"][:preview_count],
        columns=feature_names,
    )
    preview["binary_label"] = data["y"][:preview_count]
    preview["original_label"] = data["original_labels"][:preview_count]

    st.dataframe(preview, use_container_width=True)

st.caption(
    "The displayed records are already transformed using the saved "
    "training-fitted preprocessing pipeline. This page does not refit "
    "the preprocessor."
)
