from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
CLIENTS_DIR = ROOT / "data" / "processed" / "clients"

st.set_page_config(page_title="Federated Clients | HFL-DSE-DP", page_icon="🧩", layout="wide")
st.title("Federated Clients")
st.caption("Explore the actual client partitions used in the federated-learning experiments.")

partition_label = st.radio(
    "Client partition strategy",
    ["IID", "Non-IID"],
    horizontal=True,
    help="Non-IID uses Dirichlet label skew with alpha = 0.3.",
)
partition = "iid" if partition_label == "IID" else "noniid"

domain_label = st.selectbox("Domain", ["UAV-IDS", "UNSW-NB15"])
domain = "uav" if domain_label == "UAV-IDS" else "unsw"

metadata_path = CLIENTS_DIR / partition / domain / "metadata.json"
if not metadata_path.exists():
    st.error(f"Metadata file not found: {metadata_path}")
    st.stop()

with metadata_path.open(encoding="utf-8") as file:
    metadata = json.load(file)

clients = metadata.get("clients", [])
if not clients:
    st.error("No client records found in metadata.")
    st.stop()

rows = []
category_rows = []

for client in clients:
    binary = client.get("binary_class_counts", {})
    normal = int(binary.get("0", 0))
    attacks = int(binary.get("1", 0))

    rows.append({
        "Client": f"Client {client['client_id']}",
        "Samples": int(client["samples"]),
        "Features": int(client["features"]),
        "Normal": normal,
        "Attacks": attacks,
        "Attack share (%)": round(100 * attacks / max(normal + attacks, 1), 2),
    })

    for category, count in client.get("original_category_counts", {}).items():
        category_rows.append({
            "Client": f"Client {client['client_id']}",
            "Original category": category,
            "Records": int(count),
        })

client_df = pd.DataFrame(rows).sort_values("Client")
category_df = pd.DataFrame(category_rows)

st.subheader(f"{partition_label} partition — {domain_label}")

if partition == "noniid":
    st.info("Dirichlet non-IID partition (alpha = 0.3): clients have different original attack-category distributions.")
else:
    st.info("IID partition: clients have approximately balanced sample counts and category distributions.")

total_samples = int(client_df["Samples"].sum())
total_normal = int(client_df["Normal"].sum())
total_attacks = int(client_df["Attacks"].sum())

a, b, c, d = st.columns(4)
a.metric("Clients", len(clients))
b.metric("Total records", f"{total_samples:,}")
c.metric("Normal records", f"{total_normal:,}")
d.metric("Attack records", f"{total_attacks:,}")

st.subheader("Client-level distribution")
st.dataframe(client_df, use_container_width=True, hide_index=True)

left, right = st.columns(2)

with left:
    fig = px.bar(
        client_df, x="Client", y="Samples", text="Samples",
        title="Records assigned to each client",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(yaxis_title="Number of records", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with right:
    binary_long = client_df.melt(
        id_vars="Client",
        value_vars=["Normal", "Attacks"],
        var_name="Binary class",
        value_name="Records",
    )
    fig = px.bar(
        binary_long, x="Client", y="Records",
        color="Binary class", barmode="stack",
        title="Normal versus attack records",
    )
    fig.update_layout(yaxis_title="Number of records", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Original attack-category distribution")
st.caption(
    "These are the original dataset categories used to construct client partitions. "
    "The HFL classifier itself uses binary normal/attack labels."
)

if not category_df.empty:
    fig = px.bar(
        category_df,
        x="Client",
        y="Records",
        color="Original category",
        barmode="stack",
        title=f"Category composition across {domain_label} clients",
    )
    fig.update_layout(yaxis_title="Number of records", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    pivot = category_df.pivot_table(
        index="Original category",
        columns="Client",
        values="Records",
        aggfunc="sum",
        fill_value=0,
    )
    st.markdown("**Category counts by client**")
    st.dataframe(pivot, use_container_width=True)

st.subheader("How this affects federated learning")

if partition == "noniid":
    st.write(
        "Clients observe different mixtures of traffic categories. "
        "This creates heterogeneous local data distributions. "
        "Client sizes also differ, so sample-weighted aggregation gives larger clients more weight."
    )
else:
    st.write(
        "Clients have broadly similar category mixtures and nearly equal sample counts. "
        "This provides a more balanced comparison against the non-IID experiment."
    )

st.caption(
    "Source: data/processed/clients/{iid|noniid}/{uav|unsw}/metadata.json. "
    "This page visualizes saved partitions; it does not create or modify clients."
)
