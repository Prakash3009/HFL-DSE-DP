import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.services.data_loader import (
    get_dataset_summary,
    get_project_counts,
)

st.set_page_config(
    page_title="HFL-DSE-DP | Overview",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ HFL-DSE-DP")
st.subheader("Privacy-Preserving Federated Intrusion Detection")

st.markdown(
    """
    A research dashboard for **Feature-Heterogeneous Federated Learning**
    across UAV networks and enterprise intrusion-detection data.

    Each domain has its own feature encoder. Both encoders produce a shared
    32-dimensional representation that is processed by a common classifier.
    """
)

st.divider()

counts = get_project_counts()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Datasets / domains", counts["domains"])
col2.metric("Federated clients", counts["total_clients"])
col3.metric("Clients per domain", counts["clients_per_domain"])
col4.metric("Recorded global rounds", counts["federated_rounds"])

st.divider()

st.header("System architecture")

left, middle, right = st.columns([1, 1, 1])

with left:
    st.markdown("### UAV domain")
    st.info("15 input features")
    st.markdown("Domain-specific encoder")
    st.markdown("**32-dimensional embedding**")

with middle:
    st.markdown("### Enterprise domain")
    st.info("191 input features")
    st.markdown("Domain-specific encoder")
    st.markdown("**32-dimensional embedding**")

with right:
    st.markdown("### Shared classifier")
    st.success("Common 32-dimensional input")
    st.markdown("32 → 16 → 1")
    st.markdown("**Normal / Attack**")

st.caption(
    "Conceptual architecture: the two domain encoders are trained with "
    "their respective client groups; the shared classifier is aggregated "
    "across the federated clients."
)

st.divider()

st.header("Dataset overview")

uav = get_dataset_summary("uav")
unsw = get_dataset_summary("unsw")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### UAV-IDS")
    st.metric("Training records", f'{uav["train"]["records"]:,}')
    st.metric("Input features", uav["train"]["features"])
    st.write("Original categories:", len(uav["train"]["categories"]))

with col2:
    st.markdown("### UNSW-NB15")
    st.metric("Training records", f'{unsw["train"]["records"]:,}')
    st.metric("Transformed input features", unsw["train"]["features"])
    st.write("Original categories:", len(unsw["train"]["categories"]))

st.divider()

st.header("Research scope and interpretation")

st.warning(
    """
    **Important research distinction**

    The HFL system performs binary normal-versus-attack classification.
    The separate architecture benchmark evaluates 10 architectures on
    multiclass labels. Those benchmark results are not the output of the
    federated model and must be interpreted separately.
    """
)

st.info(
    """
    **Privacy interpretation:** The historical DP experiments use
    record-level DP-SGD with client-specific accounting. The recorded
    epsilon values depend on the accountant used. The historical runs used
    non-secure randomness (`secure_mode=False`), so this dashboard will
    not present them as a fully hardened production privacy deployment.
    """
)

st.caption("Research prototype • Local dataset-driven dashboard")
