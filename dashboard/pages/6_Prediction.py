import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.domain_encoder import DomainEncoder
from src.models.shared_network import SharedNetwork

st.set_page_config(page_title="Prediction | HFL-DSE-DP", page_icon="🔎", layout="wide")

st.title("Intrusion Detection")
st.caption("Run inference using saved federated global models.")

MODEL_OPTIONS = {
    "Non-private — IID": {
        "checkpoint": "hfl_global_model_iid.pt",
        "uav_threshold": 0.70,
        "unsw_threshold": 0.75,
    },
    "Non-private — Non-IID": {
        "checkpoint": "hfl_global_model_noniid.pt",
        "uav_threshold": 0.85,
        "unsw_threshold": 0.90,
    },
    "Differential Privacy — Noise 0.5": {
        "checkpoint": "hfl_dp_global_model_noise_0.5.pt",
        "uav_threshold": 0.85,
        "unsw_threshold": 0.95,
    },
    "Differential Privacy — Noise 1.0": {
        "checkpoint": "hfl_dp_global_model_noise_1.0.pt",
        "uav_threshold": 0.85,
        "unsw_threshold": 0.90,
    },
    "Differential Privacy — Noise 1.5": {
        "checkpoint": "hfl_dp_global_model_noise_1.5.pt",
        "uav_threshold": 0.85,
        "unsw_threshold": 0.90,
    },
}

DOMAINS = {
    "UAV": {
        "input_dim": 15,
        "preprocessor": "data/processed/uav/preprocessor.joblib",
        "encoder_key": "uav_encoder",
    },
    "UNSW-NB15": {
        "input_dim": 191,
        "preprocessor": "data/processed/unsw/preprocessor.joblib",
        "encoder_key": "unsw_encoder",
    },
}


@st.cache_resource
def load_artifacts(domain, model_name):
    domain_info = DOMAINS[domain]
    model_info = MODEL_OPTIONS[model_name]

    preprocessor = joblib.load(ROOT / domain_info["preprocessor"])
    checkpoint = torch.load(
        ROOT / "experiments/results" / model_info["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    encoder = DomainEncoder(input_dim=domain_info["input_dim"], embedding_dim=32)
    classifier = SharedNetwork(embedding_dim=32)

    encoder.load_state_dict(checkpoint[domain_info["encoder_key"]])
    classifier.load_state_dict(checkpoint["shared"])

    encoder.eval()
    classifier.eval()

    return preprocessor, encoder, classifier


def run_predictions(raw_df, domain, model_name):
    preprocessor, encoder, classifier = load_artifacts(domain, model_name)

    required_columns = list(preprocessor.feature_names_in_)
    missing = [column for column in required_columns if column not in raw_df.columns]

    if missing:
        raise ValueError(
            "The uploaded CSV is missing required columns: " + ", ".join(missing)
        )

    input_df = raw_df[required_columns].copy()

    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    features = torch.tensor(
        np.asarray(transformed, dtype=np.float32),
        dtype=torch.float32,
    )

    with torch.no_grad():
        embeddings = encoder(features)
        logits = classifier(embeddings)
        probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    return probabilities


st.subheader("1. Configure inference")

left, right = st.columns(2)

with left:
    domain = st.selectbox("Dataset domain", list(DOMAINS.keys()))

with right:
    model_name = st.selectbox("Federated model", list(MODEL_OPTIONS.keys()))

model_info = MODEL_OPTIONS[model_name]
default_threshold = model_info[
    "uav_threshold" if domain == "UAV" else "unsw_threshold"
]

threshold = st.slider(
    "Attack classification threshold",
    min_value=0.05,
    max_value=0.99,
    value=float(default_threshold),
    step=0.01,
    help="A record is classified as an attack when its predicted probability is at least this threshold.",
)

st.caption(
    f"Selected checkpoint: `{model_info['checkpoint']}` · "
    f"Expected transformed features: {DOMAINS[domain]['input_dim']}"
)

st.divider()
st.subheader("2. Upload raw records")

st.info(
    "Upload a CSV containing the original raw input columns—not the transformed "
    "or standardized feature matrix. The saved preprocessing pipeline will "
    "transform the records automatically."
)

uploaded_file = st.file_uploader(
    f"Upload {domain} CSV",
    type=["csv"],
    help="The CSV must contain the original feature columns used during training.",
)

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)

        st.write(f"Uploaded records: **{len(raw_df):,}**")
        st.write("Input preview")
        st.dataframe(raw_df.head(5), use_container_width=True)

        if len(raw_df) == 0:
            st.warning("The uploaded CSV contains no records.")
        elif len(raw_df) > 10000:
            st.warning(
                "For this first version, inference is limited to the first "
                "10,000 rows to avoid excessive memory use."
            )
            raw_df = raw_df.head(10000).copy()

        if len(raw_df) > 0:
            if st.button("Run intrusion detection", type="primary"):
                with st.spinner("Loading saved model and running inference..."):
                    probabilities = run_predictions(raw_df, domain, model_name)

                results = raw_df.copy()
                results["attack_probability"] = probabilities
                results["prediction"] = np.where(
                    probabilities >= threshold, "Attack", "Normal"
                )

                st.session_state["prediction_results"] = results
                st.session_state["prediction_context"] = (
                    domain, model_name, threshold
                )

    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")

if "prediction_results" in st.session_state:
    results = st.session_state["prediction_results"]
    saved_domain, saved_model, saved_threshold = st.session_state["prediction_context"]

    if saved_domain == domain and saved_model == model_name and saved_threshold == threshold:
        st.divider()
        st.subheader("3. Prediction results")

        total = len(results)
        attacks = int((results["prediction"] == "Attack").sum())
        normal = total - attacks

        c1, c2, c3 = st.columns(3)
        c1.metric("Records analysed", f"{total:,}")
        c2.metric("Predicted attacks", f"{attacks:,}")
        c3.metric("Predicted normal", f"{normal:,}")

        st.write("Prediction distribution")
        st.bar_chart(
            pd.DataFrame(
                {"Records": [normal, attacks]},
                index=["Normal", "Attack"],
            )
        )

        st.write("Prediction details")
        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction results",
            data=csv,
            file_name=f"{domain.lower().replace('-', '_')}_predictions.csv",
            mime="text/csv",
        )

        st.caption(
            "These are model predictions on the uploaded records. "
            "They are not live network traffic detections."
        )
    else:
        st.info("Run inference again to update results for the current selections.")

st.divider()
st.caption(
    "Binary classification: 0 = Normal, 1 = Attack. "
    "DP model labels refer to the saved experimental checkpoints; "
    "they should not be interpreted as proof of an end-to-end privacy guarantee."
)
