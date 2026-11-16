# HFL-DSE-DP

## Feature-Heterogeneous Federated Learning for Intrusion Detection

An academic research project investigating federated intrusion detection across UAV and enterprise network domains.

### Key Features
- UAV-IDS and UNSW-NB15 datasets.
- Domain-specific encoders and shared classifier.
- IID and Non-IID federated learning.
- Sample-weighted federated averaging.
- Experimental differential privacy using Opacus.
- RDP and PRV privacy accounting.
- Binary intrusion prediction.
- Ten-architecture benchmark.
- Interactive Streamlit dashboard.

### Architecture
- UAV input: 15 features.
- UNSW input: 191 transformed features.
- Embedding dimension: 32.
- 5 clients per domain, 10 total.
- Dirichlet alpha: 0.3.
- 2 global rounds, 1 local epoch per round.

### Differential Privacy
Noise multipliers: 0.5, 1.0, and 1.5.

Accounting methods: RDP and PRV, delta = 1e-5.

These are experimental privacy results, not a verified end-to-end production privacy guarantee. The recorded experiments used secure_mode=False and non-cryptographic random seeding.

### Research Benchmark
Ten architectures, two datasets, and three seeds (42, 123, 2026), producing 60 runs.

Architectures: MLP, CNN1D, ResNet1D, GRU, LSTM, BiLSTM, CNN-GRU, CNN-LSTM, TCN, Tiny Transformer.

This benchmark is separate from the federated experiments and is non-federated and non-private.

### Dashboard
Run:

    streamlit run dashboard/Home.py

Pages: Home, Datasets, Federated Clients, Training, Privacy, Evaluation, Prediction, Reports, Research Benchmark.

Training shows saved experiment histories, not live training. Prediction uses uploaded dataset records, not live network traffic.

### Project Structure
- dashboard/ — Streamlit application.
- data/ — raw and processed datasets.
- experiments/ — configurations and results.
- reports/ — figures and tables.
- src/data/ — preprocessing.
- src/federated/ — federated learning.
- src/models/ — neural network models.
- src/privacy/ — privacy experiments.
- src/evaluation/ — evaluation.
- src/benchmarks/ — architecture benchmark.

### Installation
Python 3.12 is recommended.

1. Obtain the authorized UAV-IDS and UNSW-NB15 datasets.
2. Prepare processed datasets, checkpoints, and preprocessing artifacts.
3. Install dependencies using requirements.txt.
4. Launch the dashboard.

### Limitations
- Two global rounds do not establish convergence.
- Secure randomness was not enabled in recorded DP experiments.
- Privacy accounting depends on implementation and assumptions.
- Classification thresholds affect false-positive and false-negative rates.
- The architecture benchmark is not federated or differentially private.
- Dataset duplication and ambiguous labels should be considered.

### Technologies
Python, PyTorch, Opacus, NumPy, Pandas, SciPy, Scikit-learn, Streamlit, Plotly, Matplotlib, Joblib.

### Research Scope
This is an academic research prototype, not a production network-security deployment.
