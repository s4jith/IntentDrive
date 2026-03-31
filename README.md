# BEV Vulnerable Road User Trajectory Prediction

An end-to-end Bird's-Eye-View (BEV) trajectory forecasting system for vulnerable road users (VRUs). The system connects camera-based perception, lightweight multi-agent tracking, and a transformer-based social forecasting model through a structured FastAPI backend and a React visualization dashboard.

---

## Project Overview

This project addresses the problem of safety-critical motion forecasting for pedestrians, cyclists, and motorcyclists in autonomous driving scenarios. Given a short observed history of agent positions, the system predicts K=3 multimodal 6-second future trajectories along with per-mode probability scores.

The full pipeline includes:

- Object detection and optional keypoint extraction from camera frames
- Image-to-BEV coordinate conversion using camera intrinsics and scene geometry
- Temporal tracking to build per-agent motion histories
- Social context construction from neighboring agent tracks within a 50-meter radius
- Transformer-based trajectory forecasting with goal-conditioned multimodal decoding
- Optional LiDAR and radar fusion for improved short-term kinematic estimation
- FastAPI backend serving inference, live frame access, and health endpoints
- React + TypeScript dashboard for BEV scene visualization, trajectory rendering, and sensor overlay

---

## Model Architecture

### Base Model: TrajectoryTransformer

The base model (`backend/app/ml/model.py`) is a goal-conditioned multimodal trajectory forecaster operating on 4-step observed windows with 7 features per timestep: x, y, velocity_x, velocity_y, speed, heading_sin, heading_cos.

**Components:**

| Component | Description |
|---|---|
| Feature Embedding | Linear projection from 7 input features to d_model=64 |
| Positional Encoding | Sinusoidal positional encoding over the observed sequence |
| Temporal Encoder | 2-layer TransformerEncoder, 4 attention heads, feedforward dim 256 |
| Social Attention | Multi-head attention pooling over encoded neighbor agent representations, 4 heads |
| Goal Head | MLP that predicts K=3 distinct 2D endpoint goals from the combined context |
| Trajectory Head | MLP conditioned on the base context concatenated with each predicted goal; outputs a 12-step path per mode |
| Probability Head | Linear layer with softmax producing per-mode confidence scores |

**Forward pass summary:**

1. Each agent's 4-step observed sequence is embedded and positionally encoded.
2. The TransformerEncoder produces a context vector from the final timestep.
3. Each neighboring agent within the social radius is independently encoded and pooled into a social context vector via cross-attention.
4. Target and social context vectors are concatenated to form a 128-dimensional hidden state.
5. K=3 goal endpoints are predicted from the hidden state.
6. Each goal is concatenated back to the hidden state to condition the trajectory decoder, producing 3 independent 12-step trajectory modes.
7. Mode probabilities are produced via a linear + softmax head.

**Loss function:**

The training objective combines four terms:

- Best-of-K trajectory loss (minimum L2 error over K modes)
- Goal loss (L2 distance from the best-mode predicted endpoint to ground truth endpoint)
- Probability cross-entropy loss (supervising the mode probability head)
- Diversity regularization loss (penalizes mode collapse via exponential repulsion between mode trajectories)

### Fusion Model: TrajectoryTransformerFusion

The fusion variant (`backend/app/ml/model_fusion.py`) extends the base model with a sensor-aware input branch. In addition to the standard 7-feature kinematic input, per-timestep fusion features of dimension 3 are accepted: normalized LiDAR point count, normalized radar point count, and composite sensor strength. These fusion features are projected to d_model=64 via a separate linear layer, added to the base kinematic embedding, and normalized with LayerNorm before entering the shared TransformerEncoder. The fusion model supports loading weights from a base model checkpoint for initialization.

---

## Dataset

**Source:** nuScenes mini split (annotations loaded via nuScenes JSON tables)

**Target classes:** pedestrian, bicycle, motorcycle

**Windowing:**
- Takes a 2-second history of motion as input.
- Outputs K=3 multimodal trajectory predictions over a 3-second prediction horizon, each with an associated probability score.

**Input features per observed step:**
- x, y position (BEV meters)
- velocity_x, velocity_y (m/s)
- speed (m/s)
- heading_sin, heading_cos (unit circle encoding)

**Social context radius:** 50 meters

**Data augmentation (training split only):** random rotation, horizontal reflection, coordinate noise injection

**Split protocol:** deterministic 80/20 train/validation split (seed 42)

---

## Performance

### Base Model (best_social_model.pth)

| Metric | Value |
|---|---|
| Validation trajectories | 468 |
| minADE (K=3) | 0.55 m |
| minFDE (K=3) | 1.09 m |
| Miss Rate (>2.0 m) | 13.0 % |

**Constant-velocity baseline (same split):**

| Metric | Value |
|---|---|
| minADE (K=3) | 0.65 m |
| minFDE (K=3) | 1.35 m |
| Miss Rate (>2.0 m) | 19.9 % |

### Fusion Model (best_social_model_fusion.pth)

| Metric | Value |
|---|---|
| Validation trajectories | 468 |
| minADE (K=3) | 0.54 m |
| minFDE (K=3) | 1.07 m |
| Miss Rate (>2.0 m) | 12.4 % |

### Runtime Benchmark

| Stage | Latency |
|---|---|
| Detection model (per frame) | 52.7 ms |
| Sensor fusion lookup | 16 ms |
| Transformer prediction head | 18.6 ms |
|Full 2-frame loop (approximate) | 137.3 ms |
| Equivalent throughput | ~7.28 FPS |

---

## Repository Structure

```
bev/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/          # FastAPI route modules: health, live, predict
│   │   ├── core/                # Serialization and shared utilities
│   │   ├── ml/
│   │   │   ├── model.py         # TrajectoryTransformer (base)
│   │   │   ├── model_fusion.py  # TrajectoryTransformerFusion (sensor-aware)
│   │   │   ├── inference.py     # Inference pipeline
│   │   │   └── sensor_fusion.py # LiDAR/radar feature extraction
│   │   ├── services/            # Business logic layer
│   │   └── main.py              # FastAPI application factory
│   └── scripts/
│       ├── data/                # Dataset construction from nuScenes images
│       ├── training/
│       │   ├── train.py                  # Base model training
│       │   ├── train_phase2_fusion.py    # Fusion model training
│       │   └── finetune_cv_pipeline.py   # CV-synced fine-tuning
│       ├── evaluation/
│       │   ├── evaluate.py               # Base model evaluation
│       │   ├── evaluate_phase2_fusion.py # Fusion model evaluation
│       │   └── benchmark_perf.py         # Runtime latency benchmarking
│       └── tools/
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main dashboard component
│   │   ├── types.ts             # TypeScript type definitions
│   │   ├── api/                 # API client layer
│   │   ├── components/          # UI components
│   │   └── styles.css           # Global styles
│   ├── package.json
│   └── vite.config.ts
├── models/
│   ├── best_social_model.pth          # Trained base model checkpoint
│   ├── best_social_model_fusion.pth   # Trained fusion model checkpoint
│   ├── best_cv_synced_model.pth       # CV-pipeline fine-tuned checkpoint
│   └── best_social_model_fusion_smoke.pth
├── extracted_training_data.json       # Preprocessed nuScenes trajectory data
└── log/                               # Training logs
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or later
- Node.js 18 or later and npm
- nuScenes dataset (mini split) if retraining from scratch; pretrained checkpoints are included in `models/`

### Backend

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

```


### Frontend

```bash
cd frontend
npm install
```

---

## How to Run

### 1. Start the Backend API Server

From the repository root with the virtual environment active:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

Interactive API documentation is available at `http://localhost:8000/docs`.

### 2. Start the Frontend Dashboard

```bash
cd frontend
npm run dev
```

The dashboard will be available at `http://localhost:5173`.

### 3. Train the Base Model

Ensure `extracted_training_data.json` is present at the repository root (or rebuild it using `backend/scripts/data/build_dataset_from_images.py`).

```bash
python -m backend.scripts.training.train
```

Checkpoints are saved to `models/best_social_model.pth`. Training logs are written to `log/`.

### 4. Train the Fusion Model

```bash
python -m backend.scripts.training.train_phase2_fusion
```

The fusion model initializes from the base checkpoint and trains with sensor features. The output checkpoint is saved to `models/best_social_model_fusion.pth`.

### 5. Evaluate Models

```bash
# Base model
python -m backend.scripts.evaluation.evaluate

# Fusion model
python -m backend.scripts.evaluation.evaluate_phase2_fusion

# Runtime latency benchmark
python -m backend.scripts.evaluation.benchmark_perf
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Service health check |
| GET | `/api/live/frame` | Retrieve the latest processed camera frame |
| POST | `/api/predict` | Run trajectory prediction on a submitted scene |

The prediction endpoint returns a structured payload including multimodal trajectories, per-mode probabilities, agent detections, sensor summary, and scene geometry.

---




### Validation Metrics Output

```
Train Loss: 2.1834
ADE: 0.5491, FDE: 1.0873
Current Learning Rate: 0.0005

###Model Efficiency
Transformer parameters: ~146K (very lightweight)
Base: ~0.6 MB
Fusion: ~0.6 MB
```

### Output

![Output visualization](public/output.jpeg)
![Output visualization2](public/output2.png)
![Output visualization3](public/output4.png)
![Output visualization4](public/output3.png)



## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.