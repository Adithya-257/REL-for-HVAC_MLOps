# REL-for-HVAC_MLOps

# RL-HVAC: Reinforcement Learning for Smart HVAC Control

**Course:** Reinforcement Learning (24AM6PCREL) + Machine Learning Operations (24AM6AEMLO)  
**Institute:** BMS College of Engineering, Bangalore  
**SDGs Addressed:** SDG 7 (Clean Energy) · SDG 11 (Sustainable Cities) · SDG 13 (Climate Action)

---

## Problem Statement

HVAC systems account for ~40% of building energy consumption. Traditional rule-based controllers waste energy by operating on fixed schedules regardless of actual conditions. This project trains a **Deep Q-Network (DQN)** agent to learn an optimal HVAC control policy that keeps indoor temperature within a comfort zone (20–24°C) while minimizing energy consumption.

## Architecture

```
Custom Gym Env (HVAC Simulator)
        ↓
DQN Agent (PyTorch)
  - Q-Network: 4 → 64 → 64 → 3
  - Replay Buffer (10,000 transitions)
  - Target Network (hard update every 50 episodes)
        ↓
MLflow Tracking (metrics, plots, model artifacts)
        ↓
FastAPI Inference Server  ←→  Docker + docker-compose
        ↓
GitHub Actions CI/CD (test → train → build → verify)
```

## Results

| Metric | Value |
|--------|-------|
| Training episodes | 600 |
| Episode 1 reward | -1121 |
| Best reward | -5.10 |
| Final 50-ep average | -10.96 |
| Improvement | ~200x |

**Agent behavior learned:**
- Hot afternoon (29°C indoor, 37°C outdoor) → **Cool** aggressively
- Comfortable night (19°C indoor, empty building) → **Off** (energy saving)

## Project Structure

```
REL-for-HVAC_MLOps/
├── env/                    # Custom Gymnasium HVAC environment
│   └── hvac_env.py
├── agent/                  # DQN agent components
│   ├── dqn_agent.py
│   ├── dqn_network.py
│   └── replay_buffer.py
├── api/                    # FastAPI inference server
│   └── main.py
├── tests/                  # Pytest test suite (17 tests)
│   └── test_hvac.py
├── models/                 # Saved model checkpoint
│   └── dqn_hvac.pth
├── results/                # Training plots
├── .github/workflows/      # GitHub Actions CI/CD
│   └── ci.yml
├── train.py                # Main training script
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup and Run

### 1. Local (without Docker)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Train the agent
python train.py

# Start the API
python -m uvicorn api.main:app --reload --port 8000
```

### 2. Docker (recommended)

```bash
docker-compose up --build
```

- API: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`

### 3. API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"indoor_temp": 29.0, "outdoor_temp": 37.0, "hour_of_day": 14.0, "occupancy": 1.0}'
```

Response:
```json
{
  "action_id": 1,
  "action_name": "cool",
  "action_description": "Activate cooling — reduces indoor temperature by ~2°C per hour",
  "q_values": {"off": -66.66, "cool": -21.55, "heat": -131.40},
  "comfort_status": "too hot by 5.0°C",
  "timestamp": "2026-05-10T12:21:26Z"
}
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## MLOps Pipeline

| Component | Tool | Purpose |
|-----------|------|---------|
| Experiment tracking | MLflow | Log reward, loss, epsilon per episode |
| Containerization | Docker + docker-compose | API + MLflow UI as services |
| CI/CD | GitHub Actions | Auto test → train → build on push |
| Version control | Git (main/dev branches) | Code and model versioning |

## Experiment Versions (Git Tags)

| Tag | Config | Key difference | Best Reward |
|-----|--------|---------------|-------------|
| `exp-dqn-v1` | `configs/dqn_v1.yaml` | Baseline: eps_decay=0.995, batch=64 | -5.10 |
| `exp-dqn-v2` | `configs/dqn_v2.yaml` | Tuned: eps_decay=0.990, batch=128 | TBD |


## Reproducing an Experiment

Every experiment is fully reproducible. Clone the repo and run with a named config:

```bash
git clone https://github.com/Adithya-257/REL-for-HVAC_MLOps.git
cd REL-for-HVAC_MLOps
pip install -r requirements.txt

# Reproduce baseline experiment (git tag: exp-dqn-v1)
python train.py --config configs/dqn_v1.yaml

# Reproduce tuned experiment (git tag: exp-dqn-v2)
python train.py --config configs/dqn_v2.yaml
```

Each run produces:
- `results/results_log.csv` — per-episode metrics (run_id, reward, loss, epsilon, action counts, all hyperparameters)
- `results/run_summary.json` — single-file experiment summary
- `results/*.png` — 5 training plots
- `models/dqn_hvac.pth` — best model checkpoint
- MLflow run logged to `mlruns/`

To view MLflow UI after training:
```bash
mlflow ui
# Open http://localhost:5000
```

The `seed: 42` in each config ensures deterministic environment resets. Anyone who clones this repo and runs the same config file will get the same reward curve.



## Monitoring Plan (Production Design)

If this system were deployed in a real building with live IoT sensors, the following monitoring strategy would be implemented:

**What we would monitor:**
- **Average comfort violation rate** — percentage of occupied hours where indoor temperature falls outside [20,24]°C, tracked per day and per week
- **Energy consumption per episode** — total HVAC on-time (cool + heat actions) per day; a sudden increase indicates the agent is overcooling or the building's thermal properties have changed
- **Q-value confidence** — the gap between the highest and lowest Q-values per prediction; a narrowing gap indicates the agent is becoming uncertain, which may signal concept drift
- **Action distribution drift** — if the proportion of off actions drops significantly from the trained baseline, the building environment may have changed
- **Prediction latency** — FastAPI endpoint response time; target under 50ms for real-time control
- **Model staleness** — days since last retraining; trigger automatic retraining if comfort violation rate exceeds 10% over a 7-day rolling window

**Retraining trigger:** Automated retraining triggered by GitHub Actions when a monitoring alert fires, using the same `python train.py --config` reproducibility framework.

## SDG Mapping

- **SDG 7 (Clean Energy):** Reduces HVAC energy waste through intelligent control

- **SDG 11 (Sustainable Cities):** Scalable to smart building management systems
  
- **SDG 13 (Climate Action):** Lower energy use → reduced carbon emissions from buildings
