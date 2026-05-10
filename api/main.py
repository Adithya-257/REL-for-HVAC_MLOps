"""
api/main.py — FastAPI inference server for the trained DQN HVAC agent

Endpoints:
  GET  /         — health check
  GET  /info     — model info and action descriptions
  POST /predict  — given building state, return recommended HVAC action
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent.dqn_agent import DQNAgent

# -----------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hvac-api")

# -----------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------

app = FastAPI(
    title="RL-HVAC Inference API",
    description="DQN-based HVAC control agent. Send building state, get action recommendation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------
# Load model at startup
# -----------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "models/dqn_hvac.pth")

agent = DQNAgent()
agent.epsilon = 0.0   # fully greedy — no random exploration during inference

try:
    agent.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    MODEL_LOADED = False

ACTION_NAMES = {0: "off", 1: "cool", 2: "heat"}
ACTION_DESCRIPTIONS = {
    0: "Turn HVAC off — indoor temperature will drift toward outdoor",
    1: "Activate cooling — reduces indoor temperature by ~2°C per hour",
    2: "Activate heating — increases indoor temperature by ~2°C per hour",
}

# -----------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------

class StateInput(BaseModel):
    indoor_temp: float = Field(
        ..., ge=10.0, le=45.0,
        description="Current indoor temperature in Celsius",
        example=28.5,
    )
    outdoor_temp: float = Field(
        ..., ge=10.0, le=45.0,
        description="Current outdoor temperature in Celsius",
        example=36.0,
    )
    hour_of_day: float = Field(
        ..., ge=0.0, le=23.0,
        description="Current hour (0–23)",
        example=14.0,
    )
    occupancy: float = Field(
        ..., ge=0.0, le=1.0,
        description="Building occupancy (0 = empty, 1 = occupied)",
        example=1.0,
    )


class PredictionResponse(BaseModel):
    action_id: int
    action_name: str
    action_description: str
    q_values: dict
    comfort_status: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="ok" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED,
        model_path=MODEL_PATH,
        device=str(agent.device),
    )


@app.get("/info", tags=["Info"])
def model_info():
    return {
        "model": "Deep Q-Network (DQN)",
        "state_space": {
            "indoor_temp": "10–45°C",
            "outdoor_temp": "10–45°C",
            "hour_of_day": "0–23",
            "occupancy": "0 or 1",
        },
        "actions": ACTION_DESCRIPTIONS,
        "comfort_zone": "20–24°C",
        "trained_episodes": 600,
        "best_training_reward": -5.10,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(state: StateInput):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build state vector
    state_vector = np.array(
        [state.indoor_temp, state.outdoor_temp, state.hour_of_day, state.occupancy],
        dtype=np.float32,
    )

    # Get Q-values for all actions
    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_vals = agent.q_net(state_tensor).squeeze(0).cpu().numpy()

    action_id = int(np.argmax(q_vals))

    # Comfort status
    if 20.0 <= state.indoor_temp <= 24.0:
        comfort_status = "comfortable"
    elif state.indoor_temp < 20.0:
        comfort_status = f"too cold by {20.0 - state.indoor_temp:.1f}°C"
    else:
        comfort_status = f"too hot by {state.indoor_temp - 24.0:.1f}°C"

    logger.info(
        f"Predict | indoor={state.indoor_temp}°C outdoor={state.outdoor_temp}°C "
        f"hour={int(state.hour_of_day)} occupancy={int(state.occupancy)} "
        f"→ action={ACTION_NAMES[action_id]}"
    )

    return PredictionResponse(
        action_id=action_id,
        action_name=ACTION_NAMES[action_id],
        action_description=ACTION_DESCRIPTIONS[action_id],
        q_values={
            ACTION_NAMES[i]: round(float(q_vals[i]), 4)
            for i in range(3)
        },
        comfort_status=comfort_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )