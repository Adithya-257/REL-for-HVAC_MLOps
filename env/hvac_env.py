import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HVACEnv(gym.Env):
    """
    Custom Gymnasium environment simulating HVAC control in a building.

    Goal: keep indoor temperature in a comfort zone (20-24 C) across a
    24-hour day while minimizing energy consumption.

    State:  [indoor_temp, outdoor_temp, hour_of_day, occupancy]
    Action: 0 = off | 1 = cool | 2 = heat
    Reward: -(discomfort_penalty + lambda * energy_cost)
    """

    metadata = {"render_modes": ["human"]}

    # Comfort zone bounds (degrees C)
    TEMP_MIN = 20.0
    TEMP_MAX = 24.0

    # How strongly the agent penalizes discomfort vs energy
    LAMBDA_ENERGY = 0.3

    # How much each action costs in energy units
    ENERGY_COST = {0: 0.0, 1: 1.0, 2: 1.0}

    # How much the HVAC shifts indoor temp per timestep (degrees C)
    HVAC_EFFECT = {0: 0.0, 1: -2.0, 2: +2.0}

    # How strongly indoor temp drifts toward outdoor temp (thermal leakage)
    DRIFT_RATE = 0.15

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 4 continuous observations
        # [indoor_temp (10-45), outdoor_temp (10-45), hour (0-23), occupancy (0 or 1)]
        self.observation_space = spaces.Box(
            low=np.array([10.0, 10.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([45.0, 45.0, 23.0, 1.0], dtype=np.float32),
        )

        # 3 discrete actions: off, cool, heat
        self.action_space = spaces.Discrete(3)

        self.state = None
        self.hour = None
        self.outdoor_temps = None  # pre-generated outdoor profile for the day
        self.occupancy_schedule = None

    # ------------------------------------------------------------------
    # Core Gym methods
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start at a random comfortable-ish indoor temperature
        indoor_temp = self.np_random.uniform(19.0, 26.0)

        # Generate a realistic outdoor temperature profile for the day
        # Cooler at night, peaks in the afternoon (~34 C peak for Bangalore summer)
        self.outdoor_temps = self._generate_outdoor_profile()

        # Occupancy: building occupied 9am-6pm
        self.occupancy_schedule = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0,   # 0-8: unoccupied
             1, 1, 1, 1, 1, 1, 1, 1, 1,   # 9-17: occupied
             0, 0, 0, 0, 0, 0],            # 18-23: unoccupied
            dtype=np.float32,
        )

        self.hour = 0
        outdoor_temp = self.outdoor_temps[self.hour]
        occupancy = self.occupancy_schedule[self.hour]

        self.state = np.array(
            [indoor_temp, outdoor_temp, float(self.hour), occupancy],
            dtype=np.float32,
        )

        return self.state, {}

    def step(self, action):
        assert self.state is not None, "Call reset() before step()"

        indoor_temp, _, _, _ = self.state

        # --- Physics update ---
        outdoor_temp = self.outdoor_temps[self.hour]

        # Natural thermal drift: indoor moves toward outdoor
        drift = self.DRIFT_RATE * (outdoor_temp - indoor_temp)

        # HVAC effect
        hvac_delta = self.HVAC_EFFECT[int(action)]

        # New indoor temp (clipped to physical bounds)
        new_indoor_temp = float(
            np.clip(indoor_temp + drift + hvac_delta, 10.0, 45.0)
        )

        # --- Reward ---
        discomfort = self._discomfort_penalty(new_indoor_temp)
        energy = self.ENERGY_COST[int(action)]
        reward = -(discomfort + self.LAMBDA_ENERGY * energy)

        # --- Advance time ---
        self.hour += 1
        done = self.hour >= 24

        if not done:
            next_outdoor = self.outdoor_temps[self.hour]
            next_occupancy = self.occupancy_schedule[self.hour]
        else:
            next_outdoor = self.outdoor_temps[23]
            next_occupancy = 0.0

        self.state = np.array(
            [new_indoor_temp, next_outdoor, float(self.hour % 24), next_occupancy],
            dtype=np.float32,
        )

        info = {
            "indoor_temp": new_indoor_temp,
            "outdoor_temp": outdoor_temp,
            "hour": self.hour - 1,
            "discomfort": discomfort,
            "energy": energy,
        }

        if self.render_mode == "human":
            self._render_frame(action, info, reward)

        return self.state, reward, done, False, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discomfort_penalty(self, temp):
        """
        0 if temp is within [TEMP_MIN, TEMP_MAX].
        Increases quadratically the further outside the comfort zone.
        """
        if temp < self.TEMP_MIN:
            return (self.TEMP_MIN - temp) ** 2
        elif temp > self.TEMP_MAX:
            return (temp - self.TEMP_MAX) ** 2
        return 0.0

    def _generate_outdoor_profile(self):
        """
        Sinusoidal outdoor temp profile.
        Min at 5am (~26 C), peak at 3pm (~38 C) — Bangalore summer.
        Adds small random noise for variability across episodes.
        """
        hours = np.arange(24)
        # Peak at hour 15 (3pm), trough at hour 5 (5am)
        base = 32.0 + 6.0 * np.sin(2 * np.pi * (hours - 5) / 24)
        noise = self.np_random.normal(0, 0.5, size=24)
        return np.clip(base + noise, 10.0, 45.0).astype(np.float32)

    def _render_frame(self, action, info, reward):
        action_names = {0: "OFF ", 1: "COOL", 2: "HEAT"}
        print(
            f"Hour {info['hour']:02d} | "
            f"Indoor: {info['indoor_temp']:5.1f}C | "
            f"Outdoor: {info['outdoor_temp']:5.1f}C | "
            f"Action: {action_names[int(action)]} | "
            f"Reward: {reward:6.2f} | "
            f"Discomfort: {info['discomfort']:.2f}"
        )