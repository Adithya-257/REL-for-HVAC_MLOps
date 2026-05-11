"""
train.py — DQN training loop for HVAC control
Logs all metrics to MLflow and saves plots to results/

Usage:
    python train.py                              # default config
    python train.py --config configs/dqn_v1.yaml  # named experiment config
"""

import os
import sys
import csv
import json
import random
import argparse
import numpy as np
import torch
import mlflow
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from env.hvac_env import HVACEnv
from agent.dqn_agent import DQNAgent
import csv
import json
import argparse


# -----------------------------------------------------------------------
# Argument parsing — supports --config flag for reproducibility
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent for HVAC control")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (e.g. configs/dqn_v1.yaml)")
    return parser.parse_args()

def load_config(config_path):
    """Load experiment config from YAML. Falls back to defaults if no file given."""
    defaults = {
        "episodes":         600,
        "seed":             42,
        "lr":               1e-3,
        "gamma":            0.99,
        "epsilon_start":    1.0,
        "epsilon_end":      0.05,
        "epsilon_decay":    0.995,
        "batch_size":       64,
        "buffer_capacity":  10_000,
        "target_update_freq": 50,
        "run_name":         "dqn-hvac-run",
    }
    if config_path is None:
        return defaults
    try:
        import yaml
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        defaults.update(user_cfg)
        print(f"  Loaded config: {config_path}")
    except ImportError:
        print("  PyYAML not installed — using default config. Install with: pip install pyyaml")
    except FileNotFoundError:
        print(f"  Config file {config_path} not found — using defaults")
    return defaults

# -----------------------------------------------------------------------
# Config (defaults — overridden by --config if provided)
# -----------------------------------------------------------------------

EPISODES        = int(os.environ.get("EPISODES", 600))
SEED            = 42
MODEL_SAVE_PATH = "models/dqn_hvac.pth"
RESULTS_DIR     = "results"
MLFLOW_EXP      = "rl-hvac-dqn"

# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def run_episode(env, agent, training=True):
    """
    Run one full 24-hour episode.
    If training=True, stores transitions and does train steps.
    Returns (total_reward, avg_loss, avg_indoor_temp, action_counts)
    """
    obs, _ = env.reset(seed=None)
    total_reward = 0.0
    losses = []
    indoor_temps = []
    action_counts = {0: 0, 1: 0, 2: 0}

    for _ in range(24):
        action = agent.select_action(obs)
        next_obs, reward, done, _, info = env.step(action)

        if training:
            agent.store(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        obs = next_obs
        total_reward += reward
        indoor_temps.append(info["indoor_temp"])
        action_counts[action] += 1

        if done:
            break

    return (
        total_reward,
        float(np.mean(losses)) if losses else 0.0,
        float(np.mean(indoor_temps)),
        action_counts,
    )


def smooth(values, window=20):
    """Moving average for cleaner plots."""
    if len(values) < window:
        return values
    return [
        float(np.mean(values[max(0, i - window): i + 1]))
        for i in range(len(values))
    ]


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def save_reward_curve(rewards, smoothed, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, color="#B0BEC5", alpha=0.45, linewidth=0.8, label="Raw reward")
    ax.plot(smoothed, color="#1E88E5", linewidth=2.2, label="Smoothed (20-ep window)")
    ax.axhline(y=np.mean(rewards[-50:]), color="#43A047", linestyle="--",
               linewidth=1.4, label=f"Last-50 avg: {np.mean(rewards[-50:]):.1f}")
    ax.set_title("DQN Training — Episode Reward over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def save_epsilon_curve(epsilons, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epsilons, color="#FB8C00", linewidth=2)
    ax.set_title("Epsilon Decay (Exploration → Exploitation)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def save_loss_curve(losses, path):
    smoothed = smooth(losses, window=30)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, color="#EF9A9A", alpha=0.4, linewidth=0.8, label="Raw loss")
    ax.plot(smoothed, color="#E53935", linewidth=2, label="Smoothed (30-ep window)")
    ax.set_title("DQN Training Loss (MSE Bellman Error)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def save_comparison_plot(env, trained_agent, path):
    """
    Side-by-side comparison of indoor temperature across 24 hours:
    random agent vs trained DQN agent.
    """
    hours = list(range(24))
    comfort_min = HVACEnv.TEMP_MIN
    comfort_max = HVACEnv.TEMP_MAX

    # --- Random agent ---
    random_agent = DQNAgent()
    random_agent.epsilon = 1.0   # always random
    env.reset(seed=SEED)
    obs, _ = env.reset(seed=SEED)
    random_temps, random_actions = [], []
    for _ in range(24):
        action = random_agent.select_action(obs)
        obs, _, done, _, info = env.step(action)
        random_temps.append(info["indoor_temp"])
        random_actions.append(action)
        if done:
            break

    # --- Trained agent ---
    trained_agent.epsilon = 0.0  # fully greedy
    obs, _ = env.reset(seed=SEED)
    trained_temps, trained_actions = [], []
    for _ in range(24):
        action = trained_agent.select_action(obs)
        obs, _, done, _, info = env.step(action)
        trained_temps.append(info["indoor_temp"])
        trained_actions.append(action)
        if done:
            break

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    action_colors = {0: "#90A4AE", 1: "#42A5F5", 2: "#EF5350"}
    action_labels = {0: "Off", 1: "Cool", 2: "Heat"}

    for ax, temps, actions, title in [
        (axes[0], random_temps,  random_actions,  "Random Agent"),
        (axes[1], trained_temps, trained_actions, "Trained DQN Agent"),
    ]:
        # Comfort band
        ax.axhspan(comfort_min, comfort_max, alpha=0.12, color="#43A047",
                   label="Comfort zone (20–24°C)")
        ax.axhline(comfort_min, color="#43A047", linewidth=0.8, linestyle="--")
        ax.axhline(comfort_max, color="#43A047", linewidth=0.8, linestyle="--")

        # Temp line
        ax.plot(hours[:len(temps)], temps, color="#212121", linewidth=2,
                label="Indoor temp", zorder=3)

        # Action color bars at bottom
        for h, act in enumerate(actions):
            ax.axvspan(h - 0.5, h + 0.5, ymin=0, ymax=0.04,
                       color=action_colors[act], alpha=0.85)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Indoor temperature (°C)")
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.25)

    # Shared legend
    patches = [mpatches.Patch(color=c, label=f"{action_labels[a]} action")
               for a, c in action_colors.items()]
    patches.append(mpatches.Patch(color="#43A047", alpha=0.3, label="Comfort zone"))
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Random Agent vs Trained DQN — Indoor Temperature Control",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_action_distribution(action_log, path):
    """
    Stacked bar chart showing how action distribution shifts
    from early training (random) to late training (learned policy).
    """
    early = action_log[:100]
    late  = action_log[-100:]

    def counts(log):
        flat = [a for ep in log for a in ep]
        total = len(flat)
        return [flat.count(i) / total * 100 for i in range(3)]

    early_pct = counts(early)
    late_pct  = counts(late)

    labels = ["Early training\n(ep 1–100)", "Late training\n(ep 501–600)"]
    colors = ["#90A4AE", "#42A5F5", "#EF5350"]
    action_names = ["Off", "Cool", "Heat"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = [0, 0]
    for i, (color, name) in enumerate(zip(colors, action_names)):
        vals = [early_pct[i], late_pct[i]]
        bars = ax.bar(labels, vals, bottom=bottoms, color=color,
                      label=name, width=0.45, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 4:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white")
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylim(0, 105)
    ax.set_ylabel("Action share (%)")
    ax.set_title("Action Distribution — Early vs Late Training",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()

def load_config(path):
    cfg = {
        "episodes": EPISODES, "seed": 42, "lr": 1e-3, "gamma": 0.99,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.995,
        "batch_size": 64, "buffer_capacity": 10_000,
        "target_update_freq": 50, "run_name": "dqn-hvac-run",
    }
    if path:
        try:
            import yaml
            with open(path) as f:
                cfg.update(yaml.safe_load(f))
            print(f"  Loaded config: {path}")
        except Exception as e:
            print(f"  Could not load config {path}: {e} — using defaults")
    return cfg

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    episodes = int(os.environ.get("EPISODES", cfg["episodes"]))

    env   = HVACEnv()
    agent = DQNAgent(
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay=cfg["epsilon_decay"],
        batch_size=cfg["batch_size"],
        buffer_capacity=cfg["buffer_capacity"],
        target_update_freq=cfg["target_update_freq"],
    )

    all_rewards  = []
    all_losses   = []
    all_epsilons = []
    all_actions  = []

    mlflow.set_experiment(MLFLOW_EXP)

    print(f"\n{'='*55}")
    print(f"  RL-HVAC DQN Training — {episodes} episodes")
    print(f"  Device : {agent.device}")
    print(f"  MLflow : {MLFLOW_EXP}")
    print(f"{'='*55}\n")

    with mlflow.start_run(run_name=cfg["run_name"]):

        mlflow.log_params({
            "episodes":           episodes,
            "learning_rate":      cfg["lr"],
            "gamma":              cfg["gamma"],
            "epsilon_start":      cfg["epsilon_start"],
            "epsilon_end":        cfg["epsilon_end"],
            "epsilon_decay":      cfg["epsilon_decay"],
            "batch_size":         cfg["batch_size"],
            "buffer_capacity":    cfg["buffer_capacity"],
            "target_update_freq": cfg["target_update_freq"],
            "hidden_dim":         64,
            "seed":               cfg["seed"],
        })

        best_reward = -float("inf")

        for ep in range(1, episodes + 1):
            reward, avg_loss, avg_temp, action_counts = run_episode(env, agent)
            agent.end_episode()

            all_rewards.append(reward)
            all_losses.append(avg_loss)
            all_epsilons.append(agent.epsilon)
            all_actions.append(list(action_counts.values()))

            mlflow.log_metrics({
                "episode_reward":    reward,
                "avg_loss":          avg_loss,
                "epsilon":           agent.epsilon,
                "avg_indoor_temp":   avg_temp,
                "action_off_count":  action_counts[0],
                "action_cool_count": action_counts[1],
                "action_heat_count": action_counts[2],
                "buffer_size":       len(agent.replay_buffer),
            }, step=ep)

            if reward > best_reward:
                best_reward = reward
                agent.save(MODEL_SAVE_PATH)

            if ep % 50 == 0 or ep == 1:
                avg_last20 = np.mean(all_rewards[-20:])
                print(
                    f"  Ep {ep:4d}/{episodes} | "
                    f"Reward: {reward:8.1f} | "
                    f"Avg(20): {avg_last20:8.1f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"eps: {agent.epsilon:.3f}"
                )

        mlflow.log_metrics({
            "best_reward":      best_reward,
            "final_avg_reward": float(np.mean(all_rewards[-50:])),
            "final_epsilon":    agent.epsilon,
        })

        # --- Save results_log.csv ---
        run_id   = mlflow.active_run().info.run_id
        smoothed = smooth(all_rewards, window=20)
        csv_path = os.path.join(RESULTS_DIR, "results_log.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id","episode","episode_reward","avg_reward_20ep",
                "avg_loss","epsilon","action_off","action_cool","action_heat",
                "lr","gamma","epsilon_decay","batch_size","buffer_capacity"
            ])
            writer.writeheader()
            for i, (r, s, l, e, a) in enumerate(
                zip(all_rewards, smoothed, all_losses, all_epsilons, all_actions)
            ):
                writer.writerow({
                    "run_id":           run_id,
                    "episode":          i + 1,
                    "episode_reward":   round(r, 4),
                    "avg_reward_20ep":  round(s, 4),
                    "avg_loss":         round(l, 4),
                    "epsilon":          round(e, 4),
                    "action_off":       a[0],
                    "action_cool":      a[1],
                    "action_heat":      a[2],
                    "lr":               cfg["lr"],
                    "gamma":            cfg["gamma"],
                    "epsilon_decay":    cfg["epsilon_decay"],
                    "batch_size":       cfg["batch_size"],
                    "buffer_capacity":  cfg["buffer_capacity"],
                })
        mlflow.log_artifact(csv_path)
        print(f"  Saved: {csv_path}")

        # --- Save run_summary.json ---
        summary = {
            "run_id":           run_id,
            "run_name":         cfg["run_name"],
            "episodes":         episodes,
            "best_reward":      round(best_reward, 4),
            "final_avg_reward": round(float(np.mean(all_rewards[-50:])), 4),
            "baseline_reward":  round(all_rewards[0], 4),
            "parameters":       {
                "lr":               cfg["lr"],
                "gamma":            cfg["gamma"],
                "epsilon_start":    cfg["epsilon_start"],
                "epsilon_end":      cfg["epsilon_end"],
                "epsilon_decay":    cfg["epsilon_decay"],
                "batch_size":       cfg["batch_size"],
                "buffer_capacity":  cfg["buffer_capacity"],
                "target_update_freq": cfg["target_update_freq"],
                "seed":             cfg["seed"],
            },
        }
        json_path = os.path.join(RESULTS_DIR, "run_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(json_path)
        print(f"  Saved: {json_path}")

        # --- Plots ---
        print(f"\nGenerating plots...")
        plots = {
            "reward_curve":        (save_reward_curve,       (all_rewards, smoothed)),
            "epsilon_curve":       (save_epsilon_curve,       (all_epsilons,)),
            "loss_curve":          (save_loss_curve,          (all_losses,)),
            "agent_comparison":    (save_comparison_plot,     (env, agent)),
            "action_distribution": (save_action_distribution, (all_actions,)),
        }
        for name, (fn, args_) in plots.items():
            path = os.path.join(RESULTS_DIR, f"{name}.png")
            fn(*args_, path)
            mlflow.log_artifact(path)

        mlflow.log_artifact(MODEL_SAVE_PATH)

        print(f"\n{'='*55}")
        print(f"  Training complete!")
        print(f"  Best reward : {best_reward:.2f}")
        print(f"  Final avg   : {np.mean(all_rewards[-50:]):.2f}")
        print(f"  Model saved : {MODEL_SAVE_PATH}")
        print(f"  Plots saved : {RESULTS_DIR}/")
        print(f"  CSV saved   : {csv_path}")
        print(f"  JSON saved  : {json_path}")
        print(f"{'='*55}\n")
        print("  Run 'mlflow ui' to view the experiment dashboard.\n")


if __name__ == "__main__":
    main()