import torch
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import wandb
from tqdm import trange
from typing import List, Tuple, Any

def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_rewards(rewards_dict: dict, title: str, save_path: str) -> None:
    """Plot reward curves for multiple algorithms."""
    plt.figure(figsize=(10, 6))
    for label, rewards in rewards_dict.items():
        plt.plot(rewards, label=label)
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def collect_rollouts(
    env: gym.Env,
    policy: torch.nn.Module,
    steps: int,
    critic: Any = None,
    device: str = "cpu"
) -> Tuple[List[torch.Tensor], List[int], List[float], List[bool], List[float], List[float]]:
    """Collect rollouts for training. Supports actor-critic with values."""
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    done = False
    for _ in range(steps):
        if done:
            state, _ = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            done = False
        states.append(state)
        with torch.no_grad():
            # Call the policy; it may return (logits, value) or just logits.
            out = policy(state)
            try:
                logits, value_tensor = out
            except Exception:
                logits = out
                value_tensor = None

            # If a separate critic was provided and we don't have a value yet, call it.
            if critic is not None and value_tensor is None:
                crit_out = critic(state)
                try:
                    # critic may return (logits, value) or a tensor
                    _, crit_value = crit_out
                except Exception:
                    crit_value = crit_out
                value_tensor = crit_value

            # Ensure tensors are on same device
            if isinstance(logits, torch.Tensor):
                logits = logits.to(state.device)
            if isinstance(value_tensor, torch.Tensor):
                value_tensor = value_tensor.to(state.device)

            dist = torch.distributions.Categorical(logits=logits)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor).item()
            action = action_tensor.item()
            value = value_tensor.squeeze().item() if value_tensor is not None else 0.0
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value)
        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    return states, actions, rewards, dones, log_probs, values

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95
) -> np.ndarray:
    """Generalized Advantage Estimation (GAE)."""
    advantages = []
    gae = 0.0
    values += [0.0]  # Bootstrap last value as 0
    for r, v, next_v, done in zip(reversed(rewards), reversed(values[:-1]), reversed(values[1:]), reversed(dones)):
        delta = r + gamma * next_v * (1 - int(done)) - v
        gae = delta + gamma * lam * (1 - int(done)) * gae
        advantages.insert(0, gae)
    return np.array(advantages, dtype=np.float32)