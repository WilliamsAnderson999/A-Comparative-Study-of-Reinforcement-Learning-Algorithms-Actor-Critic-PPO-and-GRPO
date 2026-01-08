import gymnasium as gym
import torch
import wandb
from tqdm import trange
from ..common.utils import set_seed, collect_rollouts
from .ppo_rl import PPO
from ..common.config import ppo_config as config
import argparse

def train(env_name: str, episodes: int, device: str = ("cuda" if torch.cuda.is_available() else "cpu"), use_wandb: bool = False):
    set_seed()
    if use_wandb:
        wandb.init(project="rl-study-ppo", config=config)
    else:
        wandb.init(mode="disabled")
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = PPO(state_dim, action_dim, config["hidden"], config["lr"], use_amp=config.get("use_amp", False)).to(device)
    rewards = []

    for ep in trange(episodes):
        rollouts = collect_rollouts(env, model.actor, config["rollout_steps"], model.critic, device)
        states, actions, rews, dones, log_probs, values = rollouts
        loss = model.update(states, actions, log_probs, rews, dones, values, config["gamma"], config["lam"], config["clip_ratio"], config["ent_coef"], config["epochs"], config["batch_size"])
        ep_reward = sum(rews)
        rewards.append(ep_reward)
        wandb.log({"episode": ep, "reward": ep_reward, "loss": loss})
    torch.save(model.state_dict(), f"models/ppo_{env_name}.pt")
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()
    train(args.env, config["episodes"], device=args.device, use_wandb=args.use_wandb)