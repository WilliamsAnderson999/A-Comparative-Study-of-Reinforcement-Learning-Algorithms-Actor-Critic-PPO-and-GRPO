import gymnasium as gym
import torch
import wandb
from tqdm import trange
from ..common.utils import set_seed, collect_rollouts
from .actor_critic import ActorCritic
from ..common.config import actor_critic_config as config
import argparse

def train(env_name: str, episodes: int, device: str = "cpu", use_wandb: bool = False):
    set_seed()
    if use_wandb:
        wandb.init(project="rl-study-ac", config=config)
    else:
        wandb.init(mode="disabled")
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim, config["hidden"], config["lr"]).to(device)
    rewards = []

    for ep in trange(episodes):
        states, actions, rewards_ep, dones, log_probs, values = collect_rollouts(env, model, config["max_steps"], model.forward, device)
        loss = model.update(states, actions, log_probs, rewards_ep, dones, values, config["gamma"], 0.95, config["ent_coef"], config["clip_ratio"], config["epochs"])
        ep_reward = sum(rewards_ep)  # rewards
        rewards.append(ep_reward)
        wandb.log({"episode": ep, "reward": ep_reward, "loss": loss})
    torch.save(model.state_dict(), f"models/ac_{env_name}.pt")
    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.env, config["episodes"], device=args.device)