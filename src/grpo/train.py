import gymnasium as gym
import torch
import torch.optim as optim
import wandb
from .grpo import ActorNetwork, GRPOTrainer
from ..common.config import grpo_config as config
import argparse

def train(env_name: str, total_updates: int, device: str = ("cuda" if torch.cuda.is_available() else "cpu"), use_wandb: bool = False):
    if use_wandb:
        wandb.init(project="rl-study-grpo", config=config)
    else:
        wandb.init(mode="disabled")
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor = ActorNetwork(state_dim, action_dim, config["hidden"]).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=config["lr"])

    trainer = GRPOTrainer(
        env, actor, optimizer,
        config["clip_ratio"], config["beta"], config["gamma"],
        config["epochs"], config["batch_size"], config["group_size"]
    )
    trainer.train(total_updates)
    torch.save(actor.state_dict(), f"models/grpo_{env_name}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()
    train(args.env, config["total_updates"], device=args.device, use_wandb=args.use_wandb)