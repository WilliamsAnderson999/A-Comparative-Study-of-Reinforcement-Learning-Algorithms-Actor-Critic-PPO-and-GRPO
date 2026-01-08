import gymnasium as gym
import torch
from .grpo import ActorNetwork
from ..common.utils import set_seed
from ..common.config import grpo_config as config
import argparse

def test(env_name: str, episodes: int = 10, render: bool = False, device: str = ("cuda" if torch.cuda.is_available() else "cpu")):
    set_seed()
    env = gym.make(env_name, render_mode="human" if render else None)
    # Use the environment's spec id (handles version auto-upgrade, e.g., CartPole → CartPole-v1)
    env_id = env.spec.id if env.spec else env_name
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor = ActorNetwork(state_dim, action_dim, config["hidden"]).to(device)
    actor.load_state_dict(torch.load(f"models/grpo_{env_id}.pt", map_location=device))
    actor.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        total_rew = 0.0
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits = actor(state_t).squeeze(0)
            action = torch.argmax(logits).item()
            state, rew, terminated, truncated, _ = env.step(action)
            total_rew += rew
            done = terminated or truncated
        print(f"Episode {ep + 1}: Reward = {total_rew}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    test(args.env, episodes=args.episodes, render=args.render, device=args.device)