import gymnasium as gym
import torch
from ..actor_critic.actor_critic import ActorCritic
from ..common.utils import set_seed
from ..common.config import actor_critic_config as config
import argparse

def test(env_name: str, episodes: int = 10, render: bool = False, device: str = "cpu"):
    set_seed()
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = ActorCritic(state_dim, action_dim, config["hidden"]).to(device)
    model.load_state_dict(torch.load(f"models/ac_{env_name}.pt", map_location=device))
    model.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _, _ = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: Reward = {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    test(args.env, episodes=args.episodes, render=args.render, device=args.device)