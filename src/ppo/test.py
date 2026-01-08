# Combined test for RL and LM
import gymnasium as gym
import torch
from .ppo_rl import PPO
from .ppo_lm import PPOLM
from ..common.utils import set_seed
from ..common.config import ppo_config as config
import argparse

def test_rl(env_name: str, episodes: int = 10, render: bool = False, device: str = ("cuda" if torch.cuda.is_available() else "cpu")):
    set_seed()
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = PPO(state_dim, action_dim, config["hidden"]).to(device)
    model.load_state_dict(torch.load(f"models/ppo_{env_name}.pt", map_location=device))
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
        print(f"RL Episode {ep + 1}: Reward = {total_reward}")

def test_lm(prompt: str = "Hello, world!", max_new: int = 30):
    model = PPOLM()
    # Try loading a combined checkpoint (model + value_head) first, then fall back to model-only
    try:
        ckpt = torch.load("models/ppo_lm_distilgpt2_full.pt", map_location=model.device)
        model.model.load_state_dict(ckpt['model'])
        model.value_head.load_state_dict(ckpt['value_head'])
        print("Loaded full checkpoint")
    except Exception:
        try:
            model.model.load_state_dict(torch.load("models/ppo_lm_distilgpt2.pt", map_location=model.device))
            print("Loaded model-only checkpoint")
        except Exception:
            print("No checkpoint found; using base model")

    # Tokenize with attention mask for reliable behavior
    tok_out = model.tokenizer(prompt, return_tensors='pt', padding=True)
    tok_out = {k: v.to(model.device) for k, v in tok_out.items()}
    sequence, _, _ = model.generate(tok_out, max_new)
    text = model.tokenizer.decode(sequence[0, tok_out['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Generated: {text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rl", choices=["rl", "lm"])
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--prompt", type=str, default="Hello, world!")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "rl":
        test_rl(args.env, episodes=args.episodes, render=args.render, device=args.device)
    else:
        test_lm(args.prompt)