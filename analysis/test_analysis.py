# analysis/test_analysis.py

import gymnasium as gym
import torch
import numpy as np
import os
from tqdm import tqdm
from src.common.utils import set_seed

# Chemins
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Environnements (v3 pour LunarLander comme dans ton config.py)
ENVS = ["CartPole-v1", "LunarLander-v3"]

# Seuils de résolution officiels
SOLVED_THRESHOLD = {
    "CartPole-v1": 475.0,
    "LunarLander-v3": 200.0
}

# Hyperparamètres EXACTEMENT comme dans ton common/config.py
HIDDEN_SIZES = {
    "Actor-Critic": 256,
    "PPO": 128,
    "GRPO": 128
}

def evaluate_model(algo_name: str, model_path: str, env_name: str, num_episodes: int = 50, render: bool = False):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    if not os.path.exists(model_path):
        print(f"⚠️  Modèle non trouvé : {model_path}")
        env.close()
        return None
    
    hidden = HIDDEN_SIZES.get(algo_name, 128)
    model = None
    
    try:
        if "actor-critic" in algo_name.lower():
            from src.actor_critic.actor_critic import ActorCritic
            model = ActorCritic(state_dim, action_dim, hidden=hidden).to(device)
        
        elif "ppo" in algo_name.lower():
            from src.ppo.ppo_rl import PPO
            model = PPO(state_dim, action_dim, hidden=hidden).to(device)
        
        elif "grpo" in algo_name.lower():
            from src.grpo.grpo import ActorNetwork
            model = ActorNetwork(state_dim, action_dim, hidden=hidden).to(device)
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ {algo_name} chargé avec hidden={hidden}")
        model.eval()
    
    except Exception as e:
        print(f"❌ Erreur chargement {algo_name} : {e}")
        env.close()
        return None
    
    rewards = []
    successes = 0
    
    for ep in tqdm(range(num_episodes), desc=f"{algo_name} → {env_name}", leave=False):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if "grpo" in algo_name.lower():
                    logits = model(state_tensor).squeeze(0)
                else:
                    logits, _ = model(state_tensor)
                action = torch.argmax(logits).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        rewards.append(total_reward)
        if total_reward >= SOLVED_THRESHOLD.get(env_name, 200):
            successes += 1
    
    env.close()
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards),
        "success_rate": successes / num_episodes * 100,
        "rewards_list": rewards
    }

def full_analysis():
    print("🚀 DÉBUT DE L'ANALYSE COMPLÈTE DES MODÈLES ENTRAÎNÉS\n")
    
    model_mapping = [
        ("Actor-Critic", "ac_{env}.pt"),
        ("PPO", "ppo_{env}.pt"),
        ("GRPO", "grpo_{env}.pt"),
    ]
    
    for env in ENVS:
        print(f"\n📊 ENVIRONNEMENT : {env}")
        print("-" * 70)
        
        for algo_name, filename_pattern in model_mapping:
            # Pour CartPole : env_file = "CartPole-v1"
            # Pour LunarLander : env_file = "LunarLander-v3" (correspond à tes fichiers)
            env_file = env
            model_path = os.path.join(MODELS_DIR, filename_pattern.format(env=env_file))
            
            eval_result = evaluate_model(algo_name, model_path, env, num_episodes=50, render=False)
            
            if eval_result:
                success_icon = "✅ RÉSOLU" if eval_result["success_rate"] >= 90 else "⚠️  Partiel" if eval_result["success_rate"] > 30 else "❌ Échec"
                print(f"{algo_name:15} → Moyenne: {eval_result['mean_reward']:6.1f} ± {eval_result['std_reward']:5.1f} | "
                      f"Max: {eval_result['max_reward']:6.1f} | Succès: {eval_result['success_rate']:5.1f}% {success_icon}")
            else:
                print(f"{algo_name:15} → Échec chargement ou modèle absent")

    print("\n🎉 Analyse terminée ! Tous les modèles compatibles ont été évalués sur 50 épisodes.")
    print("   Conseil : Si PPO atteint >250 sur LunarLander-v3 → c'est excellent !")

if __name__ == "__main__":
    full_analysis()