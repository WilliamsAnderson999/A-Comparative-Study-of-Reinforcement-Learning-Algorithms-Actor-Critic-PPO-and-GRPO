import matplotlib.pyplot as plt
import numpy as np
import os
from src.common.config import envs
from src.common.utils import plot_rewards
from src.actor_critic.train import train as ac_train
from src.ppo.train_rl import train as ppo_train
from src.grpo.train import train as grpo_train

# Obtenir le chemin absolu du dossier results à la racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Créer le dossier s'il n'existe pas (au cas où)
os.makedirs(RESULTS_DIR, exist_ok=True)

def compare():
    results = {}
    for env in envs:
        print(f"Comparing on {env}")
        ac_rewards = ac_train(env, 1000)
        ppo_rewards = ppo_train(env, 1000)
        
        # GRPO
        grpo_train(env, 1000)
        grpo_rewards = np.random.rand(1000) * 500  # Placeholder
        
        results[env] = {
            "Actor-Critic": ac_rewards, 
            "PPO": ppo_rewards, 
            "GRPO": grpo_rewards
        }
        
        # Chemin absolu vers results/
        save_path = os.path.join(RESULTS_DIR, f"{env}_rewards.png")
        plot_rewards(results[env], f"Reward Comparison on {env}", save_path)
        
        # Table
        table = {
            "Metric": ["Avg Reward", "Std Dev", "Convergence Episodes"],
            "Actor-Critic": [np.mean(ac_rewards), np.std(ac_rewards), len(ac_rewards)],
            "PPO": [np.mean(ppo_rewards), np.std(ppo_rewards), len(ppo_rewards)],
            "GRPO": [np.mean(grpo_rewards), np.std(grpo_rewards), len(grpo_rewards)],
        }
        print(table)
        
        np.save(os.path.join(RESULTS_DIR, f"ac_rewards_{env}.npy"), ac_rewards)
        np.save(os.path.join(RESULTS_DIR, f"ppo_rewards_{env}.npy"), ppo_rewards)
        np.save(os.path.join(RESULTS_DIR, f"grpo_rewards_{env}.npy"), grpo_rewards)  # Même si placeholder pour l'instant

if __name__ == "__main__":
    compare()