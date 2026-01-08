# Hyperparameters as dicts for easy loading
actor_critic_config = {
    "lr": 3e-4,
    "hidden": 256,
    "gamma": 0.99,
    "ent_coef": 0.01,
    "epochs": 10,  # Reduced for speed
    "clip_ratio": 0.2,  # Optional for stability
    "max_steps": 4096,  # Increased for better GPU utilization
    "episodes": 1000,
}

ppo_config = {
    "lr": 2.5e-4,              # Slightly lower LR for stability with small batches
    "hidden": 128,              # Reduced from 64 → saves VRAM and speeds up
    "gamma": 0.99,
    "lam": 0.95,
    "clip_ratio": 0.2,
    "ent_coef": 0.02,
    "epochs": 4,               # Reduced from 10 → big VRAM saver
    "batch_size": 32,          # Safe on 3050 (was 64)
    "rollout_steps": 1024,     # Reduced from 2048 → half the memory per update
    "episodes": 1500,
    "use_amp": True,           # Enable Automatic Mixed Precision (huge speedup + memory save)
}

grpo_config = {
    "lr": 3e-4,
    "hidden": 128,
    "gamma": 0.99,
    "clip_ratio": 0.2,
    "beta": 0.001,
    "epochs": 5,
    "batch_size": 32,
    "group_size": 200,
    "total_updates": 1000,
}

envs = ["CartPole-v1", "LunarLander-v3"]