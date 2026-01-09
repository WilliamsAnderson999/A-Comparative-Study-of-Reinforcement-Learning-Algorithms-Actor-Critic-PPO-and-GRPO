# Hyperparameters as dicts for easy loading
actor_critic_config = {
    "lr": 3e-4,
    "hidden": 256,
    "gamma": 0.99,
    "ent_coef": 0.01,
    "epochs": 10,  
    "clip_ratio": 0.2,  
    "max_steps": 4096,  
    "episodes": 1000,
}

ppo_config = {
    "lr": 2.5e-4,              
    "hidden": 128,              
    "gamma": 0.99,
    "lam": 0.95,
    "clip_ratio": 0.2,
    "ent_coef": 0.02,
    "epochs": 4,               
    "batch_size": 32,          
    "rollout_steps": 1024,     
    "episodes": 1500,
    "use_amp": True,           
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
