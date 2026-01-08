A Comparative Study of Reinforcement Learning Algorithms: Actor-Critic, PPO, and GRPO
This project presents a comparative study of several Reinforcement Learning (RL) algorithms, specifically Actor-Critic, Proximal Policy Optimization (PPO), and Generalized Reinforcement Policy Optimization (GRPO).
The goal is to analyze their performance, stability, and efficiency across standard environments such as CartPole and LunarLander, as well as on language modeling tasks with DistilGPT2.

🎯 Objectives
Implement and train the Actor-Critic, PPO, and GRPO algorithms.

Compare their results in terms of convergence, robustness, and efficiency.

Provide an experimental foundation to understand the strengths and limitations of each approach.

Explore the application of PPO and GRPO in Language Modeling contexts.

📂 Project Structure
src/actor_critic/ → Actor-Critic implementation

src/ppo/ → PPO implementation (RL and LM)

src/grpo/ → GRPO implementation

src/common/ → utilities and configuration

analysis/ → scripts for results analysis

notebooks/ → comparison and visualization notebooks

models/ → trained models (CartPole, LunarLander, DistilGPT2)

references/ → bibliography and related articles

⚡ Technologies
Python (PyTorch, NumPy, Matplotlib)

Gymnasium/OpenAI Gym for RL environments

Weights & Biases (wandb) for experiment tracking

Jupyter Notebooks for analysis and visualization

🚀 Expected Results
Comparative benchmarks across multiple RL environments.

Visualizations of learning curves and performance metrics.

Discussion on the applicability of these algorithms in diverse contexts (control, NLP).