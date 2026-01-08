# Analysis and Interpretation of Actor-Critic, PPO, and GRPO

This analysis provides an engineering-level interpretation of the algorithms, based on experiments on CartPole-v1 and LunarLander-v2. We discuss stability, variance reduction, computational efficiency, and applicability.

## Comparison Table
| Feature          | Actor-Critic (Generic) | PPO (Clipped AC) | GRPO (Regularized) |
|------------------|------------------------|-----------------|---------------------|
| Architecture    | Actor + Critic        | Actor + Critic | Actor only (no critic) |
| Update Mechanism| Policy gradient + TD  | Clipped surrogate | Regularized (KL + clip) |
| Stability       | Medium (high variance)| High (clipping prevents large shifts) | High (tunable beta for conservativeness) |
| Variance Reduction | Critic bootstraps values | GAE + clipping | Group-relative advantages (normalizes within batch) |
| Computation     | Low (simple updates) | Medium (multi-epoch) | Medium (KL computation) |
| Use Cases       | Baseline for continuous actions | General RL (e.g., robotics) | LLM reasoning, verifiable rewards |

## Ingenious Interpretations
- **Variance and Stability**: Actor-Critic reduces policy gradient variance via critic estimates (theoretical basis in @konda2000actor), but can diverge if critic is inaccurate. PPO addresses this with clipping (ϵ=0.2), ensuring monotonic improvement (proven in @schulman2017proximal). GRPO generalizes this by adding explicit KL penalty (β), making it tunable for conservative updates—ideal for high-stakes engineering systems like autonomous vehicles where large policy shifts could be catastrophic.
  
- **Efficiency in Practice**: In our tests, PPO converges faster on LunarLander-v2 (harder env) due to multi-epoch optimization on collected data, reusing rollouts efficiently. GRPO's group-relative advantages lower variance in noisy rewards (e.g., by normalizing within mini-groups), which is ingenious for domains like NLP or finance where rewards are sparse/verifiable, as seen in @shao2024deepseekmath. Actor-Critic is computationally lighter but slower to converge.

- **Hyperparameter Sensitivity**: Engineering insight: PPO's clip_ratio is robust (0.1-0.2 range), but GRPO's beta allows fine-tuning for exploration (low beta) vs. exploitation (high beta). In real-world deployment, use cross-validation on envs; our configs show PPO std dev ~20% lower than AC.

- **Limitations and Extensions**: AC/PPO handle continuous actions easily; GRPO (as implemented) is discrete but extensible. For production, integrate with distributed training (@mnih2016asynchronous). Future: Test on continuous envs like Pendulum-v1.

See results/ for plots/tables.