import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from src.common.utils import compute_gae
from typing import Tuple, List
from torch.cuda.amp import autocast, GradScaler

class PPO(nn.Module):
    """
    Proximal Policy Optimization for RL environments.
    - Actor and Critic networks.
    - Clipped surrogate loss (Schulman et al., 2017).
    - GAE advantages.
    - Citation: @schulman2017proximal
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64, lr: float = 3e-4, use_amp: bool = True):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value

    def select_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).item()
        return action.item(), log_prob, value.item()

    def update(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        old_log_probs: List[float],
        rewards: List[float],
        dones: List[bool],
        values: List[float],
        gamma: float,
        lam: float,
        clip_ratio: float,
        ent_coef: float,
        epochs: int,
        batch_size: int
    ) -> float:
        # Ensure tensors are on the model device
        device = next(self.parameters()).device
        with torch.no_grad():
            _, next_value = self.forward(states[-1].to(device))
            values.append(next_value.item())
        advantages = compute_gae(rewards, values[:-1], dones, gamma, lam)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32, device=device)

        # Flatten and move to device with correct dtypes
        states = torch.cat(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)

        # Mini-batch updates
        dataset = torch.utils.data.TensorDataset(states, actions, advantages, returns, old_log_probs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for _ in range(epochs):
            for batch in loader:
                s_b, a_b, adv_b, ret_b, old_lp_b = batch
                self.optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    logits, vals = self.forward(s_b)
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(a_b)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - old_lp_b)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(vals.squeeze(), ret_b)
                    loss = policy_loss + 0.5 * value_loss - ent_coef * entropy

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                losses.append(loss.item())
        return sum(losses) / len(losses)