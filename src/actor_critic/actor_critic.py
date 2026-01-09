import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from src.common.utils import compute_gae
from typing import Tuple, List

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, lr: float = 3e-4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, action_dim)  # Logits for policy
        self.critic_head = nn.Linear(hidden, 1)  # Value estimate
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
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
        log_probs: List[float],
        rewards: List[float],
        dones: List[bool],
        values: List[float],
        gamma: float,
        lam: float,
        ent_coef: float,
        clip_ratio: float,
        epochs: int
    ) -> float:
        # Compute GAE and returns
        device = next(self.parameters()).device
        with torch.no_grad():
            _, next_value = self.forward(states[-1].to(device))
            values.append(next_value.item())
        advantages = compute_gae(rewards, values[:-1], dones, gamma, lam)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32, device=device)

        # Flatten tensors and move to model device
        states = torch.cat(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=device)

        losses = []
        for _ in range(epochs):
            logits, vals = self.forward(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(vals.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss - ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)
