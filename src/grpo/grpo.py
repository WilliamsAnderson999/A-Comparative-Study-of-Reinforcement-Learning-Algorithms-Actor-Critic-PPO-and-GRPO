import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.distributions.kl as kl
from typing import Tuple
import gymnasium as gym
from tqdm import trange
import wandb
from src.common.utils import set_seed

class ActorNetwork(nn.Module):
    """
    Actor network for GRPO.
    - Orthogonal init for stability (Engstrom et al., 2020).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GRPOTrainer:
    def __init__(
        self, env: gym.Env, actor: nn.Module, optimizer: optim.Optimizer,
        clip_ratio: float = 0.2, beta: float = 0.001, gamma: float = 0.99,
        epochs: int = 10, batch_size: int = 32, group_size: int = 200
    ):
        self.env = env
        self.actor = actor
        self.optimizer = optimizer
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.group_size = group_size
        self.device = next(actor.parameters()).device
        self.ep_rewards = []

    def _calc_returns(self, rewards: list, dones: list) -> torch.Tensor:
        returns = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - int(d))
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def collect_rollout(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, dones = [], [], [], []
        old_logits = []
        state, _ = self.env.reset()
        state = torch.FloatTensor(state).to(self.device)
        ep_rew = 0.0
        done = False

        for _ in range(self.group_size):
            if done:
                state, _ = self.env.reset()
                state = torch.FloatTensor(state).to(self.device)
                self.ep_rewards.append(ep_rew)
                ep_rew = 0.0
                done = False

            with torch.no_grad():
                logits = self.actor(state.unsqueeze(0)).squeeze(0)
                dist = Categorical(logits=logits)
                action = dist.sample()

            next_state, rew, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            ep_rew += rew

            states.append(state)
            actions.append(action)
            rewards.append(rew)
            dones.append(done)
            old_logits.append(logits)

            state = torch.FloatTensor(next_state).to(self.device)

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_logits = torch.stack(old_logits)
        returns = self._calc_returns(rewards, dones)
        advantages = returns - returns.mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalized

        return states, actions, advantages, old_logits

    def train(self, total_updates: int = 1000):
        set_seed()
        self.actor.train()
        for update in range(total_updates):
            states, actions, advantages, old_logits = self.collect_rollout()

            dataset = torch.utils.data.TensorDataset(states, actions, advantages, old_logits)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            policy_losses, kl_divs = [], []
            for _ in range(self.epochs):
                for batch in loader:
                    s_b, a_b, adv_b, old_l_b = batch
                    new_logits = self.actor(s_b)
                    old_dist = Categorical(logits=old_l_b.detach())
                    new_dist = Categorical(logits=new_logits)

                    logp_new = new_dist.log_prob(a_b)
                    logp_old = old_dist.log_prob(a_b).detach()
                    ratio = torch.exp(logp_new - logp_old)

                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()

                    kl_div = kl.kl_divergence(old_dist, new_dist).mean()
                    loss = policy_loss + self.beta * kl_div

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer.step()

                    policy_losses.append(policy_loss.item())
                    kl_divs.append(kl_div.item())

            if self.ep_rewards:
                avg_rew = sum(self.ep_rewards[-20:]) / len(self.ep_rewards[-20:])
                wandb.log({"update": update, "avg_reward": avg_rew, "policy_loss": sum(policy_losses)/len(policy_losses), "kl": sum(kl_divs)/len(kl_divs)})
