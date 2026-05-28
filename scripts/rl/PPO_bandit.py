# ppo_bandit.py
"""
ppo_bandit.py

一次决策（contextual bandit / 1-step episode）版 PPO。
- 每个 episode 只有一次 act：a = pi(x)
- roll-out 跑完整段仿真得到一个标量 reward R
- advantage A = R - V(x)
- PPO 更新策略（clip objective）与 value network

注意：
- 这里用“动作裁剪到 [-1,1]”的近似做法（工程上足够用）
- 如果想要严格的 squashed-Gaussian logprob 修正，可后续再加
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math


@dataclass
class PPOConfig:
    obs_dim: int
    act_dim: int

    hidden_sizes: Tuple[int, int] = (128, 128)
    lr: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    update_epochs: int = 10
    minibatch_size: int = 64


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, int], out_dim: int):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOBanditAgent(nn.Module):
    """
    Actor-Critic for 1-step PPO.
    Actor outputs mean and log_std for Normal distribution over actions.
    """
    def __init__(self, cfg: PPOConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        self.actor_mean = MLP(cfg.obs_dim, cfg.hidden_sizes, cfg.act_dim)
        # 独立可训练的 log_std（每个动作维度一个）
        self.actor_logstd = nn.Parameter(torch.zeros(cfg.act_dim))
        self.critic = MLP(cfg.obs_dim, cfg.hidden_sizes, 1)

        self.to(self.device)

        self.optim = optim.Adam(self.parameters(), lr=cfg.lr)

    def _atanh(self, x: torch.Tensor) -> torch.Tensor:
        # clamp 避免 atanh 爆炸
        eps = 1e-6
        x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _squashed_logprob(
        self,
        dist: torch.distributions.Normal,
        a_raw: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        # log p(a_raw) - sum log|da/da_raw|, where a=tanh(a_raw)
        logp_raw = dist.log_prob(a_raw).sum(dim=-1)
        # Jacobian correction: log(1 - tanh^2) = log(1 - a^2)
        eps = 1e-6
        log_det = torch.log(1.0 - a.pow(2) + eps).sum(dim=-1)
        return logp_raw - log_det

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean = self.actor_mean(o)
        logstd = self.actor_logstd.unsqueeze(0)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)

        a_raw = dist.sample()
        a = torch.tanh(a_raw)  # (-1,1)
        logp = self._squashed_logprob(dist, a_raw, a)
        v = self.critic(o).squeeze(-1)

        return (
            a.squeeze(0).cpu().numpy(),
            float(logp.item()),
            float(v.item()),
        )


    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean = self.actor_mean(o)
        logstd = self.actor_logstd.unsqueeze(0)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)

        a_raw = mean
        a = torch.tanh(a_raw)
        logp = self._squashed_logprob(dist, a_raw, a)
        v = self.critic(o).squeeze(-1)

        return (
            a.squeeze(0).cpu().numpy(),
            float(logp.item()),
            float(v.item()),
        )

    @torch.no_grad()
    def act_batch(self, obs_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        o = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        if o.ndim == 1:
            o = o.unsqueeze(0)

        mean = self.actor_mean(o)
        logstd = self.actor_logstd.unsqueeze(0).expand_as(mean)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)

        a_raw = dist.sample()
        a = torch.tanh(a_raw)
        logp = self._squashed_logprob(dist, a_raw, a)
        v = self.critic(o).squeeze(-1)

        return (a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy())


    def _evaluate_actions(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs_t)
        logstd = self.actor_logstd.unsqueeze(0).expand_as(mean)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)

        a = act_t
        a_raw = self._atanh(a)
        logp = self._squashed_logprob(dist, a_raw, a)

        # entropy 对 squashed 分布严格计算较麻烦；这里保留 Normal entropy 作为近似项
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy

    def _value(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.critic(obs_t).squeeze(-1)

    def update(self, buffer: "RolloutBuffer") -> dict:
        """
        用 buffer 中的一批 (obs, act, logp_old, value_old, reward) 做 PPO 更新。
        """
        cfg = self.cfg
        data = buffer.to_torch(self.device)

        obs = data["obs"]      # [N,obs_dim]
        act = data["act"]      # [N,act_dim]
        logp_old = data["logp"]  # [N]
        v_old = data["val"]      # [N]
        r = data["rew"]          # [N]

        # 1-step：return=reward
        ret = r
        adv = (ret - v_old).detach()
        # advantage 归一化（很重要，训练更稳）
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = obs.shape[0]
        idxs = torch.arange(N, device=self.device)

        losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}

        for _ in range(cfg.update_epochs):
            perm = idxs[torch.randperm(N)]
            for start in range(0, N, cfg.minibatch_size):
                mb = perm[start:start + cfg.minibatch_size]
                obs_b = obs[mb]
                act_b = act[mb]
                logp_old_b = logp_old[mb]
                adv_b = adv[mb]
                ret_b = ret[mb]

                logp, ent = self._evaluate_actions(obs_b, act_b)
                ratio = torch.exp(logp - logp_old_b)

                # clipped surrogate objective
                clip = cfg.clip_ratio
                obj1 = ratio * adv_b
                obj2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_b
                policy_loss = -torch.min(obj1, obj2).mean()

                v = self._value(obs_b)
                value_loss = ((v - ret_b) ** 2).mean()

                entropy_loss = -ent.mean()

                total_loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                self.optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), cfg.max_grad_norm)
                self.optim.step()

                losses["policy"] += float(policy_loss.item())
                losses["value"] += float(value_loss.item())
                losses["entropy"] += float(entropy_loss.item())
                losses["total"] += float(total_loss.item())

        # 平均一下（便于打印）
        denom = max(1, cfg.update_epochs * math.ceil(N / cfg.minibatch_size))
        for k in losses:
            losses[k] /= denom
        return losses


class RolloutBuffer:
    """
    1-step rollout buffer: 每条样本 = 一个 episode 的 (obs, act, logp, val, reward).
    """
    def __init__(self, obs_dim: int, act_dim: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._obs: List[np.ndarray] = []
        self._act: List[np.ndarray] = []
        self._logp: List[float] = []
        self._val: List[float] = []
        self._rew: List[float] = []

    def add(self, obs: np.ndarray, act: np.ndarray, logp: float, val: float, rew: float) -> None:
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._act.append(np.asarray(act, dtype=np.float32))
        self._logp.append(float(logp))
        self._val.append(float(val))
        self._rew.append(float(rew))

    def __len__(self) -> int:
        return len(self._rew)

    def clear(self) -> None:
        self._obs.clear()
        self._act.clear()
        self._logp.clear()
        self._val.clear()
        self._rew.clear()

    def to_torch(self, device: torch.device) -> dict:
        obs = torch.as_tensor(np.stack(self._obs, axis=0), dtype=torch.float32, device=device)
        act = torch.as_tensor(np.stack(self._act, axis=0), dtype=torch.float32, device=device)
        logp = torch.as_tensor(np.array(self._logp, dtype=np.float32), dtype=torch.float32, device=device)
        val = torch.as_tensor(np.array(self._val, dtype=np.float32), dtype=torch.float32, device=device)
        rew = torch.as_tensor(np.array(self._rew, dtype=np.float32), dtype=torch.float32, device=device)
        return {"obs": obs, "act": act, "logp": logp, "val": val, "rew": rew}
