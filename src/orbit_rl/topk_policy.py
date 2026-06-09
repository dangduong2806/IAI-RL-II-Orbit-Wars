from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .mission_encoder import (
    GLOBAL_FEATURE_DIM,
    MISSION_FEATURE_DIM,
    MISSION_RANK_FEATURE_INDEX,
    MISSION_RAW_SCORE_FEATURE_INDEX,
)


class TopKMissionPolicy(nn.Module):
    """Actor-critic policy selecting STOP or one mission from top-K candidates."""

    def __init__(self, top_k: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.top_k = int(top_k)
        self.global_dim = GLOBAL_FEATURE_DIM
        self.mission_dim = MISSION_FEATURE_DIM
        self.heuristic_prior_strength = 0.0
        self.heuristic_prior_rank_weight = 0.25
        self.heuristic_stop_penalty = 0.0

        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mission_encoder = nn.Sequential(
            nn.Linear(self.mission_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.mission_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def configure_heuristic_prior(
        self,
        strength: float = 0.0,
        rank_weight: float = 0.25,
        stop_penalty: float = 0.0,
    ):
        self.heuristic_prior_strength = max(0.0, float(strength))
        self.heuristic_prior_rank_weight = max(0.0, float(rank_weight))
        self.heuristic_stop_penalty = max(0.0, float(stop_penalty))

    def _heuristic_logit_prior(self, mission_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, top_k, _ = mission_x.shape
        prior = mission_x.new_zeros((batch, top_k + 1))
        strength = float(getattr(self, "heuristic_prior_strength", 0.0) or 0.0)
        stop_penalty = float(getattr(self, "heuristic_stop_penalty", 0.0) or 0.0)
        if strength <= 0.0 and stop_penalty <= 0.0:
            return prior

        valid = mask.float().clamp(0.0, 1.0)
        has_valid = (valid.sum(dim=1, keepdim=True) > 0.0).float()

        score = mission_x[:, :, MISSION_RAW_SCORE_FEATURE_INDEX].float()
        score = score.masked_fill(valid <= 0.0, 0.0)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        score_mean = (score * valid).sum(dim=1, keepdim=True) / denom
        centered_score = (score - score_mean) * valid

        rank = mission_x[:, :, MISSION_RANK_FEATURE_INDEX].float().clamp(0.0, 1.0)
        rank_bonus = (1.0 - rank) * valid
        rank_weight = float(getattr(self, "heuristic_prior_rank_weight", 0.25) or 0.0)

        prior[:, 1:] = strength * (centered_score + rank_weight * rank_bonus)
        best_score = score.masked_fill(valid <= 0.0, -1e9).max(dim=1, keepdim=True).values
        has_good_mission = has_valid * (best_score > 0.0).float()
        prior[:, :1] = -stop_penalty * has_good_mission
        return prior

    def forward(self, global_x: torch.Tensor, mission_x: torch.Tensor, mask: torch.Tensor):
        if global_x.dim() == 1:
            global_x = global_x.unsqueeze(0)
        if mission_x.dim() == 2:
            mission_x = mission_x.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        g = self.global_encoder(global_x.float())
        batch, top_k, mission_dim = mission_x.shape
        m = self.mission_encoder(mission_x.float().reshape(batch * top_k, mission_dim))
        m = m.reshape(batch, top_k, -1)

        g_expand = g.unsqueeze(1).expand(-1, top_k, -1)
        pair = torch.cat([g_expand, m], dim=-1)
        mission_logits = self.mission_head(pair).squeeze(-1)
        mission_logits = mission_logits.masked_fill(mask.float() <= 0.0, -1e9)

        stop_logit = self.stop_head(g)
        logits = torch.cat([stop_logit, mission_logits], dim=1)
        logits = logits + self._heuristic_logit_prior(mission_x, mask)
        value = self.value_head(g).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, global_x, mission_x, mask, deterministic: bool = False):
        logits, value = self.forward(global_x, mission_x, mask)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value, dist.entropy()


def save_policy(path: str, policy: TopKMissionPolicy, extra: dict | None = None):
    payload = {
        "model_state_dict": policy.state_dict(),
        "top_k": policy.top_k,
        "global_dim": policy.global_dim,
        "mission_dim": policy.mission_dim,
        "heuristic_prior_strength": float(getattr(policy, "heuristic_prior_strength", 0.0) or 0.0),
        "heuristic_prior_rank_weight": float(getattr(policy, "heuristic_prior_rank_weight", 0.25) or 0.25),
        "heuristic_stop_penalty": float(getattr(policy, "heuristic_stop_penalty", 0.0) or 0.0),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_policy(path: str, map_location="cpu") -> TopKMissionPolicy:
    payload = torch.load(path, map_location=map_location)
    top_k = int(payload.get("top_k", 8))
    policy = TopKMissionPolicy(top_k=top_k)
    state = payload.get("model_state_dict", payload)
    current = policy.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in current and tuple(value.shape) == tuple(current[name].shape)
    }
    policy.load_state_dict(compatible, strict=False)
    policy.configure_heuristic_prior(
        strength=float(payload.get("heuristic_prior_strength", 0.0) or 0.0),
        rank_weight=float(payload.get("heuristic_prior_rank_weight", 0.25) or 0.25),
        stop_penalty=float(payload.get("heuristic_stop_penalty", 0.0) or 0.0),
    )
    policy.checkpoint_extra = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "top_k", "global_dim", "mission_dim"}
    }
    return policy
