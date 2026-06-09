from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .mission_encoder import encode_topk
from .mission_interface import build_world, generate_top_missions, plan_with_selector, select_mission_bundle
from .topk_policy import TopKMissionPolicy, load_policy


def _read(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class MissionDecision:
    global_x: np.ndarray
    mission_x: np.ndarray
    mask: np.ndarray
    action: int
    logprob: float
    value: float


class TopKMissionAgent:
    """lb-1200 + PPO top-K mission selector.

    Action convention:
        0    = STOP / execute no top-K mission
        1..K = execute the chosen mission rank from lb-1200's sorted top-K list

    The selected mission can be bundled with compatible high-score follow-ups.
    PPO still learns the primary action; the bundle lets lb-1200 execute more
    than one disjoint opportunity in a turn.
    """

    def __init__(self, top_k: int = 8, device: str | torch.device = "cpu", policy: TopKMissionPolicy | None = None,
                 safe_fallback: bool = False, max_selected_missions: int = 2,
                 min_extra_score_ratio: float = 0.65):
        self.top_k = int(top_k)
        self.device = torch.device(device)
        self.policy = policy if policy is not None else TopKMissionPolicy(top_k=self.top_k)
        self.policy.to(self.device)
        self.safe_fallback = bool(safe_fallback)
        self.max_selected_missions = max(1, int(max_selected_missions))
        self.min_extra_score_ratio = float(min_extra_score_ratio)
        self._last_decisions: list[MissionDecision] = []

    @classmethod
    def from_checkpoint(cls, path: str, device: str | torch.device = "cpu", safe_fallback: bool | None = None):
        policy = load_policy(path, map_location=device)
        extra = getattr(policy, "checkpoint_extra", {}) or {}
        if safe_fallback is None:
            safe_fallback = bool(extra.get("safe_fallback", False))
        return cls(
            top_k=policy.top_k,
            device=device,
            policy=policy,
            safe_fallback=safe_fallback,
            max_selected_missions=int(extra.get("max_selected_missions", 2)),
            min_extra_score_ratio=float(extra.get("min_extra_score_ratio", 0.65)),
        )

    def consume_last_decisions(self) -> list[MissionDecision]:
        out = self._last_decisions
        self._last_decisions = []
        return out

    def _select_action(self, world: Any, missions: list[Any], deterministic: bool, training: bool) -> int:
        global_x, mission_x, mask = encode_topk(world, missions, top_k=self.top_k)
        g = torch.tensor(global_x, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.tensor(mission_x, dtype=torch.float32, device=self.device).unsqueeze(0)
        ms = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        if training:
            action_t, logprob_t, value_t, _ = self.policy.act(g, m, ms, deterministic=deterministic)
        else:
            with torch.no_grad():
                action_t, logprob_t, value_t, _ = self.policy.act(g, m, ms, deterministic=deterministic)

        action = int(action_t.item())
        if training:
            self._last_decisions.append(
                MissionDecision(
                    global_x=global_x,
                    mission_x=mission_x,
                    mask=mask,
                    action=action,
                    logprob=float(logprob_t.item()),
                    value=float(value_t.item()),
                )
            )
        return action

    def mission_selector(self, world=None, missions=None, top_k=None, deterministic: bool = False, training: bool = False):
        missions = list(missions or [])
        if not missions:
            return []
        action = self._select_action(world, missions, deterministic=deterministic, training=training)
        return select_mission_bundle(
            missions,
            action,
            top_k=self.top_k,
            safe_fallback=self.safe_fallback and not training,
            max_missions=self.max_selected_missions,
            min_extra_score_ratio=self.min_extra_score_ratio,
        )

    def estimate_value(self, obs: Any, deadline=None) -> float:
        try:
            world = build_world(obs)
            if not getattr(world, "my_planets", []):
                return 0.0
            missions = generate_top_missions(world, top_k=self.top_k, deadline=deadline)
            global_x, mission_x, mask = encode_topk(world, missions, top_k=self.top_k)
            g = torch.tensor(global_x, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.tensor(mission_x, dtype=torch.float32, device=self.device).unsqueeze(0)
            ms = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, value = self.policy(g, m, ms)
            return float(value.item())
        except Exception:
            return 0.0

    def act(self, obs: Any, config: Any = None, deterministic: bool = True, training: bool = False):
        world = build_world(obs)
        if not getattr(world, "my_planets", []):
            return []
        deadline = None
        if config is not None:
            start_time = time.perf_counter()
            act_timeout = float(_read(config, "actTimeout", 1.0) or 1.0)
            soft_budget = min(0.82, max(0.55, act_timeout * 0.82))
            deadline = start_time + soft_budget
        return plan_with_selector(
            world,
            selector=lambda world=None, missions=None, top_k=None, **kwargs: self.mission_selector(
                world=world,
                missions=missions,
                top_k=top_k,
                deterministic=deterministic,
                training=training,
            ),
            top_k=self.top_k,
            deadline=deadline,
        )
