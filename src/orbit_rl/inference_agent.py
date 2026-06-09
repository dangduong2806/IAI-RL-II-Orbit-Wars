from __future__ import annotations

import os
from pathlib import Path

import torch

from . import lb1200_strategy as lb
from .ppo_mission_agent import TopKMissionAgent

_AGENT = None
_LOAD_TRIED = False


def _candidate_paths():
    env_path = os.environ.get("ORBIT_PPO_MISSION_MODEL")
    paths = []
    if env_path:
        paths.append(env_path)
    paths.extend([
        "checkpoints/ppo_selfplay_best.pt",
        "checkpoints/ppo_selfplay.pt",
        "checkpoints/bc_policy.pt",
        "/kaggle_simulations/agent/checkpoints/ppo_selfplay_best.pt",
        "/kaggle_simulations/agent/checkpoints/ppo_selfplay.pt",
        "/kaggle_simulations/agent/checkpoints/bc_policy.pt",
    ])
    return paths


def _load_agent():
    global _AGENT, _LOAD_TRIED
    if _LOAD_TRIED:
        return _AGENT
    _LOAD_TRIED = True
    for path in _candidate_paths():
        if path and Path(path).exists():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _AGENT = TopKMissionAgent.from_checkpoint(path, device=device)
                _AGENT.policy.eval()
                return _AGENT
            except Exception:
                _AGENT = None
    return None


def agent(obs, config=None):
    model_agent = _load_agent()
    if model_agent is not None:
        try:
            return model_agent.act(obs, config=config, deterministic=True, training=False)
        except Exception:
            pass
    # Safety fallback: never crash the submission.
    try:
        return lb.agent(obs, config)
    except Exception:
        return []
