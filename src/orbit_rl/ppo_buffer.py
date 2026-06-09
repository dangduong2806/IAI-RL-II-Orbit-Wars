from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PPOBuffer:
    global_x: list = field(default_factory=list)
    mission_x: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def add(self, global_x, mission_x, mask, action, logprob, value, reward, done):
        self.global_x.append(np.asarray(global_x, dtype=np.float32))
        self.mission_x.append(np.asarray(mission_x, dtype=np.float32))
        self.mask.append(np.asarray(mask, dtype=np.float32))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.actions)

    def clear(self):
        self.global_x.clear(); self.mission_x.clear(); self.mask.clear()
        self.actions.clear(); self.logprobs.clear(); self.values.clear()
        self.rewards.clear(); self.dones.clear()

    def as_arrays(self, gamma: float = 0.99, gae_lambda: float = 0.95, bootstrap_value: float = 0.0):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values + [float(bootstrap_value)], dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        adv = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "global_x": np.asarray(self.global_x, dtype=np.float32),
            "mission_x": np.asarray(self.mission_x, dtype=np.float32),
            "mask": np.asarray(self.mask, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "old_logprobs": np.asarray(self.logprobs, dtype=np.float32),
            "returns": returns.astype(np.float32),
            "advantages": adv.astype(np.float32),
        }
