from __future__ import annotations

from typing import Any, Callable

from kaggle_environments import make

from .reward import extract_observation, extract_reward, extract_status, shaped_reward


class OrbitWarsSelfPlayEnv:
    def __init__(self, opponent_fn: Callable[[Any], list], seed: int | None = None, debug: bool = False):
        self.opponent_fn = opponent_fn
        self.seed = seed
        self.debug = debug
        self.env = None
        self.last_obs = None
        self.last_opp_obs = None

    def reset(self):
        cfg = {}
        if self.seed is not None:
            cfg["seed"] = int(self.seed)
            cfg["randomSeed"] = int(self.seed)
        self.env = make("orbit_wars", configuration=cfg, debug=self.debug)
        states = self.env.reset(num_agents=2)
        self.last_obs = extract_observation(states[0])
        self.last_opp_obs = extract_observation(states[1])
        return self.last_obs

    def step(self, learner_moves: list):
        if self.env is None:
            raise RuntimeError("Call reset() first.")
        opponent_moves = self.opponent_fn(self.last_opp_obs)
        prev_obs = self.last_obs
        states = self.env.step([learner_moves, opponent_moves])
        learner_state = states[0]
        opp_state = states[1]
        self.last_obs = extract_observation(learner_state)
        self.last_opp_obs = extract_observation(opp_state)
        done = extract_status(learner_state) != "ACTIVE"
        terminal = extract_reward(learner_state) if done else 0.0
        reward = shaped_reward(prev_obs, self.last_obs, done=done, terminal_reward=terminal)
        return self.last_obs, reward, done, {"terminal_reward": terminal, "status": extract_status(learner_state)}
