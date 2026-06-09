from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import lb1200_strategy as lb
from .mission_encoder import encode_topk
from .mission_interface import generate_top_missions, infer_action_from_moves


def _load_replay(path: str) -> dict[str, Any] | None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get("name") == "orbit_wars" and isinstance(data.get("steps"), list):
        return data
    return None


def _agent_indices(replay: dict[str, Any], mode: str) -> list[int]:
    rewards = replay.get("rewards") or []
    if mode == "all":
        return list(range(len(rewards) or 2))
    if mode == "winner":
        best = max(float(r or 0.0) for r in rewards)
        return [i for i, r in enumerate(rewards) if float(r or 0.0) == best]
    if mode == "loser":
        worst = min(float(r or 0.0) for r in rewards)
        return [i for i, r in enumerate(rewards) if float(r or 0.0) == worst]
    return [int(mode)]


def _observation_with_step(obs: Any, step_idx: int) -> Any:
    if isinstance(obs, dict) and "step" not in obs:
        obs = copy.deepcopy(obs)
        obs["step"] = int(step_idx)
    return obs


def collect_from_replays(paths: list[str], top_k: int, agent_mode: str, skip_unmatched: bool):
    globals_x = []
    missions_x = []
    masks = []
    actions = []
    meta = []
    matched = unmatched = 0

    for path in paths:
        replay = _load_replay(path)
        if replay is None:
            print(f"skip non-replay json: {path}")
            continue
        agent_ids = set(_agent_indices(replay, agent_mode))
        steps = replay.get("steps") or []

        for step_idx, states in enumerate(steps[:-1]):
            if not isinstance(states, list):
                continue
            for agent_id, state in enumerate(states):
                if agent_id not in agent_ids or not isinstance(state, dict):
                    continue
                obs = state.get("observation")
                if obs is None:
                    continue

                moves = state.get("action") or []
                obs = _observation_with_step(obs, step_idx)
                try:
                    world = lb.build_world(obs)
                    top_missions = generate_top_missions(world, top_k=top_k)
                    action = infer_action_from_moves(world, top_missions, moves, top_k=top_k)
                except Exception:
                    unmatched += 1
                    continue

                if action is None:
                    unmatched += 1
                    if skip_unmatched:
                        continue
                    action = 0
                else:
                    matched += 1

                g, m, mask = encode_topk(world, top_missions, top_k=top_k)
                globals_x.append(g)
                missions_x.append(m)
                masks.append(mask)
                actions.append(action)
                meta.append((str(path), int(step_idx), int(agent_id)))

    return {
        "global_x": np.asarray(globals_x, dtype=np.float32),
        "mission_x": np.asarray(missions_x, dtype=np.float32),
        "mask": np.asarray(masks, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.int64),
        "top_k": np.asarray([top_k], dtype=np.int64),
        "meta": np.asarray(meta, dtype=object),
        "matched": matched,
        "unmatched": unmatched,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("replays", nargs="+")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--agent", type=str, default="winner", help="'winner', 'loser', 'all', or numeric agent index")
    p.add_argument("--save", type=str, default="data/replay_bc_dataset.npz")
    p.add_argument("--keep-unmatched-as-stop", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = collect_from_replays(
        args.replays,
        top_k=args.top_k,
        agent_mode=args.agent,
        skip_unmatched=not args.keep_unmatched_as_stop,
    )
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **out)
    print(
        f"saved replay dataset: {save_path} | samples={len(out['action'])} "
        f"matched={out['matched']} unmatched={out['unmatched']}"
    )


if __name__ == "__main__":
    main()
