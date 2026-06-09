from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from . import lb1200_strategy as lb

MISSION_KINDS = [
    "single",
    "capture",
    "swarm",
    "snipe",
    "rescue",
    "recapture",
    "reinforce",
    "crash_exploit",
    "other",
]

MISSION_RANK_FEATURE_INDEX = 0
MISSION_NORM_SCORE_FEATURE_INDEX = 1
MISSION_RAW_SCORE_FEATURE_INDEX = 2

GLOBAL_FEATURE_DIM = 18
MISSION_FEATURE_DIM = 58


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-8:
        return default
    return float(a) / float(b)


def _clip(x: float, lo: float = -10.0, hi: float = 10.0) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(lo, min(hi, x))


def _norm_score(score: float) -> float:
    # Scores can vary a lot; tanh keeps the scale stable.
    return math.tanh(float(score) / 5.0)


def _sum_attr(items: Iterable[Any], name: str) -> float:
    total = 0.0
    for item in items:
        total += float(getattr(item, name, 0.0) or 0.0)
    return total


def _owner_flags(owner: int, player: int) -> list[float]:
    return [
        1.0 if owner == player else 0.0,
        1.0 if owner == -1 else 0.0,
        1.0 if owner not in (-1, player) else 0.0,
    ]


def encode_global(world: Any) -> np.ndarray:
    planets = list(getattr(world, "planets", []) or [])
    my_planets = list(getattr(world, "my_planets", []) or [])
    enemy_planets = list(getattr(world, "enemy_planets", []) or [])
    neutral_planets = list(getattr(world, "neutral_planets", []) or [])
    fleets = list(getattr(world, "fleets", []) or [])
    my_fleets = list(getattr(world, "my_fleets", []) or [])
    enemy_fleets = list(getattr(world, "enemy_fleets", []) or [])

    my_ships = _sum_attr(my_planets, "ships") + _sum_attr(my_fleets, "ships")
    enemy_ships = _sum_attr(enemy_planets, "ships") + _sum_attr(enemy_fleets, "ships")
    neutral_ships = _sum_attr(neutral_planets, "ships")
    my_prod = _sum_attr(my_planets, "production")
    enemy_prod = _sum_attr(enemy_planets, "production")
    neutral_prod = _sum_attr(neutral_planets, "production")

    total_planets = max(1, len(planets))
    total_ships = max(1.0, my_ships + enemy_ships + neutral_ships)
    total_prod = max(1.0, my_prod + enemy_prod + neutral_prod)

    step = float(getattr(world, "step", 0) or 0)
    remaining = float(getattr(world, "remaining_steps", max(0.0, lb.TOTAL_STEPS - step)) or 0)

    feats = [
        step / float(lb.TOTAL_STEPS),
        remaining / float(lb.TOTAL_STEPS),
        len(my_planets) / total_planets,
        len(enemy_planets) / total_planets,
        len(neutral_planets) / total_planets,
        my_ships / total_ships,
        enemy_ships / total_ships,
        neutral_ships / total_ships,
        my_prod / total_prod,
        enemy_prod / total_prod,
        neutral_prod / total_prod,
        len(my_fleets) / max(1, len(fleets)),
        len(enemy_fleets) / max(1, len(fleets)),
        _safe_div(my_ships - enemy_ships, my_ships + enemy_ships + 1.0),
        _safe_div(my_prod - enemy_prod, my_prod + enemy_prod + 1.0),
        1.0 if getattr(world, "is_late", False) else 0.0,
        1.0 if getattr(world, "is_very_late", False) else 0.0,
        1.0 if getattr(world, "is_four_player", False) else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def encode_mission(world: Any, mission: Any, rank: int, top_k: int) -> np.ndarray:
    player = int(getattr(world, "player", 0) or 0)
    target = getattr(world, "planet_by_id", {}).get(getattr(mission, "target_id", -999))
    options = list(getattr(mission, "options", []) or [])

    if target is None:
        target_owner = -999
        target_ships = target_prod = target_radius = target_x = target_y = 0.0
        is_comet = is_static = 0.0
    else:
        target_owner = int(getattr(target, "owner", -1))
        target_ships = float(getattr(target, "ships", 0.0) or 0.0)
        target_prod = float(getattr(target, "production", 0.0) or 0.0)
        target_radius = float(getattr(target, "radius", 0.0) or 0.0)
        target_x = float(getattr(target, "x", 50.0) or 50.0)
        target_y = float(getattr(target, "y", 50.0) or 50.0)
        is_comet = 1.0 if getattr(target, "id", None) in getattr(world, "comet_ids", set()) else 0.0
        try:
            is_static = 1.0 if lb.is_static_planet(target) else 0.0
        except Exception:
            is_static = 0.0

    opt_count = len(options)
    opt_scores = [float(getattr(o, "score", 0.0) or 0.0) for o in options]
    opt_turns = [float(getattr(o, "turns", 0.0) or 0.0) for o in options]
    opt_needed = [float(getattr(o, "needed", 0.0) or 0.0) for o in options]
    opt_caps = [float(getattr(o, "send_cap", 0.0) or 0.0) for o in options]

    source_ships = []
    source_prod = []
    source_dist = []
    unique_sources = set()
    for option in options:
        src_id = getattr(option, "src_id", None)
        unique_sources.add(src_id)
        src = getattr(world, "planet_by_id", {}).get(src_id)
        if src is None or target is None:
            continue
        source_ships.append(float(getattr(src, "ships", 0.0) or 0.0))
        source_prod.append(float(getattr(src, "production", 0.0) or 0.0))
        try:
            source_dist.append(float(lb.planet_distance(src, target)))
        except Exception:
            pass

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def mx(xs: list[float]) -> float:
        return float(max(xs)) if xs else 0.0

    kind = getattr(mission, "kind", "other") or "other"
    if kind not in MISSION_KINDS:
        kind = "other"
    kind_one_hot = [1.0 if kind == k else 0.0 for k in MISSION_KINDS]

    mission_score = float(getattr(mission, "score", 0.0) or 0.0)
    mission_turns = float(getattr(mission, "turns", 0.0) or 0.0)
    eval_turn = max(1, int(math.ceil(mission_turns or (mean(opt_turns) if opt_turns else 1.0))))

    incoming_friend = incoming_enemy = 0.0
    try:
        for eta, owner, ships in getattr(world, "arrivals_by_planet", {}).get(getattr(mission, "target_id"), []):
            if int(math.ceil(float(eta))) > eval_turn:
                continue
            if int(owner) == player:
                incoming_friend += float(ships)
            elif int(owner) != -1:
                incoming_enemy += float(ships)
    except Exception:
        pass

    projected_owner = target_owner
    projected_ships = target_ships
    needed_at_arrival = 0.0
    try:
        projected_owner, projected_ships = world.projected_state(getattr(mission, "target_id"), eval_turn)
        needed_at_arrival = float(
            world.min_ships_to_own_at(
                getattr(mission, "target_id"),
                eval_turn,
                player,
                upper_bound=max(1, int(sum(opt_caps) or target_ships + 1)),
            )
        )
    except Exception:
        pass

    my_reaction = enemy_reaction = float(lb.HORIZON)
    try:
        my_reaction, enemy_reaction = world.reaction_times(getattr(mission, "target_id"))
    except Exception:
        pass
    my_reaction = min(float(my_reaction), float(lb.HORIZON))
    enemy_reaction = min(float(enemy_reaction), float(lb.HORIZON))

    source_keep = []
    source_surplus = []
    for option in options:
        src_id = getattr(option, "src_id", None)
        src = getattr(world, "planet_by_id", {}).get(src_id)
        if src is None:
            continue
        keep = float(getattr(world, "keep_needed_map", {}).get(src_id, 0.0) or 0.0)
        ships = float(getattr(src, "ships", 0.0) or 0.0)
        source_keep.append(keep)
        source_surplus.append(max(0.0, ships - keep))

    hold = getattr(world, "holds_full_map", {}).get(getattr(mission, "target_id"), True)
    fall_turn = getattr(world, "fall_turn_map", {}).get(getattr(mission, "target_id"), None)
    first_enemy = getattr(world, "first_enemy_map", {}).get(getattr(mission, "target_id"), None)

    feats = [
        float(rank) / max(1.0, float(top_k - 1)),
        _norm_score(mission_score),
        _clip(mission_score / 25.0, -5.0, 5.0),
        mission_turns / 120.0,
        opt_count / 4.0,
        len(unique_sources) / 4.0,
        target_x / 100.0,
        target_y / 100.0,
        target_radius / 10.0,
        target_ships / 300.0,
        target_prod / 10.0,
        is_comet,
        is_static,
        *(_owner_flags(target_owner, player)),
        mean(opt_scores) / 10.0,
        mx(opt_scores) / 10.0,
        mean(opt_turns) / 120.0,
        mx(opt_turns) / 120.0,
        mean(opt_needed) / 300.0,
        sum(opt_needed) / 600.0,
        mean(opt_caps) / 300.0,
        sum(opt_caps) / 600.0,
        mean(source_ships) / 300.0,
        mx(source_ships) / 300.0,
        mean(source_prod) / 10.0,
        mean(source_dist) / 100.0,
        _safe_div(sum(opt_caps), target_ships + 1.0),
        _safe_div(sum(opt_needed), sum(opt_caps) + 1.0),
        incoming_friend / 300.0,
        incoming_enemy / 300.0,
        _safe_div(incoming_friend - incoming_enemy, incoming_friend + incoming_enemy + 1.0),
        projected_ships / 300.0,
        needed_at_arrival / 300.0,
        _safe_div(sum(opt_caps) - needed_at_arrival, needed_at_arrival + 1.0),
        *(_owner_flags(int(projected_owner), player)),
        my_reaction / float(lb.HORIZON),
        enemy_reaction / float(lb.HORIZON),
        _safe_div(enemy_reaction - my_reaction, float(lb.HORIZON)),
        mean(source_keep) / 300.0,
        mean(source_surplus) / 300.0,
        1.0 if hold else 0.0,
        (float(fall_turn) / float(lb.HORIZON)) if fall_turn is not None else 1.0,
        (float(first_enemy) / float(lb.HORIZON)) if first_enemy is not None else 1.0,
    ]
    feats.extend(kind_one_hot)

    if len(feats) < MISSION_FEATURE_DIM:
        feats.extend([0.0] * (MISSION_FEATURE_DIM - len(feats)))
    else:
        feats = feats[:MISSION_FEATURE_DIM]
    return np.asarray(feats, dtype=np.float32)


def encode_topk(world: Any, missions: list[Any], top_k: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_x = encode_global(world)
    mission_x = np.zeros((top_k, MISSION_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((top_k,), dtype=np.float32)

    for i, mission in enumerate((missions or [])[:top_k]):
        mission_x[i] = encode_mission(world, mission, rank=i, top_k=top_k)
        mask[i] = 1.0
    return global_x, mission_x, mask
