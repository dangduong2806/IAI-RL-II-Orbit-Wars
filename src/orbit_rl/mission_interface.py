from __future__ import annotations

import math
from typing import Any, Callable

from . import lb1200_strategy as lb


def build_world(obs: Any):
    return lb.build_world(obs)


def heuristic_agent(obs: Any, config: Any = None):
    return lb.agent(obs, config)


def generate_top_missions(world: Any, top_k: int = 8, deadline=None) -> list[Any]:
    """Collect top-K missions from the refactored lb-1200 planner.

    This function uses the RL hook inside lb.plan_moves. The selector receives
    the complete sorted mission list and returns [] so no candidate mission is
    executed during this collection call.
    """
    box: dict[str, list[Any]] = {}

    def collector(world=None, missions=None, top_k=None, **kwargs):
        box["missions"] = list(missions or [])[: int(top_k or 8)]
        return []

    try:
        lb.plan_moves(world, deadline=deadline, mission_selector=collector, top_k=top_k)
    except Exception:
        # Mission collection should never crash training/submission.
        return []
    return box.get("missions", [])


def plan_with_selector(world: Any, selector: Callable, top_k: int = 8, deadline=None):
    """Execute lb-1200 planner with a policy selector inserted after mission generation."""
    return lb.plan_moves(world, deadline=deadline, mission_selector=selector, top_k=top_k)


def _angle_diff(a: float, b: float) -> float:
    d = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def _move_matches_option(world: Any, move: list, mission: Any, angle_tol: float = 0.35) -> bool:
    if not move or len(move) < 3:
        return False
    src_id = int(move[0])
    move_angle = float(move[1])
    for option in getattr(mission, "options", []) or []:
        if int(getattr(option, "src_id", -999)) != src_id:
            continue
        # First try the option's cached aim.
        opt_angle = getattr(option, "angle", None)
        if opt_angle is not None and _angle_diff(move_angle, float(opt_angle)) <= angle_tol:
            return True
        # The final planner can re-aim after choosing exact send size, so also
        # try a fresh plan_shot to the target.
        try:
            send_guess = max(1, int(getattr(option, "needed", 1) or 1))
            aim = world.plan_shot(src_id, int(getattr(mission, "target_id")), send_guess)
            if aim is not None and _angle_diff(move_angle, float(aim[0])) <= angle_tol:
                return True
        except Exception:
            pass
    return False


def infer_teacher_action(world: Any, missions: list[Any], top_k: int = 8) -> int | None:
    """Approximate which top-K mission lb-1200 actually executed.

    Action convention:
        0      = STOP / no selected top-K mission
        1..K   = choose mission at rank action-1

    lb-1200 executes several missions per turn. For behavior cloning, we label
    the first top-K mission whose source/angle matches one of the teacher moves.
    If none matches, return None so dataset collection can skip the noisy sample.
    """
    missions = list(missions or [])[:top_k]
    if not missions:
        return 0
    try:
        teacher_moves = lb.plan_moves(world, deadline=None)
    except Exception:
        return None

    for i, mission in enumerate(missions):
        for move in teacher_moves or []:
            if _move_matches_option(world, move, mission):
                return i + 1
    return None


def infer_action_from_moves(world: Any, missions: list[Any], moves: list[Any], top_k: int = 8) -> int | None:
    """Infer which top-K mission an arbitrary move list appears to execute.

    Returns:
        0       if both the bot and top-K selector effectively stop.
        1..K    for the first top-K mission whose source/angle matches a move.
        None    when the move list does not map to the generated top-K missions.
    """
    missions = list(missions or [])[:top_k]
    moves = list(moves or [])
    if not missions:
        return 0 if not moves else None
    if not moves:
        return 0

    for i, mission in enumerate(missions):
        for move in moves:
            if _move_matches_option(world, move, mission):
                return i + 1
    return None


def select_single_mission(missions: list[Any], action: int, top_k: int = 8, safe_fallback: bool = False) -> list[Any]:
    top = list(missions or [])[:top_k]
    action = int(action)
    if action <= 0:
        return top[:1] if safe_fallback and top else []
    idx = action - 1
    if 0 <= idx < len(top):
        return [top[idx]]
    return top[:1] if safe_fallback and top else []


def _mission_sources(mission: Any) -> set[int]:
    return {
        int(getattr(option, "src_id"))
        for option in getattr(mission, "options", []) or []
        if getattr(option, "src_id", None) is not None
    }


def select_mission_bundle(
    missions: list[Any],
    action: int,
    top_k: int = 8,
    safe_fallback: bool = False,
    max_missions: int = 2,
    min_extra_score_ratio: float = 0.65,
) -> list[Any]:
    """Select the policy mission plus compatible high-score follow-ups.

    PPO still chooses the primary mission. The bundle only adds disjoint-source,
    distinct-target follow-ups from the same ranked list, so the policy can act
    on more than one opportunity per turn without changing its action space.
    """
    selected = select_single_mission(missions, action, top_k=top_k, safe_fallback=safe_fallback)
    if not selected or max_missions <= 1:
        return selected

    top = list(missions or [])[:top_k]
    base = selected[0]
    base_score = float(getattr(base, "score", 0.0) or 0.0)
    min_score = base_score * float(min_extra_score_ratio)
    used_sources = set(_mission_sources(base))
    used_targets = {int(getattr(base, "target_id", -999))}

    for mission in top:
        if len(selected) >= int(max_missions):
            break
        if mission is base:
            continue
        if float(getattr(mission, "score", 0.0) or 0.0) < min_score:
            continue
        target_id = int(getattr(mission, "target_id", -999))
        if target_id in used_targets:
            continue
        sources = _mission_sources(mission)
        if not sources or used_sources.intersection(sources):
            continue

        selected.append(mission)
        used_targets.add(target_id)
        used_sources.update(sources)

    return selected
