from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch

from . import lb1200_strategy as lb
from .mission_encoder import MISSION_RAW_SCORE_FEATURE_INDEX
from .ppo_buffer import PPOBuffer
from .ppo_mission_agent import MissionDecision, TopKMissionAgent
from .ppo_update import ppo_update
from .selfplay_env import OrbitWarsSelfPlayEnv
from .topk_policy import TopKMissionPolicy, load_policy, save_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", type=str, default="checkpoints/bc_policy.pt")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--rollout-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--selfplay-start", type=int, default=200)
    p.add_argument("--snapshot-interval", type=int, default=100)
    p.add_argument("--max-snapshots", type=int, default=5)
    p.add_argument("--save", type=str, default="checkpoints/ppo_selfplay.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-best", type=str, default="checkpoints/ppo_selfplay_best.pt")
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-seed-base", type=int, default=10000)
    p.add_argument("--teacher-opponent-prob", type=float, default=0.40)
    p.add_argument("--stochastic-snapshot-prob", type=float, default=0.20)
    p.add_argument("--current-opponent-prob", type=float, default=0.10)
    p.add_argument("--max-selected-missions", type=int, default=2)
    p.add_argument("--min-extra-score-ratio", type=float, default=0.65)
    p.add_argument("--safe-fallback", action="store_true")
    p.add_argument("--heuristic-prior-strength", type=float, default=0.35)
    p.add_argument("--heuristic-prior-rank-weight", type=float, default=0.25)
    p.add_argument("--heuristic-stop-penalty", type=float, default=0.15)
    p.add_argument("--heuristic-reward-coef", type=float, default=0.03)
    p.add_argument("--heuristic-stop-reward-penalty", type=float, default=1.0)
    return p.parse_args()


def make_snapshot_agent(
    policy: TopKMissionPolicy,
    device,
    safe_fallback=False,
    max_selected_missions: int = 2,
    min_extra_score_ratio: float = 0.65,
):
    snap_policy = copy.deepcopy(policy).to(device)
    snap_policy.eval()
    return TopKMissionAgent(
        top_k=snap_policy.top_k,
        device=device,
        policy=snap_policy,
        safe_fallback=safe_fallback,
        max_selected_missions=max_selected_missions,
        min_extra_score_ratio=min_extra_score_ratio,
    )


def evaluate_policy(
    policy: TopKMissionPolicy,
    device,
    episodes: int = 20,
    seed_base: int = 10000,
    max_selected_missions: int = 2,
    min_extra_score_ratio: float = 0.65,
    safe_fallback: bool = False,
):
    eval_agent = make_snapshot_agent(
        policy,
        device=device,
        safe_fallback=safe_fallback,
        max_selected_missions=max_selected_missions,
        min_extra_score_ratio=min_extra_score_ratio,
    )
    total_terminal = 0.0
    total_shaped = 0.0
    wins = ties = losses = 0

    for i in range(int(episodes)):
        env = OrbitWarsSelfPlayEnv(opponent_fn=lb.agent, seed=int(seed_base) + i)
        obs = env.reset()
        done = False
        ep_shaped = 0.0
        info = {}

        while not done:
            moves = eval_agent.act(obs, deterministic=True, training=False)
            obs, reward, done, info = env.step(moves)
            ep_shaped += float(reward)

        terminal = float(info.get("terminal_reward", 0.0) or 0.0)
        total_terminal += terminal
        total_shaped += ep_shaped
        if terminal > 0.0:
            wins += 1
        elif terminal < 0.0:
            losses += 1
        else:
            ties += 1

    n = max(1, int(episodes))
    return {
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "tie_rate": ties / n,
        "avg_terminal": total_terminal / n,
        "avg_shaped": total_shaped / n,
    }


def eval_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics.get("win_rate", 0.0)),
        float(metrics.get("avg_terminal", 0.0)),
        float(metrics.get("avg_shaped", 0.0)),
    )


def load_saved_eval_score(path: str) -> tuple[float, float, float] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = torch.load(p, map_location="cpu")
        score = payload.get("eval_score")
        if score is None:
            return None
        if len(score) != 3:
            return None
        return (float(score[0]), float(score[1]), float(score[2]))
    except Exception:
        return None


def heuristic_action_reward(
    decision: MissionDecision,
    coef: float = 0.03,
    stop_penalty: float = 1.0,
) -> float:
    coef = max(0.0, float(coef))
    if coef <= 0.0:
        return 0.0

    mask = np.asarray(decision.mask, dtype=np.float32) > 0.0
    if not mask.any():
        return 0.0

    scores = np.asarray(decision.mission_x[:, MISSION_RAW_SCORE_FEATURE_INDEX], dtype=np.float32)
    valid_scores = scores[mask]
    best_score = float(valid_scores.max()) if len(valid_scores) else 0.0
    if best_score <= 0.0:
        return 0.0

    action = int(decision.action)
    if action <= 0:
        return -coef * max(0.0, float(stop_penalty))

    idx = action - 1
    if idx < 0 or idx >= len(scores) or not bool(mask[idx]):
        return -0.5 * coef

    selected_score = float(scores[idx])
    relative_quality = selected_score / max(abs(best_score), 1e-6)
    relative_quality = max(-1.0, min(1.0, relative_quality))
    return coef * relative_quality


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.init and Path(args.init).exists():
        policy = load_policy(args.init, map_location=device)
        print(f"loaded init policy: {args.init}")
    else:
        policy = TopKMissionPolicy(top_k=args.top_k)
        print("no init checkpoint found; training from random policy")

    policy.configure_heuristic_prior(
        strength=args.heuristic_prior_strength,
        rank_weight=args.heuristic_prior_rank_weight,
        stop_penalty=args.heuristic_stop_penalty,
    )
    print(
        "heuristic guidance: "
        f"prior_strength={args.heuristic_prior_strength} "
        f"rank_weight={args.heuristic_prior_rank_weight} "
        f"stop_penalty={args.heuristic_stop_penalty} "
        f"reward_coef={args.heuristic_reward_coef} "
        f"safe_fallback={args.safe_fallback}"
    )

    learner = TopKMissionAgent(
        top_k=policy.top_k,
        device=device,
        policy=policy,
        safe_fallback=args.safe_fallback,
        max_selected_missions=args.max_selected_missions,
        min_extra_score_ratio=args.min_extra_score_ratio,
    )
    optimizer = torch.optim.Adam(learner.policy.parameters(), lr=args.lr)
    buffer = PPOBuffer()
    snapshots: list[TopKMissionAgent] = []

    best_eval_score = load_saved_eval_score(args.save_best)
    if best_eval_score is None:
        best_eval_score = (float("-inf"), float("-inf"), float("-inf"))
    else:
        print(f"loaded existing best eval score from {args.save_best}: {best_eval_score}")

    def choose_opponent(ep: int):
        if ep < args.selfplay_start or not snapshots:
            return lb.agent
        # Keep the teacher in the pool, but add deterministic/stochastic snapshots
        # and a current-policy snapshot so training does not overfit one style.
        r = random.random()
        if r < args.teacher_opponent_prob:
            return lb.agent
        r -= args.teacher_opponent_prob

        if r < args.stochastic_snapshot_prob:
            snap = random.choice(snapshots)
            return lambda obs, config=None: snap.act(obs, config, deterministic=False, training=False)
        r -= args.stochastic_snapshot_prob

        if r < args.current_opponent_prob:
            current = make_snapshot_agent(
                learner.policy,
                device=device,
                safe_fallback=args.safe_fallback,
                max_selected_missions=args.max_selected_missions,
                min_extra_score_ratio=args.min_extra_score_ratio,
            )
            return lambda obs, config=None: current.act(obs, config, deterministic=True, training=False)

        snap = random.choice(snapshots)
        return lambda obs, config=None: snap.act(obs, config, deterministic=True, training=False)

    for ep in range(args.episodes):
        opponent_fn = choose_opponent(ep)
        env = OrbitWarsSelfPlayEnv(opponent_fn=opponent_fn)
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_heuristic_bonus = 0.0
        steps = 0

        while not done:
            moves = learner.act(obs, deterministic=False, training=True)
            next_obs, reward, done, info = env.step(moves)
            ep_reward += reward
            steps += 1

            decisions = learner.consume_last_decisions()
            if not decisions:
                # This can happen if no candidate mission is produced.
                pass
            for dec in decisions:
                heuristic_bonus = heuristic_action_reward(
                    dec,
                    coef=args.heuristic_reward_coef,
                    stop_penalty=args.heuristic_stop_reward_penalty,
                )
                ep_heuristic_bonus += heuristic_bonus
                guided_reward = reward + heuristic_bonus
                buffer.add(
                    dec.global_x,
                    dec.mission_x,
                    dec.mask,
                    dec.action,
                    dec.logprob,
                    dec.value,
                    guided_reward,
                    done,
                )

            if len(buffer) >= args.rollout_size:
                bootstrap_value = 0.0 if done else learner.estimate_value(next_obs)
                stats = ppo_update(
                    learner.policy,
                    optimizer,
                    buffer,
                    device=device,
                    bootstrap_value=bootstrap_value,
                )
                print(f"update ep={ep:04d}: {stats}")
                buffer.clear()

            obs = next_obs

        print(
            f"episode={ep:04d} steps={steps:03d} "
            f"shaped_reward={ep_reward:.3f} heuristic_bonus={ep_heuristic_bonus:.3f} "
            f"terminal={info.get('terminal_reward')}"
        )
        
        should_eval = args.eval_interval > 0 and (ep + 1) % args.eval_interval == 0
        if should_eval:
            metrics = evaluate_policy(
                learner.policy,
                device=device,
                episodes=args.eval_episodes,
                seed_base=args.eval_seed_base,
                max_selected_missions=args.max_selected_missions,
                min_extra_score_ratio=args.min_extra_score_ratio,
                safe_fallback=args.safe_fallback,
            )
            score = eval_score(metrics)
            print(f"eval ep={ep:04d}: {metrics}")

            if score > best_eval_score:
                best_eval_score = score
                best_path = Path(args.save_best)
                best_path.parent.mkdir(parents=True, exist_ok=True)
                save_policy(
                    str(best_path),
                    learner.policy,
                    extra={
                        "stage": "ppo_selfplay_best_eval",
                        "episode": ep,
                        "eval_metrics": metrics,
                        "eval_score": list(score),
                        "eval_episodes": int(args.eval_episodes),
                        "eval_seed_base": int(args.eval_seed_base),
                        "max_selected_missions": int(args.max_selected_missions),
                        "min_extra_score_ratio": float(args.min_extra_score_ratio),
                        "safe_fallback": bool(args.safe_fallback),
                        "heuristic_prior_strength": float(args.heuristic_prior_strength),
                        "heuristic_prior_rank_weight": float(args.heuristic_prior_rank_weight),
                        "heuristic_stop_penalty": float(args.heuristic_stop_penalty),
                        "heuristic_reward_coef": float(args.heuristic_reward_coef),
                        "heuristic_stop_reward_penalty": float(args.heuristic_stop_reward_penalty),
                    },
                )
                print(f"saved best PPO policy to {best_path} | eval={metrics}")

        if (ep + 1) % args.snapshot_interval == 0:
            snapshots.append(
                make_snapshot_agent(
                    learner.policy,
                    device=device,
                    safe_fallback=args.safe_fallback,
                    max_selected_missions=args.max_selected_missions,
                    min_extra_score_ratio=args.min_extra_score_ratio,
                )
            )
            snapshots = snapshots[-args.max_snapshots:]
            print(f"snapshot pool size={len(snapshots)}")

    if len(buffer) > 0:
        stats = ppo_update(learner.policy, optimizer, buffer, device=device, bootstrap_value=0.0)
        print(f"final update: {stats}")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_policy(
        str(save_path),
        learner.policy,
        extra={
            "stage": "ppo_selfplay",
            "max_selected_missions": int(args.max_selected_missions),
            "min_extra_score_ratio": float(args.min_extra_score_ratio),
            "safe_fallback": bool(args.safe_fallback),
            "heuristic_prior_strength": float(args.heuristic_prior_strength),
            "heuristic_prior_rank_weight": float(args.heuristic_prior_rank_weight),
            "heuristic_stop_penalty": float(args.heuristic_stop_penalty),
            "heuristic_reward_coef": float(args.heuristic_reward_coef),
            "heuristic_stop_reward_penalty": float(args.heuristic_stop_reward_penalty),
        },
    )
    print(f"saved PPO policy to {save_path}")


if __name__ == "__main__":
    main()
