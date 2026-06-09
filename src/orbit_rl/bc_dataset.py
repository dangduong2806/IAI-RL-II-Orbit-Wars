from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import lb1200_strategy as lb
from .mission_encoder import encode_topk
from .mission_interface import generate_top_missions, infer_teacher_action
from .selfplay_env import OrbitWarsSelfPlayEnv


def save_dataset(
    save_path: str,
    globals_x,
    missions_x,
    masks,
    actions,
    top_k: int,
    episodes_completed: int,
    skipped_unmatched: int,
) -> None:
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        global_x=np.asarray(globals_x, dtype=np.float32),
        mission_x=np.asarray(missions_x, dtype=np.float32),
        mask=np.asarray(masks, dtype=np.float32),
        action=np.asarray(actions, dtype=np.int64),
        top_k=np.asarray([top_k], dtype=np.int64),
        episodes_completed=np.asarray([episodes_completed], dtype=np.int64),
        skipped_unmatched=np.asarray([skipped_unmatched], dtype=np.int64),
    )


def collect_dataset(
    episodes: int,
    top_k: int,
    save_path: str,
    max_steps: int | None = None,
    save_every: int = 10,
    resume: bool = False,
):
    globals_x = []
    missions_x = []
    masks = []
    actions = []
    skipped_unmatched = 0
    start_ep = 0

    save_file = Path(save_path)
    if resume and save_file.exists():
        data = np.load(save_file)
        globals_x = list(data["global_x"])
        missions_x = list(data["mission_x"])
        masks = list(data["mask"])
        actions = list(data["action"])
        start_ep = int(data.get("episodes_completed", np.asarray([0])).item())
        skipped_unmatched = int(data.get("skipped_unmatched", np.asarray([0])).item())
        saved_top_k = int(data.get("top_k", np.asarray([top_k])).item())
        if saved_top_k != int(top_k):
            raise ValueError(f"cannot resume {save_path}: top_k={saved_top_k}, requested top_k={top_k}")
        print(
            f"resumed dataset: {save_file} episodes_completed={start_ep} "
            f"samples={len(actions)} skipped_unmatched={skipped_unmatched}",
            flush=True,
        )

    env = OrbitWarsSelfPlayEnv(opponent_fn=lb.agent)

    for ep in range(start_ep, int(episodes)):
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            world = lb.build_world(obs)
            top_missions = generate_top_missions(world, top_k=top_k)
            g, m, mask = encode_topk(world, top_missions, top_k=top_k)
            y = infer_teacher_action(world, top_missions, top_k=top_k)

            if y is None:
                skipped_unmatched += 1
            else:
                globals_x.append(g)
                missions_x.append(m)
                masks.append(mask)
                actions.append(y)

            moves = lb.agent(obs)
            obs, reward, done, info = env.step(moves)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        print(
            f"episode={ep:04d} steps={steps:03d} "
            f"samples={len(actions)} skipped_unmatched={skipped_unmatched}",
            flush=True,
        )

        episodes_completed = ep + 1
        if save_every > 0 and episodes_completed % int(save_every) == 0:
            save_dataset(
                save_path,
                globals_x,
                missions_x,
                masks,
                actions,
                top_k=top_k,
                episodes_completed=episodes_completed,
                skipped_unmatched=skipped_unmatched,
            )
            print(f"checkpointed dataset: {save_file} episodes_completed={episodes_completed}", flush=True)

    save_dataset(
        save_path,
        globals_x,
        missions_x,
        masks,
        actions,
        top_k=top_k,
        episodes_completed=int(episodes),
        skipped_unmatched=skipped_unmatched,
    )
    print(f"saved dataset: {save_file} | samples={len(actions)} skipped_unmatched={skipped_unmatched}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--save", type=str, default="data/bc_dataset.npz")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    collect_dataset(
        args.episodes,
        args.top_k,
        args.save,
        max_steps=args.max_steps,
        save_every=args.save_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
