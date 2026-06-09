from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .mission_encoder import MISSION_FEATURE_DIM


def _pad_or_trim_missions(mission_x: np.ndarray, top_k: int) -> np.ndarray:
    out = mission_x
    if out.shape[1] < top_k:
        pad = np.zeros((out.shape[0], top_k - out.shape[1], out.shape[2]), dtype=out.dtype)
        out = np.concatenate([out, pad], axis=1)
    elif out.shape[1] > top_k:
        out = out[:, :top_k, :]

    if out.shape[2] < MISSION_FEATURE_DIM:
        pad = np.zeros((out.shape[0], out.shape[1], MISSION_FEATURE_DIM - out.shape[2]), dtype=out.dtype)
        out = np.concatenate([out, pad], axis=2)
    elif out.shape[2] > MISSION_FEATURE_DIM:
        out = out[:, :, :MISSION_FEATURE_DIM]
    return out.astype(np.float32)


def _pad_or_trim_mask(mask: np.ndarray, top_k: int) -> np.ndarray:
    out = mask
    if out.shape[1] < top_k:
        pad = np.zeros((out.shape[0], top_k - out.shape[1]), dtype=out.dtype)
        out = np.concatenate([out, pad], axis=1)
    elif out.shape[1] > top_k:
        out = out[:, :top_k]
    return out.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--save", type=str, default="data/merged_bc_dataset.npz")
    return p.parse_args()


def main():
    args = parse_args()
    global_x = []
    mission_x = []
    masks = []
    actions = []

    for path in args.datasets:
        data = np.load(path, allow_pickle=True)
        global_x.append(np.asarray(data["global_x"], dtype=np.float32))
        mission_x.append(_pad_or_trim_missions(np.asarray(data["mission_x"], dtype=np.float32), args.top_k))
        masks.append(_pad_or_trim_mask(np.asarray(data["mask"], dtype=np.float32), args.top_k))
        actions.append(np.asarray(data["action"], dtype=np.int64))
        print(f"loaded {path}: {len(data['action'])} samples")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        global_x=np.concatenate(global_x, axis=0),
        mission_x=np.concatenate(mission_x, axis=0),
        mask=np.concatenate(masks, axis=0),
        action=np.concatenate(actions, axis=0),
        top_k=np.asarray([args.top_k], dtype=np.int64),
    )
    print(f"saved merged dataset: {save_path} | samples={sum(len(a) for a in actions)}")


if __name__ == "__main__":
    main()
