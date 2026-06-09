from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .mission_encoder import MISSION_FEATURE_DIM
from .topk_policy import TopKMissionPolicy, load_policy, save_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/bc_dataset.npz")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--save", type=str, default="checkpoints/bc_policy.pt")
    p.add_argument("--init", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.data)
    global_x = torch.tensor(data["global_x"], dtype=torch.float32)
    mission_x = torch.tensor(data["mission_x"], dtype=torch.float32)
    mask = torch.tensor(data["mask"], dtype=torch.float32)
    action = torch.tensor(data["action"], dtype=torch.long)

    top_k = args.top_k or int(data.get("top_k", np.asarray([mission_x.shape[1]])).item())
    if mission_x.shape[-1] < MISSION_FEATURE_DIM:
        pad = MISSION_FEATURE_DIM - mission_x.shape[-1]
        mission_x = torch.nn.functional.pad(mission_x, (0, pad))
        print(f"padded mission features from {MISSION_FEATURE_DIM - pad} to {MISSION_FEATURE_DIM}")
    elif mission_x.shape[-1] > MISSION_FEATURE_DIM:
        mission_x = mission_x[..., :MISSION_FEATURE_DIM]
        print(f"truncated mission features to {MISSION_FEATURE_DIM}")

    dataset = TensorDataset(global_x, mission_x, mask, action)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    if args.init and Path(args.init).exists():
        policy = load_policy(args.init, map_location=device)
        print(f"loaded init policy: {args.init}")
    else:
        policy = TopKMissionPolicy(top_k=top_k)
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        policy.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for g, m, ms, y in loader:
            g = g.to(device)
            m = m.to(device)
            ms = ms.to(device)
            y = y.to(device)

            logits, _ = policy(g, m, ms)
            loss = ce(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * y.numel()
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total_count += int(y.numel())

        print(
            f"epoch={epoch+1:03d}/{args.epochs} "
            f"loss={total_loss/max(1,total_count):.5f} "
            f"acc={total_correct/max(1,total_count):.4f}"
        )

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_policy(str(save_path), policy, extra={"stage": "behavior_cloning"})
    print(f"saved BC policy to {save_path}")


if __name__ == "__main__":
    main()
