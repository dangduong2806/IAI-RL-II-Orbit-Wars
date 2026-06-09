from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset


def ppo_update(
    policy,
    optimizer,
    buffer,
    device="cpu",
    bootstrap_value: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    batch_size: int = 256,
):
    if len(buffer) == 0:
        return {"loss": 0.0}
    arr = buffer.as_arrays(gamma=gamma, gae_lambda=gae_lambda, bootstrap_value=bootstrap_value)
    ds = TensorDataset(
        torch.tensor(arr["global_x"], dtype=torch.float32),
        torch.tensor(arr["mission_x"], dtype=torch.float32),
        torch.tensor(arr["mask"], dtype=torch.float32),
        torch.tensor(arr["actions"], dtype=torch.long),
        torch.tensor(arr["old_logprobs"], dtype=torch.float32),
        torch.tensor(arr["returns"], dtype=torch.float32),
        torch.tensor(arr["advantages"], dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
    device = torch.device(device)

    stats = []
    for _ in range(epochs):
        for g, m, ms, act, old_lp, ret, adv in loader:
            g = g.to(device); m = m.to(device); ms = ms.to(device)
            act = act.to(device); old_lp = old_lp.to(device)
            ret = ret.to(device); adv = adv.to(device)

            logits, value = policy(g, m, ms)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(act)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_lp - old_lp)

            pg1 = ratio * adv
            pg2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
            policy_loss = -torch.min(pg1, pg2).mean()
            value_loss = torch.nn.functional.mse_loss(value, ret)
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            stats.append([float(loss.item()), float(policy_loss.item()), float(value_loss.item()), float(entropy.item())])

    mean = np.asarray(stats, dtype=np.float32).mean(axis=0)
    return {"loss": float(mean[0]), "policy_loss": float(mean[1]), "value_loss": float(mean[2]), "entropy": float(mean[3])}
