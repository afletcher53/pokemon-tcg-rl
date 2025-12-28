# tcg/rollout.py
from __future__ import annotations

import argparse

import numpy as np
import torch

from tcg.actions import ACTION_TABLE
from tcg.env import PTCGEnv
from tcg.train_bc import PolicyNet


def select_action(logits: torch.Tensor, mask: np.ndarray) -> int:
    # mask is 0/1, logits shape [n_actions]
    masked = logits.clone()
    bad = torch.tensor(mask == 0, dtype=torch.bool)
    masked[bad] = -1e9
    return int(torch.argmax(masked).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, required=True)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    ckpt = torch.load(args.policy, map_location="cpu")
    model = PolicyNet(ckpt["obs_dim"], ckpt["n_actions"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    env = PTCGEnv(scripted_opponent=True)
    obs, info = env.reset()
    for t in range(args.steps):
        x = torch.from_numpy(obs).float()
        with torch.no_grad():
            logits = model(x)
        act = select_action(logits, info["action_mask"])
        obs, r, done, trunc, info = env.step(act)
        print(f"t={t:02d} act={act:04d} {ACTION_TABLE[act]} r={r:.3f}")
        if done:
            print("done")
            break


if __name__ == "__main__":
    main()
