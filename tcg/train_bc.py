# tcg/train_bc.py
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tcg.actions import ACTION_TABLE
from tcg.dataset import build_bc_dataset


class BCDataset(Dataset):
    def __init__(self, transitions):
        self.X = np.stack([t.obs for t in transitions], axis=0)
        self.y = np.array([t.act_idx for t in transitions], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_file", type=str, required=True)
    ap.add_argument(
        "--player",
        type=str,
        required=True,
        help="Name in log for the player to imitate, e.g. Rudyodinson",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=str, default="bc_policy.pt")
    args = ap.parse_args()

    raw = open(args.log_file, "r", encoding="utf-8").read()
    transitions = build_bc_dataset(raw, agent_player_name=args.player)
    if len(transitions) < 10:
        print(
            f"Only {len(transitions)} transitions extracted. Mapping is conservative; expand parser/engine."
        )
    ds = BCDataset(transitions)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    obs_dim = ds.X.shape[1]
    n_actions = len(ACTION_TABLE)
    model = PolicyNet(obs_dim, n_actions)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    loss_history = []
    
    model.train()
    for ep in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {ep + 1}/{args.epochs}")
        total = 0.0
        n = 0
        for x, y in pbar:
            logits = model(x.float())
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
            
        epoch_loss = total / max(n, 1)
        # pbar.set_postfix(loss=epoch_loss) # tqdm handles this inside loop? 
        # Update manually for final display
        pbar.set_postfix(loss=epoch_loss)
        loss_history.append(epoch_loss)

    torch.save(
        {"state_dict": model.state_dict(), "obs_dim": obs_dim, "n_actions": n_actions},
        args.out,
    )
    print(f"Saved {args.out}")
    
    with open("loss_history.json", "w") as f:
        json.dump(loss_history, f)
    print("Saved loss_history.json")


if __name__ == "__main__":
    main()
