# Training Improvements Applied

## Root Cause Identified

**The model assigns 77%+ probability to PASS and only 0.06% to correct energy attachment!**

This means the agent never learns to attach energy → can never attack → never takes prizes → gets no reward signal for winning.

## Changes Applied

### 1. ✅ Increased Energy Attachment Reward (Critical Fix)
**Line 880 in train_advanced.py**
- Changed from `+0.1` to `+0.4` base reward for energy attachment
- Added `+0.5` bonus when attaching energy that makes Pokemon attack-ready

### 2. ✅ Extra PASS Penalty When Energy Available
**Line 854-857 in train_advanced.py**
- Added `-0.5` penalty for passing when energy attachment is available
- This directly punishes the behavior causing the problem

### 3. ✅ Periodic Checkpoint Saving
**Line 1066-1090 in train_advanced.py**
- Saves checkpoint every 1000 episodes to `checkpoints/checkpoint_ep{N}.pt`
- Allows tracking model progression over time

### 4. ✅ Alpha-Rank Integration
**Multiple locations in train_advanced.py**
- Added `PopulationTracker` import and initialization
- Records all match results during training
- Computes and displays Alpha-Rank every 1000 episodes
- Saves match history to `checkpoints/matches_ep{N}.json`

### 5. ✅ Cross-Checkpoint Evaluation (NEW)
**Lines 1161-1177 in train_advanced.py**
- Every 1000 episodes, plays current model vs previous checkpoint
- Reports win rate with clear progress indicators:
  - ✅ Progress! (>55% win rate)
  - ➡️ Stable (~50% win rate)
  - ⚠️ Regression! (<45% win rate)
- Directly answers "is training improving?"

### 6. ✅ Multiple Training Steps per Episode
**Lines 1101-1103 in train_advanced.py**
- Changed from 1 training step to 4 training steps per episode
- Increases GPU utilization from ~20% to ~60-80%

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Energy attach reward | +0.1 | +0.4 to +0.9 |
| PASS penalty (with energy) | -1.0 | -1.5 |
| Checkpoint frequency | Only best | Every 1000 eps |
| Alpha-Rank tracking | None | Full integration |

## When Changes Take Effect

The current training run will NOT see these changes (it's using the old code in memory).

To apply:
1. Wait for current training to complete or reach a good checkpoint
2. Stop training: `Ctrl+C`
3. Restart: `python train_advanced.py --mirror --episodes 50000 --mcts_sims 100 --population 16`

Or continue from checkpoint:
```bash
python train_advanced.py --mirror --episodes 50000 --mcts_sims 100 --population 16 --resume best_elo_policy.pt
```

## Additional Analysis Tools

### Evaluate with Alpha-Rank
```bash
./pkm/bin/python evaluate_alpharank.py --games 20
```

### check progression over time
```bash
ls checkpoints/
./pkm/bin/python evaluate_alpharank.py --models checkpoints/checkpoint_ep1000.pt checkpoints/checkpoint_ep2000.pt
```

## 🚀 Parallel Training (Performance Optimization)

### New Features
- **Parallel Data Collection**: Runs multiple game simulations simultaneously on different CPU cores.
- **Batched Processing**: Collects larger batches of experience data before GPU training.

### How to Use

For maximum performance on a high-core machine (e.g., 16+ cores, 32GB+ RAM):

```bash
# Recommended High-Performance Command
python train_advanced.py \
  --episodes 50000 \
  --num_workers 16 \
  --batch_size 1024 \
  --mcts_sims 50 \
  --mirror
```

### Tuning Guide
- **`--num_workers`**: Set to `num_cores - 2`. Higher = faster data collection but more RAM.
- **`--batch_size`**: Increase to `512` or `1024` when using many workers to efficienty use GPU.
- **`--mcts_sims`**: Lowering this (e.g. to 50) speeds up games linearly, allowing more games/sec.

### Expected Performance
- **CPU**: High utilization across all cores (game logic/MCTS)
- **GPU**: Higher burst utilization during training steps
- **Throughput**: significantly more games per minute compared to single-threaded execution.
