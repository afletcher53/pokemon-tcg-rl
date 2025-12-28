# Parallel Training Guide

The training script `train_advanced.py` has been updated to support parallel execution of games using multiprocessing. This addresses the CPU bottleneck caused by Monte Carlo Tree Search (MCTS) running on a single thread.

## How to Run

Use the `--num_workers` argument to specify the number of parallel processes.

```bash
python train_advanced.py --num_workers 4
```

Recommended settings:
- **Desktop (8-core)**: `--num_workers 6`
- **Server (High core count)**: `--num_workers 16` or more.

## Changes Implemented

1. **Worker Function**: `play_single_game_worker` was updated to:
   - Accept model state dictionaries instead of shared model objects (better for multiprocessing).
   - Implement the full "Enhanced Reward Shaping" logic to match the main training loop.
   - Return detailed statistics (prizes, evolutions, etc.).

2. **Main Loop**: Refactored to execute games in batches:
   - Uses `multiprocessing.Pool` (spawn context).
   - Aggregates results from multiple games before performing training steps.
   - Maintains Population-Based Training (PBT) and League logic.

## Performance Note

- **GPU Utilization**: By running multiple games in parallel, we generate experience faster, allowing the GPU to spend more time training and less time waiting for MCTS.
- **CPU Usage**: You should see high CPU usage across multiple cores (verify with `htop` or `atop`).
