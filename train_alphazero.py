"""
AlphaZero-style training for Pokemon TCG.

Key differences from vanilla RL self-play:
1. Uses PolicyValueNet with both policy and value heads
2. Policy trained on MCTS visit distributions (not selected actions)
3. Value trained on actual game outcomes
4. Option to use value network instead of rollouts in MCTS
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import argparse
import copy
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm

from tcg.env import PTCGEnv
from tcg.state import featurize
from tcg.actions import ACTION_TABLE
from tcg.mcts import MCTS, PolicyValueNet


@dataclass
class Experience:
    """Single training example from self-play."""
    obs: np.ndarray          # State observation
    mcts_probs: np.ndarray   # MCTS visit distribution (policy target)
    value: float             # Game outcome from this player's perspective


class ReplayBuffer:
    """Experience replay buffer for training."""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, exp: Experience):
        self.buffer.append(exp)
    
    def add_game(self, game_history: List[Tuple[np.ndarray, np.ndarray, int]], winner: int):
        """
        Add all positions from a completed game with final outcome.
        
        Args:
            game_history: List of (obs, mcts_probs, player) for each move
            winner: 0, 1, or -1 (draw)
        """
        for obs, mcts_probs, player in game_history:
            # Value is +1 for win, -1 for loss, 0 for draw
            if winner == -1:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            
            self.buffer.append(Experience(obs, mcts_probs, value))
    
    def add_game_with_shaping(self, game_history: List[Tuple[np.ndarray, np.ndarray, int]], 
                               winner: int, action_rewards: List[float], gamma: float = 0.99):
        """
        Add all positions from a completed game with action-based reward shaping.
        
        Args:
            game_history: List of (obs, mcts_probs, player) for each move
            winner: 0, 1, or -1 (draw)
            action_rewards: List of rewards for each action taken
            gamma: Discount factor for action rewards
        """
        n = len(game_history)
        
        for i, (obs, mcts_probs, player) in enumerate(game_history):
            # Base value from game outcome
            if winner == -1:
                base_value = 0.0
            elif winner == player:
                base_value = 1.0
            else:
                base_value = -1.0
            
            # Add discounted action reward shaping
            # Sum future rewards with discount
            shaping_value = 0.0
            for j in range(i, min(i + 20, n)):  # Look ahead up to 20 steps
                if j < len(action_rewards):
                    # Apply shaping only for this player's moves
                    if game_history[j][2] == player:
                        shaping_value += action_rewards[j] * (gamma ** (j - i))
            
            # Blend base value with shaping (shaping weight decays)
            # Early training: more shaping; Late training: more outcome
            shaping_weight = 0.3
            value = base_value * (1 - shaping_weight) + np.clip(base_value + shaping_value, -1, 1) * shaping_weight
            value = np.clip(value, -1, 1)
            
            self.buffer.append(Experience(obs, mcts_probs, value))
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


def run_alphazero_training(
    episodes: int = 10000,
    mcts_sims: int = 50,
    batch_size: int = 256,
    train_every: int = 1,
    min_buffer_size: int = 512,
    lr: float = 1e-3,
    value_weight: float = 1.0,
    use_value_net: bool = True,
    use_policy_rollouts: bool = True,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    temperature_decay_episodes: int = 5000,
    verbose: bool = False,
    save_every: int = 100,
):
    """
    AlphaZero-style self-play training.
    
    Args:
        episodes: Number of games to play
        mcts_sims: MCTS simulations per move
        batch_size: Training batch size
        train_every: Train every N games
        min_buffer_size: Minimum experiences before training starts
        lr: Learning rate
        value_weight: Weight for value loss vs policy loss
        use_value_net: Use value network instead of rollouts
        use_policy_rollouts: Use policy network for rollout actions
        temperature_start: Initial MCTS temperature (exploration)
        temperature_end: Final MCTS temperature
        temperature_decay_episodes: Episodes over which to decay temperature
        verbose: Enable verbose output
        save_every: Save model every N episodes
    """
    if not verbose:
        os.environ['PTCG_QUIET'] = '1'
    
    print("=" * 60)
    print("AlphaZero-Style Training for Pokemon TCG")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model
    obs_dim = 156  # From state.py featurize
    n_actions = len(ACTION_TABLE)
    model = PolicyValueNet(obs_dim, n_actions).to(device)
    
    # Try to load existing model
    if os.path.exists("alphazero_policy.pt"):
        try:
            checkpoint = torch.load("alphazero_policy.pt", map_location=device)
            if checkpoint.get("n_actions") == n_actions:
                model.load_state_dict(checkpoint["state_dict"])
                print("✓ Loaded existing alphazero_policy.pt")
            else:
                print("Action space mismatch, starting fresh")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    else:
        print("Starting with fresh PolicyValueNet")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
    
    replay_buffer = ReplayBuffer(capacity=200000)
    env = PTCGEnv(scripted_opponent=False, max_turns=200)
    
    # Decks
    deck_p0 = []
    deck_p0.extend(["Abra"] * 4)
    deck_p0.extend(["Kadabra"] * 3)
    deck_p0.extend(["Alakazam"] * 4)
    deck_p0.extend(["Dunsparce"] * 4)
    deck_p0.extend(["Dudunsparce"] * 4)
    deck_p0.extend(["Fan Rotom"] * 2)
    deck_p0.extend(["Psyduck"] * 1)
    deck_p0.extend(["Fezandipiti ex"] * 1)
    deck_p0.extend(["Hilda"] * 4)
    deck_p0.extend(["Dawn"] * 4)
    deck_p0.extend(["Boss's Orders"] * 3)
    deck_p0.extend(["Lillie's Determination"] * 2)
    deck_p0.extend(["Tulip"] * 1)
    deck_p0.extend(["Buddy-Buddy Poffin"] * 4)
    deck_p0.extend(["Rare Candy"] * 3)
    deck_p0.extend(["Nest Ball"] * 2)
    deck_p0.extend(["Night Stretcher"] * 2)
    deck_p0.extend(["Wondrous Patch"] * 2)
    deck_p0.extend(["Enhanced Hammer"] * 2)
    deck_p0.extend(["Battle Cage"] * 3)
    deck_p0.extend(["Basic Psychic Energy"] * 3)
    deck_p0.extend(["Enriching Energy"] * 1)
    deck_p0.extend(["Jet Energy"] * 1)

    deck_p1 = []
    deck_p1.extend(["Charmander"] * 3)
    deck_p1.extend(["Charmeleon"] * 2)
    deck_p1.extend(["Charizard ex"] * 2)
    deck_p1.extend(["Pidgey"] * 2)
    deck_p1.extend(["Pidgeotto"] * 2)
    deck_p1.extend(["Pidgeot ex"] * 2)
    deck_p1.extend(["Psyduck"] * 1)
    deck_p1.extend(["Shaymin"] * 1)
    deck_p1.extend(["Tatsugiri"] * 1)
    deck_p1.extend(["Munkidori"] * 1)
    deck_p1.extend(["Chi-Yu"] * 1)
    deck_p1.extend(["Gouging Fire ex"] * 1)
    deck_p1.extend(["Fezandipiti ex"] * 1)
    deck_p1.extend(["Lillie's Determination"] * 4)
    deck_p1.extend(["Arven"] * 4)
    deck_p1.extend(["Boss's Orders"] * 3)
    deck_p1.extend(["Iono"] * 2)
    deck_p1.extend(["Professor Turo's Scenario"] * 1)
    deck_p1.extend(["Buddy-Buddy Poffin"] * 4)
    deck_p1.extend(["Ultra Ball"] * 3)
    deck_p1.extend(["Rare Candy"] * 2)
    deck_p1.extend(["Super Rod"] * 2)
    deck_p1.extend(["Counter Catcher"] * 1)
    deck_p1.extend(["Energy Search"] * 1)
    deck_p1.extend(["Unfair Stamp"] * 1)
    deck_p1.extend(["Technical Machine: Evolution"] * 2)
    deck_p1.extend(["Artazon"] * 1)
    deck_p1.extend(["Fire Energy"] * 5)
    deck_p1.extend(["Mist Energy"] * 2)
    deck_p1.extend(["Darkness Energy"] * 1)
    deck_p1.extend(["Jet Energy"] * 1)
    
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  MCTS Simulations: {mcts_sims}")
    print(f"  Use Value Net: {use_value_net}")
    print(f"  Use Policy Rollouts: {use_policy_rollouts}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print()
    
    # Metrics
    wins_p0 = 0
    wins_p1 = 0
    draws = 0
    recent_wins = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    recent_game_lengths = deque(maxlen=100)
    
    # CSV logging
    import csv
    metrics_file = open("alphazero_metrics.csv", "w", newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow([
        "episode", "p0_winrate", "rolling_winrate", "avg_game_length",
        "policy_loss", "value_loss", "total_loss", "buffer_size", "temperature"
    ])
    
    pbar = tqdm(range(episodes), desc="AlphaZero Training", ncols=120)
    
    for ep in pbar:
        # Anneal temperature
        if ep < temperature_decay_episodes:
            temperature = temperature_start - (temperature_start - temperature_end) * (ep / temperature_decay_episodes)
        else:
            temperature = temperature_end
        
        # Create MCTS agent
        mcts_agent = MCTS(
            policy_net=model,
            device=device,
            num_simulations=mcts_sims,
            c_puct=1.5,
            max_rollout_steps=150,
            use_value_net=use_value_net,
            use_policy_rollouts=use_policy_rollouts,
            temperature=temperature
        )
        
        # Play one game
        obs, info = env.reset(options={"decks": [deck_p0, deck_p1]})
        done = False
        game_history = []  # (obs, mcts_probs, player, action_reward)
        step_count = 0
        max_steps = 2000
        action_rewards = []  # Track rewards for each action
        
        while not done and step_count < max_steps:
            turn_player = env._gs.turn_player
            mask = info["action_mask"]
            
            # Get MCTS action and visit distribution
            action, mcts_probs = mcts_agent.search(env, return_probs=True)
            
            # Store experience (will assign value after game ends)
            game_history.append((obs.copy(), mcts_probs, turn_player))
            
            # ========== ACTION REWARD SHAPING ==========
            # This helps the value network learn to take actions
            action_reward = 0.0
            PASS_ACTION = 0
            ATTACK_ACTIONS = [1]  # Action index for attacks (adjust based on ACTION_TABLE)
            
            # Count valid actions
            valid_actions = np.sum(mask)
            
            if action == PASS_ACTION and valid_actions > 1:
                # Penalize passing when other options exist
                action_reward -= 0.3
            elif action != PASS_ACTION:
                # Reward taking any action
                action_reward += 0.05
            
            action_rewards.append(action_reward)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
        
        # Game finished - determine winner
        winner = env._gs.winner if done else -1  # -1 for draw/timeout
        
        if winner == 0:
            wins_p0 += 1
            recent_wins.append(1)
        elif winner == 1:
            wins_p1 += 1
            recent_wins.append(0)
        else:
            draws += 1
            recent_wins.append(0.5)
        
        recent_game_lengths.append(step_count)
        
        # Add game to replay buffer with outcomes + action rewards
        replay_buffer.add_game_with_shaping(game_history, winner, action_rewards)
        
        # Training
        policy_loss_val = 0.0
        value_loss_val = 0.0
        total_loss_val = 0.0
        
        if len(replay_buffer) >= min_buffer_size and (ep + 1) % train_every == 0:
            model.train()
            
            # Sample batch
            batch = replay_buffer.sample(batch_size)
            
            obs_batch = torch.from_numpy(np.stack([e.obs for e in batch])).float().to(device)
            policy_target = torch.from_numpy(np.stack([e.mcts_probs for e in batch])).float().to(device)
            value_target = torch.tensor([e.value for e in batch]).float().to(device)
            
            # Forward pass
            policy_logits, value_pred = model(obs_batch)
            
            # Policy loss: cross-entropy with MCTS probabilities
            # Use KL divergence or cross-entropy
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_target * log_probs).sum(dim=1).mean()
            
            # Value loss: MSE
            value_loss = F.mse_loss(value_pred, value_target)
            
            # Total loss
            total_loss = policy_loss + value_weight * value_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            policy_loss_val = policy_loss.item()
            value_loss_val = value_loss.item()
            total_loss_val = total_loss.item()
            recent_losses.append(total_loss_val)
            
            # Step scheduler after optimizer
            scheduler.step()
        
        # Update progress bar
        rolling_wr = sum(recent_wins) / len(recent_wins) if recent_wins else 0
        avg_length = sum(recent_game_lengths) / len(recent_game_lengths) if recent_game_lengths else 0
        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
        
        pbar.set_postfix({
            'WR': f'{rolling_wr:.0%}',
            'Len': f'{avg_length:.0f}',
            'Loss': f'{avg_loss:.3f}',
            'Buf': len(replay_buffer),
            'T': f'{temperature:.2f}'
        })
        
        # Log to CSV
        if (ep + 1) % 10 == 0:
            metrics_writer.writerow([
                ep + 1,
                wins_p0 / (ep + 1),
                rolling_wr,
                avg_length,
                policy_loss_val,
                value_loss_val,
                total_loss_val,
                len(replay_buffer),
                temperature
            ])
            metrics_file.flush()
        
        # Save model
        if (ep + 1) % save_every == 0:
            torch.save({
                "obs_dim": obs_dim,
                "n_actions": n_actions,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, "alphazero_policy.pt")
            if verbose:
                print(f"\nSaved alphazero_policy.pt at episode {ep+1}")
    
    # Final save
    torch.save({
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "state_dict": model.state_dict(),
    }, "alphazero_policy.pt")
    
    metrics_file.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final stats: P0 {wins_p0}/{episodes} ({wins_p0/episodes:.1%})")
    print(f"             P1 {wins_p1}/{episodes} ({wins_p1/episodes:.1%})")
    print(f"             Draws {draws}/{episodes}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AlphaZero-style Pokemon TCG Training')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--mcts_sims', type=int, default=50, help='MCTS simulations per move')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--no_value_net', action='store_true', help='Disable value network (use rollouts)')
    parser.add_argument('--no_policy_rollouts', action='store_true', help='Use random rollouts')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save_every', type=int, default=100, help='Save model every N episodes')
    args = parser.parse_args()
    
    run_alphazero_training(
        episodes=args.episodes,
        mcts_sims=args.mcts_sims,
        batch_size=args.batch_size,
        lr=args.lr,
        use_value_net=not args.no_value_net,
        use_policy_rollouts=not args.no_policy_rollouts,
        verbose=args.verbose,
        save_every=args.save_every,
    )
