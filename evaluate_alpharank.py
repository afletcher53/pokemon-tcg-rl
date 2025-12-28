#!/usr/bin/env python3
"""
Evaluate trained agents using Alpha-Rank.

This script:
1. Loads all saved models/checkpoints
2. Plays them against each other in a round-robin tournament
3. Computes Alpha-Rank to get game-theoretic rankings
4. Compares with ELO rankings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import glob
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime
import sys

# Import from project
from alpha_rank import AlphaRank, AlphaRankResult, visualize_payoff_matrix, PopulationTracker
from tcg.env import PTCGEnv

# Try to import MCTS (may not be available)
try:
    from tcg.mcts import MCTS
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False


# Transformer Block (same as in train_advanced.py)
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        
    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ffn(x))
        return x


class AdvancedPolicyValueNet(nn.Module):
    """Transformer-based Policy-Value Network (same as in train_advanced.py)."""
    def __init__(self, obs_dim: int = 156, n_actions: int = 322, d_model: int = 256, n_layers: int = 3):
        super().__init__()
        
        self.input_embed = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        self.n_tokens = 8
        self.token_dim = d_model // self.n_tokens
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_tokens, d_model))
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads=4, dropout=0.1)
            for _ in range(n_layers)
        ])
        
        self.pool = nn.Sequential(
            nn.Linear(d_model * self.n_tokens, d_model),
            nn.GELU(),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_actions),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )
        
        self.prize_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 7),
        )
        
        self.turn_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x, return_aux: bool = False):
        batch_size = x.shape[0]
        embedded = self.input_embed(x)
        tokens = embedded.view(batch_size, self.n_tokens, -1)
        
        if tokens.shape[-1] != self.pos_encoding.shape[-1]:
            tokens = F.pad(tokens, (0, self.pos_encoding.shape[-1] - tokens.shape[-1]))
        
        tokens = tokens + self.pos_encoding
        
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        
        pooled = self.pool(tokens.reshape(batch_size, -1))
        
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        if return_aux:
            prize_pred = self.prize_head(pooled)
            turn_pred = self.turn_head(pooled)
            return policy_logits, value, prize_pred, turn_pred
        
        return policy_logits, value


def load_model(path: str) -> Tuple[nn.Module, Dict]:
    """Load a model from checkpoint with automatic architecture detection."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    # Get dimensions from checkpoint
    from tcg.actions import ACTION_TABLE
    default_n_actions = len(ACTION_TABLE)
    obs_dim = data.get('obs_dim', 156)
    n_actions = data.get('n_actions', default_n_actions)
    
    # Detect architecture from state dict keys
    if isinstance(data, dict):
        state_dict = data.get('model_state_dict', data.get('state_dict', data))
    else:
        state_dict = data
        data = {}
    
    # Check for transformer architecture
    is_transformer = any('transformer' in k or 'pos_encoding' in k for k in state_dict.keys())
    
    if is_transformer:
        model = AdvancedPolicyValueNet(obs_dim=obs_dim, n_actions=n_actions)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown model architecture in {path}")
    
    model.eval()
    
    metadata = {
        'episode': data.get('episode', data.get('generation', 0)),
        'elo': data.get('elo', 1200),
        'win_rate': data.get('win_rate', 0.5),
        'n_actions': n_actions,
    }
    
    return model, metadata


def play_game(model1: nn.Module, 
              model2: nn.Module,
              deck1: Optional[List[str]] = None,
              deck2: Optional[List[str]] = None,
              use_mcts: bool = False,
              mcts_sims: int = 50,
              verbose: bool = False) -> int:
    """
    Play a single game between two models.
    
    Returns:
        0 if model1 wins, 1 if model2 wins, -1 if draw
    """
    env = PTCGEnv(scripted_opponent=False, max_turns=200)
    
    # Define Standard Training Decks
    deck_alakazam = []
    deck_alakazam.extend(["Abra"] * 4); deck_alakazam.extend(["Kadabra"] * 3); deck_alakazam.extend(["Alakazam"] * 4)
    deck_alakazam.extend(["Dunsparce"] * 4); deck_alakazam.extend(["Dudunsparce"] * 4)
    deck_alakazam.extend(["Fan Rotom"] * 2); deck_alakazam.extend(["Psyduck"] * 1); deck_alakazam.extend(["Fezandipiti ex"] * 1)
    deck_alakazam.extend(["Hilda"] * 4); deck_alakazam.extend(["Dawn"] * 4); deck_alakazam.extend(["Boss's Orders"] * 3)
    deck_alakazam.extend(["Lillie's Determination"] * 2); deck_alakazam.extend(["Tulip"] * 1)
    deck_alakazam.extend(["Buddy-Buddy Poffin"] * 4); deck_alakazam.extend(["Rare Candy"] * 3)
    deck_alakazam.extend(["Nest Ball"] * 2); deck_alakazam.extend(["Night Stretcher"] * 2)
    deck_alakazam.extend(["Wondrous Patch"] * 2); deck_alakazam.extend(["Enhanced Hammer"] * 2)
    deck_alakazam.extend(["Battle Cage"] * 3); deck_alakazam.extend(["Basic Psychic Energy"] * 3)
    deck_alakazam.extend(["Enriching Energy"] * 1); deck_alakazam.extend(["Jet Energy"] * 1)

    deck_charizard = []
    deck_charizard.extend(["Charmander"] * 3); deck_charizard.extend(["Charmeleon"] * 2); deck_charizard.extend(["Charizard ex"] * 2)
    deck_charizard.extend(["Pidgey"] * 2); deck_charizard.extend(["Pidgeotto"] * 2); deck_charizard.extend(["Pidgeot ex"] * 2)
    deck_charizard.extend(["Psyduck"] * 1); deck_charizard.extend(["Shaymin"] * 1); deck_charizard.extend(["Tatsugiri"] * 1)
    deck_charizard.extend(["Munkidori"] * 1); deck_charizard.extend(["Chi-Yu"] * 1)
    deck_charizard.extend(["Gouging Fire ex"] * 1); deck_charizard.extend(["Fezandipiti ex"] * 1)
    deck_charizard.extend(["Lillie's Determination"] * 4); deck_charizard.extend(["Arven"] * 4)
    deck_charizard.extend(["Boss's Orders"] * 3); deck_charizard.extend(["Iono"] * 2); deck_charizard.extend(["Professor Turo's Scenario"] * 1)
    deck_charizard.extend(["Buddy-Buddy Poffin"] * 4); deck_charizard.extend(["Ultra Ball"] * 3); deck_charizard.extend(["Rare Candy"] * 2)
    deck_charizard.extend(["Super Rod"] * 2); deck_charizard.extend(["Counter Catcher"] * 1); deck_charizard.extend(["Energy Search"] * 1)
    deck_charizard.extend(["Unfair Stamp"] * 1); deck_charizard.extend(["Technical Machine: Evolution"] * 2)
    deck_charizard.extend(["Artazon"] * 1); deck_charizard.extend(["Fire Energy"] * 5); deck_charizard.extend(["Mist Energy"] * 2)
    deck_charizard.extend(["Darkness Energy"] * 1); deck_charizard.extend(["Jet Energy"] * 1)
    
    deck_gholdengo = []
    deck_gholdengo.extend(["Gimmighoul"] * 4); deck_gholdengo.extend(["Gholdengo ex"] * 3)
    deck_gholdengo.extend(["Origin Forme Palkia V"] * 2); deck_gholdengo.extend(["Origin Forme Palkia VSTAR"] * 2)
    deck_gholdengo.extend(["Radiant Greninja"] * 1); deck_gholdengo.extend(["Manaphy"] * 1)
    deck_gholdengo.extend(["Irida"] * 4); deck_gholdengo.extend(["Boss's Orders"] * 2); deck_gholdengo.extend(["Iono"] * 2)
    deck_gholdengo.extend(["Lady"] * 4); deck_gholdengo.extend(["Cynthia's Ambition"] * 1)
    deck_gholdengo.extend(["Buddy-Buddy Poffin"] * 4); deck_gholdengo.extend(["Ultra Ball"] * 4); deck_gholdengo.extend(["Nest Ball"] * 2)
    deck_gholdengo.extend(["Earthen Vessel"] * 4); deck_gholdengo.extend(["Superior Energy Retrieval"] * 4); deck_gholdengo.extend(["Super Rod"] * 1)
    deck_gholdengo.extend(["Prime Catcher"] * 1); deck_gholdengo.extend(["Canceling Cologne"] * 1); deck_gholdengo.extend(["Hisuian Heavy Ball"] * 1)
    deck_gholdengo.extend(["Pokestop"] * 2)
    deck_gholdengo.extend(["Water Energy"] * 6); deck_gholdengo.extend(["Metal Energy"] * 6)

    # Use provided decks or Randomize from the 3 meta decks (like training)
    import random
    all_decks = [deck_alakazam, deck_charizard, deck_gholdengo]
    
    d1 = deck1 if deck1 else random.choice(all_decks)
    d2 = deck2 if deck2 else random.choice(all_decks)
    
    obs, info = env.reset(options={"decks": [d1, d2]})
    
    models = [model1, model2]
    
    if use_mcts:
        mcts_players = [MCTS(model1, env, mcts_sims), MCTS(model2, env, mcts_sims)]
    
    done = False
    steps = 0
    max_steps = 500
    
    while not done and steps < max_steps:
        current_player = env._gs.turn_player
        model = models[current_player]
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        mask = env.action_mask()
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
        
        if use_mcts:
            action = mcts_players[current_player].search(env)
        else:
            with torch.no_grad():
                logits, _ = model(obs_tensor)  # No mask in forward
                # Apply mask to logits
                logits = logits.masked_fill(~mask_tensor, float('-inf'))
                # Use temperature-based sampling for variety
                temperature = 0.5  # Lower = more deterministic
                probs = torch.softmax(logits / temperature, dim=-1)
                # Sample from distribution instead of argmax
                action = torch.multinomial(probs, 1).item()
        
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        
        if truncated:
            done = True
    
    winner = info.get('winner', -1)
    return winner


def round_robin_tournament(models: Dict[str, nn.Module],
                           games_per_matchup: int = 10,
                           use_mcts: bool = False,
                           verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Play round-robin tournament between all models.
    
    Returns:
        Payoff matrix and list of agent names
    """
    agent_names = list(models.keys())
    n = len(agent_names)
    
    wins = np.zeros((n, n))
    games = np.zeros((n, n))
    
    total_matchups = n * (n - 1) // 2 * games_per_matchup
    completed = 0
    
    if verbose:
        print(f"\n🎮 Round-Robin Tournament")
        print(f"   Agents: {n}")
        print(f"   Games per matchup: {games_per_matchup}")
        print(f"   Total games: {total_matchups}")
        print()
    
    for i in range(n):
        for j in range(i + 1, n):
            model1 = models[agent_names[i]]
            model2 = models[agent_names[j]]
            
            for g in range(games_per_matchup):
                # Alternate sides
                if g % 2 == 0:
                    winner = play_game(model1, model2, use_mcts=use_mcts)
                    if winner == 0:
                        wins[i, j] += 1
                    elif winner == 1:
                        wins[j, i] += 1
                    else:
                        wins[i, j] += 0.5
                        wins[j, i] += 0.5
                else:
                    winner = play_game(model2, model1, use_mcts=use_mcts)
                    if winner == 0:
                        wins[j, i] += 1
                    elif winner == 1:
                        wins[i, j] += 1
                    else:
                        wins[i, j] += 0.5
                        wins[j, i] += 0.5
                
                games[i, j] += 1
                games[j, i] += 1
                completed += 1
                
                if verbose and completed % 10 == 0:
                    print(f"\r   Progress: {completed}/{total_matchups} ({100*completed/total_matchups:.1f}%)", end="")
    
    if verbose:
        print()
    
    # Compute payoff matrix
    payoff = np.where(games > 0, wins / games, 0.5)
    np.fill_diagonal(payoff, 0.5)
    
    return payoff, agent_names


def find_models(directory: str = ".") -> Dict[str, str]:
    """Find all saved model files."""
    patterns = [
        "*.pt",
        "checkpoints/*.pt",
        "models/*.pt",
    ]
    
    models = {}
    for pattern in patterns:
        for path in glob.glob(os.path.join(directory, pattern)):
            name = os.path.basename(path).replace('.pt', '')
            models[name] = path
    
    return models



class Quiet:
    """Context manager for suppressing stdout/stderr."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def main():
    parser = argparse.ArgumentParser(description="Alpha-Rank evaluation of Pokemon TCG agents")
    parser.add_argument('--models', nargs='+', help="Specific model paths to evaluate")
    parser.add_argument('--games', type=int, default=10, help="Games per matchup")
    parser.add_argument('--mcts', action='store_true', help="Use MCTS for decision making")
    parser.add_argument('--mcts-sims', type=int, default=50, help="MCTS simulations")
    parser.add_argument('--alpha', type=float, default=0.1, help="Alpha-Rank alpha parameter")
    parser.add_argument('--output', type=str, default=None, help="Output JSON file")
    parser.add_argument('--quick', action='store_true', help="Quick evaluation (fewer games)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ALPHA-RANK EVALUATION")
    print("=" * 60)
    
    # Select specific models for focused evaluation if not provided
    if not args.models:
        candidate_models = []
        
        # Auto-discover checkpoints from checkpoints/ directory
        checkpoint_files = sorted(glob.glob('checkpoints/checkpoint_ep*.pt'))
        if checkpoint_files:
            # Include first, last, and a few evenly spaced checkpoints
            n_checkpoints = len(checkpoint_files)
            if n_checkpoints <= 5:
                candidate_models.extend(checkpoint_files)
            else:
                # Select ~5 evenly spaced checkpoints
                indices = [0, n_checkpoints // 4, n_checkpoints // 2, 3 * n_checkpoints // 4, n_checkpoints - 1]
                for idx in sorted(set(indices)):
                    candidate_models.append(checkpoint_files[idx])
        
        # Auto-discover league agents
        league_files = sorted(glob.glob('league/agent_ep*.pt'))
        if league_files:
            n_league = len(league_files)
            if n_league <= 3:
                candidate_models.extend(league_files)
            else:
                # Select first, middle, and last
                indices = [0, n_league // 2, n_league - 1]
                for idx in sorted(set(indices)):
                    candidate_models.append(league_files[idx])
        
        # Always include best models if they exist
        for special in ['best_elo_policy.pt', 'advanced_policy.pt']:
            if os.path.exists(special) and special not in candidate_models:
                candidate_models.append(special)
        
        # Filter for existing files (shouldn't be needed but safety check)
        args.models = [m for m in candidate_models if os.path.exists(m)]
        print("Automatically discovered models for evaluation.")

    # Find models
    if args.models:
        model_paths = {}
        for p in args.models:
            name = os.path.basename(p).replace('.pt', '')
            # Create a unique name if duplicates (though unlikely with these paths)
            if name in model_paths:
                name = f"{name}_{p.replace('/', '_')}"
            model_paths[name] = p
    else:
        model_paths = find_models()
    
    if len(model_paths) < 2:
        print("❌ Need at least 2 models for evaluation")
        print(f"   Found: {list(model_paths.keys())}")
        return
    
    print(f"\n📦 Found {len(model_paths)} models:")
    for name, path in model_paths.items():
        print(f"   - {name}")
    
    # Load models
    print("\n📥 Loading models...")
    models = {}
    metadata = {}
    
    for name, path in model_paths.items():
        try:
            model, meta = load_model(path)
            models[name] = model
            metadata[name] = meta
            elo = meta.get('elo', 'N/A')
            ep = meta.get('episode', 'N/A')
            print(f"   ✓ {name}: ELO={elo}, Episode={ep}")
        except Exception as e:
            print(f"   ✗ {name}: Failed to load ({e})")
    
    if len(models) < 2:
        print("❌ Need at least 2 valid models")
        return
    
    # Adjust games for quick mode
    games_per_matchup = 3 if args.quick else args.games
    
    # Run tournament
    print(f"\n🎮 Running tournament ({games_per_matchup} games per matchup)...")
    
    # Monkey patch play_game calls in round_robin_tournament to use Quiet()
    # Or better, just wrap the round_robin_tournament loop calls
    # But round_robin_tournament calls play_game directly.
    # We will wrap the execution inside play_game by modifying it via a wrapper or just modifying the loop in main if we could.
    # Since we can't easily modify round_robin_tournament without editing the function def above, let's redefine it here or use a global flag.
    # Simpler: Redefine play_game in this scope to wrap the original reference? 
    # No, python scoping.
    
    # We will modify the round_robin_tournament function in the file to use Quiet context.
    # But I am editing main() here.
    
    # Let's perform the tournament loop here manually or wrap the call?
    # Actually, I can just redirect stdout around the whole tournament call if I don't care about progress bars updates being visible?
    # The progress bar uses print with \r. If I suppress stdout, I lose the progress bar.
    # I want to suppress the logs from env.step which go to stdout.
    
    # Best approach: Redefine round_robin_tournament to use Quiet() around play_game().
    # Since I am replacing the end of the file, I can verify if I can replace round_robin_tournament too.
    # But I only selected lines 363-486 (main).
    
    # So I will just suppress stdout for the whole tournament and print my own progress?
    # Or just let it run. With 6 models and 5 matchups each (15 total pairings) * 10 games = 150 games.
    # It might be fast enough even with prints, but prints slow it down.
    # I will wrap the call to round_robin_tournament with Quiet() and just wait.
    
    print("   (Suppressing game logs...)")
    with Quiet():
        payoff, agent_names = round_robin_tournament(
            models, 
            games_per_matchup=games_per_matchup,
            use_mcts=args.mcts,
            verbose=False # Turn off internal verbose which prints progress
        )
    
    # Compute Alpha-Rank
    print("\n📊 Computing Alpha-Rank...")
    ranker = AlphaRank(alpha=args.alpha)
    result = ranker.compute(payoff, agent_names)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Alpha-Rank rankings
    print("\n🏆 Alpha-Rank Rankings:")
    print("-" * 40)
    for i, (name, score) in enumerate(result.top_agents(len(agent_names))):
        elo = metadata.get(name, {}).get('elo', 'N/A')
        bar = "█" * int(score * 40)
        print(f"{i+1:2}. {name:25s} {score:.4f} {bar}")
        print(f"     ELO: {elo}")
    
    # Payoff matrix
    visualize_payoff_matrix(payoff, agent_names)
    
    # Compare with ELO
    print("\n📈 ELO vs Alpha-Rank Comparison:")
    print("-" * 50)
    
    elo_ranks = sorted([(name, metadata.get(name, {}).get('elo', 1200)) 
                        for name in agent_names], key=lambda x: -x[1])
    alpha_ranks = result.top_agents(len(agent_names))
    
    print(f"{'Agent':25s} {'ELO Rank':>10s} {'Alpha Rank':>12s} {'Diff':>6s}")
    print("-" * 55)
    
    for i, (name, _) in enumerate(alpha_ranks):
        elo_rank = next((j+1 for j, (n, _) in enumerate(elo_ranks) if n == name), -1)
        alpha_rank = i + 1
        diff = elo_rank - alpha_rank
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{name:25s} {elo_rank:>10d} {alpha_rank:>12d} {diff_str:>6s}")
    
    # Save results
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'alpha': args.alpha,
            'games_per_matchup': games_per_matchup,
            'rankings': result.rankings,
            'payoff_matrix': payoff.tolist(),
            'agent_names': agent_names,
            'metadata': {k: v for k, v in metadata.items()},
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

