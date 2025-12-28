"""
Advanced AlphaZero Training for Pokemon TCG with:
1. Population-Based Training (PBT)
2. League Training (Historical Opponents)
3. Transformer-based Network Architecture
4. Mirror Training Support
5. Prioritized Experience Replay
6. Auxiliary Tasks (Prize/Turn Prediction)
7. Monte Carlo Return Estimation with Shaping
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
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, Queue
import torch.multiprocessing as tmp

from tcg.env import PTCGEnv
from tcg.state import featurize
from tcg.actions import ACTION_TABLE
from tcg.mcts import MCTS
from tcg.cards import card_def
# Replay recording imports
import json
from record_replay import record_visual_replay, VisualReplay
# Alpha-Rank for game-theoretic rankings
from alpha_rank import PopulationTracker
# Scripted agents for external pressure
from tcg.scripted_agent import ScriptedAgent, SCRIPTED_AGENTS, get_scripted_agent

# =============================================================================
# COMBO SYSTEM: Hierarchical RL via Injection & Discovery
# Bridges "Temporal Gap" - agent learns that card X now enables move Y later
# =============================================================================

@dataclass
class ComboStep:
    """Single step in a strategic combo."""
    name: str
    # Condition: lambda(env, player) -> bool. Can we start this step?
    condition: any  # Callable[[Any, Any], bool]
    # Selector: lambda(env, player) -> int. Which action index to take?
    action_selector: any  # Callable[[Any, Any], int]

@dataclass 
class StrategicCombo:
    """Multi-step strategic sequence that leads to high-value states."""
    name: str
    steps: List[ComboStep]
    reward_bonus: float = 1.0

# --- HELPER FUNCTIONS FOR COMBO CONDITIONS ---
def has_in_hand(p, card_name): return card_name in p.hand
def is_active(p, card_name): return p.active.name == card_name
def has_on_bench(p, card_name): return any(s.name == card_name for s in p.bench)
def has_in_discard(p, card_type): return any(card_type in c for c in p.discard_pile)

# --- COMBO REGISTRY: Define strategic sequences ---
COMBO_REGISTRY = [
    
    # 1. THE "GHOLDENGO REFUEL" (Superior Energy Retrieval -> Make It Rain)
    # Teaches: Retrieval enables massive damage
    StrategicCombo(
        name="Gholdengo_Refuel",
        reward_bonus=2.0,
        steps=[
            # Step A: Use Retrieval (if Gholdengo active + energy in discard)
            ComboStep(
                name="Retrieve Energy",
                condition=lambda env, p: (
                    is_active(p, "Gholdengo ex") and 
                    has_in_hand(p, "Superior Energy Retrieval") and 
                    sum(1 for c in p.discard_pile if "Energy" in c) >= 2
                ),
                action_selector=lambda env, p: next(
                    (i for i, a in enumerate(ACTION_TABLE) 
                     if a.kind == "PLAY_TRAINER" and a.a == "Superior Energy Retrieval"), 0
                )
            ),
        ]
    ),

    # 2. THE "CANDY ENGINE" (Rare Candy -> Stage 2)
    # Teaches: Candy enables fast evolution
    StrategicCombo(
        name="Rare_Candy_Charizard",
        reward_bonus=1.5,
        steps=[
            ComboStep(
                name="Play Candy to Charizard",
                condition=lambda env, p: (
                    has_in_hand(p, "Rare Candy") and 
                    has_in_hand(p, "Charizard ex") and 
                    (has_on_bench(p, "Charmander") or is_active(p, "Charmander"))
                ),
                action_selector=lambda env, p: next(
                    (i for i, a in enumerate(ACTION_TABLE) 
                     if a.kind == "PLAY_TRAINER" and a.a == "Rare Candy"), 0
                )
            )
        ]
    ),
    
    # 3. THE "ALAKAZAM CANDY" (Rare Candy -> Alakazam)
    StrategicCombo(
        name="Rare_Candy_Alakazam",
        reward_bonus=1.5,
        steps=[
            ComboStep(
                name="Play Candy to Alakazam",
                condition=lambda env, p: (
                    has_in_hand(p, "Rare Candy") and 
                    has_in_hand(p, "Alakazam") and 
                    (has_on_bench(p, "Abra") or is_active(p, "Abra"))
                ),
                action_selector=lambda env, p: next(
                    (i for i, a in enumerate(ACTION_TABLE) 
                     if a.kind == "PLAY_TRAINER" and a.a == "Rare Candy"), 0
                )
            )
        ]
    ),

    # NOTE: Alakazam/Kadabra Psychic Draw is ON-EVOLUTION, triggers automatically
    # when you evolve - not an activated ability, so no combo needed

    # 5. "WONDROUS PATCH SETUP" (Wondrous Patch when energy in discard)
    StrategicCombo(
        name="Wondrous_Patch_Attach",
        reward_bonus=1.0,
        steps=[
            ComboStep(
                name="Use Wondrous Patch",
                condition=lambda env, p: (
                    has_in_hand(p, "Wondrous Patch") and
                    any(c == "Basic Psychic Energy" for c in p.discard_pile) and
                    (is_active(p, "Alakazam") or has_on_bench(p, "Alakazam") or
                     is_active(p, "Kadabra") or has_on_bench(p, "Kadabra"))
                ),
                action_selector=lambda env, p: next(
                    (i for i, a in enumerate(ACTION_TABLE) 
                     if a.kind == "PLAY_TRAINER" and a.a == "Wondrous Patch"), 0
                )
            )
        ]
    ),
    
    # 6. "BOSS'S ORDERS SNIPE" (Boss when opponent has weak benched target)
    StrategicCombo(
        name="Boss_Snipe",
        reward_bonus=1.0,
        steps=[
            ComboStep(
                name="Boss Orders to Drag Weak Target",
                condition=lambda env, p: (
                    has_in_hand(p, "Boss's Orders") and
                    not getattr(p, 'supporter_used', False) and
                    # Check if opponent has a low-HP bench target
                    any(
                        s.name and card_def(s.name).hp - s.damage <= 100
                        for s in env._gs.players[1 - env._gs.turn_player].bench
                    )
                ),
                action_selector=lambda env, p: next(
                    (i for i, a in enumerate(ACTION_TABLE) 
                     if a.kind == "PLAY_TRAINER" and a.a == "Boss's Orders"), 0
                )
            )
        ]
    ),
    
    # 7. "ENRICHING ENERGY DRAW" (Attach Enriching Energy for +4 cards)
    # Teaches: Enriching Energy = massive card advantage
    StrategicCombo(
        name="Enriching_Energy_Draw",
        reward_bonus=1.0,
        steps=[
            ComboStep(
                name="Attach Enriching Energy",
                condition=lambda env, p: (
                    has_in_hand(p, "Enriching Energy") and
                    not getattr(p, 'energy_attached', False) and
                    len(p.deck) >= 4  # Worth it if we can draw 4
                ),
                action_selector=lambda env, p: next(
                    (i for i, a in enumerate(ACTION_TABLE) 
                     if a.kind == "ATTACH_ACTIVE" and a.a == "Enriching Energy"), 0
                )
            )
        ]
    ),
    
    # 8. "DUDUNSPARCE RUN AWAY DRAW" (Draw 3, shuffle self back to recycle)
    # Teaches: Dudunsparce recycles itself + attached cards (like Enriching Energy!)
    StrategicCombo(
        name="Dudunsparce_Run_Away_Draw",
        reward_bonus=1.5,
        steps=[
            ComboStep(
                name="Use Run Away Draw",
                condition=lambda env, p: (
                    (is_active(p, "Dudunsparce") or has_on_bench(p, "Dudunsparce")) and
                    not getattr(p, 'ability_used_this_turn', False) and
                    len(p.deck) >= 3  # Need deck for draw
                ),
                action_selector=lambda env, p: next(
                    (i for i, a in enumerate(ACTION_TABLE) 
                     if a.kind == "USE_ACTIVE_ABILITY"), 0
                )
            )
        ]
    ),
    
    # 9. "ALAKAZAM ATTACK" (Powerful Hand when ready - damage scales with YOUR hand size)
    StrategicCombo(
        name="Alakazam_Powerful_Hand",
        reward_bonus=1.5,
        steps=[
            ComboStep(
                name="Attack with Powerful Hand",
                condition=lambda env, p: (
                    is_active(p, "Alakazam") and
                    len(p.active.energy) >= 1 and  # Has Psychic energy to attack
                    len(p.hand) >= 5  # Big hand = big damage!
                ),
                action_selector=lambda env, p: next(
                    (i for i, a in enumerate(ACTION_TABLE) 
                     if a.kind == "ATTACK" and a.b == 0), 0  # First attack
                )
            )
        ]
    ),
]


# =============================================================================
# IMPROVED NETWORK ARCHITECTURE
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention."""
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
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class AdvancedPolicyValueNet(nn.Module):
    """
    Transformer-based Policy-Value Network with Auxiliary Heads.
    
    Outputs:
    - Policy logits (action probabilities)
    - Value (game outcome prediction)
    - Auxiliary: Prize prediction, Turn prediction
    """
    def __init__(self, obs_dim: int, n_actions: int, d_model: int = 256, n_layers: int = 3):
        super().__init__()
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Reshape for transformer: treat different parts of observation as "tokens"
        # We'll reshape the d_model features into a sequence
        self.n_tokens = 8  # Split into 8 virtual "card slots"
        self.token_dim = d_model // self.n_tokens
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_tokens, d_model))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads=4, dropout=0.1)
            for _ in range(n_layers)
        ])
        
        # Pooling
        self.pool = nn.Sequential(
            nn.Linear(d_model * self.n_tokens, d_model),
            nn.GELU(),
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_actions),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )
        
        # Auxiliary heads
        self.prize_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 7),  # Predict 0-6 prizes taken
        )
        
        self.turn_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),  # Predict remaining turns (regression)
        )
    
    def forward(self, x, return_aux: bool = False):
        batch_size = x.shape[0]
        
        # Embed input
        embedded = self.input_embed(x)  # [B, d_model]
        
        # Reshape to sequence of tokens
        tokens = embedded.view(batch_size, self.n_tokens, -1)  # [B, n_tokens, d_model/n_tokens]
        
        # Pad to full d_model if needed
        if tokens.shape[-1] != self.pos_encoding.shape[-1]:
            tokens = F.pad(tokens, (0, self.pos_encoding.shape[-1] - tokens.shape[-1]))
        
        # Add positional encoding
        tokens = tokens + self.pos_encoding
        
        # Apply transformer
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        
        # Pool all tokens
        pooled = tokens.reshape(batch_size, -1)
        pooled = self.pool(pooled)
        
        # Outputs
        policy = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)
        
        if return_aux:
            prize_pred = self.prize_head(pooled)
            turn_pred = self.turn_head(pooled).squeeze(-1)
            return policy, value, prize_pred, turn_pred
        
        return policy, value


# =============================================================================
# PRIORITIZED EXPERIENCE REPLAY
# =============================================================================

@dataclass
class Experience:
    """Enhanced experience with priority and auxiliary targets."""
    obs: np.ndarray
    mcts_probs: np.ndarray
    value: float
    priority: float = 1.0
    # Auxiliary targets
    prizes_taken: int = 0
    turns_remaining: int = 0


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with TD-error weighting."""
    
    def __init__(self, capacity: int = 200000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        self.max_priority = 1.0
    
    def add(self, exp: Experience):
        exp.priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.position] = exp
        
        self.priorities[self.position] = self.max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
    
    def add_game_with_shaping(self, game_history: List[Tuple], winner: int, 
                               action_rewards: List[float], gamma: float = 0.99):
        """Add game with Monte Carlo return estimation and shaping."""
        n = len(game_history)
        
        for i, (obs, mcts_probs, player) in enumerate(game_history):
            # Base value from outcome
            if winner == -1:
                base_value = 0.0
            elif winner == player:
                base_value = 1.0
            else:
                base_value = -1.0
            
            # Discounted shaping rewards
            shaping_value = 0.0
            for j in range(i, min(i + 20, n)):
                if j < len(action_rewards):
                    if game_history[j][2] == player:
                        shaping_value += action_rewards[j] * (gamma ** (j - i))
            
            # === FIX: Make rewards significant ===
            # Increased from 0.05 to 0.50 so agent cares about strategic rewards
            shaping_score = shaping_value * 0.50
            
            if base_value == 1.0:  # Win
                # Boost wins slightly, but cap at 1.0
                value = np.clip(1.0 + shaping_score, 0.8, 1.0)
            elif base_value == -1.0:  # Loss
                # Mitigate loss slightly, but NEVER let it cross -0.5 (always negative!)
                value = np.clip(-1.0 + shaping_score, -1.0, -0.5)
            else:  # Draw
                # Small adjustments for draws
                value = np.clip(shaping_score, -0.2, 0.2)
            # === END FIX ===
            
            # Calculate auxiliary targets
            # (In real implementation, extract these from game state)
            prizes_taken = 0  # Placeholder
            turns_remaining = max(0, n - i)
            
            exp = Experience(
                obs=obs,
                mcts_probs=mcts_probs,
                value=value,
                priority=1.0,
                prizes_taken=prizes_taken,
                turns_remaining=turns_remaining,
            )
            self.add(exp)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Sample with priorities, return experiences, weights, and indices."""
        n = len(self.buffer)
        if n == 0:
            return [], np.array([]), []
        
        # Calculate sampling probabilities
        priorities = self.priorities[:n]
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(n, size=min(batch_size, n), p=probs, replace=False)
        
        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, weights, indices.tolist()
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# LEAGUE TRAINING (Historical Opponents)
# =============================================================================

class League:
    """Maintains a league of historical agents + exploiters."""
    
    def __init__(self, max_agents: int = 20, save_dir: str = "league"):
        self.max_agents = max_agents
        self.save_dir = save_dir
        self.agents: List[Dict] = []  # List of {path, elo, games}
        os.makedirs(save_dir, exist_ok=True)
    
    def add_agent(self, model: nn.Module, episode: int, elo: float = 1200):
        """Save a checkpoint to the league."""
        path = os.path.join(self.save_dir, f"agent_ep{episode}.pt")
        torch.save(model.state_dict(), path)
        
        self.agents.append({
            "path": path,
            "episode": episode,
            "elo": elo,
            "games": 0,
        })
        
        # Keep only top agents by ELO
        if len(self.agents) > self.max_agents:
            self.agents.sort(key=lambda x: x["elo"], reverse=True)
            removed = self.agents.pop()
            if os.path.exists(removed["path"]):
                os.remove(removed["path"])
    
    def sample_opponent(self, current_elo: float) -> Optional[str]:
        """Sample an opponent from the league (weighted by ELO proximity)."""
        if not self.agents:
            return None
        
        # Weight opponents by how close their ELO is
        weights = []
        for agent in self.agents:
            diff = abs(agent["elo"] - current_elo)
            weight = 1.0 / (1.0 + diff / 100)
            weights.append(weight)
        
        weights = np.array(weights) / sum(weights)
        idx = np.random.choice(len(self.agents), p=weights)
        return self.agents[idx]["path"]
    
    def update_elo(self, agent_path: str, won: bool, opponent_elo: float):
        """Update ELO rating after a game."""
        K = 32  # ELO K-factor
        
        for agent in self.agents:
            if agent["path"] == agent_path:
                expected = 1 / (1 + 10 ** ((opponent_elo - agent["elo"]) / 400))
                agent["elo"] += K * (int(won) - expected)
                agent["games"] += 1
                break


# =============================================================================
# GENETIC ALGORITHM POPULATION
# =============================================================================

class GeneticPopulation:
    """
    Genetic Algorithm Population with:
    - Tournament Selection
    - Crossover (weight blending)
    - Mutation (weight perturbation)
    - Elitism (preserve top performers)
    - Species/Niche protection for diversity
    """
    
    def __init__(self, n_agents: int, obs_dim: int, n_actions: int, device: torch.device,
                 mutation_rate: float = 0.1, mutation_strength: float = 0.02,
                 crossover_rate: float = 0.3, elitism_count: int = 2):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        
        # Genetic parameters
        self.mutation_rate = mutation_rate  # Probability of mutating each weight
        self.mutation_strength = mutation_strength  # Std dev of mutation noise
        self.crossover_rate = crossover_rate  # Probability of crossover vs copy
        self.elitism_count = min(elitism_count, n_agents // 2)  # Top N preserved
        
        # Initialize population
        self.models = [
            AdvancedPolicyValueNet(obs_dim, n_actions).to(device)
            for _ in range(n_agents)
        ]
        
        self.optimizers = [
            optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for model in self.models
        ]
        
        # Fitness tracking
        self.fitness = [0.0] * n_agents
        self.games_played = [0] * n_agents
        self.wins = [0] * n_agents
        self.prizes_taken = [0] * n_agents  # Track offensive capability
        self.evolutions_achieved = [0] * n_agents  # Track strategic play
        
        # Generation tracking
        self.generation = 0
        self.best_fitness_history = []
        
        # Species for diversity (simple implementation)
        self.species_id = list(range(n_agents))  # Each starts as own species
    
    def get_random_pair(self) -> Tuple[int, int]:
        """Get two different agent indices for a match."""
        i = random.randrange(self.n_agents)
        j = random.randrange(self.n_agents)
        while j == i and self.n_agents > 1:
            j = random.randrange(self.n_agents)
        return i, j
    
    def update_fitness(self, agent_idx: int, won: bool, prizes: int = 0, evolutions: int = 0):
        """
        Update fitness with multi-objective criteria.
        Fitness = wins + 0.1 * prizes_taken + 0.05 * evolutions
        """
        self.games_played[agent_idx] += 1
        if won:
            self.wins[agent_idx] += 1
        self.prizes_taken[agent_idx] += prizes
        self.evolutions_achieved[agent_idx] += evolutions
        
        # Calculate composite fitness
        games = max(self.games_played[agent_idx], 1)
        win_rate = self.wins[agent_idx] / games
        avg_prizes = self.prizes_taken[agent_idx] / games
        avg_evos = self.evolutions_achieved[agent_idx] / games
        
        # Multi-objective fitness
        self.fitness[agent_idx] = win_rate + 0.1 * avg_prizes + 0.05 * avg_evos
    
    def update_scores(self, winner_idx: int, loser_idx: int):
        """Legacy compatibility - simple win/loss update."""
        self.update_fitness(winner_idx, won=True)
        self.update_fitness(loser_idx, won=False)
    
    def tournament_select(self, tournament_size: int = 3) -> int:
        """Select an individual using tournament selection."""
        candidates = random.sample(range(self.n_agents), min(tournament_size, self.n_agents))
        best = max(candidates, key=lambda i: self.fitness[i])
        return best
    
    def mutate_model(self, model: nn.Module, strength: float = None) -> nn.Module:
        """
        Mutate a model's weights by adding Gaussian noise.
        Returns a new mutated model.
        """
        if strength is None:
            strength = self.mutation_strength
        
        mutated = AdvancedPolicyValueNet(self.obs_dim, self.n_actions).to(self.device)
        mutated.load_state_dict(copy.deepcopy(model.state_dict()))
        
        with torch.no_grad():
            for param in mutated.parameters():
                if random.random() < self.mutation_rate:
                    # Add scaled Gaussian noise
                    noise = torch.randn_like(param) * strength
                    param.add_(noise)
        
        return mutated
    
    def crossover(self, parent1: nn.Module, parent2: nn.Module, 
                  blend_ratio: float = None) -> nn.Module:
        """
        Create offspring by blending weights from two parents.
        Uses uniform crossover with optional blend ratio.
        """
        if blend_ratio is None:
            blend_ratio = random.uniform(0.3, 0.7)  # Random blend
        
        child = AdvancedPolicyValueNet(self.obs_dim, self.n_actions).to(self.device)
        
        state1 = parent1.state_dict()
        state2 = parent2.state_dict()
        child_state = {}
        
        for key in state1.keys():
            if random.random() < 0.5:
                # Uniform crossover: take from one parent
                child_state[key] = state1[key].clone()
            else:
                # Blend crossover: weighted average
                child_state[key] = blend_ratio * state1[key] + (1 - blend_ratio) * state2[key]
        
        child.load_state_dict(child_state)
        return child
    
    def calculate_diversity(self, model1: nn.Module, model2: nn.Module) -> float:
        """Calculate L2 distance between two models' weights (for species/niche)."""
        total_dist = 0.0
        count = 0
        
        state1 = model1.state_dict()
        state2 = model2.state_dict()
        
        for key in state1.keys():
            if 'weight' in key:  # Only compare weight tensors
                diff = (state1[key] - state2[key]).flatten()
                total_dist += torch.norm(diff).item()
                count += 1
        
        return total_dist / max(count, 1)
    
    def compute_population_diversity(self) -> float:
        """
        Compute population diversity using fitness variance.
        Low variance = agents are too similar = need evolution.
        Returns variance of fitness scores.
        """
        # Use fitness (which actually exists) instead of elo
        if len(self.fitness) < 2:
            return 100.0  # High diversity by default
        
        fitness_arr = np.array(self.fitness)
        # If all fitness are 0-1 range, scale variance up to be comparable to threshold
        variance = np.var(fitness_arr) * 10000  # Scale up small variances
        return variance
    
    def needs_evolution(self, diversity_threshold: float = 50.0) -> bool:
        """
        Check if population needs evolution based on diversity.
        If ELO variance is low, agents are too similar.
        """
        diversity = self.compute_population_diversity()
        return diversity < diversity_threshold
    
    def evolve_generation(self, verbose: bool = False):
        """
        Run one generation of genetic evolution:
        1. Rank by fitness
        2. Elitism: Keep top performers
        3. Selection: Tournament select parents
        4. Crossover: Create offspring
        5. Mutation: Add variation
        """
        self.generation += 1
        
        # Require minimum games before evolution
        if min(self.games_played) < 5:
            return
        
        # Sort by fitness
        rankings = np.argsort(self.fitness)[::-1]  # Descending
        
        # Track best fitness
        best_fitness = self.fitness[rankings[0]]
        self.best_fitness_history.append(best_fitness)
        
        if verbose:
            print(f"\n=== Generation {self.generation} ===")
            print(f"Best Fitness: {best_fitness:.3f}")
            print(f"Avg Fitness: {np.mean(self.fitness):.3f}")
            print(f"Top 3: {[self.fitness[i] for i in rankings[:3]]}")
        
        # === ELITISM: Preserve top performers ===
        elite_indices = rankings[:self.elitism_count]
        elite_models = [copy.deepcopy(self.models[i].state_dict()) for i in elite_indices]
        
        # === CREATE NEW POPULATION ===
        new_models = []
        
        # Add elite models unchanged
        for i, state_dict in enumerate(elite_models):
            new_model = AdvancedPolicyValueNet(self.obs_dim, self.n_actions).to(self.device)
            new_model.load_state_dict(state_dict)
            new_models.append(new_model)
        
        # Fill rest with offspring
        while len(new_models) < self.n_agents:
            # Tournament selection for parents
            parent1_idx = self.tournament_select()
            parent2_idx = self.tournament_select()
            
            # Decide: crossover or mutation
            if random.random() < self.crossover_rate and parent1_idx != parent2_idx:
                # Crossover
                child = self.crossover(self.models[parent1_idx], self.models[parent2_idx])
                # Also mutate the child
                child = self.mutate_model(child, strength=self.mutation_strength * 0.5)
            else:
                # Clone and mutate
                child = self.mutate_model(self.models[parent1_idx])
            
            new_models.append(child)
        
        # Replace population
        self.models = new_models
        
        # Create new optimizers for non-elite models
        self.optimizers = [
            optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for model in self.models
        ]
        
        # Reset fitness for new generation (but keep cumulative for elites)
        new_fitness = [0.0] * self.n_agents
        new_games = [0] * self.n_agents
        new_wins = [0] * self.n_agents
        new_prizes = [0] * self.n_agents
        new_evos = [0] * self.n_agents
        
        # Preserve some history for elites (dampened)
        for new_idx, old_idx in enumerate(elite_indices):
            new_fitness[new_idx] = self.fitness[old_idx] * 0.5  # Decay
            new_games[new_idx] = max(1, self.games_played[old_idx] // 2)
            new_wins[new_idx] = self.wins[old_idx] // 2
            new_prizes[new_idx] = self.prizes_taken[old_idx] // 2
            new_evos[new_idx] = self.evolutions_achieved[old_idx] // 2
        
        self.fitness = new_fitness
        self.games_played = new_games
        self.wins = new_wins
        self.prizes_taken = new_prizes
        self.evolutions_achieved = new_evos
        
        if verbose:
            print(f"New generation created with {len(elite_models)} elites + {self.n_agents - len(elite_models)} offspring")
    
    def get_best_model(self) -> nn.Module:
        """Return the model with highest fitness."""
        best_idx = np.argmax(self.fitness)
        return self.models[best_idx]
    
    def get_population_stats(self) -> Dict:
        """Return population statistics."""
        return {
            "generation": self.generation,
            "best_fitness": max(self.fitness),
            "avg_fitness": np.mean(self.fitness),
            "min_fitness": min(self.fitness),
            "total_games": sum(self.games_played),
            "best_win_rate": max(w/max(g,1) for w, g in zip(self.wins, self.games_played)),
        }


# Legacy alias for backwards compatibility
Population = GeneticPopulation


def evaluate_vs_checkpoint(current_model: nn.Module, 
                           checkpoint_path: str,
                           device: torch.device,
                           num_games: int = 10,
                           obs_dim: int = 156,
                           n_actions: int = 873) -> Tuple[float, int, int]:
    """
    Play current model against a saved checkpoint to measure progress.
    
    Returns:
        win_rate: float (0.0 to 1.0)
        wins: int
        losses: int
    """
    # Load checkpoint model
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_model = AdvancedPolicyValueNet(obs_dim, n_actions).to(device)
        checkpoint_model.load_state_dict(checkpoint_data['state_dict'])
        checkpoint_model.eval()
    except Exception as e:
        print(f"⚠️ Could not load checkpoint {checkpoint_path}: {e}")
        return 0.5, 0, 0
    
    wins, losses, draws = 0, 0, 0
    current_model.eval()
    
    for game_num in range(num_games):
        env = PTCGEnv(scripted_opponent=False, max_turns=30)
        obs, info = env.reset()
        done = False
        
        # Alternate which player current model plays as
        current_is_p0 = (game_num % 2 == 0)
        
        while not done:
            player = env._gs.turn_player
            model = current_model if (player == 0) == current_is_p0 else checkpoint_model
            
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            mask = env.action_mask()
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = model(obs_t)
                logits = logits.masked_fill(~mask_t, float('-inf'))
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
            obs, reward, done, truncated, info = env.step(action)
            if truncated:
                done = True
        
        winner = info.get('winner', -1)
        current_won = (winner == 0 and current_is_p0) or (winner == 1 and not current_is_p0)
        
        if winner == -1:
            draws += 1
        elif current_won:
            wins += 1
        else:
            losses += 1
    
    current_model.train()
    total_decisive = wins + losses
    win_rate = wins / total_decisive if total_decisive > 0 else 0.5
    
    return win_rate, wins, losses


def play_single_game_worker(args):
    """
    Worker function to play a single game and collect experiences.
    Designed to run in a separate process for parallelization.
    
    Returns:
        Dictionary with game results and experiences
    """
    (model_p0_state, model_p1_state, obs_dim, n_actions, deck_p0, deck_p1, 
     mcts_sims, temperature, game_id, p0_idx, p1_idx, scripted_strategy, curriculum_mode) = args
    
    # Imports restricted to this scope to ensure clean worker process
    import torch
    import numpy as np
    from tcg.env import PTCGEnv
    from tcg.actions import ACTION_TABLE
    from tcg.mcts import MCTS
    from tcg.cards import card_def
    from tcg.scripted_agent import ScriptedAgent
    
    # Preventing CPU oversubscription
    torch.set_num_threads(1)
    
    # Create models on CPU
    device = torch.device('cpu')
    
    model_p0 = AdvancedPolicyValueNet(obs_dim, n_actions).to(device)
    model_p0.load_state_dict(model_p0_state)
    model_p0.eval()
    
    if model_p1_state is None:
        model_p1 = model_p0
    else:
        model_p1 = AdvancedPolicyValueNet(obs_dim, n_actions).to(device)
        model_p1.load_state_dict(model_p1_state)
        model_p1.eval()
    
    # Create MCTS agents
    mcts_p0 = MCTS(model_p0, device, num_simulations=mcts_sims, temperature=temperature,
                use_value_net=True, use_policy_rollouts=True)
    
    # Create P1 agent - either MCTS or scripted
    use_scripted_p1 = scripted_strategy is not None
    if use_scripted_p1:
        scripted_agent = ScriptedAgent(scripted_strategy)
        mcts_p1 = None  # Not used
    else:
        mcts_p1 = MCTS(model_p1, device, num_simulations=mcts_sims, temperature=temperature,
                    use_value_net=True, use_policy_rollouts=True)
    
    # Play game
    env = PTCGEnv(scripted_opponent=False, max_turns=30)
    obs, info = env.reset(options={"decks": [deck_p0, deck_p1]})
    done = False
    
    game_history = []
    action_rewards = []
    step_count = 0
    
    # Combo Discovery: Track action chains leading to prizes
    action_chain = deque(maxlen=10)  # Rolling window of last 10 actions
    discovered_combos = []
    
    while not done and step_count < 2000:
        turn_player = env._gs.turn_player
        mask = env.action_mask()
        
        # Use proper MCTS agent or scripted agent
        if turn_player == 0:
            # Pass COMBO_REGISTRY for injection
            action, mcts_probs = mcts_p0.search(env, return_probs=True, combo_registry=COMBO_REGISTRY)
            
            # CURRICULUM LEARNING: Teach optimal action sequencing
            # Attack ends turn, so do everything else FIRST!
            if curriculum_mode:
                legal_indices = np.where(mask)[0]
                attack_indices = [i for i in legal_indices if ACTION_TABLE[i].kind == 'ATTACK']
                attach_indices = [i for i in legal_indices if 'ATTACH' in ACTION_TABLE[i].kind]
                evolve_indices = [i for i in legal_indices if 'EVOLVE' in ACTION_TABLE[i].kind]
                ability_indices = [i for i in legal_indices if 'ABILITY' in ACTION_TABLE[i].kind]
                
                forced_action = None
                
                # Priority 1: Evolve first (better attacks, more HP)
                if evolve_indices:
                    forced_action = np.random.choice(evolve_indices)
                # Priority 2: Attach energy (enables attacks)
                elif attach_indices:
                    forced_action = np.random.choice(attach_indices)
                # Priority 3: Use abilities (card advantage)
                elif ability_indices:
                    forced_action = np.random.choice(ability_indices)
                # Priority 4: Attack LAST (ends turn!)
                elif attack_indices:
                    forced_action = np.random.choice(attack_indices)
                
                if forced_action is not None:
                    action = forced_action
                    # Update probs to reflect forced action
                    mcts_probs = np.zeros(len(ACTION_TABLE), dtype=np.float32)
                    mcts_probs[action] = 1.0
        else:
            if use_scripted_p1:
                # Scripted agent - no MCTS, just heuristic
                action = scripted_agent.select_action(env, obs, mask)
                # Create dummy uniform probs over legal actions for experience
                mcts_probs = np.zeros(len(ACTION_TABLE), dtype=np.float32)
                legal_actions = np.where(mask)[0]
                if len(legal_actions) > 0:
                    mcts_probs[legal_actions] = 1.0 / len(legal_actions)
            else:
                action, mcts_probs = mcts_p1.search(env, return_probs=True)
        
        # Store experience
        game_history.append((obs.copy(), mcts_probs, turn_player))
        
        # =========================================================
        # =========================================================
        # BALANCED REWARD SHAPING (scaled to reduce variance)
        # All rewards in 0.05-0.3 range for stable gradients
        # =========================================================
        action_reward = 0.0
        act = ACTION_TABLE[action]
        
        # 1. ANTI-STALL: Heavily penalize passing when we could attack
        if action == 0:
            can_attack = any(ACTION_TABLE[i].kind == 'ATTACK' for i in np.where(mask)[0])
            if can_attack:
                action_reward -= 0.5  # Strong penalty for passing when can attack
            elif np.sum(mask) > 1:
                action_reward -= 0.05  # Small penalty for unnecessary pass
        
        # 2. BOOSTED rewards for development actions
        if act.kind == 'PLAY_BASIC_TO_BENCH':
            action_reward += 0.05
        if 'EVOLVE' in act.kind:
            action_reward += 0.3  # BOOSTED: Evolution is critical for strong attackers
        if 'ATTACH' in act.kind:
            action_reward += 0.1
        if 'ABILITY' in act.kind:
            action_reward += 0.25  # BOOSTED: Abilities are game-winning (Pidgeot, Alakazam)
        
        # 3. Reward attacking (explicit incentive to attack)
        if act.kind == 'ATTACK':
            action_reward += 0.15
        
        # 4. STEP PENALTY: Bleed reward slightly to force FAST wins
        # Prevents deck-out stalling (105 turn games should be punished)
        action_reward -= 0.005
        
        action_rewards.append(action_reward)
        
        # === COMBO DISCOVERY: Track action chain ===
        act_str = f"{act.kind}:{act.a}" if act.a else act.kind
        action_chain.append(act_str)
        
        prizes_before = len(env._gs.players[turn_player].prizes)
        obs, reward, done, truncated, info = env.step(action)
        
        # 5. MASSIVE prize reward - prizes are PRIMARY OBJECTIVE
        # Taking 1 prize = +5.0 (increased from 2.0), taking all 6 = +30.0!
        # This makes attacking WAY more valuable than stalling
        prizes_after = len(env._gs.players[turn_player].prizes)
        if prizes_after < prizes_before:
            prizes_taken = prizes_before - prizes_after
            action_rewards[-1] += prizes_taken * 5.0  # BOOSTED from 2.0
            
            # === COMBO DISCOVERY: Record successful sequences ===
            # If we took a prize, the last few actions were a "good combo"
            chain_snapshot = list(action_chain)[-4:]  # Last 4 steps
            combo_str = " -> ".join(chain_snapshot)
            discovered_combos.append(combo_str)
            
            # Extra reward for completing discovered combos
            action_rewards[-1] += 1.0
        
        step_count += 1
        if truncated:
            done = True
    
    winner = env._gs.winner if done else -1
    
    # Calculate stats
    def count_evos(p):
        count = 0
        if p.active.name:
            cd = card_def(p.active.name)
            if cd.subtype in ("Stage1", "Stage2"): count += 1
        for s in p.bench:
            if s.name:
                cd = card_def(s.name)
                if cd.subtype in ("Stage1", "Stage2"): count += 1
        return count
    
    evos_p0 = count_evos(env._gs.players[0])
    evos_p1 = count_evos(env._gs.players[1])
    
    prizes_p0 = env._gs.players[0].prizes_taken
    prizes_p1 = env._gs.players[1].prizes_taken
    
    return {
        'game_id': game_id,
        'winner': winner,
        'steps': step_count,
        'p0_prizes': prizes_p0,
        'p1_prizes': prizes_p1,
        'p0_evos': evos_p0,
        'p1_evos': evos_p1,
        'history': game_history,
        'action_rewards': action_rewards,
        'p0_idx': p0_idx,
        'p1_idx': p1_idx,
        'discovered_combos': discovered_combos,  # Track successful action sequences
    }


# =============================================================================
# QUICK BEHAVIOR SUMMARY (What has the model learned?)
# =============================================================================

def quick_behavior_summary(model, device, num_games: int = 3):
    """
    Play a few quick games and summarize model behavior.
    Shows both RAW POLICY PRIORS and MCTS ACTION CHOICES.
    """
    from tcg.mcts import MCTS
    
    model.eval()
    mcts = MCTS(model, device, num_simulations=10)  # Fast MCTS
    
    # Track MCTS action choices
    action_counts = {'ATTACK': 0, 'PASS': 0, 'EVOLVE': 0, 'ABILITY': 0, 'ATTACH': 0, 'OTHER': 0}
    
    # Track raw policy priors (what network outputs BEFORE MCTS)
    prior_sums = {'ATTACK': 0.0, 'PASS': 0.0, 'EVOLVE': 0.0, 'ABILITY': 0.0}
    prior_count = 0
    
    total_prizes = 0
    total_turns = 0
    
    for _ in range(num_games):
        env = PTCGEnv(scripted_opponent=False, max_turns=30)
        env.reset()
        
        while not env._gs.done:
            if env._gs.turn_player == 0:  # Only count P0 actions
                # Get RAW policy priors from network (before MCTS boost)
                obs = featurize(env._gs)
                mask = env.action_mask()
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(obs_t)
                    if isinstance(output, tuple):
                        logits, _ = output
                    else:
                        logits = output
                    
                    mask_t = torch.from_numpy(mask).float().to(device)
                    masked_logits = torch.where(mask_t.unsqueeze(0) > 0, logits, torch.ones_like(logits) * -1e9)
                    raw_probs = torch.softmax(masked_logits, dim=1).cpu().numpy()[0]
                
                # Sum up raw priors by category
                for idx in np.where(mask > 0)[0]:
                    act = ACTION_TABLE[idx]
                    if act.kind == 'ATTACK':
                        prior_sums['ATTACK'] += raw_probs[idx]
                    elif act.kind == 'PASS':
                        prior_sums['PASS'] += raw_probs[idx]
                    elif 'EVOLVE' in act.kind:
                        prior_sums['EVOLVE'] += raw_probs[idx]
                    elif 'ABILITY' in act.kind:
                        prior_sums['ABILITY'] += raw_probs[idx]
                prior_count += 1
                
                # Now do MCTS search and track chosen action
                action = mcts.search(env)
                act = ACTION_TABLE[action]
                
                if act.kind == 'ATTACK':
                    action_counts['ATTACK'] += 1
                elif act.kind == 'PASS':
                    action_counts['PASS'] += 1
                elif 'EVOLVE' in act.kind:
                    action_counts['EVOLVE'] += 1
                elif 'ABILITY' in act.kind:
                    action_counts['ABILITY'] += 1
                elif 'ATTACH' in act.kind:
                    action_counts['ATTACH'] += 1
                else:
                    action_counts['OTHER'] += 1
            else:
                # Just do random for opponent
                mask = env.action_mask()
                valid = np.where(mask > 0)[0]
                action = np.random.choice(valid)
            
            env.step(action)
        
        total_prizes += env._gs.players[0].prizes_taken
        total_turns += env._gs.turn_number
    
    # Generate summary
    total_actions = sum(action_counts.values()) or 1
    
    # Raw policy priors (what network learned)
    prior_count = max(prior_count, 1)
    avg_atk_prior = prior_sums['ATTACK'] / prior_count
    avg_evo_prior = prior_sums['EVOLVE'] / prior_count
    avg_abl_prior = prior_sums['ABILITY'] / prior_count
    avg_pass_prior = prior_sums['PASS'] / prior_count
    
    summary = f"📊 Policy Priors: ATK={avg_atk_prior:.0%} EVO={avg_evo_prior:.0%} ABL={avg_abl_prior:.0%} PASS={avg_pass_prior:.0%}\n"
    summary += f"   MCTS Actions:  ATK {action_counts['ATTACK']}({100*action_counts['ATTACK']//total_actions}%) "
    summary += f"EVO {action_counts['EVOLVE']}({100*action_counts['EVOLVE']//total_actions}%) "
    summary += f"ABL {action_counts['ABILITY']}({100*action_counts['ABILITY']//total_actions}%) "
    summary += f"PASS {action_counts['PASS']}({100*action_counts['PASS']//total_actions}%) "
    summary += f"| Prizes: {total_prizes}/{num_games*6} | Turns: {total_turns//num_games}"
    
    return summary


# =============================================================================
# GAUNTLET EVALUATION (Absolute Skill Measurement)
# =============================================================================

def run_evaluation_gauntlet(model, device, episode_idx, deck=None):
    """
    Runs a fixed set of validation games against benchmarks to measure True Skill.
    
    Unlike Alpha-Rank/ELO (relative metrics), this measures ABSOLUTE performance
    against fixed, non-learning opponents to catch training bugs.
    """
    print(f"\n🛡️ --- EVALUATION GAUNTLET (Episode {episode_idx}) ---")
    model.eval()
    
    # Open log file for detailed debugging
    log_file = open(f"gauntlet_log_ep{episode_idx}.txt", "w")
    log_file.write(f"=== GAUNTLET LOG Episode {episode_idx} ===\n\n")
    
    # Default deck if not provided
    if deck is None:
        deck = [
            "Abra", "Abra", "Abra", "Abra",
            "Kadabra", "Kadabra", "Kadabra",
            "Alakazam", "Alakazam", "Alakazam",
            "Dunsparce", "Dunsparce", "Dunsparce", "Dunsparce",
            "Dudunsparce", "Dudunsparce", "Dudunsparce",
            "Fan Rotom", "Fan Rotom",
            "Fezandipiti ex",
            "Hilda", "Hilda", "Hilda", "Hilda",
            "Dawn", "Dawn", "Dawn", "Dawn",
            "Buddy-Buddy Poffin", "Buddy-Buddy Poffin", "Buddy-Buddy Poffin", "Buddy-Buddy Poffin",
            "Rare Candy", "Rare Candy", "Rare Candy", "Rare Candy",
            "Ultra Ball", "Ultra Ball", "Ultra Ball", "Ultra Ball",
            "Nest Ball", "Nest Ball",
            "Basic Psychic Energy", "Basic Psychic Energy", "Basic Psychic Energy", "Basic Psychic Energy",
            "Basic Psychic Energy", "Basic Psychic Energy", "Basic Psychic Energy", "Basic Psychic Energy",
            "Basic Psychic Energy", "Basic Psychic Energy", "Basic Psychic Energy", "Basic Psychic Energy",
            "Basic Psychic Energy", "Basic Psychic Energy", "Basic Psychic Energy",
            "Enriching Energy", "Enriching Energy", "Enriching Energy",
        ]
    
    # Benchmarks to test against
    benchmarks = [
        ("Random", ScriptedAgent("random")),          # Sanity Check: Must be >80%
        ("Aggressive", ScriptedAgent("aggressive")),  # Skill Check: Should rise over time
        ("Defensive", ScriptedAgent("defensive")),    # Strategy Check  
    ]
    
    results = {}
    
    for opp_name, opp_agent in benchmarks:
        wins = 0
        prizes_taken = 0
        turns_played = 0
        deck_outs = 0
        n_games = 20  # Small sample for speed
        
        # Action tracking
        attack_count = 0
        pass_count = 0
        evolve_count = 0
        attach_count = 0
        total_agent_actions = 0
        
        for i in range(n_games):
            # Swap sides to ensure fairness (P0 vs P1)
            agent_is_p0 = (i % 2 == 0)
            
            # Create Env (manual opponent handling)
            env = PTCGEnv(scripted_opponent=False, max_turns=30)
            obs, _ = env.reset(options={"decks": [deck, deck]})
            done = False
            
            while not done:
                turn = env._gs.turn_player
                mask = env.action_mask()
                
                # Determine if it's the Agent's turn
                is_agent_turn = (turn == 0 and agent_is_p0) or (turn == 1 and not agent_is_p0)
                
                if is_agent_turn:
                    # Use MCTS for action selection (shows true capability)
                    from tcg.mcts import MCTS
                    mcts = MCTS(model, device, num_simulations=15, c_puct=1.5)
                    action = mcts.search(env)
                    
                    # LOG: Detailed turn info to file (first 3 games, all turns)
                    if i < 3:
                        me = env._gs.players[turn]
                        active_name = me.active.name if me.active.name else "None"
                        energy_count = len(me.active.energy) if me.active.name else 0
                        hand_size = len(me.hand)
                        
                        # Count legal action types
                        legal_indices = np.where(mask)[0]
                        attacks_avail = [idx for idx in legal_indices if ACTION_TABLE[idx].kind == 'ATTACK']
                        attaches_avail = [idx for idx in legal_indices if 'ATTACH' in ACTION_TABLE[idx].kind]
                        evolves_avail = [idx for idx in legal_indices if 'EVOLVE' in ACTION_TABLE[idx].kind]
                        
                        log_file.write(f"Game {i+1} vs {opp_name} | Turn {env._gs.turn_number}\n")
                        log_file.write(f"  Active: {active_name}, Energy: {energy_count}, Hand: {hand_size}\n")
                        log_file.write(f"  Legal: {len(attacks_avail)} attacks, {len(attaches_avail)} attaches, {len(evolves_avail)} evolves\n")
                        log_file.write(f"  Chose: {ACTION_TABLE[action]}\n\n")
                    
                    # DEBUG: Print to console (first game only, first 5 turns)
                    if i == 0 and env._gs.turn_number <= 5:
                        attacks_available = [idx for idx in np.where(mask)[0] 
                                            if ACTION_TABLE[idx].kind == 'ATTACK']
                        me = env._gs.players[turn]
                        active_name = me.active.name if me.active.name else "None"
                        energy_count = len(me.active.energy) if me.active.name else 0
                        
                        if attacks_available:
                            print(f"      [DEBUG] Turn {env._gs.turn_number}: {len(attacks_available)} attacks available")
                            print(f"               Active: {active_name}, Energy: {energy_count}")
                            print(f"               Chose: {ACTION_TABLE[action].kind}")
                        elif env._gs.turn_number >= 2:  # Don't spam on turn 1
                            print(f"      [DEBUG] Turn {env._gs.turn_number}: NO attacks available")
                            print(f"               Active: {active_name}, Energy: {energy_count}")
                    
                    # Track action types
                    total_agent_actions += 1
                    act = ACTION_TABLE[action]
                    if act.kind == 'ATTACK':
                        attack_count += 1
                    elif act.kind == 'PASS':
                        pass_count += 1
                    elif 'EVOLVE' in act.kind:
                        evolve_count += 1
                    elif 'ATTACH' in act.kind:
                        attach_count += 1
                else:
                    # Scripted Opponent Move
                    action = opp_agent.select_action(env, obs, mask)
                
                obs, _, done, _, info = env.step(action)
            
            # Record Stats
            winner = env._gs.winner
            if (winner == 0 and agent_is_p0) or (winner == 1 and not agent_is_p0):
                wins += 1
            
            # Track Prizes (True objective) vs Deck Out (Stalling)
            p_idx = 0 if agent_is_p0 else 1
            prizes_taken += env._gs.players[p_idx].prizes_taken
            turns_played += env._gs.turn_number
            
            win_reason = getattr(env._gs, 'win_reason', '') or ''
            if "deck" in win_reason.lower():
                deck_outs += 1

        # Metrics
        win_rate = wins / n_games
        avg_prizes = prizes_taken / n_games
        avg_turns = turns_played / n_games
        
        # Action distribution
        if total_agent_actions > 0:
            attack_pct = attack_count / total_agent_actions * 100
            pass_pct = pass_count / total_agent_actions * 100
        else:
            attack_pct = pass_pct = 0
        
        # Visual Bar
        bar = "█" * int(win_rate * 10)
        print(f"   vs {opp_name:12s}: {bar:<10} {win_rate:>5.0%} | Prizes: {avg_prizes:.1f} | Attacks: {attack_pct:.0f}% | Pass: {pass_pct:.0f}% | DeckOuts: {deck_outs}")
        results[opp_name] = {
            'win_rate': win_rate,
            'avg_prizes': avg_prizes,
            'avg_turns': avg_turns,
            'deck_outs': deck_outs,
            'attack_pct': attack_pct,
            'pass_pct': pass_pct,
        }

    model.train()
    
    # Critical Alerts
    random_wr = results["Random"]["win_rate"]
    aggro_wr = results["Aggressive"]["win_rate"]
    
    if random_wr < 0.50:
        print("   ⚠️  CRITICAL: Agent is losing to Random. Training is broken (Suicide Bug?).")
    elif random_wr < 0.80:
        print("   ⚠️  WARNING: Agent should beat Random >80%. Check value predictions.")
    elif aggro_wr == 0.0 and random_wr > 0.9:
        print("   ⚠️  WARNING: Agent beats Random but fails Strategy check. Needs more skill.")
    elif random_wr >= 0.90 and aggro_wr >= 0.30:
        print("   ✅ HEALTHY: Agent is learning proper strategy!")
    
    log_file.write(f"\n=== SUMMARY ===\n")
    log_file.write(f"Random: {random_wr*100:.0f}%, Aggressive: {aggro_wr*100:.0f}%\n")
    log_file.close()
    print(f"   📄 Detailed log: gauntlet_log_ep{episode_idx}.txt")
    
    return results


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def run_advanced_training(
    episodes: int = 5000,
    mcts_sims: int = 50,
    batch_size: int = 256,
    population_size: int = 5,
    use_league: bool = True,
    use_pbt: bool = True,
    mirror_training: bool = False,
    lr: float = 1e-3,
    aux_weight: float = 0.1,
    verbose: bool = False,
    save_every: int = 100,
    # Genetic Algorithm parameters (STABILIZED defaults)
    mutation_rate: float = 0.05,       # Reduced from 0.1 - less random exploration
    mutation_strength: float = 0.005,  # Reduced from 0.02 - smaller weight changes
    crossover_rate: float = 0.2,       # Reduced from 0.3 - less disruptive breeding
    elitism_count: int = 4,            # Increased from 2 - preserve more top performers
    # Parallelization
    num_workers: int = 4,
    # Resume from checkpoint
    resume_checkpoint: str = None,
    # Scripted opponents for external pressure
    scripted_opponent_ratio: float = 0.2,  # 20% of games vs scripted opponents
    # Curriculum Learning
    curriculum_episodes: int = 0,  # Episodes of curriculum (attack-forced) before normal training
):
    """
    Advanced training with all improvements.
    """
    if not verbose:
        os.environ['PTCG_QUIET'] = '1'
    
    print("=" * 70)
    print("ADVANCED ALPHAZERO TRAINING")
    print("=" * 70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    obs_dim = 1584  # Updated V3: 5 (glob) + 100 (hand_bow) + 1 (op_hand) + 1452 (12 slots × 121) + 8 (opp_model) + 18 (discard)
    n_actions = len(ACTION_TABLE)
    
    # Initialize population or single model
    if use_pbt:
        print(f"Genetic Algorithm Population: {population_size} agents")
        print(f"  Mutation Rate: {mutation_rate}, Strength: {mutation_strength}")
        print(f"  Crossover Rate: {crossover_rate}, Elitism: {elitism_count}")
        population = GeneticPopulation(
            population_size, obs_dim, n_actions, device,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            crossover_rate=crossover_rate,
            elitism_count=elitism_count
        )
        models = population.models
        optimizers = population.optimizers
        
        # Resume from checkpoint - load into ALL population members
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"\n🔄 Resuming from checkpoint: {resume_checkpoint}")
            try:
                ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
                base_state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
                
                # Load base model into first few population members (elites)
                for i in range(min(elitism_count, population_size)):
                    population.models[i].load_state_dict(base_state)
                    print(f"   ✓ Loaded checkpoint into agent {i}")
                
                # Slightly mutate the rest to create diversity
                for i in range(elitism_count, population_size):
                    population.models[i].load_state_dict(base_state)
                    # Add small random noise
                    with torch.no_grad():
                        for param in population.models[i].parameters():
                            noise = torch.randn_like(param) * 0.001
                            param.add_(noise)
                    print(f"   ✓ Loaded + mutated checkpoint into agent {i}")
                
                # Restore ELO if available
                if 'elo' in ckpt:
                    current_elo = float(ckpt['elo'])
                    best_elo = current_elo
                    print(f"   ✓ Restored ELO: {current_elo}")
                
                print(f"   ✓ Resume complete! Population initialized from {resume_checkpoint}")
            except Exception as e:
                print(f"   ⚠️ Failed to load checkpoint: {e}")
                print(f"   Continuing with random initialization...")
    else:
        population = None  # For consistency
        model = AdvancedPolicyValueNet(obs_dim, n_actions).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        models = [model]
        optimizers = [optimizer]
        
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"\n🔄 Resuming from checkpoint: {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            base_state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
            model.load_state_dict(base_state)
            print(f"   ✓ Resume complete!")
    
    # League
    league = League(max_agents=20) if use_league else None
    # Initialize ELO (may have been set during resume above)
    try:
        current_elo = current_elo  # Use value set during resume
    except NameError:
        current_elo = 1200.0
    try:
        best_elo = best_elo  # Use value set during resume
    except NameError:
        best_elo = 1200.0
    best_fitness = 0.0  # Track highest fitness for checkpoint saving
    
    # Replay buffer
    replay_buffer = PrioritizedReplayBuffer(capacity=300000)
    
    # Alpha-Rank match tracking
    match_tracker = PopulationTracker(max_history=50000)
    
    env = PTCGEnv(scripted_opponent=False, max_turns=30)
    
    # Decks
    alakazam_deck = []
    alakazam_deck.extend(["Abra"] * 4)
    alakazam_deck.extend(["Kadabra"] * 3)
    alakazam_deck.extend(["Alakazam"] * 4)
    alakazam_deck.extend(["Dunsparce"] * 4)
    alakazam_deck.extend(["Dudunsparce"] * 4)
    alakazam_deck.extend(["Fan Rotom"] * 2)
    alakazam_deck.extend(["Psyduck"] * 1)
    alakazam_deck.extend(["Fezandipiti ex"] * 1)
    alakazam_deck.extend(["Hilda"] * 4)
    alakazam_deck.extend(["Dawn"] * 4)
    alakazam_deck.extend(["Boss's Orders"] * 3)
    alakazam_deck.extend(["Lillie's Determination"] * 2)
    alakazam_deck.extend(["Tulip"] * 1)
    alakazam_deck.extend(["Buddy-Buddy Poffin"] * 4)
    alakazam_deck.extend(["Rare Candy"] * 3)
    alakazam_deck.extend(["Nest Ball"] * 2)
    alakazam_deck.extend(["Night Stretcher"] * 2)
    alakazam_deck.extend(["Wondrous Patch"] * 2)
    alakazam_deck.extend(["Enhanced Hammer"] * 2)
    alakazam_deck.extend(["Battle Cage"] * 3)
    alakazam_deck.extend(["Basic Psychic Energy"] * 3)
    alakazam_deck.extend(["Enriching Energy"] * 1)
    alakazam_deck.extend(["Jet Energy"] * 1)

    charizard_deck = []
    charizard_deck.extend(["Charmander"] * 3)
    charizard_deck.extend(["Charmeleon"] * 2)
    charizard_deck.extend(["Charizard ex"] * 2)
    charizard_deck.extend(["Pidgey"] * 2)
    charizard_deck.extend(["Pidgeotto"] * 2)
    charizard_deck.extend(["Pidgeot ex"] * 2)
    charizard_deck.extend(["Psyduck"] * 1)
    charizard_deck.extend(["Shaymin"] * 1)
    charizard_deck.extend(["Tatsugiri"] * 1)
    charizard_deck.extend(["Munkidori"] * 1)
    charizard_deck.extend(["Chi-Yu"] * 1)
    charizard_deck.extend(["Gouging Fire ex"] * 1)
    charizard_deck.extend(["Fezandipiti ex"] * 1)
    charizard_deck.extend(["Lillie's Determination"] * 4)
    charizard_deck.extend(["Arven"] * 4)
    charizard_deck.extend(["Boss's Orders"] * 3)
    charizard_deck.extend(["Iono"] * 2)
    charizard_deck.extend(["Professor Turo's Scenario"] * 1)
    charizard_deck.extend(["Buddy-Buddy Poffin"] * 4)
    charizard_deck.extend(["Ultra Ball"] * 3)
    charizard_deck.extend(["Rare Candy"] * 2)
    charizard_deck.extend(["Super Rod"] * 2)
    charizard_deck.extend(["Counter Catcher"] * 1)
    charizard_deck.extend(["Energy Search"] * 1)
    charizard_deck.extend(["Unfair Stamp"] * 1)
    charizard_deck.extend(["Technical Machine: Evolution"] * 2)
    charizard_deck.extend(["Artazon"] * 1)
    charizard_deck.extend(["Fire Energy"] * 5)
    charizard_deck.extend(["Mist Energy"] * 2)
    charizard_deck.extend(["Darkness Energy"] * 1)
    charizard_deck.extend(["Jet Energy"] * 1)

    gholdengo_deck = []
    gholdengo_deck.extend(["Gimmighoul"] * 4)
    gholdengo_deck.extend(["Gholdengo ex"] * 3)
    gholdengo_deck.extend(["Solrock"] * 3)
    gholdengo_deck.extend(["Lunatone"] * 2)
    gholdengo_deck.extend(["Fezandipiti ex"] * 1)
    gholdengo_deck.extend(["Genesect ex"] * 1)
    gholdengo_deck.extend(["Hop's Cramorant"] * 1)
    gholdengo_deck.extend(["Arven"] * 4)
    gholdengo_deck.extend(["Boss's Orders"] * 3)
    gholdengo_deck.extend(["Professor Turo's Scenario"] * 2)
    gholdengo_deck.extend(["Lana's Aid"] * 1)
    gholdengo_deck.extend(["Superior Energy Retrieval"] * 4)
    gholdengo_deck.extend(["Fighting Gong"] * 4)
    gholdengo_deck.extend(["Nest Ball"] * 4)
    gholdengo_deck.extend(["Earthen Vessel"] * 3)
    gholdengo_deck.extend(["Buddy-Buddy Poffin"] * 1)
    gholdengo_deck.extend(["Super Rod"] * 1)
    gholdengo_deck.extend(["Premium Power Pro"] * 1)
    gholdengo_deck.extend(["Prime Catcher"] * 1)
    gholdengo_deck.extend(["Air Balloon"] * 2)
    gholdengo_deck.extend(["Vitality Band"] * 1)
    gholdengo_deck.extend(["Artazon"] * 2)
    gholdengo_deck.extend(["Fighting Energy"] * 8)
    gholdengo_deck.extend(["Metal Energy"] * 3)
    
    print(f"\nConfiguration:")
    print(f"  Episodes: {episodes}")
    print(f"  MCTS Sims: {mcts_sims}")
    print(f"  PBT: {use_pbt} ({population_size} agents)")
    print(f"  League: {use_league}")
    print(f"  Mirror Training: {mirror_training}")
    print(f"  Aux Weight: {aux_weight}")
    print(f"  Scripted Opponent Ratio: {scripted_opponent_ratio * 100:.0f}%")
    if curriculum_episodes > 0:
        print(f"  📚 Curriculum Learning: {curriculum_episodes} episodes (force attacks)")
    print()
    
    # Metrics
    import csv
    metrics_file = open("advanced_training_metrics.csv", "w", newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow([
        "episode", "win_rate", "avg_length", "policy_loss", "value_loss", 
        "aux_loss", "total_loss", "elo", "buffer_size", "avg_prizes", "avg_evolutions",
        "checkpoint_winrate"
    ])
    
    recent_wins = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_prizes = deque(maxlen=100)
    recent_evolutions = deque(maxlen=100)
    recent_value_errors = deque(maxlen=50)  # Track value prediction accuracy
    recent_sign_accs = deque(maxlen=50)
    last_checkpoint_winrate = 0.5  # Track most recent checkpoint vs checkpoint win rate
    
    # Parallelization Setup
    if num_workers > 1:
        try:
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(num_workers)
            print(f"🚀 Started multiprocessing pool with {num_workers} workers")
        except Exception as e:
            print(f"⚠️ Failed to start multiprocessing pool: {e}")
            print("Falling back to single-threaded execution")
            num_workers = 1
            pool = None
    else:
        pool = None

    pbar = tqdm(total=episodes, desc="Advanced Training", ncols=120)
    
    total_games = 0
    match_state = {}  # Track multi-game matches between agents
    
    while total_games < episodes:
        # Determine batch size for this iteration
        batch_size_games = num_workers if pool else 1
        remaining = episodes - total_games
        batch_size_games = min(batch_size_games, remaining)
        
        # Prepare batch arguments
        batch_args = []
        
        # We need to store some metadata to handle league updates and match resolution after games
        # But for now, let's just use the returned data from workers
        
        for i in range(batch_size_games):
            current_game_id = total_games + i + 1
            
            # Select model(s)
            if use_pbt:
                p0_idx, p1_idx = population.get_random_pair()
                model_p0 = models[p0_idx]
                model_p1 = models[p1_idx]
            else:
                p0_idx, p1_idx = 0, 0
                model_p0 = models[0]
                model_p1 = models[0]
            
            # Decide opponent type: scripted, league, or neural net
            scripted_strategy = None
            league_opponent_path = None
            model_p1_state = None
            
            # Priority: Scripted > League > Neural Net (population)
            opponent_roll = random.random()
            
            if opponent_roll < scripted_opponent_ratio:
                # Use scripted opponent for external pressure
                scripted_strategies = ["aggressive", "evolution_rush", "defensive", "energy_first"]
                scripted_strategy = random.choice(scripted_strategies)
                model_p1_state = model_p1.state_dict()  # Dummy, won't be used
                p1_idx = -2  # Special marker for scripted opponent
                
            elif opponent_roll < scripted_opponent_ratio + 0.3 and use_league and league:
                # Use league opponent
                league_opponent_path = league.sample_opponent(current_elo)
                if league_opponent_path:
                    try:
                        league_state = torch.load(league_opponent_path, map_location='cpu')
                        # Handle potential nesting of state_dict (legacy vs new)
                        if 'state_dict' in league_state:
                            model_p1_state = league_state['state_dict']
                        else:
                            model_p1_state = league_state
                        # For league games, p1_idx is -1
                        p1_idx = -1 
                    except:
                        model_p1_state = model_p1.state_dict()
            
            if model_p1_state is None:
                model_p1_state = model_p1.state_dict()
                
            # Temperature annealing
            temperature = max(0.1, 1.0 - current_game_id / (episodes * 0.5))
            
            # Select decks
            all_decks = [alakazam_deck, charizard_deck, gholdengo_deck]
            deck_p0 = random.choice(all_decks)
            deck_p1 = random.choice(all_decks)
            
            # Curriculum learning: 50% of games force attacks during curriculum phase
            # Mixed approach: some games learn "attacks good", others learn "setup first"
            in_curriculum = (current_game_id <= curriculum_episodes) and (random.random() < 0.5)
            
            args = (
                model_p0.state_dict(),
                model_p1_state,
                obs_dim,
                n_actions,
                deck_p0,
                deck_p1,
                mcts_sims,
                temperature,
                current_game_id,
                p0_idx, 
                p1_idx,
                scripted_strategy,  # None for neural net, strategy name for scripted
                in_curriculum,  # Curriculum mode flag
            )
            batch_args.append(args)

        # Execute Batch - use imap_unordered for better throughput
        # This processes results as games complete, rather than waiting for all
        if pool:
            results = list(pool.imap_unordered(play_single_game_worker, batch_args))
        else:
            results = [play_single_game_worker(batch_args[0])]
            
        # Process Results (note: order may differ from batch_args with imap_unordered)
        for res in results:
            game_id = res['game_id']
            winner = res['winner']
            steps = res['steps']
            p0_idx = res['p0_idx']
            p1_idx = res['p1_idx']
            # Reconstruct league check (imperfect but functional) is p1_idx == -1
            is_league = (p1_idx == -1)
            
            # === LOG DISCOVERED COMBOS ===
            if 'discovered_combos' in res and res['discovered_combos']:
                with open("agent_combos.txt", "a") as f:
                    for combo in res['discovered_combos']:
                        f.write(f"Ep {total_games}: {combo}\n")
            
            # Update history buffers
            recent_lengths.append(steps)
            
            # Win/Loss & Match Logic
            if winner == 0:
                recent_wins.append(1)
                
                # Update PBT Match State
                if not is_league and use_pbt:
                    match_key = tuple(sorted((p0_idx, p1_idx)))
                    # match_state is initialized before the loop
                    if match_key not in match_state: match_state[match_key] = [0, 0]
                    
                    if p0_idx < p1_idx: match_state[match_key][0] += 1
                    else: match_state[match_key][1] += 1
                    
                    cur_p0, cur_p1 = match_state[match_key]
                    if cur_p0 >= 2 or cur_p1 >= 2:
                        w_idx = p0_idx if ((p0_idx < p1_idx and cur_p0>=2) or (p0_idx > p1_idx and cur_p1>=2)) else p1_idx
                        l_idx = p1_idx if w_idx == p0_idx else p0_idx
                        population.update_scores(w_idx, l_idx)
                        del match_state[match_key]
                
                # League Update
                if is_league and use_league and league:
                    # Which file was it? We didn't pass path back. 
                    # Approximation: League always updates on next batch? 
                    # Actually, since we can't easily map back the exact filename without passing it through,
                    # we might skip updating specific opponent ELO or accept a small limitation.
                    # Or we can put path in args/result.
                    # For now, just update global ELO.
                    current_elo += 32 * (1 - 0.5)

            elif winner == 1:
                recent_wins.append(0)
                if not is_league and use_pbt:
                    match_key = tuple(sorted((p0_idx, p1_idx)))
                    # match_state is initialized before the loop
                    if match_key not in match_state: match_state[match_key] = [0, 0]
                    
                    if p0_idx < p1_idx: match_state[match_key][1] += 1
                    else: match_state[match_key][0] += 1
                    
                    cur_p0, cur_p1 = match_state[match_key]
                    if cur_p0 >= 2 or cur_p1 >= 2:
                        w_idx = p0_idx if ((p0_idx < p1_idx and cur_p0>=2) or (p0_idx > p1_idx and cur_p1>=2)) else p1_idx
                        l_idx = p1_idx if w_idx == p0_idx else p0_idx
                        population.update_scores(w_idx, l_idx)
                        del match_state[match_key]
                
                if is_league:
                    current_elo += 32 * (0 - 0.5)
            else:
                recent_wins.append(0.5)
            
            # Tracks
            recent_prizes.append((res['p0_prizes'] + res['p1_prizes']) / 2)
            recent_evolutions.append((res['p0_evos'] + res['p1_evos']) / 2)
            
            # Add to Buffer
            replay_buffer.add_game_with_shaping(res['history'], winner, res['action_rewards'])
            
            # Alpha-Rank
            if not is_league:
                a1 = f"agent_{p0_idx}"
                a2 = f"agent_{p1_idx}"
                w_name = a1 if winner == 0 else (a2 if winner == 1 else None)
                gen = population.generation if use_pbt and population else 0
                match_tracker.record_match(a1, a2, w_name, gen1=gen, gen2=gen)

        # Training Step (run once per batch, or scaled?)
        # To maintain ratio, we should run training multiple times if batch size is large
        # Original: 8 steps per game.
        # New: 8 * batch_size steps?
        training_steps = 8 * batch_size_games
        
        policy_loss_val = 0.0
        value_loss_val = 0.0
        aux_loss_val = 0.0
        total_loss_val = 0.0
        
        if len(replay_buffer) >= 512:
            model_updates = 0
            # Limit training time per iteration so we don't stall too long
            # If batch=4, steps=32. That's fine.
            for _ in range(training_steps):
                for model, optimizer in zip(models, optimizers):
                    model.train()
                    
                    batch, weights, indices = replay_buffer.sample(batch_size)
                    if not batch: continue
                    
                    obs_batch = torch.from_numpy(np.stack([e.obs for e in batch])).float().to(device)
                    policy_target = torch.from_numpy(np.stack([e.mcts_probs for e in batch])).float().to(device)
                    value_target = torch.tensor([e.value for e in batch]).float().to(device)
                    weights_t = torch.from_numpy(weights).float().to(device)
                    
                    policy_logits, value_pred, _, _ = model(obs_batch, return_aux=True)
                    
                    log_probs = F.log_softmax(policy_logits, dim=1)
                    policy_loss = -(policy_target * log_probs).sum(dim=1)
                    policy_loss = (policy_loss * weights_t).mean()
                    
                    value_loss = F.mse_loss(value_pred, value_target, reduction='none')
                    value_loss = (value_loss * weights_t).mean()
                    
                    aux_loss = torch.tensor(0.0, device=device)
                    total_loss = policy_loss + value_loss + aux_weight * aux_loss
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    with torch.no_grad():
                        td_errors = (value_pred - value_target).abs().cpu().numpy()
                        
                        # === VALUE DELUSION MONITOR ===
                        # Track if we're predicting the right winner (signs match)
                        value_error = (value_pred - value_target).abs().mean().item()
                        sign_accuracy = ((value_pred * value_target) > 0).float().mean().item()
                        
                        # Accumulate for averaging (don't spam terminal)
                        recent_value_errors.append(value_error)
                        recent_sign_accs.append(sign_accuracy)
                    
                    replay_buffer.update_priorities(indices, td_errors)
                    
                    policy_loss_val = policy_loss.item()
                    value_loss_val = value_loss.item()
                    aux_loss_val = aux_loss.item()
                    total_loss_val = total_loss.item()
                    model_updates += 1

        # ADAPTIVE PBT: Evolve when population diversity is low
        # Check every 100 games, but only evolve if diversity is LOW
        if use_pbt and (total_games // 100) > ((total_games - batch_size_games) // 100):
            diversity = population.compute_population_diversity()
            if population.needs_evolution(diversity_threshold=100.0):
                print(f"   🧬 Low diversity ({diversity:.1f}) - triggering genetic evolution")
                population.evolve_generation(verbose=verbose)

        if use_league and league and (total_games // 200) > ((total_games - batch_size_games) // 200):
            best_model = population.get_best_model() if use_pbt else models[0]
            league.add_agent(best_model, total_games, current_elo)

        # Update Progress
        total_games += batch_size_games
        pbar.update(batch_size_games)
        
        # Logging
        win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
        avg_len = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0
        gen_info = f"G{population.generation}" if use_pbt and population else ""
        
        # Compute averages for display
        avg_val_err = np.mean(recent_value_errors) if recent_value_errors else 0
        avg_sign_acc = np.mean(recent_sign_accs) if recent_sign_accs else 0
        diversity = population.compute_population_diversity() if use_pbt and population else 0
        
        pbar.set_postfix({
            'WR': f'{win_rate:.0%}',
            'Len': f'{avg_len:.0f}',
            'Loss': f'{total_loss_val:.3f}',
            'ELO': f'{current_elo:.0f}',
            'VErr': f'{avg_val_err:.2f}',
            'Div': f'{diversity:.0f}',
            'Gen': gen_info,
        })
        
        # CSV Metrics (record every ~10 games)
        if total_games % 10 < batch_size_games:
             # Calculate averages
            avg_prz = np.mean(recent_prizes) if recent_prizes else 0.0
            avg_evo = np.mean(recent_evolutions) if recent_evolutions else 0.0
             
            metrics_writer.writerow([
                total_games, win_rate, avg_len, policy_loss_val, value_loss_val,
                aux_loss_val, total_loss_val, current_elo, len(replay_buffer),
                avg_prz, avg_evo, last_checkpoint_winrate
            ])
            metrics_file.flush()

        # Checkpoints
        if total_games % save_every < batch_size_games:
             # Logic for periodic checkpoint...
             # (Simplify: just save regularly)
             os.makedirs("checkpoints", exist_ok=True)
             ckpt_path = f"checkpoints/checkpoint_ep{total_games}.pt"
             # Checkpoint Evaluation
             if total_games >= save_every:
                 prev_checkpoint = total_games - save_every
                 # Adjust for batch size overruns
                 prev_checkpoint = (prev_checkpoint // save_every) * save_every
                 prev_path = f"checkpoints/checkpoint_ep{prev_checkpoint}.pt"
                 
                 if os.path.exists(prev_path):
                     print(f"\n⚔️ Evaluation vs {prev_path}...")
                     best_model = population.get_best_model() if use_pbt and population else models[0]
                     wr, w, l = evaluate_vs_checkpoint(best_model, prev_path, device, num_games=10, 
                                                       obs_dim=obs_dim, n_actions=n_actions)
                     last_checkpoint_winrate = wr
                     print(f"   Result: {w}-{l} (WR: {wr:.0%})")
             
             ckpt_path = f"checkpoints/checkpoint_ep{total_games}.pt"
             # ... save logic similar to before ...
             best_model = population.get_best_model() if use_pbt and population else models[0]
             torch.save({
                 "state_dict": best_model.state_dict(),
                 "episode": total_games,
                 "elo": current_elo,
                 "checkpoint_winrate": last_checkpoint_winrate,
                 "n_actions": n_actions
             }, ckpt_path)
             
             # Also save "latest"
             torch.save({
                 "state_dict": best_model.state_dict(), 
                 "elo": current_elo,
                 "n_actions": n_actions
             }, "advanced_policy.pt")
             
             # Show what the model has learned (async to not block training)
             import threading
             def run_behavior_async():
                 try:
                     behavior = quick_behavior_summary(best_model, device, num_games=3)
                     print(f"   {behavior}")
                 except Exception as e:
                     print(f"   ⚠️ Behavior summary failed: {e}")
             
             behavior_thread = threading.Thread(target=run_behavior_async, daemon=True)
             behavior_thread.start()
             
             # === GAUNTLET EVALUATION (ASYNC) ===
             # Run absolute skill benchmark every 500 episodes in background
             if total_games % 500 < batch_size_games:
                 best_model = population.get_best_model() if use_pbt and population else models[0]
                 
                 # Spawn gauntlet as background process so training continues
                 import threading
                 def run_gauntlet_async():
                     gauntlet_results = run_evaluation_gauntlet(best_model, device, total_games, deck_p0)
                     random_wr = gauntlet_results.get("Random", {}).get("win_rate", 0)
                     if random_wr < 0.50 and total_games >= 500:
                         print("   🚨 CRITICAL: Training appears broken - agent losing to Random!")
                 
                 gauntlet_thread = threading.Thread(target=run_gauntlet_async, daemon=True)
                 gauntlet_thread.start()
                 # Don't wait - training continues immediately!

        # Best ELO Check
        if current_elo > best_elo:
            best_elo = current_elo
            best_model = population.get_best_model() if use_pbt and population else models[0]
            torch.save({"state_dict": best_model.state_dict(), "elo": best_elo}, "best_elo_policy.pt")

    if pool:
        pool.close()
        pool.join()

    metrics_file.close()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🧬 PARALLEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Statistics:")
    print(f"   Total Games: {total_games}")
    print(f"   Final ELO: {current_elo:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced AlphaZero Training with Genetic Algorithm')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--mcts_sims', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--population', type=int, default=5, help='Population size for genetic algorithm')
    parser.add_argument('--no_league', action='store_true')
    parser.add_argument('--no_pbt', action='store_true', help='Disable genetic algorithm / PBT')
    parser.add_argument('--mirror', action='store_true', help='Mirror training (same deck vs same deck)')
    parser.add_argument('--verbose', action='store_true')
    
    # Genetic Algorithm parameters (STABILIZED defaults)
    parser.add_argument('--mutation_rate', type=float, default=0.05, 
                        help='Probability of mutating each weight (0-1)')
    parser.add_argument('--mutation_strength', type=float, default=0.005,
                        help='Standard deviation of mutation noise')
    parser.add_argument('--crossover_rate', type=float, default=0.2,
                        help='Probability of crossover vs pure mutation (0-1)')
    parser.add_argument('--elitism', type=int, default=4,
                        help='Number of top performers to preserve unchanged')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel game workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file (e.g., checkpoints/checkpoint_ep100.pt)')
    parser.add_argument('--scripted_ratio', type=float, default=0.2,
                        help='Ratio of games to play against scripted opponents (0-1, default 0.2)')
    parser.add_argument('--curriculum', type=int, default=0,
                        help='Episodes of curriculum learning (force attacks when available)')
    parser.add_argument('--ppo', action='store_true',
                        help='Use PPO instead of MCTS (experimental - learns directly from rewards)')
    
    args = parser.parse_args()
    
    if args.ppo:
        print("=" * 60)
        print("🧪 PPO MODE (Experimental)")
        print("=" * 60)
        print("PPO mode trains directly from rewards without MCTS.")
        print("This is useful when MCTS discovers degenerate strategies.")
        print("\nNote: PPO implementation coming soon.")
        print("For now, use --curriculum 1000 for aggressive curriculum learning.")
        print("=" * 60)
        # For now, just increase curriculum as a workaround
        args.curriculum = max(args.curriculum, args.episodes // 2)
    
    run_advanced_training(
        episodes=args.episodes,
        mcts_sims=args.mcts_sims,
        batch_size=args.batch_size,
        population_size=args.population,
        use_league=not args.no_league,
        use_pbt=not args.no_pbt,
        mirror_training=args.mirror,
        verbose=args.verbose,
        # Genetic Algorithm parameters
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        crossover_rate=args.crossover_rate,
        elitism_count=args.elitism,
        # Parallelization
        num_workers=args.num_workers,
        # Resume
        resume_checkpoint=args.resume,
        # Scripted opponents
        scripted_opponent_ratio=args.scripted_ratio,
        # Curriculum learning
        curriculum_episodes=args.curriculum,
    )
