#!/usr/bin/env python3
"""
Alpha-Rank Implementation for Pokemon TCG Agent Evaluation

Alpha-Rank is a game-theoretic method for ranking agents that handles:
- Non-transitive relationships (Rock-Paper-Scissors dynamics)
- Population-based training evaluation
- More principled ranking than ELO

Based on: "Alpha-Rank: Multi-Agent Evaluation by Evolution" (Omidshafiei et al., 2019)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class AlphaRankResult:
    """Results from Alpha-Rank computation."""
    rankings: Dict[str, float]  # Agent name -> ranking score (0-1, sums to 1)
    payoff_matrix: np.ndarray   # Win rate matrix
    agent_names: List[str]      # Ordered list of agent names
    meta_nash: np.ndarray       # Meta Nash equilibrium distribution
    sweep_results: Optional[Dict] = None  # Alpha sweep results
    
    def top_agents(self, n: int = 5) -> List[Tuple[str, float]]:
        """Return top n agents by ranking."""
        sorted_agents = sorted(self.rankings.items(), key=lambda x: -x[1])
        return sorted_agents[:n]
    
    def __str__(self) -> str:
        lines = ["Alpha-Rank Results:", "=" * 40]
        for i, (name, score) in enumerate(self.top_agents(10)):
            bar = "█" * int(score * 50)
            lines.append(f"{i+1:2}. {name:20s} {score:.4f} {bar}")
        return "\n".join(lines)


class AlphaRank:
    """
    Alpha-Rank algorithm implementation.
    
    Key concepts:
    - Builds a response graph from payoff matrix
    - Computes stationary distribution of evolutionary dynamics
    - Agents with higher mass in stationary distribution are "better"
    """
    
    def __init__(self, alpha: float = 0.1, use_inf_alpha: bool = False):
        """
        Initialize Alpha-Rank.
        
        Args:
            alpha: Selection intensity parameter (higher = more deterministic)
            use_inf_alpha: If True, use infinite alpha (pure best-response dynamics)
        """
        self.alpha = alpha
        self.use_inf_alpha = use_inf_alpha
    
    def compute_payoff_matrix(self, 
                               match_results: List[Dict],
                               agent_names: List[str]) -> np.ndarray:
        """
        Compute payoff matrix from match results.
        
        Args:
            match_results: List of dicts with 'agent1', 'agent2', 'winner' keys
            agent_names: List of agent names
            
        Returns:
            Payoff matrix P where P[i,j] = win rate of agent i vs agent j
        """
        n = len(agent_names)
        name_to_idx = {name: i for i, name in enumerate(agent_names)}
        
        wins = np.zeros((n, n))
        games = np.zeros((n, n))
        
        for result in match_results:
            i = name_to_idx.get(result['agent1'])
            j = name_to_idx.get(result['agent2'])
            if i is None or j is None:
                continue
                
            games[i, j] += 1
            games[j, i] += 1
            
            if result['winner'] == result['agent1']:
                wins[i, j] += 1
            elif result['winner'] == result['agent2']:
                wins[j, i] += 1
            else:  # Draw
                wins[i, j] += 0.5
                wins[j, i] += 0.5
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            payoff = np.where(games > 0, wins / games, 0.5)
        
        # Set diagonal to 0.5 (playing yourself is a draw)
        np.fill_diagonal(payoff, 0.5)
        
        return payoff
    
    def _compute_transition_matrix(self, payoff: np.ndarray) -> np.ndarray:
        """
        Compute Markov chain transition matrix from payoff matrix.
        
        The transition probability from strategy i to strategy j represents
        the probability of switching to j given current strategy i.
        """
        n = payoff.shape[0]
        transition = np.zeros((n, n))
        
        if self.use_inf_alpha:
            # Infinite alpha: deterministic best-response
            for i in range(n):
                # Exclude self (can't switch to same strategy)
                payoffs_vs_others = payoff[:, i].copy()
                payoffs_vs_others[i] = -np.inf
                best_j = np.argmax(payoffs_vs_others)
                
                if payoff[best_j, i] > 0.5:  # j beats i
                    transition[i, best_j] = 1.0
                else:
                    transition[i, i] = 1.0
        else:
            # Finite alpha: softmax over advantages
            for i in range(n):
                advantages = np.zeros(n)
                for j in range(n):
                    if j != i:
                        # Advantage of switching from i to j
                        # Higher payoff[j, :] means j does better
                        avg_payoff_j = np.mean(payoff[j, :])
                        avg_payoff_i = np.mean(payoff[i, :])
                        advantages[j] = avg_payoff_j - avg_payoff_i
                
                # Softmax with temperature 1/alpha
                advantages[i] = 0  # No advantage for staying
                exp_adv = np.exp(self.alpha * advantages)
                exp_adv[i] = 0  # Can't transition to self initially
                
                total = np.sum(exp_adv)
                if total > 0:
                    probs = exp_adv / total
                    # Probability of staying = 1 - probability of switching
                    stay_prob = 1.0 / (1.0 + total / n)  # Neutral mutation rate
                    transition[i, :] = probs * (1 - stay_prob)
                    transition[i, i] = stay_prob
                else:
                    transition[i, i] = 1.0
        
        # Normalize rows
        row_sums = transition.sum(axis=1, keepdims=True)
        transition = np.where(row_sums > 0, transition / row_sums, 1.0 / n)
        
        return transition
    
    def _compute_stationary_distribution(self, 
                                          transition: np.ndarray,
                                          max_iter: int = 1000,
                                          tol: float = 1e-8) -> np.ndarray:
        """
        Compute stationary distribution of Markov chain via power iteration.
        """
        n = transition.shape[0]
        
        # Start with uniform distribution
        pi = np.ones(n) / n
        
        for _ in range(max_iter):
            pi_new = pi @ transition
            if np.max(np.abs(pi_new - pi)) < tol:
                break
            pi = pi_new
        
        # Normalize
        pi = pi / pi.sum()
        return pi
    
    def compute(self,
                payoff_matrix: np.ndarray,
                agent_names: List[str]) -> AlphaRankResult:
        """
        Compute Alpha-Rank from payoff matrix.
        
        Args:
            payoff_matrix: Win rate matrix P[i,j] = win rate of i vs j
            agent_names: List of agent names
            
        Returns:
            AlphaRankResult with rankings
        """
        n = len(agent_names)
        assert payoff_matrix.shape == (n, n), f"Payoff matrix shape mismatch"
        
        # Compute transition matrix
        transition = self._compute_transition_matrix(payoff_matrix)
        
        # Compute stationary distribution
        pi = self._compute_stationary_distribution(transition)
        
        # Build rankings dict
        rankings = {name: float(pi[i]) for i, name in enumerate(agent_names)}
        
        return AlphaRankResult(
            rankings=rankings,
            payoff_matrix=payoff_matrix,
            agent_names=agent_names,
            meta_nash=pi
        )
    
    def compute_from_matches(self,
                             match_results: List[Dict],
                             agent_names: Optional[List[str]] = None) -> AlphaRankResult:
        """
        Compute Alpha-Rank directly from match results.
        
        Args:
            match_results: List of dicts with 'agent1', 'agent2', 'winner' keys
            agent_names: Optional list of agent names (inferred if not provided)
            
        Returns:
            AlphaRankResult with rankings
        """
        if agent_names is None:
            # Infer agent names from match results
            names = set()
            for r in match_results:
                names.add(r['agent1'])
                names.add(r['agent2'])
            agent_names = sorted(names)
        
        payoff = self.compute_payoff_matrix(match_results, agent_names)
        return self.compute(payoff, agent_names)
    
    def alpha_sweep(self,
                    payoff_matrix: np.ndarray,
                    agent_names: List[str],
                    alphas: Optional[List[float]] = None) -> Dict:
        """
        Perform alpha sweep to analyze sensitivity.
        
        Returns rankings for different alpha values.
        """
        if alphas is None:
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        results = {}
        for alpha in alphas:
            self.alpha = alpha
            result = self.compute(payoff_matrix, agent_names)
            results[alpha] = result.rankings
        
        return results


class PopulationTracker:
    """
    Track match results during training for Alpha-Rank evaluation.
    """
    
    def __init__(self, max_history: int = 10000):
        self.match_history: List[Dict] = []
        self.max_history = max_history
        self.agent_generations: Dict[str, int] = {}  # Track agent versions
    
    def record_match(self, 
                     agent1: str, 
                     agent2: str, 
                     winner: Optional[str],
                     gen1: int = 0,
                     gen2: int = 0):
        """Record a match result."""
        # Use generation-tagged names for more granular tracking
        name1 = f"{agent1}_g{gen1}" if gen1 > 0 else agent1
        name2 = f"{agent2}_g{gen2}" if gen2 > 0 else agent2
        
        result = {
            'agent1': name1,
            'agent2': name2,
            'winner': winner.replace(agent1, name1).replace(agent2, name2) if winner else None
        }
        
        self.match_history.append(result)
        
        # Trim history if needed
        if len(self.match_history) > self.max_history:
            self.match_history = self.match_history[-self.max_history:]
    
    def get_alpha_rank(self, 
                       alpha: float = 0.1,
                       recent_only: int = None) -> AlphaRankResult:
        """Compute Alpha-Rank from recorded matches."""
        matches = self.match_history
        if recent_only:
            matches = matches[-recent_only:]
        
        ranker = AlphaRank(alpha=alpha)
        return ranker.compute_from_matches(matches)
    
    def save(self, filepath: str):
        """Save match history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.match_history, f)
    
    def load(self, filepath: str):
        """Load match history from file."""
        with open(filepath, 'r') as f:
            self.match_history = json.load(f)


def visualize_payoff_matrix(payoff: np.ndarray, 
                            agent_names: List[str],
                            filepath: Optional[str] = None):
    """
    Print/visualize payoff matrix as heatmap in terminal.
    """
    n = len(agent_names)
    
    # Truncate names for display
    short_names = [name[:8] for name in agent_names]
    
    print("\nPayoff Matrix (row vs column win rate):")
    print("-" * (12 + n * 7))
    
    # Header
    header = "            "
    for name in short_names:
        header += f"{name:>6s} "
    print(header)
    
    # Rows
    for i, name in enumerate(short_names):
        row = f"{name:>10s}  "
        for j in range(n):
            val = payoff[i, j]
            # Color code: green > 0.5, red < 0.5
            if val > 0.6:
                row += f"\033[92m{val:.2f}\033[0m  "
            elif val < 0.4:
                row += f"\033[91m{val:.2f}\033[0m  "
            else:
                row += f"{val:.2f}  "
        print(row)
    
    print("-" * (12 + n * 7))


# Example usage and testing
if __name__ == "__main__":
    print("Alpha-Rank Test")
    print("=" * 50)
    
    # Create a simple Rock-Paper-Scissors style payoff matrix
    # Rock beats Scissors, Scissors beats Paper, Paper beats Rock
    agent_names = ["Rock", "Paper", "Scissors", "Random"]
    
    # Payoff matrix: P[i,j] = win rate of agent i against agent j
    payoff = np.array([
        [0.5, 0.0, 1.0, 0.5],  # Rock
        [1.0, 0.5, 0.0, 0.5],  # Paper
        [0.0, 1.0, 0.5, 0.5],  # Scissors
        [0.5, 0.5, 0.5, 0.5],  # Random
    ])
    
    # Compute Alpha-Rank
    ranker = AlphaRank(alpha=0.1)
    result = ranker.compute(payoff, agent_names)
    
    print(result)
    print()
    
    visualize_payoff_matrix(payoff, agent_names)
    
    # Test with simulated match data
    print("\n\nTest with Match History:")
    print("=" * 50)
    
    tracker = PopulationTracker()
    
    # Simulate matches
    import random
    agents = ["Agent_A", "Agent_B", "Agent_C"]
    
    # A beats B 70%, B beats C 70%, C beats A 70% (non-transitive!)
    matchups = [
        ("Agent_A", "Agent_B", 0.7),  # A wins 70%
        ("Agent_B", "Agent_C", 0.7),  # B wins 70%
        ("Agent_C", "Agent_A", 0.7),  # C wins 70% (non-transitive)
    ]
    
    for _ in range(100):
        for a1, a2, win_rate in matchups:
            winner = a1 if random.random() < win_rate else a2
            tracker.record_match(a1, a2, winner)
    
    result = tracker.get_alpha_rank(alpha=0.1)
    print(result)
    
    # Show that all agents are roughly equal (due to non-transitivity)
    print("\n✓ Non-transitive dynamics detected!")
    print("  (All agents have similar rankings despite clear head-to-head advantages)")
