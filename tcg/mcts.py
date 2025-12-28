"""
MCTS Implementation for Pokemon TCG.
Improved version with:
1. Deck-agnostic board state heuristics
2. Policy-guided rollouts (instead of random)
3. Neural network value function option
4. Visit distribution for training
"""
from __future__ import annotations
import math
import copy
import torch
import numpy as np
import random
from typing import Dict, Optional, Tuple, List

from tcg.env import PTCGEnv
from tcg.state import featurize, GameState
from tcg.cards import card_def
from tcg.actions import ACTION_TABLE


class MCTSNode:
    def __init__(self, player_idx: int, parent: Optional['MCTSNode'] = None, prior: float = 0.0):
        self.player_idx = player_idx  # Player whose turn it is at this node
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def evaluate_pokemon_value(slot, is_active: bool = False) -> float:
    """
    Deck-agnostic evaluation of a Pokemon's strategic value.
    Returns a score representing the Pokemon's contribution to winning.
    """
    if not slot or not slot.name:
        return 0.0
    
    try:
        cd = card_def(slot.name)
    except:
        return 0.5  # Unknown card, neutral
    
    score = 0.0
    
    # Stage multiplier - evolved Pokemon are inherently more valuable
    if cd.subtype == "Stage2":
        score += 2.5
    elif cd.subtype == "Stage1":
        score += 1.2
    elif cd.subtype == "Basic":
        # Check if it's a setup Pokemon (has evolution)
        score += 0.3
    
    # HP value (tankiness)
    score += cd.hp / 200.0  # Normalize: 330 HP = 1.65, 50 HP = 0.25
    
    # ex/V Pokemon are key threats
    if cd.has_rule_box:
        score += 1.0
    
    # Energy attached = attack readiness
    energy_count = len(slot.energy) if hasattr(slot, 'energy') else 0
    score += energy_count * 0.5
    
    # Active vs Bench positioning
    if is_active and energy_count > 0:
        # Active with energy = can attack this turn
        score += 0.8
    
    # Damage reduces value (closer to being KO'd)
    if hasattr(slot, 'damage') and slot.damage:
        damage_ratio = slot.damage / max(cd.hp, 1)
        score -= damage_ratio * 1.5  # Damaged Pokemon are less valuable
    
    # Has ability = utility value
    if cd.ability:
        score += 0.4
    
    return score


def evaluate_board_state(gs: GameState, for_player: int = 0) -> float:
    """
    Deck-agnostic evaluation of the game state.
    Returns value from -1 to 1 representing P0's advantage.
    """
    p0 = gs.players[0]
    p1 = gs.players[1]
    
    score = 0.0
    
    # ========== PRIZE DIFFERENTIAL (Primary objective) ==========
    p0_prizes_taken = 6 - len(p0.prizes)
    p1_prizes_taken = 6 - len(p1.prizes)
    prize_diff = p0_prizes_taken - p1_prizes_taken
    score += prize_diff * 3.0  # Strong weight on prize lead
    
    # ========== BOARD DEVELOPMENT ==========
    # Active Pokemon value
    if p0.active:
        score += evaluate_pokemon_value(p0.active, is_active=True)
    else:
        score -= 5.0  # No active = very bad
    
    if p1.active:
        score -= evaluate_pokemon_value(p1.active, is_active=True)
    else:
        score += 5.0  # Opponent no active = very good
    
    # Bench presence and quality
    p0_bench_value = 0.0
    p0_bench_count = 0
    for slot in p0.bench:
        if slot and slot.name:
            p0_bench_value += evaluate_pokemon_value(slot, is_active=False)
            p0_bench_count += 1
    
    p1_bench_value = 0.0
    p1_bench_count = 0
    for slot in p1.bench:
        if slot and slot.name:
            p1_bench_value += evaluate_pokemon_value(slot, is_active=False)
            p1_bench_count += 1
    
    score += p0_bench_value * 0.5
    score -= p1_bench_value * 0.5
    
    # Empty bench is dangerous
    if p0_bench_count == 0:
        score -= 2.0
    if p1_bench_count == 0:
        score += 2.0
    
    # ========== HAND SIZE (Resource advantage) ==========
    # More cards = more options
    p0_hand = len(p0.hand) if hasattr(p0, 'hand') else 0
    p1_hand = len(p1.hand) if hasattr(p1, 'hand') else 0
    hand_diff = (p0_hand - p1_hand) * 0.1
    score += hand_diff
    
    # ========== DECK SIZE (Avoid deck-out) ==========
    p0_deck = len(p0.deck) if hasattr(p0, 'deck') else 30
    p1_deck = len(p1.deck) if hasattr(p1, 'deck') else 30
    
    # Low deck is risky
    if p0_deck < 5:
        score -= 1.0
    if p1_deck < 5:
        score += 1.0
    
    # ========== ATTACK POTENTIAL ==========
    # Estimate max damage this turn
    if p0.active and hasattr(p0.active, 'energy') and len(p0.active.energy) > 0:
        try:
            cd = card_def(p0.active.name)
            if cd.attacks:
                best_damage = max(atk.damage for atk in cd.attacks)
                score += best_damage / 100.0  # Normalize
        except:
            pass
    
    # ========== EVOLUTION POTENTIAL ==========
    # Having unevolved basics on bench is potential (but they need to evolve)
    for slot in p0.bench:
        if slot and slot.name:
            try:
                cd = card_def(slot.name)
                if cd.subtype == "Basic" and cd.evolves_from is None:
                    # Check if hand has evolution
                    for card in p0.hand:
                        try:
                            evo_cd = card_def(card)
                            if evo_cd.evolves_from == slot.name:
                                score += 0.3  # Potential to evolve
                                break
                        except:
                            pass
            except:
                pass
    
    # Normalize to -1 to 1 range
    return math.tanh(score * 0.15)


class MCTS:
    def __init__(self, policy_net: torch.nn.Module, device: torch.device, 
                 num_simulations: int = 50, c_puct: float = 1.5, 
                 max_rollout_steps: int = 150,
                 use_value_net: bool = False,
                 use_policy_rollouts: bool = True,
                 temperature: float = 1.0):
        """
        MCTS with improved evaluation.
        
        Args:
            policy_net: Policy network (or PolicyValueNet with value head)
            device: Torch device
            num_simulations: Number of MCTS simulations per search
            c_puct: Exploration constant for UCB formula
            max_rollout_steps: Maximum steps in rollout
            use_value_net: If True, use value head instead of rollouts
            use_policy_rollouts: If True, use policy network for rollout actions
            temperature: Temperature for action selection (higher = more exploration)
        """
        self.policy_net = policy_net
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_rollout_steps = max_rollout_steps
        self.use_value_net = use_value_net
        self.use_policy_rollouts = use_policy_rollouts
        self.temperature = temperature
        
    def search(self, env: PTCGEnv, return_probs: bool = False, combo_registry=None) -> int | Tuple[int, np.ndarray]:
        """
        Run MCTS search from the current environment state.
        
        Args:
            env: Current environment
            return_probs: If True, return (action, visit_distribution) for training
            combo_registry: Optional list of StrategicCombo objects for injection
            
        Returns:
            Best action index, optionally with visit distribution
        """
        # === COMBO INJECTION: Check for forced strategic moves ===
        if combo_registry:
            forced_action = self._check_combo_injection(env, combo_registry)
            if forced_action is not None:
                if return_probs:
                    # Return a "confident" probability distribution to train the policy to mimic
                    probs = np.zeros(len(ACTION_TABLE), dtype=np.float32)
                    probs[forced_action] = 1.0
                    return forced_action, probs
                return forced_action
        
        root_player = env._gs.turn_player
        root = MCTSNode(player_idx=root_player, prior=0.0)
        
        # Expand root immediately
        obs = featurize(env._gs)
        mask = env.action_mask()
        self._expand(root, obs, mask)
        
        for _ in range(self.num_simulations):
            node = root
            sim_env = copy.deepcopy(env)  # Clone for simulation
            
            # 1. Selection - traverse tree using UCB
            path = []
            while node.is_expanded and node.children:
                act_idx, child = self._select_child(node)
                path.append((node, act_idx, child))
                node = child
                sim_env.step(act_idx)
                
                # Update node player if first visit
                if node.player_idx == -1:
                    node.player_idx = sim_env._gs.turn_player
                
                if sim_env._gs.done:
                    break
            
            # 2. Expansion & Evaluation
            value = 0.0
            if not sim_env._gs.done:
                mask = sim_env.action_mask()
                if np.sum(mask) > 0:
                    if not node.is_expanded:
                        obs = featurize(sim_env._gs)
                        leaf_value = self._expand(node, obs, mask)
                        
                        if self.use_value_net:
                            # Use value network prediction
                            value = leaf_value
                        else:
                            # Use rollout
                            value = self._rollout(sim_env)
                    else:
                        value = self._rollout(sim_env)
            else:
                # Terminal state
                winner = sim_env._gs.winner
                if winner == 0:
                    value = 1.0
                elif winner == 1:
                    value = -1.0
                else:
                    value = 0.0
                
            # 3. Backpropagation
            self._backpropagate(node, value)
            
        # Select action based on visit counts
        if not root.children:
            mask = env.action_mask()
            valid = np.where(mask > 0)[0]
            action = valid[0] if len(valid) > 0 else 0
            if return_probs:
                probs = np.zeros(mask.shape[0])
                probs[action] = 1.0
                return action, probs
            return action
        
        # Get visit counts as probabilities
        n_actions = len(mask)
        visit_counts = np.zeros(n_actions)
        for act_idx, child in root.children.items():
            visit_counts[act_idx] = child.visit_count
        
        # Apply temperature
        if self.temperature > 0:
            visit_counts_temp = visit_counts ** (1.0 / self.temperature)
            probs = visit_counts_temp / (visit_counts_temp.sum() + 1e-8)
        else:
            # Greedy selection
            probs = np.zeros(n_actions)
            best_action = np.argmax(visit_counts)
            probs[best_action] = 1.0
        
        # Select action
        if self.temperature > 0:
            best_action = np.random.choice(n_actions, p=probs)
        else:
            best_action = np.argmax(visit_counts)
        
        if return_probs:
            return best_action, probs
        return best_action

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select child using PUCT formula with proper two-player handling."""
        from tcg.actions import ACTION_TABLE
        
        is_p0_turn = (node.player_idx == 0)
        
        best_score = -float('inf')
        best_act = -1
        best_child = None
        
        sqrt_parent = math.sqrt(node.visit_count)
        
        for act_idx, child in node.children.items():
            # UCB exploration term
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            
            # Q value (child's average value from P0's perspective)
            q = child.value
            
            # === STRATEGIC BONUS: Bias toward good actions ===
            # This helps overcome value network delusion about PASS
            act = ACTION_TABLE[act_idx]
            strategic_bonus = 0.0
            if act.kind == 'ATTACK':
                strategic_bonus = 0.3  # Strong bonus for attacks
            elif 'EVOLVE' in act.kind:
                strategic_bonus = 0.1  # Bonus for evolving
            elif 'ABILITY' in act.kind:
                strategic_bonus = 0.1  # Bonus for abilities
            elif act.kind == 'PASS':
                strategic_bonus = -0.1  # Penalty for passing
            
            # P0 maximizes, P1 minimizes (so flip Q for P1)
            if is_p0_turn:
                score = q + u + strategic_bonus
            else:
                score = -q + u + strategic_bonus  # P1 wants low P0 value
            
            if score > best_score:
                best_score = score
                best_act = act_idx
                best_child = child
        
        if best_child is None:
            if not node.children:
                return -1, None
            return list(node.children.keys())[0], list(node.children.values())[0]

        return best_act, best_child
        
    def _expand(self, node: MCTSNode, obs: np.ndarray, mask: np.ndarray) -> float:
        """
        Expand node by creating children for all valid actions.
        Returns value estimate if using value network.
        """
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.policy_net.eval()
            output = self.policy_net(obs_t)
            
            # Handle both PolicyNet and PolicyValueNet
            if isinstance(output, tuple):
                logits, value = output
                value = value.item()
            else:
                logits = output
                value = 0.0  # No value head
        
        # === CRITICAL FIX: VALUE PERSPECTIVE ===
        # The network predicts value for the CURRENT player (node.player_idx).
        # MCTS requires the value to be relative to PLAYER 0.
        # If it is Player 1's turn, a high value means P1 wins (which is -1.0 for P0).
        if node.player_idx == 1:
            value = -value
        # === END FIX ===
            
        mask_t = torch.from_numpy(mask).float().to(self.device)
        huge_neg = torch.ones_like(logits) * -1e9
        masked_logits = torch.where(mask_t.unsqueeze(0) > 0, logits, huge_neg)
        probs = torch.softmax(masked_logits, dim=1).cpu().numpy()[0]
        
        valid_indices = np.where(mask > 0)[0]
        
        # === EXPLORATION FIX: Force strategic action exploration ===
        # Without this, agent gets stuck in PASS-only equilibrium
        from tcg.actions import ACTION_TABLE
        MIN_ATTACK_PRIOR = 0.50  # 50% prior for attacks - MUST explore attacks!
        MIN_EVOLVE_PRIOR = 0.20  # 20% prior for evolutions
        MIN_ABILITY_PRIOR = 0.15  # 15% prior for abilities
        MIN_ATTACH_PRIOR = 0.25  # 25% prior for energy attachment - CRITICAL for enabling attacks!
        MAX_PASS_PRIOR = 0.10   # CAP pass at 10% to force action!
        
        # Boost attack actions
        attack_indices = [idx for idx in valid_indices if ACTION_TABLE[idx].kind == 'ATTACK']
        if attack_indices:
            for idx in attack_indices:
                if probs[idx] < MIN_ATTACK_PRIOR:
                    probs[idx] = MIN_ATTACK_PRIOR
        
        # Boost evolution actions  
        evolve_indices = [idx for idx in valid_indices if 'EVOLVE' in ACTION_TABLE[idx].kind]
        for idx in evolve_indices:
            if probs[idx] < MIN_EVOLVE_PRIOR:
                probs[idx] = MIN_EVOLVE_PRIOR
        
        # Boost ability actions
        ability_indices = [idx for idx in valid_indices if 'ABILITY' in ACTION_TABLE[idx].kind]
        for idx in ability_indices:
            if probs[idx] < MIN_ABILITY_PRIOR:
                probs[idx] = MIN_ABILITY_PRIOR
        
        # Boost ATTACH actions (critical for enabling attacks!)
        attach_indices = [idx for idx in valid_indices if 'ATTACH' in ACTION_TABLE[idx].kind]
        for idx in attach_indices:
            if probs[idx] < MIN_ATTACH_PRIOR:
                probs[idx] = MIN_ATTACH_PRIOR
        
        # PENALIZE PASS - cap it so other actions get explored
        pass_idx = 0  # PASS is always action 0
        if pass_idx in valid_indices and len(valid_indices) > 1:
            if probs[pass_idx] > MAX_PASS_PRIOR:
                probs[pass_idx] = MAX_PASS_PRIOR
        
        # Renormalize
        prob_sum = probs[valid_indices].sum()
        if prob_sum > 0:
            probs[valid_indices] /= prob_sum
        
        node.children = {}
        for idx in valid_indices:
            node.children[idx] = MCTSNode(player_idx=-1, parent=node, prior=probs[idx])
            
        node.is_expanded = True
        return value
        
    def _rollout(self, env: PTCGEnv) -> float:
        """
        Rollout with 50% CURRICULUM-GUIDED + 50% RANDOM actions.
        Curriculum shows winning sequence, random explores alternatives.
        """
        from tcg.actions import ACTION_TABLE
        steps = 0
        
        while not env._gs.done and steps < self.max_rollout_steps:
            mask = env.action_mask()
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                break
            
            # 50% curriculum-guided, 50% random for exploration
            if np.random.random() < 0.5:
                # CURRICULUM-GUIDED: Show optimal action sequence
                evolve_indices = [i for i in valid if 'EVOLVE' in ACTION_TABLE[i].kind]
                attach_indices = [i for i in valid if 'ATTACH' in ACTION_TABLE[i].kind]
                ability_indices = [i for i in valid if 'ABILITY' in ACTION_TABLE[i].kind]
                attack_indices = [i for i in valid if ACTION_TABLE[i].kind == 'ATTACK']
                
                action = None
                
                # Priority: Evolve → Attach → Ability → Attack → Random
                if evolve_indices:
                    action = np.random.choice(evolve_indices)
                elif attach_indices:
                    action = np.random.choice(attach_indices)
                elif ability_indices:
                    action = np.random.choice(ability_indices)
                elif attack_indices:
                    action = np.random.choice(attack_indices)
                else:
                    action = np.random.choice(valid)
            else:
                # RANDOM: Explore alternatives (but avoid PASS if possible)
                non_pass = [i for i in valid if i != 0]
                action = np.random.choice(non_pass) if non_pass else np.random.choice(valid)
                
            env.step(action)
            steps += 1
            
        if env._gs.done:
            if env._gs.winner == 0:
                return 1.0
            elif env._gs.winner == 1:
                return -1.0
            else:
                return 0.0
        
        # Use heuristic evaluation for non-terminal states
        return evaluate_board_state(env._gs, for_player=0)

    def _backpropagate(self, node: MCTSNode, value_p0: float):
        """Backpropagate value up the tree."""
        curr = node
        while curr:
            curr.visit_count += 1
            curr.value_sum += value_p0  # Always from P0's perspective
            curr = curr.parent
    
    def _check_combo_injection(self, env: PTCGEnv, combo_registry) -> Optional[int]:
        """
        Check if current state allows a scripted combo step.
        Returns action index if combo should be injected, None otherwise.
        """
        me = env._gs.players[env._gs.turn_player]
        mask = env.action_mask()
        
        for combo in combo_registry:
            for step in combo.steps:
                try:
                    # Check if combo condition is met
                    if step.condition(env, me):
                        # Get the action to take
                        action_idx = step.action_selector(env, me)
                        
                        # Verify the action is legal in current state
                        if 0 <= action_idx < len(mask) and mask[action_idx]:
                            # 70% chance to FORCE the combo move (30% exploration)
                            if np.random.random() < 0.70:
                                return action_idx
                except Exception:
                    # If condition check fails, skip this combo
                    continue
        
        return None


class PolicyValueNet(torch.nn.Module):
    """
    Combined Policy and Value network for AlphaZero-style MCTS.
    Policy head outputs action logits.
    Value head outputs expected game outcome from P0's perspective.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 512):
        super().__init__()
        
        # Shared trunk
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.ReLU(),
        )
        
        # Policy head
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )
        
        # Value head  
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Tanh(),  # Output in [-1, 1]
        )
    
    def forward(self, x):
        """
        Forward pass.
        Returns: (policy_logits, value)
        """
        shared = self.shared(x)
        policy = self.policy_head(shared)
        value = self.value_head(shared)
        return policy, value.squeeze(-1)
    
    def policy_only(self, x):
        """Get just the policy logits (for compatibility)."""
        shared = self.shared(x)
        return self.policy_head(shared)
