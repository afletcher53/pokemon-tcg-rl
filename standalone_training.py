#!/usr/bin/env python3
"""
STANDALONE Pokemon TCG AlphaZero Training Script

This is a self-contained training script that combines all necessary components:
- Game environment (PTCGEnv)
- Card definitions
- State representation
- MCTS with neural network
- Genetic Algorithm population
- Scripted opponents for external pressure

Usage:
    python standalone_training.py --episodes 5000 --resume checkpoint.pt

Author: Consolidated from multi-file Pokemon TCG RL project
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
from typing import List, Tuple, Dict, Optional, Callable
from tqdm import tqdm
import multiprocessing as mp
import gymnasium as gym
from gymnasium import spaces


# =============================================================================
# CARD DEFINITIONS
# =============================================================================

@dataclass
class Attack:
    name: str
    damage: int
    cost: List[str]
    text: str = ""

@dataclass  
class CardDef:
    name: str
    supertype: str  # Pokemon, Trainer, Energy
    subtype: str    # Basic, Stage1, Stage2, Supporter, Item, Stadium, Basic (energy)
    hp: int = 0
    type: str = "Colorless"
    evolves_from: Optional[str] = None
    tags: tuple = ()
    attacks: Tuple[Attack, ...] = ()
    weakness: Optional[str] = None
    resistance: Optional[str] = None
    retreat_cost: int = 0
    ability: Optional[str] = None
    
    def has_rule_box(self) -> bool:
        return "ex" in self.tags or "V" in self.tags

# Minimal card registry for training
CARD_REGISTRY: Dict[str, CardDef] = {
    # Pokemon - Alakazam Line
    "Abra": CardDef("Abra", "Pokemon", "Basic", hp=50, type="Psychic",
                    attacks=(Attack("Teleportation Attack", 10, ["Psychic"]),),
                    weakness="Darkness", retreat_cost=1),
    "Kadabra": CardDef("Kadabra", "Pokemon", "Stage1", hp=90, type="Psychic", evolves_from="Abra",
                      attacks=(Attack("Psychic", 60, ["Psychic", "Colorless"]),),
                      weakness="Darkness", retreat_cost=2),
    "Alakazam": CardDef("Alakazam", "Pokemon", "Stage2", hp=150, type="Psychic", evolves_from="Kadabra",
                       attacks=(Attack("Psychic", 90, ["Psychic", "Colorless"]),),
                       weakness="Darkness", retreat_cost=2, ability="Mindful Mastery"),
    
    # Pokemon - Charizard Line  
    "Charmander": CardDef("Charmander", "Pokemon", "Basic", hp=60, type="Fire",
                         attacks=(Attack("Ember", 30, ["Fire"]),),
                         weakness="Water", retreat_cost=1),
    "Charmeleon": CardDef("Charmeleon", "Pokemon", "Stage1", hp=90, type="Fire", evolves_from="Charmander",
                         attacks=(Attack("Flamethrower", 60, ["Fire", "Colorless"]),),
                         weakness="Water", retreat_cost=2),
    "Charizard ex": CardDef("Charizard ex", "Pokemon", "Stage2", hp=330, type="Fire", evolves_from="Charmeleon",
                           tags=("ex",), attacks=(Attack("Burn Brightly", 180, ["Fire", "Fire"]),),
                           weakness="Water", retreat_cost=2),
    
    # Support Pokemon
    "Dunsparce": CardDef("Dunsparce", "Pokemon", "Basic", hp=60, type="Colorless",
                        attacks=(Attack("Tackle", 20, ["Colorless", "Colorless"]),),
                        weakness="Fighting", retreat_cost=1),
    "Dudunsparce": CardDef("Dudunsparce", "Pokemon", "Stage1", hp=140, type="Colorless", evolves_from="Dunsparce",
                          attacks=(Attack("Land Crush", 90, ["Colorless", "Colorless", "Colorless"]),),
                          weakness="Fighting", retreat_cost=3, ability="Run Away Draw"),
    "Fan Rotom": CardDef("Fan Rotom", "Pokemon", "Basic", hp=70, type="Colorless",
                        attacks=(Attack("Spin Storm", 20, ["Colorless"]),),
                        retreat_cost=1, ability="Fan Call"),
    "Fezandipiti ex": CardDef("Fezandipiti ex", "Pokemon", "Basic", hp=210, type="Psychic", tags=("ex",),
                             attacks=(Attack("Cruel Arrow", 100, ["Psychic", "Colorless"]),),
                             weakness="Darkness", retreat_cost=1, ability="Flip the Script"),
    "Psyduck": CardDef("Psyduck", "Pokemon", "Basic", hp=70, type="Water",
                      attacks=(Attack("Ram", 20, ["Colorless", "Colorless"]),),
                      weakness="Lightning", retreat_cost=1, ability="Damp"),
                      
    # Trainers
    "Rare Candy": CardDef("Rare Candy", "Trainer", "Item"),
    "Buddy-Buddy Poffin": CardDef("Buddy-Buddy Poffin", "Trainer", "Item"),
    "Ultra Ball": CardDef("Ultra Ball", "Trainer", "Item"),
    "Nest Ball": CardDef("Nest Ball", "Trainer", "Item"),
    "Super Rod": CardDef("Super Rod", "Trainer", "Item"),
    "Night Stretcher": CardDef("Night Stretcher", "Trainer", "Item"),
    "Wondrous Patch": CardDef("Wondrous Patch", "Trainer", "Item"),
    "Enhanced Hammer": CardDef("Enhanced Hammer", "Trainer", "Item"),
    "Hilda": CardDef("Hilda", "Trainer", "Supporter"),
    "Dawn": CardDef("Dawn", "Trainer", "Supporter"),
    "Arven": CardDef("Arven", "Trainer", "Supporter"),
    "Iono": CardDef("Iono", "Trainer", "Supporter"),
    "Boss's Orders": CardDef("Boss's Orders", "Trainer", "Supporter"),
    "Lillie's Determination": CardDef("Lillie's Determination", "Trainer", "Supporter"),
    "Tulip": CardDef("Tulip", "Trainer", "Supporter"),
    "Counter Catcher": CardDef("Counter Catcher", "Trainer", "Item"),
    "Battle Cage": CardDef("Battle Cage", "Trainer", "Stadium"),
    "Artazon": CardDef("Artazon", "Trainer", "Stadium"),
    
    # Energy
    "Basic Psychic Energy": CardDef("Basic Psychic Energy", "Energy", "Basic", type="Psychic"),
    "Basic Fire Energy": CardDef("Basic Fire Energy", "Energy", "Basic", type="Fire"),
    "Fire Energy": CardDef("Fire Energy", "Energy", "Basic", type="Fire"),
    "Jet Energy": CardDef("Jet Energy", "Energy", "Special_Energy", type="Colorless"),
    "Enriching Energy": CardDef("Enriching Energy", "Energy", "Special_Energy", type="Colorless"),
}

def card_def(name: str) -> CardDef:
    if name in CARD_REGISTRY:
        return CARD_REGISTRY[name]
    # Fallback for unknown cards
    return CardDef(name, "Unknown", "Unknown")


# =============================================================================
# GAME STATE
# =============================================================================

MAX_HAND = 30
MAX_BENCH = 5

@dataclass
class PokemonSlot:
    name: Optional[str] = None
    energy: List[str] = field(default_factory=list)
    tool: Optional[str] = None
    damage: int = 0
    status: Dict[str, bool] = field(default_factory=lambda: {
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False
    })
    turn_played: int = 0
    damage_reduction: int = 0

@dataclass
class PlayerState:
    hand: List[str] = field(default_factory=list)
    deck: List[str] = field(default_factory=list)
    discard_pile: List[str] = field(default_factory=list)
    prizes: List[str] = field(default_factory=list)
    active: PokemonSlot = field(default_factory=PokemonSlot)
    bench: List[PokemonSlot] = field(default_factory=lambda: [PokemonSlot() for _ in range(MAX_BENCH)])
    energy_attached: bool = False
    supporter_used: bool = False
    quick_search_used: bool = False
    ability_used_this_turn: bool = False
    
    def prizes_taken(self) -> int:
        return 6 - len(self.prizes)

@dataclass
class GameState:
    players: List[PlayerState] = field(default_factory=lambda: [PlayerState(), PlayerState()])
    turn_player: int = 0
    turn_number: int = 1
    done: bool = False
    winner: Optional[int] = None
    ko_last_turn: bool = False
    active_stadium: Optional[str] = None


def featurize(gs: GameState) -> np.ndarray:
    """Convert game state to fixed-size observation vector."""
    features = []
    
    # Current player and turn info
    features.extend([gs.turn_player, gs.turn_number / 100.0, float(gs.ko_last_turn)])
    
    for p_idx in range(2):
        p = gs.players[p_idx]
        
        # Hand/deck/discard counts
        features.extend([
            len(p.hand) / MAX_HAND,
            len(p.deck) / 60.0,
            len(p.discard_pile) / 60.0,
            len(p.prizes) / 6.0,
            float(p.energy_attached),
            float(p.supporter_used),
        ])
        
        # Active Pokemon
        if p.active.name:
            cd = card_def(p.active.name)
            features.extend([
                cd.hp / 300.0,
                p.active.damage / 300.0,
                len(p.active.energy) / 6.0,
                float(cd.subtype == "Stage2"),
                float(cd.subtype == "Stage1"),
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # Bench summary
        bench_count = sum(1 for s in p.bench if s.name)
        bench_energy = sum(len(s.energy) for s in p.bench if s.name)
        features.extend([bench_count / 5.0, bench_energy / 20.0])
    
    # Pad to fixed size (156 features)
    while len(features) < 156:
        features.append(0.0)
    
    return np.array(features[:156], dtype=np.float32)


# =============================================================================
# ACTIONS
# =============================================================================

@dataclass(frozen=True)
class Action:
    kind: str
    a: Optional[str] = None  # Card name
    b: Optional[int] = None  # Target index
    c: Optional[int] = None  # Secondary target
    d: Optional[int] = None
    e: Optional[int] = None
    f: Optional[int] = None

def build_action_table(max_bench: int = 5) -> List[Action]:
    """Build complete action table for the game."""
    actions = [Action("PASS")]
    
    # Get all card names
    pokemon_cards = [n for n, c in CARD_REGISTRY.items() if c.supertype == "Pokemon"]
    trainer_cards = [n for n, c in CARD_REGISTRY.items() if c.supertype == "Trainer"]
    energy_cards = [n for n, c in CARD_REGISTRY.items() if c.supertype == "Energy"]
    
    # Play basic to bench
    for card in pokemon_cards:
        if card_def(card).subtype == "Basic":
            for slot in range(max_bench):
                actions.append(Action("PLAY_BASIC_TO_BENCH", a=card, b=slot))
    
    # Evolve active
    for card in pokemon_cards:
        if card_def(card).subtype in ("Stage1", "Stage2"):
            actions.append(Action("EVOLVE_ACTIVE", a=card))
            for slot in range(max_bench):
                actions.append(Action(f"EVOLVE_BENCH_{slot}", a=card, b=slot))
    
    # Attach energy
    for energy in energy_cards:
        actions.append(Action("ATTACH_ACTIVE", a=energy))
        for slot in range(max_bench):
            actions.append(Action(f"ATTACH_BENCH_{slot}", a=energy, b=slot))
    
    # Play trainers (simplified - just target 6 for most)
    for trainer in trainer_cards:
        for target in range(7):  # 0-4 bench, 5 active, 6 self/global
            actions.append(Action("PLAY_TRAINER", a=trainer, b=target))
    
    # Retreat
    for slot in range(max_bench):
        actions.append(Action("RETREAT_TO", b=slot))
    
    # Use ability
    actions.append(Action("USE_ACTIVE_ABILITY"))
    for target in range(7):
        actions.append(Action("USE_ACTIVE_ABILITY", c=target))
    
    # Attack
    actions.append(Action("ATTACK"))
    
    return actions

ACTION_TABLE = build_action_table()
ACTION_INDEX = {a: i for i, a in enumerate(ACTION_TABLE)}


# =============================================================================
# ENVIRONMENT
# =============================================================================

class PTCGEnv(gym.Env):
    """Simplified Pokemon TCG Environment for training."""
    
    def __init__(self, scripted_opponent: bool = False, max_turns: int = 200):
        super().__init__()
        self.scripted_opponent = scripted_opponent
        self.max_turns = max_turns
        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(156,), dtype=np.float32)
        self._gs = GameState()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._gs = GameState()
        
        # Get custom decks if provided
        custom_decks = options.get("decks") if options else None
        
        for p_idx in range(2):
            if custom_decks and p_idx < len(custom_decks):
                deck = list(custom_decks[p_idx])
            else:
                deck = self._create_default_deck()
            
            random.shuffle(deck)
            p = self._gs.players[p_idx]
            p.deck = deck
            
            # Draw 7 cards
            p.hand = [p.deck.pop() for _ in range(min(7, len(p.deck)))]
            
            # Ensure at least one basic
            basics = [c for c in p.hand if card_def(c).subtype == "Basic" and card_def(c).supertype == "Pokemon"]
            while not basics and p.deck:
                p.deck.extend(p.hand)
                random.shuffle(p.deck)
                p.hand = [p.deck.pop() for _ in range(min(7, len(p.deck)))]
                basics = [c for c in p.hand if card_def(c).subtype == "Basic" and card_def(c).supertype == "Pokemon"]
            
            # Set active
            if basics:
                card = basics[0]
                p.hand.remove(card)
                p.active.name = card
                p.active.turn_played = 0
            
            # Set prizes
            p.prizes = [p.deck.pop() for _ in range(min(6, len(p.deck)))]
        
        self._gs.turn_player = random.randint(0, 1)
        obs = featurize(self._gs)
        return obs, {"action_mask": self.action_mask()}
    
    def _create_default_deck(self) -> List[str]:
        """Create a simple training deck."""
        deck = []
        deck.extend(["Abra"] * 4)
        deck.extend(["Kadabra"] * 3)
        deck.extend(["Alakazam"] * 3)
        deck.extend(["Dunsparce"] * 4)
        deck.extend(["Dudunsparce"] * 3)
        deck.extend(["Fan Rotom"] * 2)
        deck.extend(["Hilda"] * 4)
        deck.extend(["Dawn"] * 4)
        deck.extend(["Buddy-Buddy Poffin"] * 4)
        deck.extend(["Rare Candy"] * 4)
        deck.extend(["Ultra Ball"] * 4)
        deck.extend(["Basic Psychic Energy"] * 8)
        while len(deck) < 60:
            deck.append("Basic Psychic Energy")
        return deck[:60]
    
    def action_mask(self) -> np.ndarray:
        """Return boolean mask of valid actions."""
        gs = self._gs
        me = gs.players[gs.turn_player]
        mask = np.zeros(len(ACTION_TABLE), dtype=np.int8)
        
        if gs.done:
            return mask
        
        # PASS is always valid
        mask[0] = 1
        
        empty_bench = [i for i, s in enumerate(me.bench) if s.name is None]
        
        for i, act in enumerate(ACTION_TABLE):
            if act.kind == "PASS":
                continue
            
            if act.kind == "PLAY_BASIC_TO_BENCH":
                if act.a in me.hand and empty_bench and act.b in empty_bench:
                    if card_def(act.a).subtype == "Basic" and card_def(act.a).supertype == "Pokemon":
                        mask[i] = 1
            
            elif act.kind == "EVOLVE_ACTIVE":
                if gs.turn_number > 2 and act.a in me.hand:
                    evo = card_def(act.a)
                    if evo.evolves_from and me.active.name == evo.evolves_from:
                        if me.active.turn_played < gs.turn_number:
                            mask[i] = 1
            
            elif act.kind.startswith("EVOLVE_BENCH_"):
                if gs.turn_number > 2 and act.a in me.hand:
                    evo = card_def(act.a)
                    idx = act.b
                    if idx is not None and 0 <= idx < MAX_BENCH:
                        slot = me.bench[idx]
                        if slot.name and evo.evolves_from == slot.name:
                            if slot.turn_played < gs.turn_number:
                                mask[i] = 1
            
            elif act.kind == "ATTACH_ACTIVE":
                if not me.energy_attached and act.a in me.hand and me.active.name:
                    if card_def(act.a).supertype == "Energy":
                        mask[i] = 1
            
            elif act.kind.startswith("ATTACH_BENCH_"):
                if not me.energy_attached and act.a in me.hand:
                    idx = act.b
                    if idx is not None and 0 <= idx < MAX_BENCH and me.bench[idx].name:
                        if card_def(act.a).supertype == "Energy":
                            mask[i] = 1
            
            elif act.kind == "PLAY_TRAINER":
                if act.a in me.hand:
                    cd = card_def(act.a)
                    if cd.subtype == "Supporter":
                        if me.supporter_used or gs.turn_number == 1:
                            continue
                    mask[i] = 1
            
            elif act.kind == "RETREAT_TO":
                if me.active.name:
                    retreat_cost = card_def(me.active.name).retreat_cost
                    idx = act.b
                    if len(me.active.energy) >= retreat_cost and idx is not None:
                        if 0 <= idx < MAX_BENCH and me.bench[idx].name:
                            mask[i] = 1
            
            elif act.kind == "USE_ACTIVE_ABILITY":
                if me.active.name and not me.ability_used_this_turn:
                    mask[i] = 1
            
            elif act.kind == "ATTACK":
                if me.active.name and me.active.name:
                    if not (gs.turn_number == 1 and gs.turn_player == 0):
                        cd = card_def(me.active.name)
                        if cd.attacks:
                            cost = len(cd.attacks[0].cost) if cd.attacks[0].cost else 0
                            if len(me.active.energy) >= cost:
                                mask[i] = 1
        
        return mask
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return new state."""
        gs = self._gs
        me = gs.players[gs.turn_player]
        op = gs.players[1 - gs.turn_player]
        act = ACTION_TABLE[action]
        
        reward = 0.0
        
        if act.kind == "PASS":
            # End turn
            self._end_turn()
        
        elif act.kind == "PLAY_BASIC_TO_BENCH":
            if act.a in me.hand and act.b is not None:
                me.hand.remove(act.a)
                me.bench[act.b].name = act.a
                me.bench[act.b].turn_played = gs.turn_number
        
        elif act.kind == "EVOLVE_ACTIVE":
            if act.a in me.hand:
                me.hand.remove(act.a)
                me.active.name = act.a
        
        elif act.kind.startswith("EVOLVE_BENCH_"):
            if act.a in me.hand and act.b is not None:
                me.hand.remove(act.a)
                me.bench[act.b].name = act.a
        
        elif act.kind == "ATTACH_ACTIVE":
            if act.a in me.hand and not me.energy_attached:
                me.hand.remove(act.a)
                me.active.energy.append(act.a)
                me.energy_attached = True
        
        elif act.kind.startswith("ATTACH_BENCH_"):
            if act.a in me.hand and act.b is not None and not me.energy_attached:
                me.hand.remove(act.a)
                me.bench[act.b].energy.append(act.a)
                me.energy_attached = True
        
        elif act.kind == "PLAY_TRAINER":
            if act.a in me.hand:
                me.hand.remove(act.a)
                cd = card_def(act.a)
                if cd.subtype == "Supporter":
                    me.supporter_used = True
                    # Simple supporter effect: draw cards
                    if act.a in ("Hilda", "Dawn"):
                        for _ in range(5):
                            if me.deck:
                                me.hand.append(me.deck.pop())
                me.discard_pile.append(act.a)
        
        elif act.kind == "RETREAT_TO":
            if act.b is not None:
                # Swap active with bench
                old_active = (me.active.name, me.active.energy.copy(), me.active.damage)
                me.active.name = me.bench[act.b].name
                me.active.energy = me.bench[act.b].energy.copy()
                me.active.damage = me.bench[act.b].damage
                me.bench[act.b].name = old_active[0]
                me.bench[act.b].energy = old_active[1]
                me.bench[act.b].damage = old_active[2]
        
        elif act.kind == "USE_ACTIVE_ABILITY":
            me.ability_used_this_turn = True
            # Simple ability: draw a card
            if me.deck:
                me.hand.append(me.deck.pop())
        
        elif act.kind == "ATTACK":
            if me.active.name and op.active.name:
                cd = card_def(me.active.name)
                if cd.attacks:
                    damage = cd.attacks[0].damage
                    op.active.damage += damage
                    
                    # Check KO
                    opp_hp = card_def(op.active.name).hp
                    if op.active.damage >= opp_hp:
                        # KO - take prize
                        if me.prizes:
                            me.hand.append(me.prizes.pop())
                            reward = 1.0
                        
                        gs.ko_last_turn = True
                        
                        # Promote from bench
                        promoted = False
                        for slot in op.bench:
                            if slot.name:
                                op.active.name = slot.name
                                op.active.energy = slot.energy.copy()
                                op.active.damage = slot.damage
                                slot.name = None
                                slot.energy = []
                                slot.damage = 0
                                promoted = True
                                break
                        
                        if not promoted:
                            # No bench - opponent loses
                            gs.done = True
                            gs.winner = gs.turn_player
                            reward = 10.0
                
                # Check win by prizes
                if len(me.prizes) == 0:
                    gs.done = True
                    gs.winner = gs.turn_player
                    reward = 10.0
                
                # End turn after attack
                self._end_turn()
        
        # Check if game over by deck out
        if not gs.done and not me.deck and len([a for a in range(len(ACTION_TABLE)) if self.action_mask()[a]]) <= 1:
            gs.done = True
            gs.winner = 1 - gs.turn_player
        
        # Check turn limit
        if gs.turn_number > self.max_turns:
            gs.done = True
            # Draw - whoever has more prizes taken wins
            p0_prizes = 6 - len(gs.players[0].prizes)
            p1_prizes = 6 - len(gs.players[1].prizes)
            if p0_prizes > p1_prizes:
                gs.winner = 0
            elif p1_prizes > p0_prizes:
                gs.winner = 1
            else:
                gs.winner = -1  # Draw
        
        obs = featurize(gs)
        return obs, reward, gs.done, False, {"winner": gs.winner, "action_mask": self.action_mask()}
    
    def _end_turn(self):
        """End current turn and switch to opponent."""
        gs = self._gs
        me = gs.players[gs.turn_player]
        
        # Reset per-turn flags
        me.energy_attached = False
        me.supporter_used = False
        me.ability_used_this_turn = False
        
        # Switch player
        gs.turn_player = 1 - gs.turn_player
        
        # New turn - draw card
        new_player = gs.players[gs.turn_player]
        if new_player.deck:
            new_player.hand.append(new_player.deck.pop())
        
        gs.turn_number += 1
        gs.ko_last_turn = False


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class PolicyValueNet(nn.Module):
    """Transformer-based Policy-Value Network."""
    
    def __init__(self, obs_dim: int, n_actions: int, d_model: int = 256, n_layers: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads=4) for _ in range(n_layers)
        ])
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, n_actions),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = x + self.pos_encoding
        
        for layer in self.transformer:
            x = layer(x)
        
        x = x.squeeze(1)
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value


# =============================================================================
# MCTS
# =============================================================================

class MCTSNode:
    def __init__(self, player_idx: int, parent=None, prior: float = 0.0):
        self.player_idx = player_idx
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, policy_net: nn.Module, device: torch.device, 
                 num_simulations: int = 50, c_puct: float = 1.5, temperature: float = 1.0):
        self.policy_net = policy_net
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
    
    def search(self, env: PTCGEnv, return_probs: bool = False):
        """Run MCTS search from current state."""
        root = MCTSNode(env._gs.turn_player)
        obs = featurize(env._gs)
        mask = env.action_mask()
        
        # Expand root
        self._expand(root, obs, mask)
        
        for _ in range(self.num_simulations):
            node = root
            sim_env = copy.deepcopy(env)
            
            # Selection
            while node.is_expanded and node.children:
                action, node = self._select_child(node)
                obs, _, done, _, _ = sim_env.step(action)
                if done:
                    break
            
            # Expansion and evaluation
            if not sim_env._gs.done and not node.is_expanded:
                mask = sim_env.action_mask()
                value = self._expand(node, obs, mask)
            else:
                # Terminal node
                if sim_env._gs.winner == 0:
                    value = 1.0
                elif sim_env._gs.winner == 1:
                    value = -1.0
                else:
                    value = 0.0
            
            # Backpropagate
            self._backpropagate(node, value)
        
        # Select action
        visit_counts = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in range(len(ACTION_TABLE))
        ])
        
        if self.temperature > 0:
            visit_counts = visit_counts ** (1 / self.temperature)
        
        probs = visit_counts / (visit_counts.sum() + 1e-8)
        
        if return_probs:
            action = np.random.choice(len(ACTION_TABLE), p=probs)
            return action, probs
        else:
            return np.argmax(visit_counts)
    
    def _select_child(self, node: MCTSNode):
        best_score = -float('inf')
        best_action = 0
        best_child = None
        
        for action, child in node.children.items():
            # PUCT formula
            prior_score = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            
            # Flip value for opponent
            if child.player_idx != node.player_idx:
                value_score = -child.value()
            else:
                value_score = child.value()
            
            score = value_score + prior_score
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _expand(self, node: MCTSNode, obs: np.ndarray, mask: np.ndarray) -> float:
        """Expand node using policy net."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, value = self.policy_net(obs_t)
        
        policy = policy.cpu().numpy().flatten()
        value = value.cpu().item()
        
        # Mask illegal actions
        policy = np.where(mask, policy, -1e9)
        probs = np.exp(policy - policy.max())
        probs = probs / probs.sum()
        
        for action in range(len(ACTION_TABLE)):
            if mask[action]:
                child = MCTSNode(1 - node.player_idx, parent=node, prior=probs[action])
                node.children[action] = child
        
        node.is_expanded = True
        return value
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            # Value is from P0's perspective
            if node.player_idx == 0:
                node.value_sum += value
            else:
                node.value_sum -= value
            node = node.parent


# =============================================================================
# SCRIPTED AGENTS
# =============================================================================

class ScriptedAgent:
    """Heuristic-based opponent for external pressure."""
    
    def __init__(self, strategy: str = "aggressive"):
        self.strategy = strategy
    
    def select_action(self, env: PTCGEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        legal_actions = np.where(mask)[0]
        
        if len(legal_actions) <= 1:
            return legal_actions[0] if len(legal_actions) == 1 else 0
        
        if self.strategy == "aggressive":
            # Priority: Attack > Attach > Evolve > Trainer > Bench > Pass
            for action in legal_actions:
                if ACTION_TABLE[action].kind == "ATTACK":
                    return action
            for action in legal_actions:
                if ACTION_TABLE[action].kind == "ATTACH_ACTIVE":
                    return action
            for action in legal_actions:
                if "EVOLVE" in ACTION_TABLE[action].kind:
                    return action
            for action in legal_actions:
                if ACTION_TABLE[action].kind == "PLAY_TRAINER":
                    return action
            for action in legal_actions:
                if ACTION_TABLE[action].kind == "PLAY_BASIC_TO_BENCH":
                    return action
        
        # Default: random non-pass
        non_pass = [a for a in legal_actions if a != 0]
        return random.choice(non_pass) if non_pass else 0


# =============================================================================
# GENETIC ALGORITHM POPULATION
# =============================================================================

class GeneticPopulation:
    """Population-based training with genetic evolution."""
    
    def __init__(self, n_agents: int, obs_dim: int, n_actions: int, device: torch.device,
                 mutation_rate: float = 0.05, mutation_strength: float = 0.005, elitism_count: int = 2):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism_count = min(elitism_count, n_agents // 2)
        
        self.models = [
            PolicyValueNet(obs_dim, n_actions).to(device)
            for _ in range(n_agents)
        ]
        
        self.optimizers = [
            optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for model in self.models
        ]
        
        self.fitness = [0.0] * n_agents
        self.games_played = [0] * n_agents
        self.generation = 0
    
    def get_random_pair(self) -> Tuple[int, int]:
        i = random.randrange(self.n_agents)
        j = random.randrange(self.n_agents)
        while j == i and self.n_agents > 1:
            j = random.randrange(self.n_agents)
        return i, j
    
    def update_fitness(self, agent_idx: int, won: bool):
        self.games_played[agent_idx] += 1
        if won:
            self.fitness[agent_idx] = (self.fitness[agent_idx] * (self.games_played[agent_idx] - 1) + 1.0) / self.games_played[agent_idx]
        else:
            self.fitness[agent_idx] = (self.fitness[agent_idx] * (self.games_played[agent_idx] - 1)) / self.games_played[agent_idx]
    
    def evolve_generation(self):
        """Run one generation of evolution."""
        self.generation += 1
        
        if min(self.games_played) < 5:
            return
        
        rankings = np.argsort(self.fitness)[::-1]
        elite_indices = rankings[:self.elitism_count]
        elite_states = [copy.deepcopy(self.models[i].state_dict()) for i in elite_indices]
        
        # Replace weak agents with mutated elites
        for i in range(self.elitism_count, self.n_agents):
            parent_idx = random.choice(elite_indices)
            self.models[i].load_state_dict(copy.deepcopy(self.models[parent_idx].state_dict()))
            with torch.no_grad():
                for param in self.models[i].parameters():
                    if random.random() < self.mutation_rate:
                        noise = torch.randn_like(param) * self.mutation_strength
                        param.add_(noise)
        
        # Reset fitness
        self.fitness = [0.0] * self.n_agents
        self.games_played = [0] * self.n_agents
    
    def get_best_model(self) -> nn.Module:
        return self.models[np.argmax(self.fitness)]


# =============================================================================
# TRAINING LOOP
# =============================================================================

def play_game(model_p0: nn.Module, model_p1: nn.Module, device: torch.device,
              deck_p0: List[str], deck_p1: List[str], mcts_sims: int = 50,
              scripted_strategy: Optional[str] = None) -> Dict:
    """Play a single game and return experiences."""
    
    mcts_p0 = MCTS(model_p0, device, num_simulations=mcts_sims)
    
    if scripted_strategy:
        scripted = ScriptedAgent(scripted_strategy)
        mcts_p1 = None
    else:
        mcts_p1 = MCTS(model_p1, device, num_simulations=mcts_sims)
    
    env = PTCGEnv(max_turns=200)
    obs, _ = env.reset(options={"decks": [deck_p0, deck_p1]})
    
    history = []
    step = 0
    
    while not env._gs.done and step < 2000:
        mask = env.action_mask()
        player = env._gs.turn_player
        
        if player == 0:
            action, probs = mcts_p0.search(env, return_probs=True)
        else:
            if scripted_strategy:
                action = scripted.select_action(env, obs, mask)
                probs = np.zeros(len(ACTION_TABLE))
                probs[action] = 1.0
            else:
                action, probs = mcts_p1.search(env, return_probs=True)
        
        history.append((obs.copy(), probs, player))
        obs, _, done, _, info = env.step(action)
        step += 1
    
    return {
        "winner": env._gs.winner,
        "history": history,
        "steps": step,
    }


def train(episodes: int = 5000, mcts_sims: int = 50, population_size: int = 8,
          resume_path: Optional[str] = None, scripted_ratio: float = 0.2):
    """Main training function."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    obs_dim = 156
    n_actions = len(ACTION_TABLE)
    
    # Initialize population
    population = GeneticPopulation(population_size, obs_dim, n_actions, device)
    
    # Resume if provided
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        for i in range(population_size):
            population.models[i].load_state_dict(state)
            if i >= population.elitism_count:
                with torch.no_grad():
                    for param in population.models[i].parameters():
                        param.add_(torch.randn_like(param) * 0.001)
        print("Resume complete!")
    
    # Default deck
    default_deck = []
    default_deck.extend(["Abra"] * 4)
    default_deck.extend(["Kadabra"] * 3)
    default_deck.extend(["Alakazam"] * 3)
    default_deck.extend(["Dunsparce"] * 4)
    default_deck.extend(["Dudunsparce"] * 3)
    default_deck.extend(["Fan Rotom"] * 2)
    default_deck.extend(["Hilda"] * 4)
    default_deck.extend(["Dawn"] * 4)
    default_deck.extend(["Buddy-Buddy Poffin"] * 4)
    default_deck.extend(["Rare Candy"] * 4)
    default_deck.extend(["Ultra Ball"] * 4)
    default_deck.extend(["Basic Psychic Energy"] * 21)
    
    # Experience buffer
    buffer = []
    
    # Training loop
    pbar = tqdm(range(episodes), desc="Training")
    recent_wins = deque(maxlen=100)
    
    for ep in pbar:
        # Select agents
        p0_idx, p1_idx = population.get_random_pair()
        model_p0 = population.models[p0_idx]
        model_p1 = population.models[p1_idx]
        
        # Maybe use scripted opponent
        scripted_strategy = None
        if random.random() < scripted_ratio:
            scripted_strategy = random.choice(["aggressive", "defensive"])
        
        # Play game
        model_p0.eval()
        model_p1.eval()
        result = play_game(model_p0, model_p1, device, default_deck, default_deck, 
                          mcts_sims, scripted_strategy)
        
        # Update fitness
        if result["winner"] == 0:
            population.update_fitness(p0_idx, won=True)
            population.update_fitness(p1_idx, won=False)
            recent_wins.append(1)
        elif result["winner"] == 1:
            population.update_fitness(p0_idx, won=False)
            population.update_fitness(p1_idx, won=True)
            recent_wins.append(0)
        else:
            recent_wins.append(0.5)
        
        # Store experiences
        for obs, probs, player in result["history"]:
            if result["winner"] == player:
                value = 1.0
            elif result["winner"] == 1 - player:
                value = -1.0
            else:
                value = 0.0
            buffer.append((obs, probs, value))
        
        # Train
        if len(buffer) >= 256:
            batch_idx = random.sample(range(len(buffer)), 256)
            batch = [buffer[i] for i in batch_idx]
            
            obs_batch = torch.FloatTensor([b[0] for b in batch]).to(device)
            probs_batch = torch.FloatTensor([b[1] for b in batch]).to(device)
            value_batch = torch.FloatTensor([b[2] for b in batch]).to(device)
            
            for model, optimizer in zip(population.models, population.optimizers):
                model.train()
                optimizer.zero_grad()
                
                policy, value = model(obs_batch)
                
                # Policy loss (cross entropy with MCTS probs)
                policy_loss = -torch.mean(torch.sum(probs_batch * F.log_softmax(policy, dim=-1), dim=-1))
                
                # Value loss (MSE)
                value_loss = F.mse_loss(value.squeeze(), value_batch)
                
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()
        
        # Trim buffer
        if len(buffer) > 100000:
            buffer = buffer[-50000:]
        
        # Evolution
        if (ep + 1) % 100 == 0:
            population.evolve_generation()
            
            # Save checkpoint
            best_model = population.get_best_model()
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "state_dict": best_model.state_dict(),
                "episode": ep + 1,
            }, f"checkpoints/checkpoint_ep{ep + 1}.pt")
        
        # Update progress bar
        win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.5
        pbar.set_postfix({"WR": f"{win_rate:.0%}", "Gen": population.generation})
    
    print("Training complete!")
    
    # Save final
    best_model = population.get_best_model()
    torch.save({"state_dict": best_model.state_dict()}, "final_policy.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standalone Pokemon TCG Training')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--mcts_sims', type=int, default=50)
    parser.add_argument('--population', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--scripted_ratio', type=float, default=0.2)
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        mcts_sims=args.mcts_sims,
        population_size=args.population,
        resume_path=args.resume,
        scripted_ratio=args.scripted_ratio,
    )
