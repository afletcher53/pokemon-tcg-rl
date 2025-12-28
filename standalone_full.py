#!/usr/bin/env python3
"""
STANDALONE Pokemon TCG AlphaZero Training Script

This is a complete, self-contained training script combining all source files.
Generated automatically from the multi-file Pokemon TCG RL project.

Usage:
    python standalone_full.py --episodes 5000 --resume checkpoint.pt
"""

from __future__ import annotations
import argparse
import copy
import csv
import gymnasium as gym
import json
import math
import multiprocessing as mp
import numpy as np
import os
import random
import torch
import torch.multiprocessing as tmp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, Counter
from dataclasses import dataclass, field
from gymnasium import spaces
from multiprocessing import Pool, Queue
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Callable, Any


# =============================================================================
# CARDS
# Source: tcg/cards.py
# =============================================================================

# tcg/cards.py


@dataclass(frozen=True)
class Attack:
    name: str
    damage: int
    cost: List[str] # e.g. ["Fire", "Fire", "Colorless"]
    text: str = ""

@dataclass(frozen=True)
class CardDef:
    name: str
    supertype: str  # "Pokemon" | "Trainer" | "Energy"
    subtype: str  # "Basic", "Stage1", "Stage2", "Item", "Supporter", "Stadium"
    hp: int = 0
    type: str = "Colorless" 
    evolves_from: Optional[str] = None
    tags: tuple[str, ...] = ()
    attacks: Tuple[Attack, ...] = ()
    weakness: Optional[str] = None
    resistance: Optional[str] = None
    retreat_cost: int = 0
    ability: Optional[str] = None # Name of ability

    @property
    def has_rule_box(self):
        return "ex" in self.tags or "V" in self.tags

# Minimal registry (expand per deck)
CARD_REGISTRY: Dict[str, CardDef] = {
    # Pokemon
    "Psyduck": CardDef("Psyduck", "Pokemon", "Basic", hp=70, type="Water",
                       attacks=(Attack("Ram", 20, ["Colorless", "Colorless"]),),
                       weakness="Lightning", retreat_cost=1, ability="Damp"),

    "Charmander": CardDef("Charmander", "Pokemon", "Basic", hp=70, type="Fire", 
                          attacks=(Attack("Ember", 30, ["Fire", "Colorless"]),),
                          weakness="Water", retreat_cost=1),
                          
    "Charmeleon": CardDef("Charmeleon", "Pokemon", "Stage1", hp=90, type="Fire", evolves_from="Charmander",
                          attacks=(Attack("Combustion", 50, ["Fire", "Fire", "Colorless"]),),
                          weakness="Water", retreat_cost=2),
                          
    "Charizard ex": CardDef("Charizard ex", "Pokemon", "Stage2", hp=330, type="Darkness", evolves_from="Charmeleon", tags=("ex",),
                            attacks=(Attack("Burning Darkness", 180, ["Fire", "Fire"]),),
                            weakness="Grass", retreat_cost=2, ability="Infernal Reign"),
    
    "Pidgey": CardDef("Pidgey", "Pokemon", "Basic", hp=50, type="Colorless",
                      attacks=(Attack("Call for Family", 0, ["Colorless"]),
                               Attack("Tackle", 20, ["Colorless", "Colorless"])),
                      weakness="Lightning", resistance="Fighting", retreat_cost=1),
                      
    "Pidgeotto": CardDef("Pidgeotto", "Pokemon", "Stage1", hp=90, type="Colorless", evolves_from="Pidgey",
                         attacks=(Attack("Wing Attack", 40, ["Colorless", "Colorless"]),),
                         weakness="Lightning", retreat_cost=1),
                         
    "Pidgeot ex": CardDef("Pidgeot ex", "Pokemon", "Stage2", hp=280, type="Colorless", evolves_from="Pidgeotto", tags=("ex",),
                          attacks=(Attack("Gust", 120, ["Colorless", "Colorless"]),),
                          weakness="Lightning", retreat_cost=0, ability="Quick Search"),
    
    "Duskull": CardDef("Duskull", "Pokemon", "Basic", hp=60, type="Psychic",
                       attacks=(Attack("Rain of Pain", 10, ["Psychic"]),),
                       weakness="Darkness", retreat_cost=1),
                       
    "Dusclops": CardDef("Dusclops", "Pokemon", "Stage1", hp=90, type="Psychic", evolves_from="Duskull",
                        attacks=(Attack("Will-O-Wisp", 30, ["Psychic", "Colorless"]),),
                        weakness="Darkness", retreat_cost=2),
                        
    "Dusknoir": CardDef("Dusknoir", "Pokemon", "Stage2", hp=160, type="Psychic", evolves_from="Dusclops",
                        attacks=(Attack("Shadow Bind", 150, ["Psychic", "Psychic", "Colorless"]),),
                        weakness="Darkness", retreat_cost=3),
    
    "Tatsugiri": CardDef("Tatsugiri", "Pokemon", "Basic", hp=70, type="Dragon",
                         attacks=(Attack("Surf", 50, ["Fire", "Water"]),),
                         retreat_cost=1, ability="Attract Customers"),
                         
    "Klefki": CardDef("Klefki", "Pokemon", "Basic", hp=70, type="Psychic",
                      attacks=(Attack("Mischievous Lock", 10, ["Colorless"]),),
                      weakness="Metal", retreat_cost=1),
                      
    "Fan Rotom": CardDef("Fan Rotom", "Pokemon", "Basic", hp=70, type="Colorless",
                         attacks=(Attack("Assault Landing", 70, ["Colorless"]),),
                         weakness="Lightning", retreat_cost=1, ability="Fan Call"),
    
    "Dunsparce": CardDef("Dunsparce", "Pokemon", "Basic", hp=70, type="Colorless",
                         attacks=(Attack("Gnaw", 20, ["Colorless"]),),
                         weakness="Fighting", retreat_cost=1),
                         
    "Dudunsparce": CardDef("Dudunsparce", "Pokemon", "Stage1", hp=140, type="Colorless", evolves_from="Dunsparce",
                           attacks=(Attack("Land Crush", 90, ["Colorless", "Colorless", "Colorless"]),),
                           weakness="Fighting", retreat_cost=3, ability="Run Away Draw"),
    
    "Abra": CardDef("Abra", "Pokemon", "Basic", hp=50, type="Psychic",
                    attacks=(Attack("Teleportation Attack", 10, ["Psychic"]),),
                    weakness="Darkness", resistance="Fighting", retreat_cost=1),
                    
    "Kadabra": CardDef("Kadabra", "Pokemon", "Stage1", hp=80, type="Psychic", evolves_from="Abra",
                       attacks=(Attack("Super Psy Bolt", 30, ["Psychic"]),),
                       weakness="Darkness", retreat_cost=1, ability="Psychic Draw"),
                       
    "Alakazam": CardDef("Alakazam", "Pokemon", "Stage2", hp=140, type="Psychic", evolves_from="Kadabra",
                        attacks=(Attack("Powerful Hand", 0, ["Psychic"]),),
                        weakness="Darkness", retreat_cost=1, ability="Psychic Draw"),
    
    "Fezandipiti ex": CardDef("Fezandipiti ex", "Pokemon", "Basic", hp=210, type="Darkness", tags=("ex",),
                              attacks=(Attack("Cruel Arrow", 0, ["Colorless", "Colorless", "Colorless"]),),
                              weakness="Fighting", retreat_cost=1, ability="Flip the Script"),
    
    # New Pokemon for Charizard deck
    "Shaymin": CardDef("Shaymin", "Pokemon", "Basic", hp=80, type="Grass",
                       attacks=(Attack("Smash Kick", 30, ["Colorless", "Colorless"]),),
                       weakness="Fire", retreat_cost=1, ability="Flower Curtain"),
    
    "Munkidori": CardDef("Munkidori", "Pokemon", "Basic", hp=110, type="Psychic",
                         attacks=(Attack("Mind Bend", 60, ["Psychic", "Colorless"]),),
                         weakness="Darkness", resistance="Fighting", retreat_cost=1, ability="Adrena-Brain"),
    
    "Chi-Yu": CardDef("Chi-Yu", "Pokemon", "Basic", hp=110, type="Fire",
                      attacks=(Attack("Megafire of Envy", 50, ["Fire", "Fire"]),),  # 50+90 if KO'd last turn
                      weakness="Water", retreat_cost=1),
    
    "Gouging Fire ex": CardDef("Gouging Fire ex", "Pokemon", "Basic", hp=230, type="Fire", tags=("ex",),
                               attacks=(Attack("Blaze Blitz", 260, ["Fire", "Fire", "Colorless"]),),
                               weakness="Water", retreat_cost=2),
                               
    "Mega Charizard X ex": CardDef("Mega Charizard X ex", "Pokemon", "Stage2", hp=360, type="Fire", evolves_from="Charmeleon", tags=("ex",),
                                   attacks=(Attack("Inferno X", 90, ["Fire", "Fire"]),), # 90x per discarded Fire Energy
                                   weakness="Water", retreat_cost=2),
                               
    # Gholdengo / Solrock / Lunatone Deck
    "Gimmighoul": CardDef("Gimmighoul", "Pokemon", "Basic", hp=70, type="Psychic",
                          attacks=(Attack("Minor Errand-Running", 0, ["Colorless"]), 
                                   Attack("Tackle", 50, ["Colorless", "Colorless", "Colorless"])),
                          weakness="Darkness", resistance="Fighting", retreat_cost=2),
                          
    "Gholdengo ex": CardDef("Gholdengo ex", "Pokemon", "Stage1", hp=260, type="Metal", evolves_from="Gimmighoul", tags=("ex",),
                            attacks=(Attack("Make It Rain", 50, ["Metal"]),), # 50x
                            weakness="Fire", resistance="Grass", retreat_cost=2, ability="Coin Bonus"),
                            
    "Solrock": CardDef("Solrock", "Pokemon", "Basic", hp=110, type="Fighting",
                       attacks=(Attack("Cosmic Beam", 70, ["Fighting"]),), # Condition: Lunatone on bench
                       weakness="Grass", retreat_cost=1),
                       
    "Lunatone": CardDef("Lunatone", "Pokemon", "Basic", hp=110, type="Fighting",
                        attacks=(Attack("Power Gem", 50, ["Fighting", "Fighting"]),),
                        weakness="Grass", retreat_cost=1, ability="Lunar Cycle"),
                        
    "Genesect ex": CardDef("Genesect ex", "Pokemon", "Basic", hp=220, type="Metal", tags=("ex",),
                           attacks=(Attack("Protect Charge", 150, ["Metal", "Metal", "Colorless"]),),
                           weakness="Fire", resistance="Grass", retreat_cost=2, ability="Metallic Signal"),
                           
    "Hop's Cramorant": CardDef("Hop's Cramorant", "Pokemon", "Basic", hp=110, type="Colorless",
                               attacks=(Attack("Fickle Spitting", 120, ["Colorless"]),),
                               weakness="Lightning", resistance="Fighting", retreat_cost=1),
    
    # Trainers
    "Buddy-Buddy Poffin": CardDef("Buddy-Buddy Poffin", "Trainer", "Item"),
    "Rare Candy": CardDef("Rare Candy", "Trainer", "Item"),
    "Ultra Ball": CardDef("Ultra Ball", "Trainer", "Item"),
    "Arven": CardDef("Arven", "Trainer", "Supporter"),
    "Iono": CardDef("Iono", "Trainer", "Supporter"),
    "Boss's Orders": CardDef("Boss's Orders", "Trainer", "Supporter"),
    "Artazon": CardDef("Artazon", "Trainer", "Stadium"),
    "Battle Cage": CardDef("Battle Cage", "Trainer", "Stadium"),
    "Technical Machine: Evolution": CardDef("Technical Machine: Evolution", "Trainer", "Tool",
                                            attacks=(Attack("Evolution", 0, ["Colorless"]),)),
    "Counter Catcher": CardDef("Counter Catcher", "Trainer", "Item"),
    "Super Rod": CardDef("Super Rod", "Trainer", "Item"),
    "Night Stretcher": CardDef("Night Stretcher", "Trainer", "Item"),
    "Hilda": CardDef("Hilda", "Trainer", "Supporter"),
    "Dawn": CardDef("Dawn", "Trainer", "Supporter"),
    "Lillie's Determination": CardDef("Lillie's Determination", "Trainer", "Supporter"),
    "Tulip": CardDef("Tulip", "Trainer", "Supporter"),
    "Enhanced Hammer": CardDef("Enhanced Hammer", "Trainer", "Item"),
    "Professor's Research": CardDef("Professor's Research", "Trainer", "Supporter"),
    "Nest Ball": CardDef("Nest Ball", "Trainer", "Item"),
    "Wondrous Patch": CardDef("Wondrous Patch", "Trainer", "Item"),
    "Bill": CardDef("Bill", "Trainer", "Supporter"),  # Debug
    "Energy Search": CardDef("Energy Search", "Trainer", "Item"),
    "Professor Turo's Scenario": CardDef("Professor Turo's Scenario", "Trainer", "Supporter"),
    "Unfair Stamp": CardDef("Unfair Stamp", "Trainer", "Item"),
    "Earthen Vessel": CardDef("Earthen Vessel", "Trainer", "Item"),
    "Superior Energy Retrieval": CardDef("Superior Energy Retrieval", "Trainer", "Item"),
    "Fighting Gong": CardDef("Fighting Gong", "Trainer", "Item"), 
    "Lana's Aid": CardDef("Lana's Aid", "Trainer", "Supporter"),
    "Premium Power Pro": CardDef("Premium Power Pro", "Trainer", "Item"),
    "Prime Catcher": CardDef("Prime Catcher", "Trainer", "Item"),
    "Air Balloon": CardDef("Air Balloon", "Trainer", "Tool"),
    "Vitality Band": CardDef("Vitality Band", "Trainer", "Tool"),
    "Maximum Belt": CardDef("Maximum Belt", "Trainer", "Tool"),
    "Switch": CardDef("Switch", "Trainer", "Item"),
    "Escape Rope": CardDef("Escape Rope", "Trainer", "Item"),
    
    # Energy
    "Basic Fire Energy": CardDef("Basic Fire Energy", "Energy", "Basic", type="Fire"),
    "Fire Energy": CardDef("Fire Energy", "Energy", "Basic", type="Fire"),  # Alias
    "Basic Psychic Energy": CardDef("Basic Psychic Energy", "Energy", "Basic", type="Psychic"),
    "Jet Energy": CardDef("Jet Energy", "Energy", "Special", type="Colorless"),
    "Enriching Energy": CardDef("Enriching Energy", "Energy", "Special", type="Colorless"),
    "Mist Energy": CardDef("Mist Energy", "Energy", "Special", type="Colorless"),
    "Darkness Energy": CardDef("Darkness Energy", "Energy", "Basic", type="Darkness"),
    "Basic Water Energy": CardDef("Basic Water Energy", "Energy", "Basic", type="Water"),
    "Basic Metal Energy": CardDef("Basic Metal Energy", "Energy", "Basic", type="Metal"),
    "Metal Energy": CardDef("Metal Energy", "Energy", "Basic", type="Metal"),
    "Basic Fighting Energy": CardDef("Basic Fighting Energy", "Energy", "Basic", type="Fighting"),
    "Fighting Energy": CardDef("Fighting Energy", "Energy", "Basic", type="Fighting"),
}


def card_def(name: str) -> CardDef:
    if name not in CARD_REGISTRY:
        # Unknown card, keep system alive. Add it later.
        CARD_REGISTRY[name] = CardDef(name, "Unknown", "Unknown")
    return CARD_REGISTRY[name]


# =============================================================================
# STATE
# Source: tcg/state.py
# =============================================================================

# tcg/state.py


MAX_HAND = 30
MAX_BENCH = 5


@dataclass
class PokemonSlot:
    name: Optional[str] = None
    energy: List[str] = field(default_factory=list)
    tool: Optional[str] = None
    damage: int = 0
    # status: Poisoned, Burned, Asleep, Paralyzed, Confused
    status: Dict[str, bool] = field(default_factory=lambda: {
        "poisoned": False, "burned": False, "asleep": False, "paralyzed": False, "confused": False
    })
    turn_played: int = 0  # Turn number when this pokemon entered play or last evolved
    damage_reduction: int = 0  # Damage reduction for next attack (e.g., Protect Charge)
    ability_used_this_turn: bool = False  # Per-Pokemon ability tracking (resets each turn)


@dataclass
class PlayerState:
    hand: List[str] = field(default_factory=list)
    deck: List[str] = field(default_factory=list) # Full list of card names
    discard_pile: List[str] = field(default_factory=list)
    prizes: List[str] = field(default_factory=list)
    
    # helper for compatibility with old "count" checks
    @property
    def deck_count(self): return len(self.deck)
    @property
    def discard_count(self): return len(self.discard_pile)
    @property
    def prizes_taken(self): return 6 - len(self.prizes) # Assuming start with 6
    
    active: PokemonSlot = field(default_factory=PokemonSlot)
    bench: List[PokemonSlot] = field(
        default_factory=lambda: [PokemonSlot() for _ in range(MAX_BENCH)]
    )
    stadium: Optional[str] = None
    supporter_used: bool = False
    energy_attached: bool = False
    quick_search_used: bool = False  # Track Pidgeot ex ability
    ability_used_this_turn: bool = False
    fan_call_used: bool = False
    fighting_buff: bool = False  # Track Premium Power Pro (+30 dmg)
    
    # --- OPPONENT MODELING ---
    # Track opponent's visible actions for prediction
    last_searched_pokemon: Optional[str] = None  # What they searched for with Ultra Ball etc.
    last_searched_type: Optional[str] = None  # "Basic", "Stage1", "Stage2", "Evolution"
    turns_since_supporter: int = 0  # How many turns since they played a supporter
    total_supporters_played: int = 0  # Total supporters played this game
    likely_threats: List[str] = field(default_factory=list)  # Predicted key Pokemon coming
    
    # --- KNOWN OPPONENT HAND (Revealed Cards) ---
    # Cards revealed via restricted searches (Ultra Ball, Nest Ball, etc.)
    # NOT populated by unrestricted searches (Quick Search = any card)
    known_hand: List[str] = field(default_factory=list)  # Cards opponent revealed when searching


@dataclass
class GameState:
    # index 0 = agent, 1 = opponent
    players: List[PlayerState] = field(
        default_factory=lambda: [PlayerState(), PlayerState()]
    )
    turn_player: int = 0
    turn_number: int = 1
    done: bool = False
    winner: Optional[int] = None
    win_reason: Optional[str] = None
    ko_last_turn: bool = False  # Did a KO happen on the opponent's last turn? (Used for Fezandipiti ex)
    active_stadium: Optional[str] = None


# Fixed Vocabulary for Bag-of-Words hand encoding
# This allows the agent to 'see' which cards it has
CARD_VOCAB = sorted([
    'Abra', 'Air Balloon', 'Alakazam', 'Artazon', 'Arven', 'Basic Fighting Energy', 
    'Basic Fire Energy', 'Basic Metal Energy', 'Basic Psychic Energy', 'Basic Water Energy', 
    'Battle Cage', 'Bill', "Boss's Orders", 'Buddy-Buddy Poffin', 'Charizard ex', 
    'Charmander', 'Charmeleon', 'Chi-Yu', 'Counter Catcher', 'Darkness Energy', 'Dawn', 
    'Dudunsparce', 'Dunsparce', 'Dusclops', 'Dusknoir', 'Duskull', 'Earthen Vessel', 
    'Energy Search', 'Enhanced Hammer', 'Enriching Energy', 'Escape Rope', 'Fan Rotom', 
    'Fezandipiti ex', 'Fighting Energy', 'Fighting Gong', 'Fire Energy', 'Genesect ex', 
    'Gholdengo ex', 'Gimmighoul', 'Gouging Fire ex', 'Hilda', "Hop's Cramorant", 'Iono', 
    'Jet Energy', 'Kadabra', 'Klefki', "Lana's Aid", "Lillie's Determination", 'Lunatone', 
    'Maximum Belt', 'Mega Charizard X ex', 'Metal Energy', 'Mist Energy', 'Munkidori', 
    'Nest Ball', 'Night Stretcher', 'Pidgeot ex', 'Pidgeotto', 'Pidgey', 'Premium Power Pro', 
    'Prime Catcher', "Professor Turo's Scenario", "Professor's Research", 'Psyduck', 
    'Rare Candy', 'Shaymin', 'Solrock', 'Super Rod', 'Superior Energy Retrieval', 'Switch', 
    'Tatsugiri', 'Technical Machine: Evolution', 'Tulip', 'Ultra Ball', 'Unfair Stamp', 
    'Vitality Band', 'Wondrous Patch'
])
CARD_TO_IDX = {name: i for i, name in enumerate(CARD_VOCAB)}
CARD_VOCAB_SIZE = len(CARD_VOCAB)  # ~77 cards
CARD_VOCAB_SIZE_PADDED = 100  # Padded to 100 for future card additions

# Slot dimensions: 10 base features + 11 energy types + 100 identity = 121 per slot
SLOT_VEC_SIZE = 10 + 11 + CARD_VOCAB_SIZE_PADDED  # 121

# Energy types for tracking (prevents "energy blindness")
ENERGY_TYPES = ["Grass", "Fire", "Water", "Lightning", "Psychic", 
                "Fighting", "Darkness", "Metal", "Fairy", "Dragon", "Colorless"]
ENERGY_TO_IDX = {t: i for i, t in enumerate(ENERGY_TYPES)}


def featurize(gs: GameState) -> np.ndarray:
    """
    Compact numeric observation for the agent.
    
    Features:
    - Hand: Bag-of-words encoding (which specific cards in hand)
    - Board: Each slot has base features + energy type breakdown + Pokemon identity one-hot
    - All values normalized to 0-1 range
    
    Total size: 5 (glob) + 100 (hand_bow) + 1 (op_hand) + 1452 (12 slots × 121) + 8 (opp_model) + 18 (discard) = 1584
    """
    me = gs.players[gs.turn_player]
    op = gs.players[1 - gs.turn_player]

    # Hand: Bag-of-Words encoding (which specific cards) - PADDED
    def hand_bow(hand: List[str]) -> np.ndarray:
        """Bag of words encoding - count of each card type in hand."""
        vec = np.zeros(CARD_VOCAB_SIZE_PADDED, dtype=np.float32)  # Use padded size
        for c in hand:
            if c in CARD_TO_IDX:
                vec[CARD_TO_IDX[c]] += 1.0
        # Normalize counts (max 4 copies of any card)
        return vec / 4.0

    # Board state - NORMALIZED with ENERGY TYPE BREAKDOWN
    def slot_vec(slot: PokemonSlot) -> np.ndarray:
        # [has_pokemon, energy_total, damage, hp_ratio, has_tool, status_bits (5)] + 11 energy types + 100 identity
        if not slot.name:
            return np.zeros(SLOT_VEC_SIZE, dtype=np.float32)  # 10 base + 11 energy + 100 identity = 121
        hp_max = card_def(slot.name).hp
        if hp_max == 0: hp_max = 1
        
        # Count specific energy types attached (FIXES ENERGY BLINDNESS)
        energy_counts = np.zeros(len(ENERGY_TYPES), dtype=np.float32)
        for e_card in slot.energy:
            try:
                etype = card_def(e_card).type
            except:
                etype = "Colorless"
            if etype in ENERGY_TO_IDX:
                energy_counts[ENERGY_TO_IDX[etype]] += 1.0
        # Normalize (cap at 4 to keep inputs 0-1 range)
        energy_counts = energy_counts / 4.0
        
        status_vec = [
            float(slot.status.get("poisoned", False)),
            float(slot.status.get("burned", False)),
            float(slot.status.get("asleep", False)),
            float(slot.status.get("paralyzed", False)),
            float(slot.status.get("confused", False)),
        ]
        
        # Base features (10 dims)
        base_vec = np.array([
            1.0, 
            float(len(slot.energy)) / 5.0,    # Normalized: max ~5 energy
            float(slot.damage) / 300.0,       # Normalized: max ~300 damage
            slot.damage / hp_max,             # HP ratio already 0-1
            1.0 if slot.tool else 0.0,
            *status_vec
        ], dtype=np.float32)
        
        # Pokemon identity encoding (CARD_VOCAB_SIZE_PADDED dims)
        identity_vec = np.zeros(CARD_VOCAB_SIZE_PADDED, dtype=np.float32)
        if slot.name in CARD_TO_IDX:
            identity_vec[CARD_TO_IDX[slot.name]] = 1.0
        
        # Combined: base (10) + energy types (11) + identity (100) = 121
        return np.concatenate([base_vec, energy_counts, identity_vec])
    
    # Discard Analysis (Memory/Resource Tracking) - NORMALIZED
    def discard_vec(discard: List[str]) -> np.ndarray:
        # [Pokemon, Trainer, Energy, Boss, Iono, Arven, RareCandy, CharizardEx, PidgeotEx]
        counts = np.zeros(9, dtype=np.float32)
        for c in discard:
            cd = card_def(c)
            # Types
            if cd.supertype == "Pokemon": counts[0] += 1
            if cd.supertype == "Trainer": counts[1] += 1
            if cd.supertype == "Energy": counts[2] += 1
            # Key Cards
            if c == "Boss's Orders": counts[3] += 1
            if c == "Iono": counts[4] += 1
            if c == "Arven": counts[5] += 1
            if c == "Rare Candy": counts[6] += 1
            if c == "Charizard ex": counts[7] += 1
            if c == "Pidgeot ex": counts[8] += 1
        return counts / 10.0  # Normalize
    
    # Global features - NORMALIZED
    glob = np.array([
        gs.turn_number / 50.0,              # Normalized: games rarely go past 50 turns
        (6 - len(me.prizes)) / 6.0,         # Prizes taken ratio
        (6 - len(op.prizes)) / 6.0,         # Opponent prizes taken ratio
        len(me.deck) / 60.0,                # Deck size ratio
        len(op.deck) / 60.0                 # Opponent deck size ratio
    ], dtype=np.float32)
    
    # Hand encoding
    me_hand = hand_bow(me.hand)
    op_hand_count = np.array([len(op.hand) / 10.0], dtype=np.float32)  # Normalized
    
    # Board encoding
    me_active = slot_vec(me.active)
    me_bench = np.concatenate([slot_vec(s) for s in me.bench])
    op_active = slot_vec(op.active)
    op_bench = np.concatenate([slot_vec(s) for s in op.bench])
    
    # Opponent modeling (already normalized)
    turns_since_sup = min(op.turns_since_supporter, 5) / 5.0
    likely_hand_disruption = 1.0 if (op.turns_since_supporter == 0 and op.total_supporters_played > 0) else 0.0
    
    all_op_pokemon = [op.active] + [s for s in op.bench if s.name]
    highest_stage_hp = 0
    main_attacker_energy = 0
    unevolved_basics_count = 0
    evolved_pokemon_count = 0
    evolution_lines = {}
    
    for slot in all_op_pokemon:
        if not slot.name:
            continue
        cd = card_def(slot.name)
        if cd.subtype in ("Stage1", "Stage2"):
            evolved_pokemon_count += 1
            if cd.hp > highest_stage_hp:
                highest_stage_hp = cd.hp
                main_attacker_energy = len(slot.energy)
            base = cd.evolves_from
            while base:
                base_def = card_def(base)
                if base_def.evolves_from:
                    base = base_def.evolves_from
                else:
                    break
            if base:
                if base not in evolution_lines:
                    evolution_lines[base] = []
                evolution_lines[base].append(cd.subtype)
        elif cd.subtype == "Basic" and cd.hp <= 70:
            unevolved_basics_count += 1
    
    main_line_stage = 0.0
    if evolved_pokemon_count > 0:
        main_line_stage = 1.0 if any("Stage2" in stages for stages in evolution_lines.values()) else 0.5
    
    active_evolved_ready = 0.0
    if op.active.name:
        active_cd = card_def(op.active.name)
        if active_cd.subtype in ("Stage1", "Stage2") and len(op.active.energy) >= 1:
            active_evolved_ready = 1.0
    
    opponent_model = np.array([
        turns_since_sup,
        likely_hand_disruption,
        main_line_stage,
        min(unevolved_basics_count, 5) / 5.0,
        min(highest_stage_hp, 300) / 300.0,
        1.0 if main_attacker_energy >= 1 else 0.0,
        1.0 if op.last_searched_type in ("Stage1", "Stage2", "Evolution") else 0.0,
        active_evolved_ready
    ], dtype=np.float32)
    
    me_discard = discard_vec(me.discard_pile)
    op_discard = discard_vec(op.discard_pile)
    
    # Total: 5 (glob) + 100 (hand) + 1 (op_hand) + 1320 (12 slots × 110) + 8 (opp_model) + 18 (discard) = 1452
    return np.concatenate([
        glob, me_hand, op_hand_count, me_active, me_bench, op_active, op_bench, 
        opponent_model, me_discard, op_discard
    ])


# =============================================================================
# ACTIONS
# Source: tcg/actions.py
# =============================================================================

# tcg/actions.py


@dataclass(frozen=True)
class Action:
    kind: str
    a: Optional[str] = None  # card or attack/ability name
    b: Optional[int] = None  # target slot index / primary selection
    c: Optional[int] = None  # secondary target / card selection 1
    d: Optional[int] = None  # tertiary target / card selection 2
    e: Optional[int] = None  # card selection 3 (for 3+ card selections)
    f: Optional[int] = None  # card selection 4 (future expansion)
    
    def __repr__(self):
        parts = [f"'{self.kind}'"]
        if self.a is not None: parts.append(f"a={self.a}")
        if self.b is not None: parts.append(f"b={self.b}")
        if self.c is not None: parts.append(f"c={self.c}")
        if self.d is not None: parts.append(f"d={self.d}")
        if self.e is not None: parts.append(f"e={self.e}")
        if self.f is not None: parts.append(f"f={self.f}")
        return f"Action({', '.join(parts)})"


def build_action_table(max_bench: int = 5) -> List[Action]:
    actions: List[Action] = []
    actions.append(Action("PASS"))

    # Play Basic Pokemon from hand to bench slots
    # IMPORTANT: Sort keys for deterministic action ordering across processes
    for name in sorted(CARD_REGISTRY.keys()):
        cd = CARD_REGISTRY[name]
        if cd.supertype == "Pokemon" and cd.subtype == "Basic":
            for slot in range(max_bench):
                actions.append(Action("PLAY_BASIC_TO_BENCH", a=name, b=slot))

    # Evolve bench/active: choose evo card name + target index (active=-1, bench=0..)
    for name in sorted(CARD_REGISTRY.keys()):
        cd = CARD_REGISTRY[name]
        if cd.supertype == "Pokemon" and cd.evolves_from is not None:
            actions.append(Action("EVOLVE_ACTIVE", a=name))
            for slot in range(max_bench):
                actions.append(Action("EVOLVE_BENCH", a=name, b=slot))

    # Attach energy or tool to active or bench
    for name in sorted(CARD_REGISTRY.keys()):
        cd = CARD_REGISTRY[name]
        if cd.supertype == "Energy":
            actions.append(Action("ATTACH_ACTIVE", a=name))
            for slot in range(max_bench):
                actions.append(Action("ATTACH_BENCH", a=name, b=slot))
        elif cd.subtype == "Tool":
            actions.append(Action("ATTACH_TOOL_ACTIVE", a=name))
            for slot in range(max_bench):
                actions.append(Action("ATTACH_TOOL_BENCH", a=name, b=slot))

    # Play Trainer: define actions for each trainer with potential targets (Active=-1, Bench=0..4)
    # Target 0..4 = Bench, Target 5 = Active, Target 6 = None (self-targeting/global)
    for name in sorted(CARD_REGISTRY.keys()):
        cd = CARD_REGISTRY[name]
        if cd.supertype == "Trainer":
            # Special Handling for Discard Trainers (Learnt Discard)
            if name in ("Ultra Ball", "Superior Energy Retrieval", "Earthen Vessel"):
                continue

            for target in range(7):
                actions.append(Action("PLAY_TRAINER", a=name, b=target))
                
    # Learnt Discard Trainers
    # Ultra Ball: b=SearchType(0-5), c=Discard1(0-6), d=Discard2(0-6)
    # Limit discard indices to 0-4 (First 5 cards) to keep action space reasonable (~150 actions)
    # Note: If hand > 5, agent can only discard from first 5.
    for target in range(6): # Search Type
        for d1 in range(5):
             for d2 in range(d1+1, 5): # Unique pairs, d1 < d2
                 actions.append(Action("PLAY_TRAINER", a="Ultra Ball", b=target, c=d1, d=d2))
    
    # Superior Energy Retrieval: b=Discard1(0-4), c=Discard2(0-4)
    for d1 in range(5):
        for d2 in range(d1+1, 5):
             actions.append(Action("PLAY_TRAINER", a="Superior Energy Retrieval", b=d1, c=d2))
             
    # Earthen Vessel: b=Discard(0-4)
    for d1 in range(5):
        actions.append(Action("PLAY_TRAINER", a="Earthen Vessel", b=d1))
    
    # --- Card Selection Trainers ---
    # These trainers let the agent choose specific cards from deck or discard
    
    # Fighting Gong: Choose Energy (b=0) or Pokemon (b=1)
    if "Fighting Gong" in CARD_REGISTRY:
        actions.append(Action("PLAY_TRAINER", a="Fighting Gong", b=0))  # Energy
        actions.append(Action("PLAY_TRAINER", a="Fighting Gong", b=1))  # Pokemon
        # Note: b=6 (either) is already generated in the main trainer loop
    
    # Night Stretcher: Select 1 card from discard (first 15 indices)
    # b = discard pile index (0-14, representing last 15 cards)
    if "Night Stretcher" in CARD_REGISTRY:
        for disc_idx in range(15):
            actions.append(Action("PLAY_TRAINER", a="Night Stretcher", b=disc_idx))
    
    # Lana's Aid: Select up to 3 cards from discard (first 15 indices)
    # c, d, e = discard pile indices
    # Generate common patterns to keep action space reasonable
    if "Lana's Aid" in CARD_REGISTRY:
        # 1. Single card selections (15 actions)
        for c in range(15):
            actions.append(Action("PLAY_TRAINER", a="Lana's Aid", b=6, c=c))
        
        # 2. Two card selections (105 actions, but limit to 50 most common)
        # Prioritize nearby indices (recently discarded cards)
        two_card_combos = []
        for c in range(15):
            for d in range(c+1, 15):
                # Prioritize high indices (recent cards)
                priority = (c + d) / 2  # Average index
                two_card_combos.append((priority, c, d))
        
        # Take top 50 combinations
        two_card_combos.sort(reverse=True)
        for _, c, d in two_card_combos[:50]:
            actions.append(Action("PLAY_TRAINER", a="Lana's Aid", b=6, c=c, d=d))
        
        # 3. Three card selections (limit to 50 most common)
        # Prioritize high indices (recent cards)
        three_card_combos = []
        for c in range(15):
            for d in range(c+1, 15):
                for e in range(d+1, 15):
                    # Prioritize high indices
                    priority = (c + d + e) / 3
                    three_card_combos.append((priority, c, d, e))
        
        # Take top 50 combinations
        three_card_combos.sort(reverse=True)
        for _, c, d, e in three_card_combos[:50]:
            actions.append(Action("PLAY_TRAINER", a="Lana's Aid", b=6, c=c, d=d, e=e))
        
        # 4. Add fallback action (no selections = take last 3)
        actions.append(Action("PLAY_TRAINER", a="Lana's Aid", b=6))
    
    # Super Rod: Select up to 3 cards from discard to shuffle back
    # Same pattern as Lana's Aid
    if "Super Rod" in CARD_REGISTRY:
        # Single card
        for c in range(15):
            actions.append(Action("PLAY_TRAINER", a="Super Rod", b=6, c=c))
        
        # Two cards (top 30)
        two_card_combos = []
        for c in range(15):
            for d in range(c+1, 15):
                priority = (c + d) / 2
                two_card_combos.append((priority, c, d))
        two_card_combos.sort(reverse=True)
        for _, c, d in two_card_combos[:30]:
            actions.append(Action("PLAY_TRAINER", a="Super Rod", b=6, c=c, d=d))
        
        # Three cards (top 30)
        three_card_combos = []
        for c in range(15):
            for d in range(c+1, 15):
                for e in range(d+1, 15):
                    priority = (c + d + e) / 3
                    three_card_combos.append((priority, c, d, e))
        three_card_combos.sort(reverse=True)
        for _, c, d, e in three_card_combos[:30]:
            actions.append(Action("PLAY_TRAINER", a="Super Rod", b=6, c=c, d=d, e=e))
        
        # Fallback
        actions.append(Action("PLAY_TRAINER", a="Super Rod", b=6))
    
    # Buddy-Buddy Poffin: Select 2 Pokemon from deck (first 20 indices)
    # c, d = deck indices
    if "Buddy-Buddy Poffin" in CARD_REGISTRY:
        # Generate selections for indices 0-19 (top 20 cards in deck)
        # Limit to 60 combinations to keep action space reasonable
        poffin_combos = []
        for c in range(20):
            for d in range(c+1, 20):
                # Prioritize lower indices (top of shuffled deck)
                priority = -(c + d)  # Negative so lower indices are higher priority
                poffin_combos.append((priority, c, d))
        
        poffin_combos.sort(reverse=True)
        for _, c, d in poffin_combos[:60]:
            actions.append(Action("PLAY_TRAINER", a="Buddy-Buddy Poffin", b=6, c=c, d=d))
        
        # Fallback (take first 2 valid)
        actions.append(Action("PLAY_TRAINER", a="Buddy-Buddy Poffin", b=6))

    # Retreat: choose bench swap target
    for slot in range(max_bench):
        actions.append(Action("RETREAT_TO", b=slot))

    # Attack: support up to 2 attacks per Pokemon
    # Attack: support up to 2 attacks per Pokemon
    # b = Attack Index (0 or 1)
    # c = Target Slot (0-4 Bench, 5 Opponent Active)
    # Note: Actions without 'c' default to untargeted (usually targeting active)
    actions.append(Action("ATTACK", b=0))
    actions.append(Action("ATTACK", b=1))
    
    # Targeted Attacks (for Snipers like Fezandipiti)
    for target_slot in range(6): # 0-4 Bench, 5 Active
        actions.append(Action("ATTACK", b=0, c=target_slot))
        actions.append(Action("ATTACK", b=1, c=target_slot))
        
    # Magnitude Attacks (for Gholdengo ex "Make It Rain")
    # b = Attack Index, c = Amount (1-10)
    for amount in range(1, 11):
        actions.append(Action("ATTACK_MAGNITUDE", b=0, c=amount))
        actions.append(Action("ATTACK_MAGNITUDE", b=1, c=amount))

    # ability: keep generic “USE_ACTIVE_ABILITY” 
    # (Many abilities are passive or automatic, this is for activated ones like Munkidori)
    actions.append(Action("USE_ACTIVE_ABILITY"))
    
    # Targeted Abilities (for Dusknoir/Munkidori)
    for target_slot in range(6):
         actions.append(Action("USE_ACTIVE_ABILITY", c=target_slot))

    return actions


ACTION_TABLE: List[Action] = build_action_table()
ACTION_INDEX: Dict[Action, int] = {a: i for i, a in enumerate(ACTION_TABLE)}


# =============================================================================
# EFFECTS
# Source: tcg/effects.py
# =============================================================================

# Helper to check if we should print verbose output
def should_print():
    return os.environ.get('PTCG_QUIET') != '1'

# --- Helpers ---

def _search_and_bench(env: 'PTCGEnv', player: 'PlayerState', count: int, filter_func):
    """
    Find cards in actual 'deck', remove them, and put on bench.
    Cards benched via search are REVEALED to opponent (restricted search).
    """
    gs = env._gs
    opponent = gs.players[1] if player == gs.players[0] else gs.players[0]
    
    for _ in range(count):
        # find empty slot
        slot_idx = -1
        for i, s in enumerate(player.bench):
            if s.name is None:
                slot_idx = i
                break
        
        if slot_idx >= 0:
            # Find in deck
            found_idx = -1
            for i, c_name in enumerate(player.deck):
                cd = card_def(c_name)
                if filter_func(cd):
                    found_idx = i
                    break
            
            if found_idx >= 0:
                card = player.deck.pop(found_idx)
                player.bench[slot_idx].name = card
                player.bench[slot_idx].turn_played = env._gs.turn_number
                player.bench[slot_idx].energy = []
                player.bench[slot_idx].damage = 0
                
                # REVEAL TO OPPONENT (benching via search is always restricted)
                opponent.known_hand.append(card)  # Track what they're setting up
                player.last_searched_pokemon = card
                player.last_searched_type = card_def(card).subtype
                
                if player == env._gs.players[0]:
                    if should_print():
                        print(f"    -> Search & Bench: \033[96m{card}\033[0m (revealed)")
                random.shuffle(player.deck)
        else:
             if player == env._gs.players[0]:
                 if should_print():
                     print(f"    -> Search & Bench: \033[90mNothing found\033[0m")

def _search_to_hand(env: 'PTCGEnv', player: 'PlayerState', count: int, type_filter=None):
    """
    Search deck for specific card type, remove, add to hand.
    
    If type_filter is provided (restricted search like Ultra Ball), the card is 
    revealed to the opponent and tracked in their known_hand.
    If type_filter is None (unrestricted like Quick Search), card is NOT revealed.
    """
    gs = env._gs
    opponent = gs.players[1] if player == gs.players[0] else gs.players[0]
    
    for _ in range(count):
        found_idx = -1
        for i, c_name in enumerate(player.deck):
            cd = card_def(c_name)
            if type_filter:
                if callable(type_filter):
                    if type_filter(cd):
                        found_idx = i
                        break
                elif cd.supertype == type_filter:
                    found_idx = i
                    break
            else:
                # No filter means any card (unrestricted search)
                found_idx = i
                break
        
        if found_idx >= 0:
            card = player.deck.pop(found_idx)
            player.hand.append(card)
            
            # REVEAL TO OPPONENT if search was restricted (has type_filter)
            if type_filter is not None:
                opponent.known_hand.append(card)
                # Also update last_searched tracking
                cd = card_def(card)
                if cd.supertype == "Pokemon":
                    player.last_searched_pokemon = card
                    player.last_searched_type = cd.subtype
            
            if player == env._gs.players[0]:
                if should_print():
                    revealed = "(revealed)" if type_filter else "(hidden)"
                    print(f"    -> Search & Draw: \033[96m{card}\033[0m {revealed}")
            random.shuffle(player.deck)
        else:
            if player == env._gs.players[0]:
                if should_print():
                    print(f"    -> Search & Draw: \033[90mNothing found\033[0m")

def _card_score(card_name, hand):
    cd = card_def(card_name)
    score = 50 # Default generic score
    
    # Priority 1: Basic Energy (High priority to discard, usually recoverable)
    if cd.supertype == "Energy" and cd.subtype == "Basic":
        return 0
        
    # Priority 2: Duplicates
    count = hand.count(card_name)
    if count >= 3: return 1
    if count == 2: return 5
    
    # Priority 3: Type based
    if cd.supertype == "Pokemon":
        if cd.tags and "ex" in cd.tags:
            return 100 # Keep ex
        if cd.subtype == "Stage2":
            return 45
        return 30
    
    if cd.subtype == "Tool":
        return 10
    
    if cd.subtype == "Stadium":
        return 15
        
    if cd.name == "Rare Candy":
        return 90
        
    return score

def _discard_from_hand(player: 'PlayerState', count: int):
    # Smart Discard: Discard 'count' lowest-value cards
    for _ in range(count):
        if not player.hand: break
        
        # Scored list: (score, index, name)
        scored = []
        for i, c in enumerate(player.hand):
            # Recalculate score each time as hand changes (duplicates count)
            s = _card_score(c, player.hand)
            scored.append((s, i, c))
            
        # Sort by score ascending (lowest first)
        scored.sort(key=lambda x: x[0])
        
        # Pop lowest (Must pop by index, so find index in current hand)
        # We grabbed index 'i' but sorting changes order? No, i is index in original list.
        # But popping invalidates indices.
        # Safer: Find item by value again or pop using the index from the enumerate IF we assume list order didn't change during loop?
        # Actually, simpler: Just find the card name to remove? No, duplicate names exist.
        # Best: Identify the target index among current hand.
        target_idx = scored[0][1]
        
        # Pop
        c = player.hand.pop(target_idx)
        player.discard_pile.append(c)

def _switch_opponent_active(env: 'PTCGEnv', op: 'PlayerState'):
    # Force switch opponent active with a bencher.
    # Pick first available bencher (simple deterministic logic)
    benchers = [i for i, s in enumerate(op.bench) if s.name]
    if benchers:
        target = benchers[0] # Deterministic
        # Clear status
        op.active.status = {k:False for k in op.active.status}
        op.active, op.bench[target] = op.bench[target], op.active

def apply_energy_effect(env: 'PTCGEnv', player_idx: int, energy_name: str, attached_to_active: bool, bench_slot: int = None):
    """
    Apply special energy attachment effects.
    Called after energy is attached to a Pokemon.
    
    Args:
        env: Game environment
        player_idx: Index of player attaching energy
        energy_name: Name of energy card
        attached_to_active: True if attached to active, False if to bench
        bench_slot: Bench slot index if attached to bench
    """
    me = env._gs.players[player_idx]
    
    if energy_name == "Enriching Energy":
        # Draw 4 cards when attached
        env._draw_cards(me, 4)
        if me == env._gs.players[0] and should_print():
            print(f"    -> Enriching Energy: Drew 4 cards")
    
    elif energy_name == "Jet Energy":
        # When attached to bench, switch that Pokemon to active
        if not attached_to_active and bench_slot is not None:
            # Check if there's a Pokemon to switch with
            if me.bench[bench_slot].name:
                me.active, me.bench[bench_slot] = me.bench[bench_slot], me.active
                if me == env._gs.players[0] and should_print():
                    print(f"    -> Jet Energy: Switched {me.active.name} to active")

def apply_trainer_effect(env: 'PTCGEnv', player_idx: int, card_name: str, target_idx: int = 6, secondary_idx: int = None, tertiary_idx: int = None, quaternary_idx: int = None, quinary_idx: int = None):
    """
    Apply effect of a Trainer card. 
    Assumes card has already been played/consumed from hand.
    """
    gs = env._gs
    me = gs.players[player_idx]
    op = gs.players[1 - player_idx]
    
    # --- ITEMS ---
    if card_name == "Buddy-Buddy Poffin":
        # Search deck for up to 2 Basic Pokemon with <= 70 HP and bench them
        # Agent selects specific Pokemon via secondary_idx (c) and tertiary_idx (d)
        
        # Get valid candidates (Basic Pokemon with HP <= 70)
        candidates = [i for i, c in enumerate(me.deck)
                     if card_def(c).supertype == "Pokemon" and 
                        card_def(c).subtype == "Basic" and 
                        card_def(c).hp <= 70]
        
        # Parse selections: c, d from action parameters
        selected_indices = []
        for sel_idx in [secondary_idx, tertiary_idx]:  # c, d
            if sel_idx is not None and sel_idx in candidates and sel_idx not in selected_indices:
                selected_indices.append(sel_idx)
        
        # If no selections, use fallback (first 2 valid)
        if not selected_indices:
            selected_indices = candidates[:2]
        
        # Bench selected Pokemon (remove from deck, high to low to preserve indices)
        for deck_idx in sorted(selected_indices, reverse=True):
            card = me.deck.pop(deck_idx)
            # Find empty bench slot
            benched = False
            for bench_idx in range(5):
                if not me.bench[bench_idx].name:
                    me.bench[bench_idx].name = card
                    me.bench[bench_idx].damage = 0
                    me.bench[bench_idx].energy = []
                    me.bench[bench_idx].turn_played = gs.turn_number
                    me.bench[bench_idx].tool = None
                    me.bench[bench_idx].status = {k: False for k in me.bench[bench_idx].status}
                    benched = True
                    if me == env._gs.players[0] and should_print():
                        print(f"    -> Buddy-Buddy Poffin: Benched {card}")
                    break
            
            if not benched:
                # No bench space, put back in deck
                me.deck.append(card)
        
        random.shuffle(me.deck)
        
    elif card_name == "Rare Candy":
        # Use target_idx: 0-4 Bench, 5 Active
        target = None
        if target_idx == 5: target = me.active
        elif 0 <= target_idx < 5: target = me.bench[target_idx]
        
        if target and target.name:
            # Find a Stage 2 in hand that evolves from this Basic
            stage2_name = None
            for c in me.hand:
                if c == "Rare Candy": continue
                cd_e = card_def(c)
                if cd_e.supertype == "Pokemon" and cd_e.subtype == "Stage2":
                    s1 = cd_e.evolves_from
                    if s1:
                        s1_def = card_def(s1)
                        if s1_def.evolves_from == target.name or s1 == target.name:
                            stage2_name = c; break
            
            if stage2_name:
                me.hand.remove(stage2_name)
                target.name = stage2_name
                target.turn_played = gs.turn_number
                apply_on_evolve_ability(env, player_idx, stage2_name)
        
    elif card_name == "Ultra Ball":
        if len(me.hand) >= 2:
            # Learnt Discard: Use secondary_idx (c) and tertiary_idx (d)
            if secondary_idx is not None and tertiary_idx is not None:
                # Indices in current hand (post-play)
                # Sort descending to pop correctly
                to_discard = sorted([secondary_idx, tertiary_idx], reverse=True)
                # Validate bounds (safety)
                if to_discard[0] < len(me.hand):
                     for idx in to_discard:
                         c = me.hand.pop(idx)
                         me.discard_pile.append(c)
                else:
                     # Fallback (should be masked out)
                     _discard_from_hand(me, 2)
            else:
                 _discard_from_hand(me, 2)
            
            # Target 6 = intentional fail (hand thinning)
            if target_idx == 6:
                if me == env._gs.players[0]:
                    if should_print():
                        print(f"    -> Ultra Ball: Intentionally failed search (hand thinning)")
            else:
                # Target-based search (agent chooses what to search for)
                # 0 = Basic, 1 = Stage1, 2 = Stage2, 3 = Evo of active, 
                # 4 = Evo of bench, 5 = Key attacker
                def ultra_ball_filter(cd):
                    if cd.supertype != "Pokemon":
                        return False
                        
                    if target_idx == 0:
                        return cd.subtype == "Basic"
                    elif target_idx == 1:
                        return cd.subtype == "Stage1"
                    elif target_idx == 2:
                        return cd.subtype == "Stage2"
                    elif target_idx == 3:
                        # Evolution of active
                        return me.active.name and cd.evolves_from == me.active.name
                    elif target_idx == 4:
                        # Evolution of any bench
                        for slot in me.bench:
                            if slot.name and cd.evolves_from == slot.name:
                                return True
                        return False
                    elif target_idx == 5:
                        # Key attackers
                        return cd.name in ("Alakazam", "Charizard ex", "Pidgeot ex", "Dudunsparce")
                    else:
                        # Fallback - any Pokemon
                        return True
                
                _search_to_hand(env, me, 1, type_filter=ultra_ball_filter)
        
    elif card_name == "Nest Ball":
        # Target-based search for basic Pokemon to bench
        # 0 = Evolution starters, 1 = Support, 2 = Tech, 3 = Any
        evo_starters = ("Abra", "Charmander", "Pidgey", "Charcadet")
        support_mons = ("Fan Rotom", "Dunsparce")
        tech_mons = ("Fezandipiti ex", "Tatsugiri", "Psyduck")
        
        def nest_ball_filter(cd):
            if cd.supertype != "Pokemon" or cd.subtype != "Basic":
                return False
            if target_idx == 0:
                return cd.name in evo_starters
            elif target_idx == 1:
                return cd.name in support_mons
            elif target_idx == 2:
                return cd.name in tech_mons
            else:  # target_idx == 3 or default
                return True
        
        _search_and_bench(env, me, count=1, filter_func=nest_ball_filter)
        
    elif card_name == "Super Rod":
        # Shuffle up to 3 Pokemon/Basic Energy from discard back to deck
        # Agent selects specific cards via secondary_idx (c), tertiary_idx (d), quaternary_idx (e)
        
        # Get valid candidates
        candidates = [i for i, c in enumerate(me.discard_pile) 
                      if card_def(c).supertype == "Pokemon" or 
                         (card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic")]
        
        # Parse selections: c, d, e from action parameters
        selected_indices = []
        for sel_idx in [secondary_idx, tertiary_idx, quaternary_idx]:  # c, d, e
            if sel_idx is not None and sel_idx in candidates and sel_idx not in selected_indices:
                selected_indices.append(sel_idx)
        
        # If no selections provided, use fallback (last 3)
        if not selected_indices:
            selected_indices = candidates[-3:]
        
        # Shuffle selected cards back to deck (remove from discard, high to low to preserve indices)
        for idx in sorted(selected_indices, reverse=True):
            card = me.discard_pile.pop(idx)
            me.deck.append(card)
            if me == env._gs.players[0] and should_print():
                print(f"    -> Super Rod: Shuffled back {card}")
        
        random.shuffle(me.deck)
        
    elif card_name == "Night Stretcher":
        # Put 1 Pokemon or Basic Energy from discard to hand
        # Agent selects specific card via target_idx (b) = discard pile index
        
        # Get valid candidates (Pokemon or Basic Energy)
        candidates = [i for i, c in enumerate(me.discard_pile)
                     if card_def(c).supertype == "Pokemon" or 
                        (card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic")]
        
        # If target_idx is valid candidate index, use it directly
        if target_idx in candidates:
            idx = target_idx
            card = me.discard_pile.pop(idx)
            me.hand.append(card)
            if me == env._gs.players[0] and should_print():
                print(f"    -> Night Stretcher: Recovered {card}")
        elif candidates:
            # Fallback: take last valid card
            idx = candidates[-1]
            card = me.discard_pile.pop(idx)
            me.hand.append(card)
            if me == env._gs.players[0] and should_print():
                print(f"    -> Night Stretcher: Recovered {card} (fallback)")
            
    elif card_name in ("Counter Catcher", "Boss's Orders"):
        # Counter Catcher check: Must be behind on prizes
        if card_name == "Counter Catcher":
            if me.prizes_taken >= op.prizes_taken:
                if me == env._gs.players[0] and should_print():
                    print(f"    -> Counter Catcher: ❌ Not behind on prizes")
                return

        # Target opponent's bench member to switch to active
        switched = False
        if 0 <= target_idx < 5:
            target = op.bench[target_idx]
            if target and target.name:
                # Clear status of current active before moving to bench
                op.active.status = {k:False for k in op.active.status}
                
                op.active, op.bench[target_idx] = op.bench[target_idx], op.active
                switched = True
        
        if not switched:
            # Fallback for generic actions or missing targets
            _switch_opponent_active(env, op)
            
    elif card_name == "Enhanced Hammer":
        # Discard a Special Energy from opponent's active
        special_indices = [i for i, e in enumerate(op.active.energy) 
                          if card_def(e).subtype == "Special"]
        if special_indices:
            # Discard first special energy found
            idx = special_indices[0]
            discarded = op.active.energy.pop(idx)
            op.discard_pile.append(discarded)
            if me == env._gs.players[0] and should_print():
                print(f"    -> Enhanced Hammer: Discarded {discarded}")
        else:
            if me == env._gs.players[0] and should_print():
                print(f"    -> Enhanced Hammer: No Special Energy to discard")
            
    elif card_name == "Wondrous Patch":
        # Attach Basic Psychic Energy from discard to Benched Psychic Pokemon.
        # Robust check for Basic Psychic Energy
        psychic_energies = [i for i, c in enumerate(me.discard_pile) 
                            if card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic" and card_def(c).type == "Psychic"]
                            
        targets = [b for b in me.bench if b.name and card_def(b.name).type == "Psychic"]
        
        if psychic_energies and targets:
             idx = psychic_energies[0]
             # Note: We take the specific card found at that index.
             # If multiple exist, we take the first found (which is early in discard).
             # Usually in TCG you pick, but picking first available is fine for agent.
             card_to_attach = me.discard_pile.pop(idx)
             
             target = targets[0] # Default to first
             # Intelligent selection: Prioritize active if it needs energy to attack
             if me.active in targets:
                 active_def = card_def(me.active.name)
                 if active_def.attacks and not env._check_energy(me.active.energy, active_def.attacks[0].cost):
                     target = me.active
            
             target.energy.append(card_to_attach)
             if me == env._gs.players[0]:
                 if should_print():
                     print(f"    -> Wondrous Patch: Attached \033[96m{card_to_attach}\033[0m to \033[96m{target.name}\033[0m")
        else:
             if me == env._gs.players[0]:
                 if should_print():
                     print(f"    -> Wondrous Patch: \033[90mFailed (No valid target/energy)\033[0m")
            
    # --- SUPPORTERS ---
    elif card_name == "Professor's Research":
        # Discard hand, Draw 7.
        while me.hand:
            me.discard_pile.append(me.hand.pop())
        env._draw_cards(me, 7)
        
    elif card_name == "Iono":
        # Shuffle hand to bottom. Draw = Prizes Remaining.
        # Check condition: "If either player put any cards on the bottom of their deck in this way"
        any_moved = False
        
        # Me
        if me.hand:
            any_moved = True
            for c in me.hand:
                me.deck.insert(0, c) # Insert at 0 (bottom)
            me.hand = []
            
        # Op
        if op.hand:
            any_moved = True
            for c in op.hand:
                op.deck.insert(0, c)
            op.hand = []
            
        if any_moved:
            draw_me = 6 - me.prizes_taken
            env._draw_cards(me, max(1, draw_me))
            
            draw_op = 6 - op.prizes_taken
            env._draw_cards(op, max(1, draw_op))
        else:
             if me == env._gs.players[0]:
                 if should_print():
                     print(f"    -> Iono: \033[90mNo cards moved to bottom, no draw triggered.\033[0m")
        
    elif card_name == "Arven":
        # Search Item and Tool.
        _search_to_hand(env, me, 1, type_filter=lambda c: c.subtype == "Item") # Item
        _search_to_hand(env, me, 1, type_filter=lambda c: c.subtype == "Tool") # Tool
        
    elif card_name == "Lillie's Determination":
        count = 8 if me.prizes_taken == 0 else 6 # Note: prizes_taken is usually 0 at start. Logic: "If you have exactly 6 Prize cards remaining"
        # 6 prizes remaining means 0 taken.
        # Shuffle hand into deck
        me.deck.extend(me.hand)
        me.hand = []
        random.shuffle(me.deck)
        env._draw_cards(me, count)
        
    elif card_name == "Hilda":
        # Search 1 Evolution Pokemon (Stage 1 or Stage 2)
        _search_to_hand(env, me, 1, type_filter=lambda c: c.supertype == "Pokemon" and c.subtype in ("Stage1", "Stage2"))
        # Search 1 Energy
        _search_to_hand(env, me, 1, type_filter="Energy")
        # Ensure shuffle
        random.shuffle(me.deck)
        
    elif card_name == "Dawn":
        # Search 1 Basic, 1 Stage 1, 1 Stage 2
        _search_to_hand(env, me, 1, type_filter=lambda c: c.supertype == "Pokemon" and c.subtype == "Basic")
        _search_to_hand(env, me, 1, type_filter=lambda c: c.supertype == "Pokemon" and c.subtype == "Stage1")
        _search_to_hand(env, me, 1, type_filter=lambda c: c.supertype == "Pokemon" and c.subtype == "Stage2") 
        
    elif card_name == "Tulip":
        # Recover 4 Psychic from discard.
        candidates = [i for i, c in enumerate(me.discard_pile) if c == "Basic Psychic Energy" or (card_def(c).supertype == "Pokemon" and card_def(c).type == "Psychic")]
        for idx in sorted(candidates[-4:], reverse=True):
            me.hand.append(me.discard_pile.pop(idx))
        if me == env._gs.players[0]:
            if should_print():
                print(f"    -> Tulip Recovered: \033[96m{len(candidates[-4:])} cards\033[0m")

    elif card_name == "Artazon":
        me.stadium = card_name
        _search_and_bench(env, me, 1, filter_func=lambda c: c.supertype == "Pokemon" and c.subtype == "Basic" and not c.has_rule_box)
        
    elif card_name == "Battle Cage":
        # Effect: Prevent all damage counters from being placed on Benched Pokémon 
        # (both yours and opponent's) by effects of attacks and Abilities from opponent's Pokémon.
        # Note: The actual prevention is checked in apply_attack_effect and apply_ability_effect
        # by checking if either player's stadium == "Battle Cage"
        me.stadium = card_name
        # Also set as active stadium on game state for easy checking
        env._gs.active_stadium = card_name

    elif card_name == "Technical Machine: Evolution":
        pass

    elif card_name == "Bill": # Testing
        env._draw_cards(me, 2)
    
    elif card_name == "Energy Search":
        # Search deck for a Basic Energy card, put it into hand
        _search_to_hand(env, me, 1, type_filter=lambda c: c.supertype == "Energy" and c.subtype == "Basic")
    
    elif card_name == "Professor Turo's Scenario":
        # Target 0-4 Bench, 5 Active
        target = None
        if target_idx == 5: target = me.active
        elif 0 <= target_idx < 5: target = me.bench[target_idx]
        
        if target and target.name:
            me.hand.append(target.name)
            # Discard all attached energy
            me.discard_pile.extend(target.energy)
            # Infer pre-evolution rescue if possible (Heuristic)
            cd = card_def(target.name)
            if cd.evolves_from:
                # If we are returning an evolution to hand, effectively we are also returning the pre-evolution?
                # TCG Rules: Base goes to hand too.
                # Sim limitation: "target.name" is just the top card.
                # Hack: Create the pre-evolution card in hand.
                me.hand.append(cd.evolves_from)
                
            # Reset slot
            if target_idx == 5:
                # Need to promote from bench
                me.active = PokemonSlot()
                for i, s in enumerate(me.bench):
                    if s.name:
                        me.active, me.bench[i] = me.bench[i], PokemonSlot()
                        break
            else:
                me.bench[target_idx] = PokemonSlot()
    
    elif card_name == "Unfair Stamp":
        # Only usable if your Pokemon was KO'd last turn (checked in action mask ideally)
        # Each player shuffles hand into deck, you draw 5, opponent draws 2
        if gs.ko_last_turn:
            # Shuffle my hand into deck
            me.deck.extend(me.hand)
            random.shuffle(me.deck)
            me.hand = []
            env._draw_cards(me, 5)
            
            # Shuffle opponent's hand into deck
            op.deck.extend(op.hand)
            random.shuffle(op.deck)
            op.hand = []
            env._draw_cards(op, 2)
            
             
             
    elif card_name == "Earthen Vessel":
        # Cost: Discard 1. Use target_idx.
        if len(me.hand) >= 1:
            if target_idx < len(me.hand):
                # Specific discard
                c = me.hand.pop(target_idx)
                me.discard_pile.append(c)
                if me == env._gs.players[0] and should_print():
                    print(f"    -> Earthen Vessel: Discarded {c}")
            else:
                _discard_from_hand(me, 1) # Fallback
            
            # Search 2 Basic Energy
            _search_to_hand(env, me, 2, type_filter=lambda c: c.supertype == "Energy" and c.subtype == "Basic")
            
    elif card_name == "Superior Energy Retrieval":
        # Cost: Discard 2. Use indices b/c (target_idx, secondary_idx).
        if len(me.hand) >= 2:
            if secondary_idx is not None:
                # Wait, Env passed b and c. b=target_idx, c=secondary_idx.
                to_discard = sorted([target_idx, secondary_idx], reverse=True)
                if to_discard[0] < len(me.hand):
                     for idx in to_discard:
                         me.discard_pile.append(me.hand.pop(idx))
                else:
                     _discard_from_hand(me, 2)
            else:
                _discard_from_hand(me, 2)
        
        # Recover 4 Basic Energy
        # Logic: Find up to 4, move to hand.
        candidates = [i for i, c in enumerate(me.discard_pile) if card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic"]
        
        # Recover last 4
        to_recover = sorted(candidates[-4:], reverse=True)
        for idx in to_recover:
            me.hand.append(me.discard_pile.pop(idx))
            
        if me == env._gs.players[0] and should_print():
             print(f"    -> Superior Energy Retrieval: Recovered {len(to_recover)} energies")

    elif card_name == "Premium Power Pro":
        # +30 damage for Fighting Pokemon this turn
        me.fighting_buff = True
        if me == env._gs.players[0] and should_print():
             print(f"    -> Premium Power Pro: +30 Fighting Dmg this turn!")
             
    elif card_name == "Fighting Gong":
        # Search Basic [F] Energy OR Basic [F] Pokemon.
        # Agent chooses via target_idx: 0 = Energy, 1 = Pokemon
        
        if target_idx == 0:
            # Search for Basic Fighting Energy
            def gong_filter(c):
                return c.supertype == "Energy" and c.subtype == "Basic" and c.type == "Fighting"
        elif target_idx == 1:
            # Search for Basic Fighting Pokemon
            def gong_filter(c):
                return c.supertype == "Pokemon" and c.subtype == "Basic" and c.type == "Fighting"
        else:
            # Fallback: take either (first valid found)
            def gong_filter(c):
                is_f_energy = (c.supertype == "Energy" and c.subtype == "Basic" and c.type == "Fighting")
                is_f_poke = (c.supertype == "Pokemon" and c.subtype == "Basic" and c.type == "Fighting")
                return is_f_energy or is_f_poke
        
        _search_to_hand(env, me, 1, type_filter=gong_filter)
        if me == env._gs.players[0] and should_print():
            choice = "Energy" if target_idx == 0 else "Pokemon" if target_idx == 1 else "Either"
            print(f"    -> Fighting Gong: Searching for {choice}")
        
    elif card_name == "Lana's Aid":
        # Recover up to 3 Non-Rule-Box Pokemon/Basic Energy from discard
        # Agent selects specific cards via secondary_idx (c), tertiary_idx (d), quaternary_idx (e)
        
        # Get valid candidates
        candidates = [i for i, c in enumerate(me.discard_pile) 
                      if (card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic") or
                         (card_def(c).supertype == "Pokemon" and not card_def(c).has_rule_box)]
        
        # Parse selections: c, d, e from action parameters
        selected_indices = []
        for sel_idx in [secondary_idx, tertiary_idx, quaternary_idx]:  # c, d, e
            if sel_idx is not None and sel_idx in candidates and sel_idx not in selected_indices:
                selected_indices.append(sel_idx)
        
        # If no selections provided, use fallback (last 3)
        if not selected_indices:
            selected_indices = candidates[-3:]
        
        # Recover selected cards (remove from discard, high to low to preserve indices)
        for idx in sorted(selected_indices, reverse=True):
            card = me.discard_pile.pop(idx)
            me.hand.append(card)
            if me == env._gs.players[0] and should_print():
                print(f"    -> Lana's Aid: Recovered {card}")
             
    elif card_name == "Prime Catcher":
        # Switch Opponent Active with Bench (Targeted)
        # AND Switch Self Active with Bench (Targeted?)
        # Card says: "Switch in 1 of your opponent's Benched... If you do, switch your Active..."
        # Logic: Target idx is Opponent Bench. Self switch needs a second target?
        # Sim limitation: We only have 1 target index.
        # Heuristic: Agent targets opponent to switch in. We AUTO-SWITCH self with first available or random?
        # Or: Use target_idx for Op, and random/smart self switch.
        
        # 1. Switch Opponent
        switched_op = False
        if 0 <= target_idx < 5 and op.bench[target_idx].name:
             op.active, op.bench[target_idx] = op.bench[target_idx], op.active
             switched_op = True
        elif not op.active.name:
             pass # Op active empty? rare
        else:
             # Fallback
             _switch_opponent_active(env, op)
             switched_op = True
             
        # 2. Switch Self if Op switched
        if switched_op:
             # Just swap with first bencher for now (Agent control limitation)
             # Ideally we need Action.c for self target, but we used it for attacks.
             # Trainers usually act on `target_idx`.
             benchers = [i for i, s in enumerate(me.bench) if s.name]
             if benchers:
                 t = benchers[0]
                 me.active, me.bench[t] = me.bench[t], me.active
                 if me == env._gs.players[0] and should_print():
                     print(f"    -> Prime Catcher: Swapped both actives!")


def apply_ability_effect(env: 'PTCGEnv', player_idx: int, pokemon_name: str, target_idx: int = 6):
    """
    Trigger result of 'USE_ACTIVE_ABILITY' or similar, based on Active Pokemon.
    """
    gs = env._gs
    me = gs.players[player_idx]
    op = gs.players[1 - player_idx]  # Opponent reference needed for some abilities
    
    if pokemon_name == "Pidgeot ex": # Quick Search
        # Check limit
        if me.quick_search_used:
            return
        if me == env._gs.players[0]:
            if should_print():
                print(f"    -> Ability: \033[96mQuick Search\033[0m")
        # Search 1 card
        _search_to_hand(env, me, 1)
        me.quick_search_used = True
        

        
    elif pokemon_name == "Tatsugiri": # Attract Customers
        # Must be in the Active Spot
        if me.active.name != "Tatsugiri":
            if me == env._gs.players[0] and should_print():
                print(f"    -> Tatsugiri: ❌ Must be Active to use Attract Customers")
            return
        
        # Reveal top 6 cards, put a Supporter into hand, shuffle ONLY the other revealed cards back.
        # Implementation: Check top 6
        top_6 = me.deck[-6:]
        found_idx = -1
        for i, c in enumerate(reversed(top_6)): # Look from top down
            if card_def(c).subtype == "Supporter":
                found_idx = len(me.deck) - 1 - i
                break
        
        if found_idx >= 0:
            card = me.deck.pop(found_idx)
            me.hand.append(card)
            if me == env._gs.players[0]:
                if should_print():
                    print(f"    -> Tatsugiri found: \033[96m{card}\033[0m")
            # Shuffle remaining 5? Technically reveal top 6, take 1, shuffle others.
            # Sim: Just shuffle whole deck minus the 1 taken.
            random.shuffle(me.deck)
        else:
            if me == env._gs.players[0]:
                if should_print():
                    print(f"    -> Tatsugiri: \033[90mNo Supporter in top 6.\033[0m")
            random.shuffle(me.deck)
        
    elif pokemon_name == "Dudunsparce": # Run Away Draw
        if me == env._gs.players[0]:
            if should_print():
                print(f"    -> Ability: \033[96mRun Away Draw\033[0m")
        # Draw 3, shuffle self into deck.
        env._draw_cards(me, 3)
        # Infer pre-evolution refund
        # If Dudunsparce shuffles in, the Dunsparce underneath also goes to deck.
        # We manually add "Dunsparce" to deck to simulate this.
        me.deck.append("Dunsparce")
        random.shuffle(me.deck)
        
        # Shuffle active into deck
        if me.active.name == "Dudunsparce":
            me.deck.append("Dudunsparce")
            # simplified: assume energies return to deck as strings. 
            me.deck.extend(me.active.energy)
            random.shuffle(me.deck)
            
            me.active = PokemonSlot() # Open spot
            # Auto-promote first bench:
            for i, s in enumerate(me.bench):
                if s.name:
                    me.active, me.bench[i] = me.bench[i], PokemonSlot()
                    break

    elif pokemon_name == "Fan Rotom": # Fan Call
        # First turn for each player
        is_first_turn = (gs.turn_number == 1 if player_idx == 0 else gs.turn_number == 2)
        if is_first_turn:
            if me == env._gs.players[0]:
                if should_print():
                    print(f"    -> Ability: \033[96mFan Call\033[0m")
            _search_to_hand(env, me, 3, type_filter=lambda c: c.supertype == "Pokemon" and c.type == "Colorless" and c.hp <= 100)
        else:
             if me == env._gs.players[0]:
                 if should_print():
                     print(f"    -> Ability: \033[90mFan Call Failed (Not First Turn)\033[0m")
        
    elif pokemon_name == "Munkidori":
         # Adrena-Brain: If this Pokemon has any Darkness Energy attached, you can move 3 damage counters.
         if any(card_def(e).type == "Darkness" for e in me.active.energy):
             # Move 30 damage from one of your Pokemon to one of your opponent's Pokemon.
             
             # Source: Smart heuristic (Most damaged friendly)
             best_source = me.active
             max_dmg = me.active.damage
             for b in me.bench:
                 if b.name and b.damage > max_dmg:
                     max_dmg = b.damage
                     best_source = b
             
             if max_dmg > 0:
                 # Destination: Agent Selected Target
                 if target_idx == 6: target_idx = 5 # Default to Active
                 target = None
                 if target_idx == 5: target = op.active
                 elif 0 <= target_idx < 5: target = op.bench[target_idx]
                 
                 if target and target.name:
                     # BATTLE CAGE CHECK
                     is_protected = False
                     # "Prevent all damage counters... on Benched Pokemon by effects calls... from Opponent"
                     if env._gs.active_stadium == "Battle Cage":
                         if target != op.active: # Bench
                             is_protected = True
                     
                     if is_protected:
                         if me == env._gs.players[0] and should_print():
                             print(f"    -> Munkidori blocked by Battle Cage!")
                     else:
                         amount = min(30, max_dmg)
                         best_source.damage -= amount
                         target.damage += amount
                         
                         if target == op.active: # Check op KO
                             op_hp = card_def(op.active.name).hp
                             if op.active.damage >= op_hp:
                                 env._handle_knockout(op)
                         # Note: Bench KO handled in strict mode or environment checkup
        
    elif pokemon_name == "Lunatone": # Lunar Cycle (Replacing Sun Energy logic)
        # Check: Solrock in play
        solrock_in_play = any(s.name == "Solrock" for s in me.bench) or me.active.name == "Solrock"
        if not solrock_in_play: return
        
        # Check: Discard F energy from hand
        f_indices = [i for i, c in enumerate(me.hand) if c in ("Basic Fighting Energy", "Fighting Energy")]
        if not f_indices: return
        
        # Execute
        card = me.hand.pop(f_indices[0])
        me.discard_pile.append(card)
        
        if me == env._gs.players[0] and should_print():
             print(f"    -> Ability: \033[96mLunar Cycle\033[0m (Discarded {card}, Draw 3)")
        env._draw_cards(me, 3)

    elif pokemon_name == "Genesect ex": # Metallic Signal
        # Search 2 Evolution [M] Pokemon
        if me == env._gs.players[0] and should_print():
             print(f"    -> Ability: \033[96mMetallic Signal\033[0m")
        _search_to_hand(env, me, 2, type_filter=lambda c: c.supertype == "Pokemon" and c.subtype in ("Stage1", "Stage2") and c.type == "Metal")
        random.shuffle(me.deck)

def _has_psyduck_damp_active(env: 'PTCGEnv') -> bool:
    """
    Check if Psyduck's Damp ability is active.
    Damp: Pokémon in play lose any Ability that requires the Pokémon using it to Knock Out itself.
    """
    gs = env._gs
    # Check both players for Psyduck in play
    for player in gs.players:
        all_pokemon = [player.active] + [s for s in player.bench if s.name]
        for slot in all_pokemon:
            if slot.name == "Psyduck":
                return True
    return False


def apply_ability_effect(env: 'PTCGEnv', player_idx: int, pokemon_name: str, target_idx: int = 6):
    """
    Apply an ability effect for a Pokemon.
    """
    gs = env._gs
    me = gs.players[player_idx]
    op = gs.players[1 - player_idx]
    
    # Check for Pidgeot ex ability
    if pokemon_name == "Pidgeot ex": # Quick Search
        if me == env._gs.players[0]:
             if should_print():
                 print(f"    -> Ability: \033[96mQuick Search\033[0m")
        # Search for any card
        _search_to_hand(env, me, 1)
        random.shuffle(me.deck)
    

     
    elif pokemon_name == "Fezandipiti ex": # Flip the Script
         if env._gs.ko_last_turn:
             if me == env._gs.players[0]:
                 if should_print():
                     print(f"    -> Ability: \033[96mFlip the Script\033[0m")
             env._draw_cards(me, 3)
         else:
              if me == env._gs.players[0]:
                  if should_print():
                      print(f"    -> Flip the Script: \033[90mNo KO last turn\033[0m")
     
    elif pokemon_name == "Tatsugiri": # Attract Customers
        # Must be in the Active Spot
        if me.active.name != "Tatsugiri":
            if me == env._gs.players[0] and should_print():
                print(f"    -> Tatsugiri: ❌ Must be Active to use Attract Customers")
            return
        
        # Reveal top 6 cards, put a  Supporter into hand, shuffle ONLY the other revealed cards back.
        # Implementation: Check top 6
        top_6 = me.deck[-6:]
        found_idx = -1
        for i, c in enumerate(reversed(top_6)): # Look from top down
            if card_def(c).subtype == "Supporter":
                found_idx = len(me.deck) - 1 - i
                break
        
        if found_idx >= 0:
            card = me.deck.pop(found_idx)
            me.hand.append(card)
            if me == env._gs.players[0]:
                if should_print():
                    print(f"    -> Tatsugiri found: \033[96m{card}\033[0m")
            # Shuffle remaining 5? Technically reveal top 6, take 1, shuffle others.
            # Sim: Just shuffle whole deck minus the 1 taken.
            random.shuffle(me.deck)
        else:
            if me == env._gs.players[0]:
                if should_print():
                    print(f"    -> Tatsugiri: \033[90mNo Supporter in top 6.\033[0m")
            random.shuffle(me.deck)
        
    elif pokemon_name == "Fan Rotom": # Fan Call
        # First turn only
        is_first_turn = (gs.turn_number == 1 if gs.turn_player == 0 else gs.turn_number == 2)
        if is_first_turn:
            if me == env._gs.players[0]:
                if should_print():
                    print(f"    -> Ability: \033[96mFan Call\033[0m")
            _search_to_hand(env, me, 3, type_filter=lambda c: c.supertype == "Pokemon" and c.type == "Colorless" and c.hp <= 100)
        else:
             if me == env._gs.players[0]:
                 if should_print():
                     print(f"    -> Ability: \033[90mFan Call Failed (Not First Turn)\033[0m")
        
    elif pokemon_name == "Munkidori":
         # Adrena-Brain: If this Pokemon has any Darkness Energy attached, you can move 3 damage counters.
         if any(card_def(e).type == "Darkness" for e in me.active.energy):
             # Move 30 damage from one of your Pokemon to one of your opponent's Pokemon.
             
             # Source: Smart heuristic (Most damaged friendly)
             best_source = me.active
             max_dmg = me.active.damage
             for b in me.bench:
                 if b.name and b.damage > max_dmg:
                     max_dmg = b.damage
                     best_source = b
             
             if max_dmg > 0:
                 # Destination: Agent Selected Target
                 if target_idx == 6: target_idx = 5 # Default to Active
                 target = None
                 if target_idx == 5: target = op.active
                 elif 0 <= target_idx < 5: target = op.bench[target_idx]
                 
                 if target and target.name:
                     # BATTLE CAGE CHECK
                     is_protected = False
                     # "Prevent all damage counters... on Benched Pokemon by effects calls... from Opponent"
                     if env._gs.active_stadium == "Battle Cage":
                         if target != op.active: # Bench
                             is_protected = True
                     
                     if is_protected:
                         if me == env._gs.players[0] and should_print():
                             print(f"    -> Munkidori blocked by Battle Cage!")
                     else:
                         # Move 30 damage
                         amount_to_move = min(30, best_source.damage)
                         best_source.damage -= amount_to_move
                         target.damage += amount_to_move
                         if me == env._gs.players[0] and should_print():
                              print(f"    -> Adrena-Brain: Moved {amount_to_move} damage!")
                         
                         # Check KO on opponent side
                         # Handled during checkup cycle normally. But immediate might matter.
                         
    elif pokemon_name == "Lunatone": # Lunar Cycle (Replacing Sun Energy logic)
        # Check: Solrock in play
        solrock_in_play = any(s.name == "Solrock" for s in me.bench) or me.active.name == "Solrock"
        if not solrock_in_play: return
        
        # Check: Discard F energy from hand
        f_indices = [i for i, c in enumerate(me.hand) if c in ("Basic Fighting Energy", "Fighting Energy")]
        if not f_indices: return
        
        # Execute
        card = me.hand.pop(f_indices[0])
        me.discard_pile.append(card)
        
        if me == env._gs.players[0] and should_print():
             print(f"    -> Ability: \033[96mLunar Cycle\033[0m (Discarded {card}, Draw 3)")
        env._draw_cards(me, 3)

    elif pokemon_name == "Genesect ex": # Metallic Signal
        # Search 2 Evolution [M] Pokemon
        if me == env._gs.players[0] and should_print():
             print(f"    -> Ability: \033[96mMetallic Signal\033[0m")
        _search_to_hand(env, me, 2, type_filter=lambda c: c.supertype == "Pokemon" and c.subtype in ("Stage1", "Stage2") and c.type == "Metal")
        random.shuffle(me.deck)

    elif pokemon_name == "Dusknoir": # Cursed Blast
        # Check Psyduck's Damp ability
        if _has_psyduck_damp_active(env):
            if me == env._gs.players[0] and should_print():
                print(f"    -> Cursed Blast: ❌ Blocked by Psyduck's Damp!")
            return
        
        if me == env._gs.players[0]:
             if should_print():
                 print(f"    -> Ability: \033[96mCursed Blast\033[0m")
        
        # Target Selection
        if target_idx == 6: target_idx = 5
        target = None
        if target_idx == 5: target = op.active
        elif 0 <= target_idx < 5: target = op.bench[target_idx]
        
        if target and target.name:
             target.damage += 130
             if me == env._gs.players[0] and should_print():
                 print(f"    -> Cursed Blast hit {target.name} for 130!")
             
             # Check KO (Immediate)
             hp = card_def(target.name).hp
             if target.damage >= hp:
                 if target == op.active:
                    env._handle_knockout(op)
    
        me.active.damage = 999 
        env._handle_knockout(me) 

    elif pokemon_name == "Gholdengo ex": # Coin Bonus
        # Once per turn. Draw 1. If active, draw 2.
        draw_count = 2 if me.active.name == "Gholdengo ex" else 1
        env._draw_cards(me, draw_count)
        if me == env._gs.players[0] and should_print():
             print(f"    -> Ability: \033[96mCoin Bonus (Draw {draw_count})\033[0m")
 


def apply_on_evolve_ability(env: 'PTCGEnv', player_idx: int, pokemon_name: str, from_hand: bool = True):
    """
    Triggered when a Pokemon evolves into this one.
    """
    gs = env._gs
    me = gs.players[player_idx]
    
    if pokemon_name == "Charizard ex": # Infernal Reign
        if not from_hand: return # Ability only triggers when played from hand
        # Search up to 3 Fire Energy -> Attach to your Pokemon in any way you like.
        if me == env._gs.players[0]:
            if should_print():
                print(f"    -> Ability: \033[96mInfernal Reign\033[0m")
        
        fire_found = []
        for _ in range(3):
            for i, c in enumerate(me.deck):
                if c in ("Basic Fire Energy", "Fire Energy"):
                    fire_found.append(me.deck.pop(i))
                    break
        
        # Shuffle deck after searching
        random.shuffle(me.deck)
        
        # RANDOM ATTACHMENT: Let the agent learn what works through outcome variance
        # This is strategic - where to put energy affects game outcome
        all_slots = [me.active] + [b for b in me.bench if b.name]
        valid_slots = [slot for slot in all_slots if slot.name]
        
        for energy in fire_found:
            if valid_slots:
                # Random distribution across all valid Pokemon
                target_slot = random.choice(valid_slots)
                target_slot.energy.append(energy)
                if me == env._gs.players[0] and should_print():
                    print(f"       Attached Fire Energy to {target_slot.name}")
            else:
                # No valid targets - discard
                me.discard_pile.append(energy)
        
    elif pokemon_name == "Alakazam": # Psychic Draw
        if not from_hand: return
        if me == env._gs.players[0]:
            if should_print():
                print(f"    -> Ability: \033[96mPsychic Draw (Alakazam)\033[0m")
        env._draw_cards(me, 3) 
        
    elif pokemon_name == "Kadabra": # Psychic Draw
        if not from_hand: return
        if me == env._gs.players[0]:
            if should_print():
                print(f"    -> Ability: \033[96mPsychic Draw (Kadabra)\033[0m")
        env._draw_cards(me, 2) 


def _has_shaymin_protection(defender: 'PlayerState', target: 'PokemonSlot') -> bool:
    """
    Check if target is protected by Shaymin's Flower Curtain ability.
    Protects benched Pokemon without Rule Boxes from attack damage.
    """
    # Must be benched (not active)
    if target == defender.active:
        return False
    
    # Must not have Rule Box
    if not target.name:
        return False
    target_def = card_def(target.name)
    if target_def.has_rule_box:
        return False
    
    # Check if Shaymin is in play for defender
    all_defender_pokemon = [defender.active] + [s for s in defender.bench if s.name]
    for slot in all_defender_pokemon:
        if slot.name == "Shaymin":
            return True
    
    return False


def apply_attack_effect(env: 'PTCGEnv', player_idx: int, pokemon_name: str, damage_out: int, atk_idx: int = 0, target_idx: int = 6, discard_amount: int = 0):
    """
    Modify attack damage or apply side effects.
    Returns: (final_damage, flags_dict)
    flags_dict can contain:
      - 'ignore_weakness_resistance': bool
      - 'damage_reduction': int (to apply to defender)
    """
    gs = env._gs
    me = gs.players[player_idx]
    op = gs.players[1 - player_idx]
    flags = {}
    
    # --- Global Damage Modifiers ---
    # 1. Vitality Band (+10)
    if me.active.tool == "Vitality Band":
        damage_out += 10
    
    # 2. Maximum Belt (+50 to ex)
    if me.active.tool == "Maximum Belt":
        if op.active.name:
            op_def = card_def(op.active.name)
            if op_def.tags and "ex" in op_def.tags:
                damage_out += 50
        
    # 3. Premium Power Pro (+30 if Fighting)
    if getattr(me, 'fighting_buff', False):
        if card_def(pokemon_name).type == "Fighting":
            damage_out += 30
    
    # Mist Energy Protection
    has_mist = op.active.name and any(e == "Mist Energy" for e in op.active.energy)
    
    if pokemon_name == "Charizard ex":
        # Burning Darkness: 180 + 30 for each Prize card your opponent has already taken.
        return (180 + (30 * op.prizes_taken), flags)
        
    if pokemon_name == "Fan Rotom":
        # Assault Landing: 70. Only if stadium in play.
        has_stadium = me.stadium is not None or op.stadium is not None
        if not has_stadium: return (0, flags)
        return (70, flags)
        
    if pokemon_name == "Alakazam":
         # Powerful Hand: Place 2 counters per card. (Ignores Weakness/Resistance)
         flags['ignore_weakness_resistance'] = True
         return (20 * len(me.hand), flags)
    
    if pokemon_name == "Chi-Yu":
        # Megafire of Envy: 50 + 90 if any of your Pokemon were KO'd last turn
        base = 50
        if gs.ko_last_turn: base += 90
        return (base, flags)

    if pokemon_name == "Fezandipiti ex":
        # Cruel Arrow: 100 damage to 1 opponent's Pokemon.
        
        if target_idx == 6: target_idx = 5
        target = None
        if target_idx == 5: target = op.active
        elif 0 <= target_idx < 5: target = op.bench[target_idx]
        
        if target and target.name:
            # Check Shaymin's Flower Curtain protection
            if _has_shaymin_protection(op, target):
                if me == gs.players[0] and should_print():
                    print(f"    -> Cruel Arrow: ✅ Blocked by Shaymin's Flower Curtain!")
            else:
                # Battle Cage only blocks damage counters. Cruel Arrow is DAMAGE. Not blocked.
                target.damage += (100 + damage_out)
                if me == gs.players[0] and should_print():
                    print(f"    -> Cruel Arrow hit {target.name} for 100!")
                 
        return (0, flags) # Damage handled manually
    
    if pokemon_name == "Gouging Fire ex":
        # Blaze Blitz: 260. Can't attack next turn.
        me.active.status["cant_attack"] = True
        return (260, flags)
        
    if pokemon_name == "Pidgey":
        if atk_idx == 0: # Call for Family
            # Search for up to 2 Basic Pokemon and put them onto Bench
            _search_and_bench(env, me, 2, filter_func=lambda c: c.supertype == "Pokemon" and c.subtype == "Basic")
            return (0, flags)
        elif atk_idx == 1: # Tackle
            return (20, flags)
            
    if pokemon_name == "Abra":
        # Teleportation Attack: 10 + Switch
        # Switch with 1 of your benched Pokemon
        benchers = [i for i, s in enumerate(me.bench) if s.name]
        if benchers:
            # Deterministic switch: Swap with first available bencher
            target = benchers[0]
            # Clear status? Usually switching clears status.
            me.active.status = {k:False for k in me.active.status}
            me.active, me.bench[target] = me.bench[target], me.active
            if me == gs.players[0]:
                if should_print():
                    print(f"    -> Abra switches with {me.active.name}")
        return (10, flags)
    if pokemon_name == "Gimmighoul":
        if atk_idx == 0: # Minor Errand-Running
            # Search deck for up to 2 Basic Energy
            _search_to_hand(env, me, 2, type_filter=lambda c: c.supertype == "Energy" and c.subtype == "Basic")
            return (0, flags)
        else:
            return (50, flags) # Tackle
            
    if pokemon_name == "Gholdengo ex":
         # Make It Rain: Discard X Basic Energy from hand. 50x damage.
         # Logic: Use discard_amount provided by action.
         
         # Identify energy candidates
         energy_indices = [i for i, c in enumerate(me.hand) if card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic"]
         
         count_to_discard = min(len(energy_indices), discard_amount)
         dmg = 0
         
         # Remove from highest index to lowest
         indices_to_remove = sorted(energy_indices, reverse=True)[:count_to_discard]
         for i in indices_to_remove:
             card = me.hand.pop(i)
             me.discard_pile.append(card)
             dmg += 50
         
         if me == gs.players[0] and should_print():
             print(f"    -> Make It Rain: Discarded {count_to_discard}/{discard_amount} energy for {dmg} damage!")
         return (dmg, flags)
         
    if pokemon_name == "Mega Charizard X ex":
         # Inferno X: Discard any amount of Fire Energy from among your Pokémon. 90x damage.
         # Logic: Use discard_amount provided by action (ATTACK_MAGNITUDE).
         
         # Collect all Fire Energy across all Pokemon
         all_slots = [me.active] + [s for s in me.bench if s.name]
         fire_energy_locations = []  # List of (slot_index, energy_index) tuples
         
         for slot_idx, slot in enumerate(all_slots):
             for e_idx, energy in enumerate(slot.energy):
                 if energy in ("Basic Fire Energy", "Fire Energy"):
                     fire_energy_locations.append((slot_idx, e_idx))
         
         # Discard up to discard_amount Fire Energy
         count_to_discard = min(len(fire_energy_locations), discard_amount)
         dmg = 0
         
         # Remove from highest index to lowest to preserve lower indices
         locations_to_remove = sorted(fire_energy_locations, reverse=True)[:count_to_discard]
         for slot_idx, e_idx in locations_to_remove:
             slot = all_slots[slot_idx]
             energy = slot.energy.pop(e_idx)
             me.discard_pile.append(energy)
             dmg += 90
         
         if me == gs.players[0] and should_print():
             print(f"    -> Inferno X: Discarded {count_to_discard}/{discard_amount} Fire Energy for {dmg} damage!")
         return (dmg, flags)
         
    if pokemon_name == "Solrock":
        # Cosmic Beam: 70 if Lunatone on Bench.
        # This attack's damage isn't affected by Weakness or Resistance.
        has_lunatone = any(s.name == "Lunatone" for s in me.bench)
        if has_lunatone:
            flags['ignore_weakness_resistance'] = True
            return (damage_out, flags)
        return (0, flags)
        
    if pokemon_name == "Hop's Cramorant":
        # Fickle Spitting: 120 if Op prizes == 3 or 4.
        rem = 6 - op.prizes_taken
        if rem == 3 or rem == 4: return (120, flags)
        return (0, flags)
        
    if pokemon_name == "Genesect ex":
        # Protect Charge: 150. During opponent's next turn, this Pokémon takes 30 less damage.
        me.active.damage_reduction = 30
        if me == gs.players[0] and should_print():
            print(f"    -> Protect Charge: +30 damage reduction next turn")
        return (150, flags)


    return (damage_out, flags)


# =============================================================================
# ENV
# Source: tcg/env.py
# =============================================================================

class PTCGEnv(gym.Env):
    """
    Minimal imperfect-information environment scaffold.
    - single-agent: player 0 is controlled, player 1 is scripted/random.
    - rewards: +1 win, -1 loss, small shaping via prize delta.
    """

    metadata = {"render_modes": []}

    def __init__(self, scripted_opponent: bool = True, max_turns: int = 50):
        super().__init__()
        self.scripted_opponent = scripted_opponent
        self.max_turns = max_turns

        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        # observation: fixed-length vector
        dummy = featurize(GameState())
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=dummy.shape, dtype=np.float32
        )

        self._gs = GameState()
        self._ko_this_turn = False  # Volatile flag

    # Define a standard deck for testing
    def _create_standard_deck(self) -> list[str]:
        # Based on the user's provided list or common cards
        deck = []
        deck.extend(["Charmander"] * 4)
        deck.extend(["Charmeleon"] * 1)
        deck.extend(["Charizard ex"] * 3)
        deck.extend(["Pidgey"] * 2)
        deck.extend(["Pidgeotto"] * 1)
        deck.extend(["Pidgeot ex"] * 2)
        deck.extend(["Duskull"] * 2)
        deck.extend(["Dusclops"] * 2)
        deck.extend(["Dusknoir"] * 1)
        deck.extend(["Tatsugiri"] * 1)
        deck.extend(["Fezandipiti ex"] * 1)
        deck.extend(["Fan Rotom"] * 1)
        deck.extend(["Rare Candy"] * 4)
        deck.extend(["Buddy-Buddy Poffin"] * 4)
        deck.extend(["Ultra Ball"] * 4)
        deck.extend(["Arven"] * 4)
        deck.extend(["Iono"] * 3)
        deck.extend(["Professor's Research"] * 2)
        deck.extend(["Boss's Orders"] * 2)
        deck.extend(["Counter Catcher"] * 1)
        deck.extend(["Super Rod"] * 1)
        deck.extend(["Basic Fire Energy"] * 6)
        deck.extend(["Basic Psychic Energy"] * 2)
        deck.extend(["Jet Energy"] * 1)
        # Fill remaining to 60 if needed
        while len(deck) < 60:
            deck.append("Basic Fire Energy")
        return deck[:60]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._gs = GameState()
        self._ko_this_turn = False
        
        # Check for custom decks in options
        # options={"decks": [deck_list_p0, deck_list_p1]}
        custom_decks = None
        if options and "decks" in options:
            custom_decks = options["decks"]
        
        # Setup each player
        p0_mulligans = 0
        p1_mulligans = 0
        for p_idx in range(2):
            if custom_decks and p_idx < len(custom_decks):
                deck = list(custom_decks[p_idx]) # Copy needed to avoid mutating source
            else:
                deck = self._create_standard_deck()
            
            random.shuffle(deck)
            p = self._gs.players[p_idx]
            p.deck = deck
            
            # Draw 7
            hand = []
            for _ in range(7):
                if p.deck:
                    hand.append(p.deck.pop())
            p.hand = hand
            
            # Mulligan check
            # Real TCG: reveal hand, shuffle back, draw 7, opponent draws 1 for each mulligan
            basics = [c for c in hand if card_def(c).subtype == "Basic" and card_def(c).supertype == "Pokemon"]
            mulligan_count = 0
            while not basics and p.deck:
                mulligan_count += 1
                # Shuffle hand back, draw 7
                p.deck.extend(p.hand)
                random.shuffle(p.deck)
                p.hand = []
                for _ in range(7):
                    if p.deck: p.hand.append(p.deck.pop())
                basics = [c for c in p.hand if card_def(c).subtype == "Basic" and card_def(c).supertype == "Pokemon"]
            
            # Store mulligan count for opponent draw later
            if p_idx == 0:
                p0_mulligans = mulligan_count
            else:
                p1_mulligans = mulligan_count
            
            hand = p.hand # Sync local variable
            
            # Auto-Play Active (Setup Phase)
            # Find a basic
            basic_idx = -1
            for i, c in enumerate(hand):
                cd = card_def(c)
                if cd.supertype == "Pokemon" and cd.subtype == "Basic":
                    basic_idx = i
                    break
            
            if basic_idx >= 0:
                card = hand.pop(basic_idx)
                p.active.name = card
                p.active.turn_played = 0 # Setup
                p.active.damage = 0
                p.active.energy = []
            
            p.hand = hand
            
            # Set Prizes
            random.shuffle(p.deck) # Shuffled before prize draw
            prizes = []
            for _ in range(6):
                if p.deck:
                    prizes.append(p.deck.pop())
            p.prizes = prizes
        
        # Apply mulligan penalty: each player draws extra cards for opponent's mulligans
        # P0 draws p1_mulligans extra, P1 draws p0_mulligans extra
        for _ in range(p0_mulligans):
            if self._gs.players[1].deck:
                self._gs.players[1].hand.append(self._gs.players[1].deck.pop())
        for _ in range(p1_mulligans):
            if self._gs.players[0].deck:
                self._gs.players[0].hand.append(self._gs.players[0].deck.pop())
            
        # Randomize starting player
        self._gs.turn_player = random.randint(0, 1)
        # Note: If P1 starts, turn count starts at 1. But effectively P0 "skipped" turn 0?
        # Actually standard TCG: Turn 1 is P(Start). 
        # gs.turn_number is global.

        obs = featurize(self._gs)
        info = {"action_mask": self.action_mask()}
        return obs, info

    def action_mask(self) -> np.ndarray:
        """Return boolean mask over ACTION_TABLE."""
        gs = self._gs
        me = gs.players[gs.turn_player]
        op = gs.players[1 - gs.turn_player]
        mask = np.zeros((len(ACTION_TABLE),), dtype=np.int8)

        # helper: find empty bench slots
        empty_bench = [i for i, s in enumerate(me.bench) if s.name is None]

        # Rule: Can only evolve if:
        # 1. Pokemon in play (Active or Bench).
        # 2. Pokemon has been in play for > 0 turns (Turn Played < Current Turn).
        #    UNLESS it involves Rare Candy (handled in effect, but mask might allow it if we treat Rare Candy as a specific action?
        #    No, Rare Candy is PLAY_TRAINER. The standard EVOLVE action is "Normal Evolution").
        # 3. Not Player 1's first turn (if we strictly follow that rule).

        can_evolve_normally = True
        # Rule: Neither player can evolve on their first turn.
        # This corresponds to Turn 1 (Player A) and Turn 2 (Player B).
        if gs.turn_number <= 2:
            can_evolve_normally = False

        for i, act in enumerate(ACTION_TABLE):
            if gs.done:
                mask[i] = 0
                continue

            if act.kind == "PASS":
                # SMART PASS MASKING: Only allow PASS if truly needed
                # This is set after scanning all other actions
                # For now, allow it - we'll reduce priority via MCTS priors
                mask[i] = 1
                continue

            if act.kind == "PLAY_BASIC_TO_BENCH":
                if act.a in me.hand and empty_bench and (act.b in empty_bench):
                    mask[i] = 1
                continue

            if act.kind.startswith("EVOLVE"):
                if not can_evolve_normally:
                    continue

                # Can evolve if evo card in hand and target has evolves_from match
                evo = act.a
                if evo not in me.hand:
                    continue
                evo_def = card_def(evo)
                if evo_def.evolves_from is None:
                    continue

                target_slot = None
                if act.kind == "EVOLVE_ACTIVE":
                    target_slot = me.active
                else:
                    idx = act.b
                    if idx is None or idx < 0 or idx >= MAX_BENCH:
                        continue
                    target_slot = me.bench[idx]

                if target_slot.name == evo_def.evolves_from:
                    # Check "Evolution Sickness": Must have been in play since BEFORE this turn.
                    if target_slot.turn_played < gs.turn_number:
                        mask[i] = 1
                continue

            if act.kind.startswith("ATTACH_") and not act.kind.startswith("ATTACH_TOOL"):
                # Rule: Once per turn
                if me.energy_attached:
                    continue
                if act.a not in me.hand:
                    continue
                # needs a target pokemon
                target = None
                if act.kind == "ATTACH_ACTIVE":
                    target = me.active
                else:
                    idx = act.b
                    if idx is not None and 0 <= idx < MAX_BENCH:
                        target = me.bench[idx]
                
                if target and target.name is not None:
                    # Heuristic Check: Prevent excessive energy (cap at 6)
                    if len(target.energy) >= 6:
                        continue
                    
                    # --- SMART ENERGY MASKING ---
                    # Prevents "energy blindness" - don't attach useless energy
                    energy_def = card_def(act.a)
                    energy_type = energy_def.type
                    poke_def = card_def(target.name)
                    poke_type = poke_def.type
                    
                    # Allow attachment if:
                    # 1. Energy type matches Pokemon type (Fire on Fire pokemon)
                    type_match = (energy_type == poke_type)
                    
                    # 2. It is Special Energy (usually works on anything)
                    is_special = (energy_def.subtype == "Special")
                    
                    # 3. Pokemon has attacks with Colorless costs (any energy works)
                    has_colorless_attacks = any(
                        "Colorless" in atk.cost for atk in poke_def.attacks
                    ) if poke_def.attacks else False
                    
                    # 4. Pokemon needs energy to pay Retreat Cost
                    needs_retreat_energy = len(target.energy) < poke_def.retreat_cost
                    
                    # 5. Energy is "Colorless" type (works for colorless costs)
                    is_colorless_energy = (energy_type == "Colorless")
                    
                    is_useful = type_match or is_special or has_colorless_attacks or needs_retreat_energy or is_colorless_energy
                    
                    if is_useful:
                        mask[i] = 1
                continue

            if act.kind.startswith("ATTACH_TOOL"):
                if act.a not in me.hand:
                    continue
                # Needs a target pokemon
                target = None
                if act.kind == "ATTACH_TOOL_ACTIVE":
                    target = me.active
                else:
                    idx = act.b
                    if idx is not None and 0 <= idx < MAX_BENCH:
                        target = me.bench[idx]
                
                # Rule: Only one tool per Pokemon
                if target and target.name and target.tool is None:
                    mask[i] = 1
                continue

            if act.kind == "PLAY_TRAINER":
                # allow trainer play if in hand; supporter once per turn
                if act.a not in me.hand:
                    continue
                cd = card_def(act.a)

                # Rule: Supporter once per turn
                if cd.subtype == "Supporter" and me.supporter_used:
                    continue

                # Rule: No Supporter on first turn of player going first
                if (
                    cd.subtype == "Supporter"
                    and gs.turn_number == 1
                ):
                    continue

                # Target validation (Target 0-4: Bench, Target 5: Active, Target 6: Global/Self)
                target = act.b
                
                # Default for non-targeted: only target 6 valid
                if act.a in ("Arven", "Iono", "Lillie's Determination", "Hilda", "Dawn", "Tulip", 
                            "Professor Turo's Scenario", "Boss's Orders", "Counter Catcher", 
                            "Buddy-Buddy Poffin", "Ultra Ball", "Nest Ball", "Rare Candy", 
                            "Super Rod", "Unfair Stamp", "Battle Cage", "Artazon", "Wondrous Patch", 
                            "Night Stretcher", "Tulip", "Enhanced Hammer"):
                    
                    if act.a == "Nest Ball":
                         # Nest Ball search targets (agent chooses which Basic to bench):
                         # 0 = Evolution starters (Abra, Charmander, Pidgey, Charcadet)
                         # 1 = Support/Draw Pokemon (Fan Rotom, Dunsparce)
                         # 2 = Tech Pokemon (Fezandipiti ex, Tatsugiri, Psyduck)
                         # 3 = Any basic (first available)
                         if target > 3: continue  # Only 0-3 valid
                         if not empty_bench: continue  # Bench full
                         
                         # Check if target type has valid cards in deck
                         has_match = False
                         evo_starters = ("Abra", "Charmander", "Pidgey", "Charcadet")
                         support_mons = ("Fan Rotom", "Dunsparce")
                         tech_mons = ("Fezandipiti ex", "Tatsugiri", "Psyduck")
                         
                         for c in me.deck:
                             cd_c = card_def(c)
                             if cd_c.supertype != "Pokemon" or cd_c.subtype != "Basic":
                                 continue
                             if target == 0 and c in evo_starters:
                                 has_match = True; break
                             elif target == 1 and c in support_mons:
                                 has_match = True; break
                             elif target == 2 and c in tech_mons:
                                 has_match = True; break
                             elif target == 3:  # Any basic
                                 has_match = True; break
                         
                         if not has_match: continue

                    elif act.a == "Buddy-Buddy Poffin":
                         # Buddy-Buddy Poffin: Search for up to 2 Pokemon with 70 HP or less
                         # Target = how many to bench (0, 1, or 2)
                         # 0 = Bench 0 (fail search on purpose for hand thinning)
                         # 1 = Bench 1 Pokemon
                         # 2 = Bench 2 Pokemon
                         if target > 2: continue
                         if not empty_bench and target > 0: continue  # Need bench space if benching any
                         
                         # Count how many slots available
                         empty_slots = sum(1 for s in me.bench if not s.name)
                         if target > empty_slots: continue  # Can't bench more than slots
                         
                         # Check if enough 70 HP or less basics in deck
                         candidates_in_deck = sum(1 for c in me.deck 
                             if card_def(c).supertype == "Pokemon" 
                             and card_def(c).subtype == "Basic" 
                             and card_def(c).hp <= 70)
                         
                         if target > candidates_in_deck: continue  # Not enough targets

                    elif act.a == "Artazon":
                         if target != 6: continue
                         # Need bench space to put searched Pokemon
                         if not empty_bench: continue
                         # Check if there are valid targets (non-Rule Box basics) in deck
                         has_valid_target = any(
                             card_def(c).supertype == "Pokemon" and 
                             card_def(c).subtype == "Basic" and 
                             not card_def(c).has_rule_box
                             for c in me.deck
                         )
                         if not has_valid_target: continue
                    
                    elif act.a == "Night Stretcher":
                        # Night Stretcher: Select specific card from discard via target_idx (b)
                        # b = discard pile index (0-14)
                        
                        # Get valid candidates
                        candidates = [i for i, c in enumerate(me.discard_pile)
                                     if card_def(c).supertype == "Pokemon" or 
                                        (card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic")]
                        
                        # If target is within bounds and is a valid candidate
                        if target < len(me.discard_pile) and target in candidates:
                            pass  # Valid selection
                        elif target >= len(me.discard_pile):
                            continue  # Out of bounds
                        elif not candidates:
                            continue  # No valid targets at all
                        # else: target is in bounds but not a valid candidate - still invalid
                        elif target not in candidates:
                            continue

                    elif act.a == "Super Rod":
                        # Super Rod: Select up to 3 cards from discard via c, d, e
                        if target != 6: continue  # Only b=6 is valid
                        
                        # Get valid candidates
                        candidates = [i for i, c in enumerate(me.discard_pile)
                                     if card_def(c).supertype == "Pokemon" or 
                                        (card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic")]
                        
                        if not candidates:
                            continue  # No valid targets
                        
                        # Validate selections if provided
                        selections = [idx for idx in [act.c, act.d, act.e] if idx is not None]
                        if selections:
                            # All selections must be within bounds
                            if any(idx >= len(me.discard_pile) for idx in selections):
                                continue
                            # All selections must be valid candidates
                            if not all(idx in candidates for idx in selections):
                                continue
                            # All selections must be unique
                            if len(selections) != len(set(selections)):
                                continue
                        # If no selections, we'll use fallback (OK)

                    elif act.a == "Lana's Aid":
                        # Lana's Aid: Select up to 3 cards from discard via c, d, e
                        if target != 6: continue  # Only b=6 is valid
                        
                        # Get valid candidates (non-Rule Box Pokemon + Basic Energy)
                        candidates = [i for i, c in enumerate(me.discard_pile)
                                     if (card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic") or
                                        (card_def(c).supertype == "Pokemon" and not card_def(c).has_rule_box())]
                        
                        if not candidates:
                            continue  # No valid targets
                        
                        # Validate selections if provided
                        selections = [idx for idx in [act.c, act.d, act.e] if idx is not None]
                        if selections:
                            # All selections must be within bounds
                            if any(idx >= len(me.discard_pile) for idx in selections):
                                continue
                            # All selections must be valid candidates
                            if not all(idx in candidates for idx in selections):
                                continue
                            # All selections must be unique
                            if len(selections) != len(set(selections)):
                                continue
                        # If no selections, we'll use fallback (OK)
                    
                    elif act.a == "Buddy-Buddy Poffin":
                        # Buddy-Buddy Poffin: Select up to 2 Pokemon from deck via c, d
                        if target != 6: continue  # Only b=6 is valid
                        
                        if not empty_bench: continue  # Need bench space
                        
                        # Get valid candidates (Basic Pokemon HP <= 70)
                        candidates = [i for i, c in enumerate(me.deck)
                                     if card_def(c).supertype == "Pokemon" and 
                                        card_def(c).subtype == "Basic" and 
                                        card_def(c).hp <= 70]
                        
                        if not candidates:
                            continue  # No valid targets
                        
                        # Validate selections if provided
                        selections = [idx for idx in [act.c, act.d] if idx is not None]
                        if selections:
                            # All selections must be within bounds
                            if any(idx >= len(me.deck) for idx in selections):
                                continue
                            # All selections must be valid candidates
                            if not all(idx in candidates for idx in selections):
                                continue
                            # All selections must be unique
                            if len(selections) != len(set(selections)):
                                continue
                            # Check we have enough bench space for selections
                            empty_slots = sum(1 for s in me.bench if not s.name)
                            if len(selections) > empty_slots:
                                continue
                        # If no selections, we'll use fallback (OK)
                    
                    elif act.a == "Fighting Gong":
                        # Fighting Gong: Choose Energy (b=0) or Pokemon (b=1)
                        if target not in (0, 1, 6):
                            continue  # Only 0, 1, or 6 are valid
                        
                        # Check if deck has the requested type
                        if target == 0:
                            # Check for Fighting Energy
                            has_target = any(
                                card_def(c).supertype == "Energy" and 
                                card_def(c).subtype == "Basic" and 
                                card_def(c).type == "Fighting"
                                for c in me.deck
                            )
                            if not has_target:
                                continue
                        elif target == 1:
                            # Check for Fighting Pokemon
                            has_target = any(
                                card_def(c).supertype == "Pokemon" and 
                                card_def(c).subtype == "Basic" and 
                                card_def(c).type == "Fighting"
                                for c in me.deck
                            )
                            if not has_target:
                                continue
                        # target == 6: either is OK (checked below)

                    elif act.a == "Tulip":
                        if target != 6: continue
                        has_valid = any(
                            (card_def(c).supertype == "Energy" and card_def(c).type == "Psychic") or 
                            (card_def(c).supertype == "Pokemon" and card_def(c).type == "Psychic")
                            for c in me.discard_pile
                        )
                        if not has_valid: continue

                    elif act.a == "Wondrous Patch":
                        if target != 6: continue
                        # Needs Basic Psychic Energy in discard AND Psychic Pokemon on bench
                        has_energy = any(
                            card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic" and card_def(c).type == "Psychic"
                            for c in me.discard_pile
                        )
                        has_target = any(
                            s.name and card_def(s.name).type == "Psychic" for s in me.bench
                        )
                        if not (has_energy and has_target): continue

                    elif act.a == "Professor Turo's Scenario":
                        # Needs to target a Pokemon in play
                        if target == 5: # Active
                            if not me.active.name: continue
                        elif target < 5: # Bench
                            if not me.bench[target].name: continue
                        else: continue # Target 6 not valid
                        
                    elif act.a == "Rare Candy":
                        # RULE: Cannot use Rare Candy on your first turn
                        # P0's first turn is turn 0, P1's first turn is turn 1
                        player_first_turn = gs.turn_player if gs.turn_player == 0 else 1
                        if gs.turn_number <= player_first_turn:
                            continue
                        
                        # Needs to target a Basic in play that has been there 1+ turns
                        if target == 5:
                            mon = me.active
                        elif target < 5:
                            mon = me.bench[target]
                        else: continue
                        
                        if not mon.name: continue
                        # Rule: Rare Candy can only be used on Basic Pokemon
                        if card_def(mon.name).subtype != "Basic": continue 
                        if mon.turn_played >= gs.turn_number: continue
                        # Must have a stage 2 in hand for this basic
                        has_evo = False
                        for c in me.hand:
                            if c == "Rare Candy": continue
                            cd_e = card_def(c)
                            if cd_e.supertype == "Pokemon" and cd_e.subtype == "Stage2":
                                # Check heritage
                                stage1 = cd_e.evolves_from
                                if stage1:
                                    s1_def = card_def(stage1)
                                    if s1_def.evolves_from == mon.name or stage1 == mon.name:
                                        has_evo = True; break
                        if not has_evo: continue

                    elif act.a == "Boss's Orders" or act.a == "Counter Catcher":
                        # Target opponent's bench (0-4)
                        if target >= 5: continue
                        if not op.bench[target].name: continue
                        # Counter Catcher extra check
                        if act.a == "Counter Catcher":
                            if len(me.prizes) <= len(op.prizes): continue

                    elif act.a == "Enhanced Hammer":
                        if target != 6: continue
                        # Must have Special Energy on opponent's pokemon
                        has_special = False
                        if op.active.name:
                             for e in op.active.energy:
                                 if card_def(e).subtype == "Special_Energy": has_special = True; break
                        if not has_special:
                             for b in op.bench:
                                 if b.name:
                                     for e in b.energy:
                                         if card_def(e).subtype == "Special_Energy": has_special = True; break
                        if not has_special: continue

                    elif act.a == "Wondrous Patch":
                        if target != 6: continue
                        # Robust check checking card definitions
                        has_energy = any(card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic" and card_def(c).type == "Psychic" for c in me.discard_pile)
                        has_target = any(b.name and card_def(b.name).type == "Psychic" for b in me.bench)
                        if not (has_energy and has_target): continue

                    elif act.a == "Night Stretcher" or act.a == "Super Rod":
                        if target != 6: continue
                        # Must have Pokemon or Basic Energy in discard
                        has_valid_target = False
                        for c in me.discard_pile:
                            cd_c = card_def(c)
                            if cd_c.supertype == "Pokemon":
                                has_valid_target = True; break
                            if cd_c.supertype == "Energy" and cd_c.subtype == "Basic":
                                has_valid_target = True; break
                        if not has_valid_target: continue

                    elif act.a == "Unfair Stamp":
                        if target != 6: continue
                        if not gs.ko_last_turn: continue

                    elif act.a == "Ultra Ball":
                        # Ultra Ball search targets (agent chooses what to search for):
                        # 0 = Search for Basic Pokemon
                        # 1 = Search for Stage 1 Pokemon
                        # 2 = Search for Stage 2 Pokemon
                        # 3 = Search for evolution of active Pokemon
                        # 4 = Search for evolution of any bench Pokemon
                        # 5 = Search for key attacker (Alakazam, Charizard ex, etc.)
                        # 6 = Fail search intentionally (hand thinning - discard 2 but find nothing)
                        if target > 6: continue  # Only 0-6 valid
                        if len(me.hand) < 3: continue  # Need 2 cards to discard + Ultra Ball
                        
                        # Target 6 is always valid (intentional fail)
                        if target == 6:
                            pass  # Always allowed - just discards 2 cards
                        else:
                            # Check if target search type has valid candidates in deck
                            has_match = False
                            for c in me.deck:
                                cd_c = card_def(c)
                                if cd_c.supertype != "Pokemon": continue
                                
                                if target == 0 and cd_c.subtype == "Basic":
                                    has_match = True; break
                                elif target == 1 and cd_c.subtype == "Stage1":
                                    has_match = True; break
                                elif target == 2 and cd_c.subtype == "Stage2":
                                    has_match = True; break
                                elif target == 3:  # Evolution of active
                                    if me.active.name and cd_c.evolves_from == me.active.name:
                                        has_match = True; break
                                elif target == 4:  # Evolution of any bench
                                    for slot in me.bench:
                                        if slot.name and cd_c.evolves_from == slot.name:
                                            has_match = True; break
                                    if has_match: break
                                elif target == 5:  # Key attackers
                                    if c in ("Alakazam", "Charizard ex", "Pidgeot ex", "Dudunsparce"):
                                        has_match = True; break
                            
                            if not has_match: continue

                    elif cd.subtype == "Stadium":
                        if target != 6: continue
                        if getattr(me, 'stadium', None) == act.a: continue

                    if act.a == "Earthen Vessel":
                        # Check discard index b
                        # Hand size after playing vessel must be > b
                        if len(me.hand) - 1 <= act.b: continue
                    
                    elif act.a == "Superior Energy Retrieval":
                         if act.c is None: continue # requires pair
                         if len(me.hand) - 1 <= max(act.b, act.c): continue
                         
                    elif act.a == "Ultra Ball":
                         if act.c is None or act.d is None: continue
                         if len(me.hand) - 1 <= max(act.c, act.d): continue
                         # Target Checks (Optional - keep existing search logic mask?)
                         # Existing mask checked `target` (act.b).
                         # I should preserve that.
                         target = act.b
                         # ... check target validity ...
                         # Copying existing search check:
                         if target == 6: pass
                         else:
                             has_match = False
                             for c in me.deck:
                                 cd_c = card_def(c)
                                 if cd_c.supertype != "Pokemon": continue
                                 if target == 0 and cd_c.subtype == "Basic": has_match = True; break
                                 elif target == 1 and cd_c.subtype == "Stage1": has_match = True; break
                                 elif target == 2 and cd_c.subtype == "Stage2": has_match = True; break
                                 elif target == 3 and me.active.name and cd_c.evolves_from == me.active.name: has_match = True; break
                                 elif target == 4:
                                     for slot in me.bench:
                                          if slot.name and cd_c.evolves_from == slot.name: has_match = True; break
                                     if has_match: break
                                 elif target == 5 and c in ("Alakazam", "Charizard ex", "Pidgeot ex", "Dudunsparce"): has_match = True; break
                             if not has_match: continue
                    
                    # --- SMART DRAW SUPPORTER VALIDATION ---
                    # Prevent wasting supporters when deck is empty/near-empty
                    elif act.a == "Hilda":
                        if target != 6: continue
                        # Hilda draws 6 cards. Skip if deck too small to benefit significantly
                        if len(me.deck) < 2: continue  # Need at least a few cards
                    
                    elif act.a == "Dawn":
                        if target != 6: continue
                        # Dawn draws based on hand size difference. Skip if deck empty
                        if len(me.deck) < 1: continue
                    
                    elif act.a == "Iono":
                        if target != 6: continue
                        # Iono shuffles hands and draws by prizes. Skip if deck too small
                        remaining_prizes = len(me.prizes)  # Draw this many
                        if len(me.deck) < remaining_prizes: continue  # Risk deck-out
                    
                    elif act.a == "Lillie's Determination":
                        if target != 6: continue
                        # Lillie draws cards if behind on prizes. Skip if deck empty
                        if len(me.deck) < 1: continue
                        # Only valuable if behind on prizes
                        if len(me.prizes) <= len(op.prizes): continue
                              
                    elif target != 6: # Most other trainers are global/self
                        if act.a not in ("Professor Turo's Scenario", "Rare Candy", "Boss's Orders", "Counter Catcher", "Prime Catcher"):
                            continue

                mask[i] = 1
                continue

            if act.kind == "RETREAT_TO":
                # Rule: Can only retreat if active has enough energy
                if me.active.name is None:
                    continue
                    
                # Status: Cannot retreat if Asleep or Paralyzed
                if me.active.status.get("asleep") or me.active.status.get("paralyzed"):
                    continue

                retreat_cost = card_def(me.active.name).retreat_cost
                if me.active.tool == "Air Balloon": retreat_cost = max(0, retreat_cost - 2)
                
                if len(me.active.energy) < retreat_cost:
                    continue
                
                idx = act.b
                if idx is None or idx < 0 or idx >= MAX_BENCH:
                    continue
                if me.bench[idx].name is None:
                    continue
                
                mask[i] = 1
                continue

            if act.kind in ("USE_ACTIVE_ABILITY", "ATTACK"):
                # super simplified: only if active exists
                if me.active.name is None:
                    continue
                    
                if act.kind == "USE_ACTIVE_ABILITY":
                    # Allow ability if active OR bench Pokemon has usable ability
                    has_ability = False
                    ability_source = None  # Which Pokemon is providing the ability
                    
                    # ===== ABILITIES THAT WORK FROM ACTIVE OR BENCH =====
                    
                    # Helper to check bench for ability
                    def check_slot_ability(slot, ability_name):
                        if not slot.name:
                            return False
                        if slot.ability_used_this_turn:
                            # All abilities are once per turn per Pokemon
                            return False
                        return True
                    
                    # 1. Check Active first
                    if me.active.name:
                        slot = me.active
                        
                        if slot.name == "Pidgeot ex" and check_slot_ability(slot, "Quick Search"):
                            has_ability = True
                            ability_source = slot
                        
                        elif slot.name == "Fezandipiti ex" and gs.ko_last_turn and check_slot_ability(slot, "Flip the Script"):
                            # Flip the Script: Only if opponent KO'd one of YOUR Pokemon on THEIR turn
                            has_ability = True
                            ability_source = slot
                        
                        elif slot.name == "Tatsugiri" and check_slot_ability(slot, "Attract Customers"):
                            has_ability = True
                            ability_source = slot
                        
                        elif slot.name == "Dudunsparce" and check_slot_ability(slot, "Run Away Draw"):
                            has_ability = True
                            ability_source = slot
                        
                        elif slot.name == "Fan Rotom":
                            is_first_turn = (gs.turn_number == 1 if gs.turn_player == 0 else gs.turn_number == 2)
                            if is_first_turn and not getattr(me, "fan_call_used", False) and check_slot_ability(slot, "Fan Call"):
                                has_ability = True
                                ability_source = slot

                        elif slot.name == "Lunatone":
                            solrock_in_play = any(s.name == "Solrock" for s in me.bench)
                            if solrock_in_play and check_slot_ability(slot, "Lunar Cycle"):
                                has_ability = True
                                ability_source = slot
                                 
                        elif slot.name == "Genesect ex" and check_slot_ability(slot, "Metallic Signal"):
                            # Metallic Signal: Once during your turn
                            has_ability = True
                            ability_source = slot
                        
                        elif slot.name == "Dusknoir" and check_slot_ability(slot, "Cursed Blast"):
                            has_ability = True
                            ability_source = slot
                        
                        elif slot.name == "Munkidori":
                            # Adrena-Brain: Needs Darkness Energy attached
                            # Note: Battle Cage only blocks BENCH targeting (checked in effects.py)
                            if any(card_def(e).type == "Darkness" for e in slot.energy) and check_slot_ability(slot, "Adrena-Brain"):
                                has_ability = True
                                ability_source = slot
                        
                        # NOTE: Alakazam/Kadabra Psychic Draw is ON-EVOLUTION, not an activated ability
                        # It triggers automatically when you evolve, handled in EVOLVE logic
                    
                    # 2. Check Bench for abilities that work from bench
                    if not has_ability:
                        for slot in me.bench:
                            if not slot.name:
                                continue
                            
                            # Fan Rotom - Fan Call (turn 1 only)
                            if slot.name == "Fan Rotom":
                                is_first_turn = (gs.turn_number == 1 if gs.turn_player == 0 else gs.turn_number == 2)
                                if is_first_turn and not getattr(me, "fan_call_used", False) and check_slot_ability(slot, "Fan Call"):
                                    has_ability = True
                                    ability_source = slot
                                    break
                            
                            # Fezandipiti ex - Flip the Script (if opponent KO'd YOUR Pokemon on THEIR turn)
                            if slot.name == "Fezandipiti ex" and gs.ko_last_turn and check_slot_ability(slot, "Flip the Script"):
                                has_ability = True
                                ability_source = slot
                                break
                            
                            # NOTE: Alakazam/Kadabra Psychic Draw is ON-EVOLUTION trigger, not here!
                            
                            # Pidgeot ex - Quick Search
                            if slot.name == "Pidgeot ex" and check_slot_ability(slot, "Quick Search"):
                                has_ability = True
                                ability_source = slot
                                break
                            
                            # Genesect ex - Metallic Signal (once per turn)
                            if slot.name == "Genesect ex" and check_slot_ability(slot, "Metallic Signal"):
                                has_ability = True
                                ability_source = slot
                                break
                            
                            # Dudunsparce - Run Away Draw
                            if slot.name == "Dudunsparce" and check_slot_ability(slot, "Run Away Draw"):
                                has_ability = True
                                ability_source = slot
                                break
                            
                            # Lunatone - Lunar Cycle (needs Solrock)
                            if slot.name == "Lunatone":
                                solrock_in_play = any(s.name == "Solrock" for s in me.bench if s != slot) or me.active.name == "Solrock"
                                if solrock_in_play and check_slot_ability(slot, "Lunar Cycle"):
                                    has_ability = True
                                    ability_source = slot
                                    break
                            
                            # Dusknoir - Cursed Blast (targeting, works from bench)
                            if slot.name == "Dusknoir" and check_slot_ability(slot, "Cursed Blast"):
                                has_ability = True
                                ability_source = slot
                                break
                            
                            # Munkidori - Adrena-Brain (targeting, works from bench)
                            # Battle Cage only blocks BENCH targeting (checked in effects.py)
                            if slot.name == "Munkidori":
                                if any(card_def(e).type == "Darkness" for e in slot.energy) and check_slot_ability(slot, "Adrena-Brain"):
                                    has_ability = True
                                    ability_source = slot
                                    break

                    if has_ability:
                        # Determine if ability requires targeting (check ability_source, not just active)
                        is_targeted = False
                        if ability_source and ability_source.name in ("Dusknoir", "Munkidori"):
                            is_targeted = True

                        if act.c is None:
                             # Untargeted action
                             # Allow if ability is Untargeted OR we are supporting untargeted fallback
                             # Prefer to force targeting for targeted abilities, but allow Untargeted as fallback to Active (c=5)
                             # Actually strict: If targeted, only allow targeted actions?
                             # Let's allow Untargeted for all (defaulting to c=5/Active) to avoid blocking naive moves.
                             mask[i] = 1
                        else:
                             # Targeted action (c=0..5)
                             # Only allow if ability IS targeted
                             if is_targeted:
                                 # Validate target existence
                                 target_exists = False
                                 if act.c == 5 and op.active.name: target_exists = True
                                 elif 0 <= act.c < 5 and op.bench[act.c].name: target_exists = True
                                 
                                 if target_exists: mask[i] = 1
                             else:
                                 # Ability is untargeted (e.g. Pidgeot), but action has target.
                                 # Block nonsense targeted actions for untargeted abilities to reduce noise.
                                 mask[i] = 0
                
                elif act.kind == "ATTACK":
                    # ATTACK - separate from ability logic!
                    # Rule: Player going first (P0) cannot attack on their first turn
                    if gs.turn_number == 1 and gs.turn_player == 0:
                        continue
                    
                    # Status: Cannot attack if Asleep or Paralyzed
                    if me.active.status.get("asleep") or me.active.status.get("paralyzed") or me.active.status.get("cant_attack"):
                        continue
                        
                    attacks = self._get_active_attacks(gs.turn_player)
                    atk_idx = act.b if act.b is not None else 0
                    if atk_idx >= len(attacks):
                        continue
                        
                    atk = attacks[atk_idx]
                    if self._check_energy(me.active.energy, atk.cost):
                        # Special condition: Fan Rotom can only attack if a stadium is in play
                        if me.active.name == "Fan Rotom" and "Evolution" not in atk.name:
                            has_stadium = me.stadium is not None or op.stadium is not None
                            if not has_stadium:
                                continue
                        mask[i] = 1
                        mask[i] = 1
                
                elif act.kind == "ATTACK_MAGNITUDE":
                    # Attack with variable amount (e.g. Gholdengo ex)
                    if me.active.name is None: continue
                    if gs.turn_number == 1 and gs.turn_player == 0: continue
                    if me.active.status.get("asleep") or me.active.status.get("paralyzed") or me.active.status.get("cant_attack"): continue
                    
                    attacks = self._get_active_attacks(gs.turn_player)
                    atk_idx = act.b if act.b is not None else 0
                    if atk_idx >= len(attacks): continue
                    
                    atk = attacks[atk_idx]
                    if not self._check_energy(me.active.energy, atk.cost): continue
                    
                    # Specific Logic for Magnitude Users
                    needed_amount = act.c
                    if me.active.name == "Gholdengo ex" and atk_idx == 0:
                        # Make It Rain: Needs enough basic energy in hand to satisfy amount
                        basic_energy_in_hand = sum(1 for c in me.hand if card_def(c).supertype == "Energy" and card_def(c).subtype == "Basic")
                        if basic_energy_in_hand < needed_amount: continue
                        mask[i] = 1
                    elif me.active.name == "Mega Charizard X ex" and atk_idx == 0:
                        # Inferno X: Needs enough Fire Energy on field to satisfy amount
                        all_slots = [me.active] + [s for s in me.bench if s.name]
                        fire_energy_count = sum(
                            1 for slot in all_slots 
                            for e in slot.energy 
                            if e in ("Basic Fire Energy", "Fire Energy")
                        )
                        if fire_energy_count < needed_amount: continue
                        mask[i] = 1
                    else:
                        # Mask out for everyone else to keep search space clean
                        continue

                continue

        return mask

    def step(self, action_idx: int):
        gs = self._gs
        me = gs.players[gs.turn_player]
        op = gs.players[1 - gs.turn_player]
        
        # Ensure _ko_this_turn initialized
        if not hasattr(self, "_ko_this_turn"): self._ko_this_turn = False

        if gs.done:
            return featurize(gs), 0.0, True, False, {"action_mask": self.action_mask()}

        act = ACTION_TABLE[action_idx]
        mask = self.action_mask()
        
        # Check legality
        if mask[action_idx] == 0:
            reward = -1.0 
            act = Action("PASS")
        else:
            reward = 0.0

        # --- Apply action ---
        if act.kind == "PASS":
            pass

        elif act.kind == "PLAY_BASIC_TO_BENCH":
            me.hand.remove(act.a)
            me.bench[act.b].name = act.a
            me.bench[act.b].turn_played = gs.turn_number 
            me.bench[act.b].damage = 0
            me.bench[act.b].energy = []
            me.bench[act.b].tool = None
            me.bench[act.b].status = {k:False for k in me.bench[act.b].status}

        elif act.kind == "EVOLVE_ACTIVE":
            me.hand.remove(act.a)
            me.active.name = act.a
            me.active.turn_played = gs.turn_number 
            # Evolution clears status
            me.active.status = {k:False for k in me.active.status}
            apply_on_evolve_ability(self, gs.turn_player, act.a)

        elif act.kind == "EVOLVE_BENCH":
            me.hand.remove(act.a)
            me.bench[act.b].name = act.a
            me.bench[act.b].turn_played = gs.turn_number
            # Evolution clears status
            me.bench[act.b].status = {k:False for k in me.bench[act.b].status}
            apply_on_evolve_ability(self, gs.turn_player, act.a)

        elif act.kind == "ATTACH_ACTIVE":
            me.hand.remove(act.a)
            me.active.energy.append(act.a)
            me.energy_attached = True
            
            # Apply special energy effects
            apply_energy_effect(self, gs.turn_player, act.a, attached_to_active=True)

        elif act.kind == "ATTACH_BENCH":
            me.hand.remove(act.a)
            me.bench[act.b].energy.append(act.a)
            me.energy_attached = True
            
            # Apply special energy effects
            apply_energy_effect(self, gs.turn_player, act.a, attached_to_active=False, bench_slot=act.b)

        elif act.kind == "ATTACH_TOOL_ACTIVE":
            me.hand.remove(act.a)
            me.active.tool = act.a

        elif act.kind == "ATTACH_TOOL_BENCH":
            me.hand.remove(act.a)
            me.bench[act.b].tool = act.a

        elif act.kind == "PLAY_TRAINER":
            card = act.a
            me.hand.remove(card)
            me.discard_pile.append(card) 
            
            cd = card_def(card)
            
            if cd.subtype == "Supporter":
                me.supporter_used = True
                me.total_supporters_played += 1  # Track for opponent modeling
                me.turns_since_supporter = 0  # Reset counter
            if cd.subtype == "Stadium":
                me.stadium = card
                op.stadium = None # Only one stadium in play
            
            # Track searches for opponent modeling
            if card == "Ultra Ball":
                target = act.b
                if target == 0:
                    me.last_searched_type = "Basic"
                elif target == 1:
                    me.last_searched_type = "Stage1"
                elif target == 2:
                    me.last_searched_type = "Stage2"
                elif target in (3, 4):
                    me.last_searched_type = "Evolution"
                elif target == 5:
                    me.last_searched_type = "KeyAttacker"
                    me.likely_threats = ["Alakazam", "Charizard ex", "Pidgeot ex"]
            
            apply_trainer_effect(self, gs.turn_player, card, target_idx=act.b, secondary_idx=act.c, tertiary_idx=act.d, quaternary_idx=act.e, quinary_idx=act.f)

        elif act.kind == "RETREAT_TO":
            cost = card_def(me.active.name).retreat_cost
            if me.active.tool == "Air Balloon": cost = max(0, cost - 2)
            for _ in range(cost):
                if me.active.energy:
                    # Preference: Discard Basic Energy first, then Special Energy
                    # Find first basic
                    found_idx = -1
                    for i, e in enumerate(me.active.energy):
                        if card_def(e).subtype == "Basic":
                            found_idx = i; break
                    
                    if found_idx == -1: # No basic, discard anything
                        me.discard_pile.append(me.active.energy.pop())
                    else:
                        me.discard_pile.append(me.active.energy.pop(found_idx))
            
            idx = act.b
            # Retreat clears status
            me.active.status = {k:False for k in me.active.status}
            me.active, me.bench[idx] = me.bench[idx], me.active

        elif act.kind == "USE_ACTIVE_ABILITY":
            # Search for which pokemon is using the ability (Active or Bench if Fan Rotom)
            using_pokemon = None
            if me.active.name:
                using_pokemon = me.active.name
                # Special case: if Active has no activation ability but Fan Rotom is on bench
                # (We'll check bench below if Active doesn't trigger)
            
            # If Fan Rotom is on bench and turn 1, use it.
            # This is a bit of a hack because we don't have separate bench ability actions yet.
            is_first_turn = (gs.turn_number == 1 if gs.turn_player == 0 else gs.turn_number == 2)
            if is_first_turn and not getattr(me, "fan_call_used", False):
                # Check if Fan Rotom is Active or on Bench
                fan_rotom_slot = None
                if me.active.name == "Fan Rotom":
                    fan_rotom_slot = me.active
                else:
                    for s in me.bench:
                        if s.name == "Fan Rotom" and not s.ability_used_this_turn:
                            fan_rotom_slot = s
                            break
                
                if fan_rotom_slot and not fan_rotom_slot.ability_used_this_turn:
                    apply_ability_effect(self, gs.turn_player, "Fan Rotom")
                    me.fan_call_used = True
                    fan_rotom_slot.ability_used_this_turn = True
            
            # If Fan Call not used/available, try Active's ability
            elif me.active.name and not me.active.ability_used_this_turn:
                # Validate Fan Rotom again to prevent fallback execution on invalid turns
                valid_to_call = True
                if me.active.name == "Fan Rotom":
                     time_ok = (gs.turn_number == 1 if gs.turn_player == 0 else gs.turn_number == 2)
                     if not time_ok: valid_to_call = False
                
                if valid_to_call:
                    # Pass target index
                    tgt = act.c if act.c is not None else 6
                    apply_ability_effect(self, gs.turn_player, me.active.name, target_idx=tgt)
                    # Mark this Pokemon's ability as used
                    me.active.ability_used_this_turn = True
                # Legacy check for Pidgeot
                if me.active.name == "Pidgeot ex":
                    me.quick_search_used = True

        elif act.kind == "ATTACK":
            if me.active.name:
                # Confusion: 50% chance to fail and take 30 damage
                confused_failed = False
                if me.active.status.get("confused"):
                    if random.random() < 0.5:
                        me.active.damage += 30
                        confused_failed = True
                
                if not confused_failed:
                    reward += self._perform_attack(me, op, gs, act)
            
            # Also check Self-KO (e.g. Cursed Blast / Recoil / Confusion fail)
            if me.active.name:
                me_hp = card_def(me.active.name).hp
                if me.active.damage >= me_hp or me.active.damage >= 900:
                    self._handle_knockout(me)

        # --- Check Win Conditions ---
        # --- Check Win Conditions ---
        if gs.players[0].prizes_taken >= 6:
            gs.done = True
            gs.winner = 0
            gs.win_reason = "Taken all Prize cards"
            reward += 10.0
        elif gs.players[1].prizes_taken >= 6:
            gs.done = True
            gs.winner = 1
            gs.win_reason = "Taken all Prize cards"
            reward -= 10.0

        # --- Turn progression ---
        if not gs.done and act.kind in ("PASS", "ATTACK"):
            self._end_turn()

        # Timeout - WINNER IS WHOEVER TOOK MORE PRIZES (no more stall advantage!)
        if gs.turn_number > self.max_turns:
            gs.done = True
            p0_prizes = gs.players[0].prizes_taken
            p1_prizes = gs.players[1].prizes_taken
            if p0_prizes > p1_prizes:
                gs.winner = 0
                gs.win_reason = f"Timeout - More prizes ({p0_prizes} vs {p1_prizes})"
                reward += 5.0
            elif p1_prizes > p0_prizes:
                gs.winner = 1
                gs.win_reason = f"Timeout - More prizes ({p1_prizes} vs {p0_prizes})"
                reward -= 5.0
            else:
                # True draw - nobody wins
                gs.win_reason = f"Timeout - Tied prizes ({p0_prizes} each)"
                reward += 0.0


        obs = featurize(gs)
        
        # === REWARD SHAPING ===
        # Add dense rewards to help MCTS and RL agents learn basic mechanics
        # 1. Penalize PASS if other actions available
        if act.kind == "PASS":
            mask = self.action_mask()
            # If there are other actions (index > 0)
            if np.sum(mask[1:]) > 0:
                 reward -= 0.5
                 # Heuristic: If we can attach energy, big penalty to pass
                 if any(ACTION_TABLE[i].kind.startswith("ATTACH") for i in range(len(mask)) if mask[i]):
                     reward -= 0.5

        # 2. Reward Playing/Bench
        elif act.kind == "PLAY_BASIC_TO_BENCH":
            reward += 0.2

        # 3. Reward Energy Attachment
        elif act.kind.startswith("ATTACH_") and not act.kind.startswith("ATTACH_TOOL"):
            reward += 0.4
            
        # 4. Reward Evolution
        elif act.kind.startswith("EVOLVE"):
            reward += 0.4
            # Bonus for Stage 2
            if "Stage2" in card_def(act.a).subtype:
                reward += 0.2

        # 5. Reward Trainers
        elif act.kind == "PLAY_TRAINER":
            reward += 0.1
            if act.a == "Rare Candy": reward += 0.3
            if act.a in ("Boss's Orders", "Prime Catcher"): reward += 0.2

        # 6. Reward Abilities
        elif act.kind == "USE_ACTIVE_ABILITY":
            reward += 0.15

        # 7. Attack Reward (already partially in self._perform_attack but boost it)
        # _perform_attack returns damage-based reward. 
        # We ensure meaningful actions get +epsilon
        
        info = {"action_mask": self.action_mask()}
        if gs.done:
            info["winner"] = gs.winner
            info["win_reason"] = getattr(gs, "win_reason", "Unknown")
        return obs, float(reward), gs.done, False, info

    def _get_active_attacks(self, player_idx):
        me = self._gs.players[player_idx]
        if not me.active.name: return []
        attacks = list(card_def(me.active.name).attacks)
        # Check Tool
        if me.active.tool == "Technical Machine: Evolution":
             attacks.append(Attack("Evolution", 0, ["Colorless"]))
        return attacks

    def _perform_attack(self, me, op, gs, act):
        reward = 0.0
        attacks = self._get_active_attacks(gs.turn_player)
        atk_idx = act.b if act.b is not None else 0
        if not attacks: return 0.0
        if atk_idx >= len(attacks): atk_idx = 0
            
        atk = attacks[atk_idx]
        
        # Special Logic: TM Evolution
        if atk.name == "Evolution":
            # Search deck for evolutions for up to 2 benched pokemon
            evolved_count = 0
            # Identify candidates in bench
            for s_idx, slot in enumerate(me.bench):
                if slot.name and evolved_count < 2:
                    # Find evolution in deck
                    evo_card = None
                    evo_idx = -1
                    for i, c in enumerate(me.deck):
                        cd = card_def(c)
                        if cd.supertype == "Pokemon" and cd.evolves_from == slot.name:
                            evo_card = c
                            evo_idx = i
                            break
                    
                    if evo_card:
                        me.deck.pop(evo_idx)
                        old_name = slot.name
                        slot.name = evo_card
                        slot.turn_played = gs.turn_number
                        slot.status = {k:False for k in slot.status}
                        apply_on_evolve_ability(self, gs.turn_player, evo_card, from_hand=False)
                        evolved_count += 1
                        # print(f"    -> TM Evolution: {old_name} evolved into {evo_card}")
            
            random.shuffle(me.deck)
            return reward

        base_dmg = atk.damage
        tgt = act.c if act.c is not None else 6
        final_dmg, atk_flags = apply_attack_effect(self, gs.turn_player, me.active.name, base_dmg, atk_idx=atk_idx, target_idx=tgt)
        
        if op.active.name and final_dmg > 0:
            # Apply damage reduction (e.g., Protect Charge)
            if op.active.damage_reduction > 0:
                final_dmg -= op.active.damage_reduction
                if final_dmg < 0: final_dmg = 0
            
            # Weakness and Resistance (unless attack ignores it)
            if not atk_flags.get('ignore_weakness_resistance', False):
                op_def = card_def(op.active.name)
                me_def = card_def(me.active.name)
                
                # Weakness: x2
                if op_def.weakness == me_def.type:
                    final_dmg *= 2
                
                # Resistance: -30
                if op_def.resistance == me_def.type:
                    final_dmg -= 30
                    if final_dmg < 0: final_dmg = 0
                
            op.active.damage += final_dmg
            
            # Check KO using opponent's actual HP
            op_hp = card_def(op.active.name).hp
            if op.active.damage >= op_hp:
                self._handle_knockout(op)
                if not gs.done: 
                    reward += 0.5
        return reward

    def _pokemon_checkup(self):
        """Standard TCG Checkup phase between turns."""
        gs = self._gs
        for p_idx in range(2):
            p = gs.players[p_idx]
            mon = p.active
            if not mon.name: continue
            
            # Poison: 10 dmg
            if mon.status.get("poisoned"):
                mon.damage += 10
                
            # Burned: 20 dmg + flip to remove
            if mon.status.get("burned"):
                mon.damage += 20
                if random.random() < 0.5:
                    mon.status["burned"] = False
                    
            # Asleep: Flip to wake up
            if mon.status.get("asleep"):
                if random.random() < 0.5:
                    mon.status["asleep"] = False
                    
            # Paralyzed: Removed at the end of the affected player's turn
            if mon.status.get("paralyzed") and gs.turn_player == p_idx:
                mon.status["paralyzed"] = False
            
            # Can't Attack: Clears at the END of the player's NEXT turn (effectively persisting 1 full round)
            # Simplification: If it's my turn ending, and I have cant_attack, clear it?
            # Blaze Blitz: "During your next turn, this Pokemon can't use Blaze Blitz."
            # Means: Use on Turn 1. Op Turn 1. My Turn 2 (Can't use). End of My Turn 2 -> Clear.
            if mon.status.get("cant_attack") and gs.turn_player == p_idx:
                 mon.status["cant_attack"] = False
                 
            # Reset temporary buffs
            if gs.turn_player == p_idx:
                p.fighting_buff = False
                # Clear damage reduction when it's YOUR turn (was applied during opponent's turn)
                p.active.damage_reduction = 0

            # Check for KO
            total_hp = card_def(mon.name).hp
            if mon.damage >= total_hp:
                self._handle_knockout(p)

    def _draw_cards(self, player: PlayerState, count: int):
        # Draw from actual deck
        to_draw = min(count, len(player.deck))
        drawn = []
        for _ in range(to_draw):
            c = player.deck.pop()
            player.hand.append(c)
            drawn.append(c)
            
        if player == self._gs.players[0] and drawn:
            # print(f"    -> You Drew: \033[96m{drawn}\033[0m")
            pass
            
        if player.deck_count <= 0 and to_draw < count:
            # Deck out check
            pass # Currently handled at start of turn turn_number increment checks if draw possible 


    def _handle_knockout(self, victim_player: PlayerState):
        gs = self._gs
        victim_name = victim_player.active.name
        
        # Determine prizes to take
        prizes_to_take = 1
        if victim_name:
             cd = card_def(victim_name)
             # Mega Pokémon (including Mega ex) give 3 Prizes
             if victim_name.startswith("Mega "):
                 prizes_to_take = 3
             # Other Rule Box Pokémon (ex, V, etc.) give 2 Prizes
             elif cd.has_rule_box:
                 prizes_to_take = 2

        # Discard stuff
        if victim_name:
            victim_player.discard_pile.append(victim_name)
        victim_player.discard_pile.extend(victim_player.active.energy)
        if victim_player.active.tool:
            victim_player.discard_pile.append(victim_player.active.tool)

        # Victim active is cleared
        victim_player.active = PokemonSlot()

        # Opponent takes prize
        attacker_idx = 0 if victim_player == gs.players[1] else 1
        attacker = gs.players[attacker_idx]
        
        for _ in range(prizes_to_take):
            if attacker.prizes:
                # Real TCG: pick a prize. Here: pop first.
                prize_card = attacker.prizes.pop(0)
                attacker.hand.append(prize_card)

        # Mark KO for Fezandipiti logic
        self._ko_this_turn = True

        # Victim must promote bench
        promoted = False
        for i, s in enumerate(victim_player.bench):
            if s.name is not None:
                victim_player.active = s
                victim_player.bench[i] = PokemonSlot()  # clear bench slot
                promoted = True
                break

        if not promoted:
            # BENCH OUT! Victim loses.
            gs.done = True
            gs.winner = attacker_idx
            gs.win_reason = f"Opponent Benched Out ({victim_name} KO'd)"

    def _end_turn(self):
        gs = self._gs
        # reset per-turn flags for current player
        p = gs.players[gs.turn_player]
        
        # Track for opponent modeling - if no supporter used, increment counter
        if not p.supporter_used:
            p.turns_since_supporter += 1
        
        p.supporter_used = False
        p.energy_attached = False
        p.quick_search_used = False
        p.ability_used_this_turn = False  # Legacy global flag (keep for compatibility)
        p.fan_call_used = False
        p.fighting_buff = False
        
        # Reset per-Pokemon ability flags
        p.active.ability_used_this_turn = False
        for slot in p.bench:
            slot.ability_used_this_turn = False

        self._pokemon_checkup()
        
        # Discard TMs at end of turn if attached
        if p.active.tool == "Technical Machine: Evolution":
            p.discard_pile.append(p.active.tool)
            p.active.tool = None
        for s in p.bench:
            if s.tool == "Technical Machine: Evolution":
                p.discard_pile.append(s.tool)
                s.tool = None

        next_ko = getattr(self, "_ko_this_turn", False)
        self._ko_this_turn = False 

        # switch turn
        gs.turn_player = 1 - gs.turn_player
        gs.ko_last_turn = next_ko
        gs.turn_number += 1

        next_p = gs.players[gs.turn_player]
        if next_p.deck_count > 0:
            self._draw_cards(next_p, 1)
        else:
            gs.done = True
            gs.winner = 1 - gs.turn_player
            gs.win_reason = "Deck Out (Cannot draw at start of turn)"

        # scripted opponent (dumb)
        if gs.turn_player == 1 and self.scripted_opponent and not gs.done:
            op = gs.players[1]
            me = gs.players[0]
            
            # 1. Play basics
            cards_to_play = []
            for i, c in enumerate(list(op.hand)): # copy
                cd = card_def(c)
                if cd.supertype == "Pokemon" and cd.subtype == "Basic":
                    if len([s for s in op.bench if s.name]) < 5:
                        cards_to_play.append(c)
            
            for c in cards_to_play:
                # Find empty slot
                for b_idx in range(5):
                    if op.bench[b_idx].name is None:
                        op.bench[b_idx].name = c
                        op.bench[b_idx].damage = 0
                        op.bench[b_idx].energy = []
                        if c in op.hand: op.hand.remove(c)
                        # print(f">> Opponent Action: \033[33mPLAY_BASIC_TO_BENCH {c}\033[0m")
                        break

            # 2. Attach energy
            energies = [c for c in op.hand if card_def(c).supertype == "Energy"]
            if energies and not op.energy_attached:
                c = energies[0]
                # attach to active
                if op.active.name:
                    op.active.energy.append(c)
                    op.hand.remove(c)
                    op.energy_attached = True
                    # print(f">> Opponent Action: \033[95mATTACH_ACTIVE {c}\033[0m")
            
            # 3. Attack if possible
            if op.active.name:
                attacks = card_def(op.active.name).attacks
                if attacks:
                    atk = attacks[0]
                    # cost check
                    is_fan_rotom = op.active.name == "Fan Rotom"
                    has_stadium = me.stadium is not None or op.stadium is not None
                    
                    if self._check_energy(op.active.energy, atk.cost) and (not is_fan_rotom or has_stadium):
                         # Damage
                         dmg = atk.damage
                         dmg, _ = apply_attack_effect(self, 1, op.active.name, dmg)  # Unpack tuple
                         
                         # Weakness/Resistance
                         if me.active.name:
                             me_def = card_def(me.active.name)
                             op_def = card_def(op.active.name)
                             if me_def.weakness == op_def.type:
                                 dmg *= 2
                             if me_def.resistance == op_def.type:
                                 dmg -= 30
                                 if dmg < 0: dmg = 0
                                 
                         me.active.damage += dmg
                         # print(f">> Opponent Action: \033[91mATTACK {atk.name} for {dmg}\033[0m")
                         
                         hp = card_def(me.active.name).hp
                         if me.active.damage >= hp:
                             self._handle_knockout(me)

            self._end_turn()  # recursive call to pass turn back

    def _check_energy(self, attached_cards: list[str], cost: list[str]) -> bool:
        """Helper to check if energy requirements are met."""
        if not cost: return True
        
        # Count types in cost
        cost_counts = Counter(cost)
        
        # Get types of attached cards
        attached_types = []
        for c in attached_cards:
            attached_types.append(card_def(c).type)
        attached_counts = Counter(attached_types)
        
        # Check specific types
        for t, count in cost_counts.items():
            if t == "Colorless": continue
            if attached_counts[t] < count:
                return False
            attached_counts[t] -= count
            
        # Check Colorless (remaining energy)
        remaining_count = sum(attached_counts.values())
        if remaining_count < cost_counts["Colorless"]:
            return False
            
        return True


# =============================================================================
# MCTS
# Source: tcg/mcts.py
# =============================================================================

"""
MCTS Implementation for Pokemon TCG.
Improved version with:
1. Deck-agnostic board state heuristics
2. Policy-guided rollouts (instead of random)
3. Neural network value function option
4. Visit distribution for training
"""


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


# =============================================================================
# SCRIPTED_AGENT
# Source: tcg/scripted_agent.py
# =============================================================================

"""
Scripted (Heuristic-based) Agents for Pokemon TCG.

These agents provide external pressure during self-play training to prevent
overfitting to the neural network's own quirks. By mixing in opponents with
known-good strategies, the trained model must learn general game skills.
"""


class ScriptedAgent:
    """
    Rule-based opponent with configurable strategy.
    
    Strategies:
    - "aggressive": Prioritize attacking and taking prizes
    - "evolution_rush": Prioritize evolution, then attack
    - "defensive": Prioritize benching and building board
    - "energy_first": Prioritize energy attachment
    - "random": Random legal actions (baseline)
    """
    
    def __init__(self, strategy: str = "aggressive"):
        self.strategy = strategy
        self.name = f"scripted_{strategy}"
    
    def select_action(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        """
        Select an action based on heuristics.
        
        Args:
            env: The PTCGEnv environment
            obs: Current observation (not used by most strategies)
            mask: Boolean mask of legal actions
            
        Returns:
            Action index
        """
        legal_actions = np.where(mask)[0]
        
        if len(legal_actions) == 0:
            return 0  # PASS if nothing legal (shouldn't happen)
        
        if len(legal_actions) == 1:
            return legal_actions[0]  # Only one choice
        
        if self.strategy == "aggressive":
            return self._aggressive_action(env, mask, legal_actions)
        elif self.strategy == "evolution_rush":
            return self._evolution_rush_action(env, mask, legal_actions)
        elif self.strategy == "defensive":
            return self._defensive_action(env, mask, legal_actions)
        elif self.strategy == "energy_first":
            return self._energy_first_action(env, mask, legal_actions)
        elif self.strategy == "control":
            return self._control_action(env, mask, legal_actions)
        elif self.strategy == "combo":
            return self._combo_action(env, mask, legal_actions)
        elif self.strategy == "random":
            return self._random_action(legal_actions)
        else:
            return self._random_action(legal_actions)
    
    def _get_actions_by_kind(self, legal_actions: np.ndarray, kind: str) -> List[int]:
        """Get legal action indices that match the given kind."""
        return [i for i in legal_actions if ACTION_TABLE[i].kind == kind]
    
    def _get_actions_by_kinds(self, legal_actions: np.ndarray, kinds: List[str]) -> List[int]:
        """Get legal action indices that match any of the given kinds."""
        return [i for i in legal_actions if ACTION_TABLE[i].kind in kinds]
    
    def _random_action(self, legal_actions: np.ndarray) -> int:
        """Random legal action (excluding PASS if other options exist)."""
        non_pass = [a for a in legal_actions if a != 0]
        if non_pass:
            return random.choice(non_pass)
        return legal_actions[0]
    
    def _aggressive_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Aggressive strategy:
        1. Attack if possible (prioritize high damage)
        2. Attach energy to active
        3. Evolve active
        4. Use abilities
        5. Play trainers
        6. Bench basics
        7. Random non-pass
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. ATTACK - highest priority
        attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
        if attacks:
            return random.choice(attacks)
        
        # 2. Attach energy to ACTIVE
        attach_active = self._get_actions_by_kind(legal_actions, "ATTACH_ACTIVE")
        if attach_active:
            return random.choice(attach_active)
        
        # 3. Evolve active
        evolve_active = self._get_actions_by_kind(legal_actions, "EVOLVE_ACTIVE")
        if evolve_active:
            # Prioritize Stage 2 evolutions
            stage2_evos = [a for a in evolve_active 
                         if card_def(ACTION_TABLE[a].a).subtype == "Stage2"]
            if stage2_evos:
                return random.choice(stage2_evos)
            return random.choice(evolve_active)
        
        # 4. Use abilities
        abilities = self._get_actions_by_kind(legal_actions, "USE_ACTIVE_ABILITY")
        if abilities:
            return random.choice(abilities)
        
        # 5. Play trainers (supporters and items)
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        if trainers:
            # Prioritize draw supporters
            for a in trainers:
                card = ACTION_TABLE[a].a
                if card in ("Hilda", "Dawn", "Lillie's Determination", "Iono", "Arven"):
                    return a
            return random.choice(trainers)
        
        # 6. Bench basics
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        if bench:
            return random.choice(bench)
        
        # 7. Attach energy to bench (if active doesn't need it)
        attach_bench = self._get_actions_by_kinds(legal_actions, 
            ["ATTACH_BENCH_0", "ATTACH_BENCH_1", "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"])
        if attach_bench:
            return random.choice(attach_bench)
        
        # 8. Evolve bench
        evolve_bench = self._get_actions_by_kinds(legal_actions,
            ["EVOLVE_BENCH_0", "EVOLVE_BENCH_1", "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"])
        if evolve_bench:
            return random.choice(evolve_bench)
        
        # Default: random non-pass
        return self._random_action(legal_actions)
    
    def _evolution_rush_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Evolution Rush strategy:
        1. Evolve (prioritize Stage 2 > Stage 1)
        2. Play Rare Candy if possible
        3. Bench evolution basics (Abra, Charmander, etc.)
        4. Use search/draw trainers
        5. Attack only when evolved
        6. Attach energy
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Evolve - HIGHEST priority
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        if all_evolves:
            # Prioritize Stage 2
            stage2 = [a for a in all_evolves 
                     if card_def(ACTION_TABLE[a].a).subtype == "Stage2"]
            if stage2:
                return random.choice(stage2)
            return random.choice(all_evolves)
        
        # 2. Play Rare Candy
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        rare_candy = [a for a in trainers if ACTION_TABLE[a].a == "Rare Candy"]
        if rare_candy:
            return random.choice(rare_candy)
        
        # 3. Bench evolution starters
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        evo_starters = [a for a in bench 
                       if ACTION_TABLE[a].a in ("Abra", "Charmander", "Pidgey", "Duskull", "Gimmighoul")]
        if evo_starters:
            return random.choice(evo_starters)
        
        # 4. Search/Draw trainers
        search_cards = [a for a in trainers 
                       if ACTION_TABLE[a].a in ("Hilda", "Dawn", "Arven", "Ultra Ball", 
                                                "Nest Ball", "Buddy-Buddy Poffin")]
        if search_cards:
            return random.choice(search_cards)
        
        # 5. Attack only if active is evolved
        if me.active.name:
            active_def = card_def(me.active.name)
            if active_def.subtype in ("Stage1", "Stage2"):
                attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
                if attacks:
                    return random.choice(attacks)
        
        # 6. Attach energy
        attach = self._get_actions_by_kinds(legal_actions, [
            "ATTACH_ACTIVE", "ATTACH_BENCH_0", "ATTACH_BENCH_1", 
            "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"
        ])
        if attach:
            return random.choice(attach)
        
        # 7. Any bench action
        if bench:
            return random.choice(bench)
        
        # Default
        return self._random_action(legal_actions)
    
    def _defensive_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Defensive strategy:
        1. Bench as many Pokemon as possible
        2. Evolve to increase HP
        3. Retreat damaged Pokemon
        4. Attach energy to bench
        5. Use draw supporters
        6. Attack only as last resort
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Bench basics - build the board
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        if bench:
            return random.choice(bench)
        
        # 2. Evolve for HP
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        if all_evolves:
            return random.choice(all_evolves)
        
        # 3. Retreat if damaged
        if me.active.name and me.active.damage > 0:
            retreat = self._get_actions_by_kind(legal_actions, "RETREAT_TO")
            if retreat:
                return random.choice(retreat)
        
        # 4. Attach energy to bench (save active for later)
        attach_bench = self._get_actions_by_kinds(legal_actions, 
            ["ATTACH_BENCH_0", "ATTACH_BENCH_1", "ATTACH_BENCH_2", 
             "ATTACH_BENCH_3", "ATTACH_BENCH_4"])
        if attach_bench:
            return random.choice(attach_bench)
        
        # 5. Draw supporters
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        draw_supporters = [a for a in trainers 
                          if ACTION_TABLE[a].a in ("Hilda", "Dawn", "Iono", 
                                                   "Lillie's Determination")]
        if draw_supporters:
            return random.choice(draw_supporters)
        
        # 6. Use abilities
        abilities = self._get_actions_by_kind(legal_actions, "USE_ACTIVE_ABILITY")
        if abilities:
            return random.choice(abilities)
        
        # 7. Attach to active if nothing else
        attach_active = self._get_actions_by_kind(legal_actions, "ATTACH_ACTIVE")
        if attach_active:
            return random.choice(attach_active)
        
        # 8. Attack as last resort
        attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
        if attacks:
            return random.choice(attacks)
        
        return self._random_action(legal_actions)
    
    def _energy_first_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Energy First strategy:
        1. Always attach energy if possible
        2. Use energy search trainers
        3. Build up for big attacks
        4. Attack when fully powered
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Attach energy - always
        attach = self._get_actions_by_kinds(legal_actions, [
            "ATTACH_ACTIVE", "ATTACH_BENCH_0", "ATTACH_BENCH_1", 
            "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"
        ])
        if attach:
            # Prefer active
            attach_active = self._get_actions_by_kind(legal_actions, "ATTACH_ACTIVE")
            if attach_active:
                return random.choice(attach_active)
            return random.choice(attach)
        
        # 2. Energy search trainers
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        energy_trainers = [a for a in trainers 
                         if ACTION_TABLE[a].a in ("Arven", "Energy Search", 
                                                  "Earthen Vessel", "Superior Energy Retrieval")]
        if energy_trainers:
            return random.choice(energy_trainers)
        
        # 3. Attack if active has enough energy
        if me.active.name:
            active_def = card_def(me.active.name)
            if active_def.attacks:
                cost = len(active_def.attacks[0].cost) if active_def.attacks[0].cost else 0
                if len(me.active.energy) >= cost:
                    attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
                    if attacks:
                        return random.choice(attacks)
        
        # 4. Evolve
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        if all_evolves:
            return random.choice(all_evolves)
        
        # 5. Bench
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        if bench:
            return random.choice(bench)
        
        # 6. Other trainers
        if trainers:
            return random.choice(trainers)
        
        return self._random_action(legal_actions)
    
    def _control_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Control strategy (Hand Disruption):
        1. Prioritize Iono, Unfair Stamp (disrupt opponent's hand)
        2. Use Boss's Orders to target key threats
        3. Slow evolution, focus on disruption
        4. Attack weak targets
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Hand disruption supporters
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        disruption = [a for a in trainers 
                     if ACTION_TABLE[a].a in ("Iono", "Unfair Stamp")]
        if disruption:
            return random.choice(disruption)
        
        # 2. Boss's Orders - target weak bench Pokemon
        boss = [a for a in trainers if ACTION_TABLE[a].a == "Boss's Orders"]
        if boss:
            return random.choice(boss)
        
        # 3. Bench basics
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        if bench:
            return random.choice(bench)
        
        # 4. Evolve
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        if all_evolves:
            return random.choice(all_evolves)
        
        # 5. Attach energy
        attach = self._get_actions_by_kinds(legal_actions, [
            "ATTACH_ACTIVE", "ATTACH_BENCH_0", "ATTACH_BENCH_1", 
            "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"
        ])
        if attach:
            return random.choice(attach)
        
        # 6. Attack
        attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
        if attacks:
            return random.choice(attacks)
        
        return self._random_action(legal_actions)
    
    def _combo_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Combo strategy (Alakazam Mind Ruler):
        1. Build large hand (draw supporters)
        2. Evolve to Alakazam quickly (Rare Candy priority)
        3. Get Pidgeot ex for consistency
        4. Attack with Mind Ruler when hand is big (7+)
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Draw supporters to build hand
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        draw_cards = [a for a in trainers 
                     if ACTION_TABLE[a].a in ("Hilda", "Dawn", "Lillie's Determination")]
        
        # Only draw if hand is small
        if len(me.hand) < 6 and draw_cards:
            return random.choice(draw_cards)
        
        # 2. Rare Candy to Alakazam
        rare_candy = [a for a in trainers if ACTION_TABLE[a].a == "Rare Candy"]
        if rare_candy:
            return random.choice(rare_candy)
        
        # 3. Evolve (priority: Alakazam > Pidgeot)
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        alakazam_evolve = [a for a in all_evolves 
                         if ACTION_TABLE[a].a in ("Alakazam", "Alakazam ex")]
        if alakazam_evolve:
            return random.choice(alakazam_evolve)
        if all_evolves:
            return random.choice(all_evolves)
        
        # 4. Use Pidgeot ex ability
        abilities = self._get_actions_by_kind(legal_actions, "USE_ACTIVE_ABILITY")
        if abilities:
            return random.choice(abilities)
        
        # 5. Bench Abra/Pidgey
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        evo_starters = [a for a in bench 
                       if ACTION_TABLE[a].a in ("Abra", "Pidgey")]
        if evo_starters:
            return random.choice(evo_starters)
        if bench:
            return random.choice(bench)
        
        # 6. Search trainers
        search = [a for a in trainers 
                 if ACTION_TABLE[a].a in ("Ultra Ball", "Nest Ball", "Arven")]
        if search:
            return random.choice(search)
        
        # 7. Attach energy
        attach = self._get_actions_by_kinds(legal_actions, [
            "ATTACH_ACTIVE", "ATTACH_BENCH_0", "ATTACH_BENCH_1", 
            "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"
        ])
        if attach:
            return random.choice(attach)
        
        # 8. Attack if hand is big (Mind Ruler scales with hand size)
        if len(me.hand) >= 5:
            attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
            if attacks:
                return random.choice(attacks)
        
        return self._random_action(legal_actions)


# Pre-built agents for easy access
SCRIPTED_AGENTS = {
    "aggressive": ScriptedAgent("aggressive"),
    "evolution_rush": ScriptedAgent("evolution_rush"),
    "defensive": ScriptedAgent("defensive"),
    "energy_first": ScriptedAgent("energy_first"),
    "control": ScriptedAgent("control"),
    "combo": ScriptedAgent("combo"),
    "random": ScriptedAgent("random"),
}


def get_scripted_agent(strategy: str = "aggressive") -> ScriptedAgent:
    """Get a scripted agent by strategy name."""
    if strategy in SCRIPTED_AGENTS:
        return SCRIPTED_AGENTS[strategy]
    return ScriptedAgent(strategy)


# =============================================================================
# ALPHA_RANK
# Source: alpha_rank.py
# =============================================================================

"""
Alpha-Rank Implementation for Pokemon TCG Agent Evaluation

Alpha-Rank is a game-theoretic method for ranking agents that handles:
- Non-transitive relationships (Rock-Paper-Scissors dynamics)
- Population-based training evaluation
- More principled ranking than ELO

Based on: "Alpha-Rank: Multi-Agent Evaluation by Evolution" (Omidshafiei et al., 2019)
"""


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


# =============================================================================
# TRAIN_ADVANCED
# Source: train_advanced.py
# =============================================================================

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

# Replay recording imports
# Alpha-Rank for game-theoretic rankings
# Scripted agents for external pressure

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

