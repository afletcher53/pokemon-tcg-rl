# tcg/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from tcg.cards import card_def

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

