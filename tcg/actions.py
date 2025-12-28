# tcg/actions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tcg.cards import CARD_REGISTRY, card_def


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
