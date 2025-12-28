from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tcg.actions import ACTION_TABLE, Action
from tcg.cards import card_def
from tcg.effects import (
    apply_ability_effect,
    apply_attack_effect,
    apply_energy_effect,
    apply_on_evolve_ability,
    apply_trainer_effect,
)
from tcg.state import MAX_BENCH, GameState, PlayerState, PokemonSlot, featurize


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
        import random
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
             from tcg.cards import Attack 
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
            
            import random
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
            import random
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
        from collections import Counter
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
