from __future__ import annotations
from typing import TYPE_CHECKING
import random
import os

if TYPE_CHECKING:
    from tcg.env import PTCGEnv
    from tcg.state import GameState, PlayerState
    from tcg.actions import Action

from tcg.cards import card_def, CARD_REGISTRY

# Helper to check if we should print verbose output
def should_print():
    return os.environ.get('PTCG_QUIET') != '1'
from tcg.state import PokemonSlot

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
