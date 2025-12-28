"""
Detailed Replay Recorder for Pokemon TCG RL Agent
Captures full game state at each step for visual replay.
Includes Stadiums, Energy Lists, and Action Consequence Analysis.
"""
from __future__ import annotations
import torch
import numpy as np
import json
import os
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from tcg.env import PTCGEnv
from tcg.train_bc import PolicyNet
from tcg.actions import ACTION_TABLE
from tcg.cards import card_def


@dataclass
class PokemonState:
    name: Optional[str]
    hp_current: int
    hp_max: int
    energy_count: int
    damage: int
    is_ex: bool = False
    pokemon_type: str = "Colorless"
    status: dict = field(default_factory=dict)
    tool: Optional[str] = None
    attached_cards: List[str] = field(default_factory=list)
    
@dataclass
class PlayerSnapshot:
    active: Optional[PokemonState]
    bench: List[Optional[PokemonState]]
    hand: List[str]
    hand_size: int
    deck_size: int
    discard_size: int
    prizes_remaining: int
    prizes_taken: int
    prizes: List[str] = None 
    stadium: Optional[str] = None
    discard_pile: List[str] = field(default_factory=list)  # Full discard pile contents
    # Opponent Model Predictions
    turns_since_supporter: int = 0
    last_searched_type: Optional[str] = None
    predicted_threats: List[str] = field(default_factory=list)  # Predicted key cards
    
@dataclass
class GameFrame:
    frame_id: int
    turn_number: int
    turn_player: int
    phase: str
    
    # Action Details
    action_kind: Optional[str] = None
    action_card: Optional[str] = None
    action_target: Optional[int] = None
    action_description: str = ""
    action_result: str = ""  # NEW: The outcome (e.g. "+2 Cards")
    is_ability: bool = False # NEW: Flag for UI styling
    
    p0: Optional[PlayerSnapshot] = None
    p1: Optional[PlayerSnapshot] = None
    winner: Optional[int] = None
    win_reason: str = ""
    
@dataclass
class VisualReplay:
    game_id: int
    p0_deck_name: str
    p1_deck_name: str
    winner: Optional[int] = None
    total_turns: int = 0
    total_frames: int = 0
    frames: List[GameFrame] = field(default_factory=list)
    p0_deck_list: List[str] = field(default_factory=list)
    p1_deck_list: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)

# --- Helper Functions ---

def get_ability_name(card_name):
    """Maps card names to their ability for the log."""
    # Add common meta cards here
    abilities = {
        "Pidgeot ex": "Quick Search",
        "Charizard ex": "Infernal Reign",
        "Fezandipiti ex": "Flip the Script",
        "Munkidori": "Adrena-Brain",
        "Rotom V": "Instant Charge",
        "Lumineon V": "Luminous Sign",
        "Squawkabilly ex": "Squawk and Seize",
        "Comfey": "Flower Selecting",
        "Kirlia": "Refinement",
        "Gardevoir ex": "Psychic Embrace",
        "Drakloak": "Recon Directive",
        "Teal Mask Ogerpon ex": "Teal Dance",
        "Radiant Greninja": "Concealed Cards"
    }
    return abilities.get(card_name, "Ability")

def get_attached_cards(slot) -> List[str]:
    cards = []
    if hasattr(slot, 'energy') and isinstance(slot.energy, list):
        cards.extend(slot.energy)
    if hasattr(slot, 'tool') and slot.tool:
        cards.append(slot.tool)
    return cards

def capture_pokemon_state(slot) -> Optional[PokemonState]:
    if slot.name is None: return None
    cd = card_def(slot.name)
    energy_count = len(slot.energy) if isinstance(slot.energy, list) else 0
    return PokemonState(
        name=slot.name,
        hp_current=max(0, cd.hp - slot.damage),
        hp_max=cd.hp,
        energy_count=energy_count,
        damage=slot.damage,
        is_ex="ex" in cd.tags,
        pokemon_type=cd.type,
        status=dict(slot.status) if hasattr(slot, 'status') else {},
        tool=slot.tool if hasattr(slot, 'tool') else None,
        attached_cards=get_attached_cards(slot)
    )

def capture_player_state(player, global_stadium=None) -> PlayerSnapshot:
    stadium_card = player.stadium if hasattr(player, 'stadium') and player.stadium else global_stadium
    
    # Capture opponent model predictions
    turns_since_sup = getattr(player, 'turns_since_supporter', 0)
    last_searched = getattr(player, 'last_searched_type', None)
    
    # Predict threats based on what we can see
    predicted_threats = []
    if last_searched == "KeyAttacker":
        predicted_threats.append("Key Attacker Soon!")
    if last_searched in ("Stage1", "Stage2", "Evolution"):
        predicted_threats.append("Evolution Coming")
    if turns_since_sup == 0:
        predicted_threats.append("May Have Hand Disruption")
    if turns_since_sup >= 3:
        predicted_threats.append("Likely No Supporters")
    
    return PlayerSnapshot(
        active=capture_pokemon_state(player.active),
        bench=[capture_pokemon_state(s) for s in player.bench],
        hand=list(player.hand),
        hand_size=len(player.hand),
        deck_size=len(player.deck),
        discard_size=len(player.discard_pile),
        prizes_remaining=len(player.prizes),
        prizes_taken=6 - len(player.prizes),
        prizes=list(player.prizes),
        stadium=stadium_card,
        discard_pile=list(player.discard_pile),
        turns_since_supporter=turns_since_sup,
        last_searched_type=last_searched,
        predicted_threats=predicted_threats
    )

def describe_action(act, player_state, opponent_state) -> tuple[str, bool]:
    """Returns (Description, IsAbility)"""
    kind = act.kind
    card = act.a
    target = act.b
    
    if kind == "PASS":
        return "Ends Turn", False
    elif kind == "PLAY_BASIC_TO_BENCH":
        return f"Bench {card}", False
    elif kind == "EVOLVE_ACTIVE":
        return f"Evolve Active -> {card}", False
    elif kind == "EVOLVE_BENCH":
        return f"Evolve Bench {target} -> {card}", False
    elif kind == "ATTACH_ACTIVE":
        return f"Attach {card} to Active", False
    elif kind == "ATTACH_BENCH":
        return f"Attach {card} to Bench {target}", False
    elif kind == "PLAY_TRAINER":
        return f"Play {card}", False
    elif kind == "RETREAT_TO":
        return f"Retreat to Slot {target}", False
    elif kind == "USE_ACTIVE_ABILITY":
        name = player_state.active.name if player_state.active else "Active"
        ability = get_ability_name(name)
        return f"Use {name}'s {ability}", True
    elif kind == "USE_ABILITY": # Generic/Bench ability
        # Try to infer which mon from target index if available, else generic
        return f"Use Ability (Slot {target})", True
    elif kind == "ATTACK":
        move = "Attack"
        if player_state.active:
             # Basic lookup for common attackers in this context
             if player_state.active.name == "Charizard ex": move = "Burning Darkness"
             elif player_state.active.name == "Alakazam": move = "Mind Jack"
             elif player_state.active.name == "Pidgeot ex": move = "Gust" # Unlikely attack but possible
        return f"{player_state.active.name} used {move}!", False
    
    return f"{kind} {card or ''}", False

def analyze_result(p0_before, p0_after, p1_before, p1_after, turn_player):
    """Compare states to generate a result string."""
    changes = []
    
    # Identify "My" state vs "Opponent" state
    my_before = p0_before if turn_player == 0 else p1_before
    my_after = p0_after if turn_player == 0 else p1_after
    op_before = p1_before if turn_player == 0 else p0_before
    op_after = p1_after if turn_player == 0 else p0_after

    # 1. Hand Delta
    d_hand = my_after.hand_size - my_before.hand_size
    if d_hand > 0: changes.append(f"+{d_hand} Cards to Hand")
    elif d_hand < 0: changes.append(f"{d_hand} Cards from Hand")

    # 2. Deck Delta (Searching)
    d_deck = my_after.deck_size - my_before.deck_size
    # If deck went down but hand didn't go up equally, we searched or milled
    if d_deck < 0 and d_hand == 0: changes.append(f"Searched/Milled {-d_deck} cards")

    # 3. Energy Attachment (Board Total)
    def count_energy(p): return (p.active.energy_count if p.active else 0) + sum(b.energy_count for b in p.bench if b)
    d_en = count_energy(my_after) - count_energy(my_before)
    if d_en > 0: changes.append(f"Attached {d_en} Energy")

    # 4. Damage Dealt
    def count_dmg(p): return (p.active.damage if p.active else 0) + sum(b.damage for b in p.bench if b)
    d_dmg = count_dmg(op_after) - count_dmg(op_before)
    if d_dmg > 0: changes.append(f"Dealt {d_dmg} Dmg")
    
    # 5. Healing
    d_heal = count_dmg(my_after) - count_dmg(my_before)
    if d_heal < 0: changes.append(f"Healed {-d_heal} HP")
    
    # 6. Prizes
    d_prize = my_after.prizes_taken - my_before.prizes_taken
    if d_prize > 0: changes.append(f"TAKEN {d_prize} PRIZE(S)!")

    # 7. Discard Recovery (Heuristic)
    # If we played a card (Hand -1, Discard +1) but Discard didn't grow, we recovered something.
    d_discard = my_after.discard_size - my_before.discard_size
    if d_discard <= 0:
        # We likely recovered cards OR shuffled them back
        recovered = my_before.discard_size - my_after.discard_size + 1 # +1 for the played card
        if recovered > 0:
            changes.append(f"Recovered {recovered} card(s)")

    if not changes: return ""
    return ", ".join(changes)

# --- Deck Definitions ---

def build_deck(deck_name: str) -> list:
    """Build a deck by name. Matches training decks."""
    deck = []
    
    if deck_name == "alakazam":
        deck.extend(["Abra"] * 4); deck.extend(["Kadabra"] * 3); deck.extend(["Alakazam"] * 4)
        deck.extend(["Dunsparce"] * 4); deck.extend(["Dudunsparce"] * 4)
        deck.extend(["Fan Rotom"] * 2); deck.extend(["Psyduck"] * 1); deck.extend(["Fezandipiti ex"] * 1)
        deck.extend(["Hilda"] * 4); deck.extend(["Dawn"] * 4); deck.extend(["Boss's Orders"] * 3)
        deck.extend(["Lillie's Determination"] * 2); deck.extend(["Tulip"] * 1)
        deck.extend(["Buddy-Buddy Poffin"] * 4); deck.extend(["Rare Candy"] * 3)
        deck.extend(["Nest Ball"] * 2); deck.extend(["Night Stretcher"] * 2)
        deck.extend(["Wondrous Patch"] * 2); deck.extend(["Enhanced Hammer"] * 2)
        deck.extend(["Battle Cage"] * 3); deck.extend(["Basic Psychic Energy"] * 3)
        deck.extend(["Enriching Energy"] * 1); deck.extend(["Jet Energy"] * 1)
        
    elif deck_name == "charizard":
        deck.extend(["Charmander"] * 3); deck.extend(["Charmeleon"] * 2); deck.extend(["Charizard ex"] * 2)
        deck.extend(["Pidgey"] * 2); deck.extend(["Pidgeotto"] * 2); deck.extend(["Pidgeot ex"] * 2)
        deck.extend(["Psyduck"] * 1); deck.extend(["Shaymin"] * 1); deck.extend(["Tatsugiri"] * 1)
        deck.extend(["Munkidori"] * 1); deck.extend(["Chi-Yu"] * 1)
        deck.extend(["Gouging Fire ex"] * 1); deck.extend(["Fezandipiti ex"] * 1)
        deck.extend(["Lillie's Determination"] * 4); deck.extend(["Arven"] * 4)
        deck.extend(["Boss's Orders"] * 3); deck.extend(["Iono"] * 2); deck.extend(["Professor Turo's Scenario"] * 1)
        deck.extend(["Buddy-Buddy Poffin"] * 4); deck.extend(["Ultra Ball"] * 3); deck.extend(["Rare Candy"] * 2)
        deck.extend(["Super Rod"] * 2); deck.extend(["Counter Catcher"] * 1); deck.extend(["Energy Search"] * 1)
        deck.extend(["Unfair Stamp"] * 1); deck.extend(["Technical Machine: Evolution"] * 2)
        deck.extend(["Artazon"] * 1); deck.extend(["Fire Energy"] * 5); deck.extend(["Mist Energy"] * 2)
        deck.extend(["Darkness Energy"] * 1); deck.extend(["Jet Energy"] * 1)
        
    elif deck_name == "gholdengo":
        deck.extend(["Gimmighoul"] * 4); deck.extend(["Gholdengo ex"] * 4)
        deck.extend(["Solrock"] * 4); deck.extend(["Lunatone"] * 2)
        deck.extend(["Genesect ex"] * 2); deck.extend(["Hop's Cramorant"] * 2)
        deck.extend(["Fezandipiti ex"] * 1); deck.extend(["Fan Rotom"] * 1)
        deck.extend(["Lillie's Determination"] * 4); deck.extend(["Arven"] * 4)
        deck.extend(["Boss's Orders"] * 3); deck.extend(["Iono"] * 2)
        deck.extend(["Buddy-Buddy Poffin"] * 4); deck.extend(["Ultra Ball"] * 3)
        deck.extend(["Rare Candy"] * 2); deck.extend(["Super Rod"] * 1)
        deck.extend(["Counter Catcher"] * 1); deck.extend(["Superior Energy Retrieval"] * 2)
        deck.extend(["Fighting Gong"] * 2); deck.extend(["Earthen Vessel"] * 2)
        deck.extend(["Basic Metal Energy"] * 4); deck.extend(["Basic Fighting Energy"] * 6)
        
    else:
        raise ValueError(f"Unknown deck: {deck_name}. Options: alakazam, charizard, gholdengo")
    
    return deck

DECK_NAMES = {
    "alakazam": "Alakazam",
    "charizard": "Charizard", 
    "gholdengo": "Gholdengo"
}

# --- Main Recording Logic ---

def record_visual_replay(model, device, game_id=0, verbose=True, deck0="alakazam", deck1="charizard"):
    # Match training config roughly (1500 steps ~ 150 turns if 10 steps/turn)
    env = PTCGEnv(scripted_opponent=False, max_turns=200)
    
    # Build decks from names
    deck_p0 = build_deck(deck0)
    deck_p1 = build_deck(deck1)
    
    p0_name = f"{DECK_NAMES[deck0]} (P0)"
    p1_name = f"{DECK_NAMES[deck1]} (P1)"

    replay = VisualReplay(game_id, p0_name, p1_name, None)
    replay.p0_deck_list = deck_p0
    replay.p1_deck_list = deck_p1
    
    obs, info = env.reset(options={"decks": [deck_p0, deck_p1]})
    done = False
    frame_id = 0
    
    # Snapshot 0
    current_stadium = getattr(env._gs, 'stadium', None)
    initial_p0 = capture_player_state(env._gs.players[0], current_stadium)
    initial_p1 = capture_player_state(env._gs.players[1], current_stadium)

    replay.frames.append(GameFrame(
        frame_id=0, turn_number=0, turn_player=env._gs.turn_player, phase="start",
        action_description="Start", p0=initial_p0, p1=initial_p1
    ))
    frame_id += 1
    
    max_steps = 400
    steps = 0
    
    while not done and steps < max_steps:
        tp = env._gs.turn_player
        
        # 1. Predict
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(obs_t)
            # Handle models that return (policy_logits, value) tuple
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
        mask_np = info["action_mask"]
        mask = torch.from_numpy(mask_np).float().to(device)
        masked_logits = torch.where(mask.unsqueeze(0) > 0, logits, torch.ones_like(logits) * -1e9)
        probs = torch.softmax(masked_logits, dim=1)[0]
        act_idx = masked_logits.argmax(dim=1).item()
        
        # Apply the same inference heuristics as training
        # Categorize valid actions
        attack_actions = [i for i, a in enumerate(ACTION_TABLE) if a.kind == 'ATTACK' and mask_np[i] == 1]
        attach_active = [i for i, a in enumerate(ACTION_TABLE) if a.kind == 'ATTACH_ACTIVE' and mask_np[i] == 1]
        constructive = [i for i, a in enumerate(ACTION_TABLE) 
                        if mask_np[i] == 1 and a.kind not in ('PASS',)]
        
        # Override PASS with better actions
        if act_idx == 0 and constructive:  # PASS is action 0
            if attack_actions:
                act_idx = attack_actions[0]
            elif attach_active:
                act_idx = attach_active[0]
            else:
                # Pick highest probability constructive action
                act_idx = max(constructive, key=lambda i: probs[i].item())
        
        act = ACTION_TABLE[act_idx]
        
        # 2. Capture Before
        curr_stadium = getattr(env._gs, 'stadium', None)
        p0_before = capture_player_state(env._gs.players[0], curr_stadium)
        p1_before = capture_player_state(env._gs.players[1], curr_stadium)
        
        # 3. Step
        obs, _, done, _, info = env.step(act_idx)
        steps += 1
        
        # 4. Capture After
        # FIX: Extract .name if stadium is an object, otherwise use it directly
        raw_stadium = getattr(env._gs, 'stadium', None)
        if raw_stadium and hasattr(raw_stadium, 'name'):
            curr_stadium = raw_stadium.name
        else:
            curr_stadium = raw_stadium if isinstance(raw_stadium, str) else None

        p0_after = capture_player_state(env._gs.players[0], curr_stadium)
        p1_after = capture_player_state(env._gs.players[1], curr_stadium)
        
        # 5. Generate Text & Analysis
        me_state = p0_before if tp == 0 else p1_before
        op_state = p1_before if tp == 0 else p0_before
        desc_str, is_abil = describe_action(act, me_state, op_state)
        result_str = analyze_result(p0_before, p0_after, p1_before, p1_after, tp)
        
        # Determine the action_card for sidebar display
        action_card = None
        if act.kind in ("ATTACK", "USE_ACTIVE_ABILITY"):
            # Show the active Pokemon for attacks/abilities
            action_card = me_state.active.name if me_state.active else None
        elif act.kind in ("PLAY_TRAINER", "PLAY_BASIC_TO_BENCH", "EVOLVE_ACTIVE", 
                          "EVOLVE_BENCH", "ATTACH_ACTIVE", "ATTACH_BENCH"):
            # Show the card being played
            action_card = act.a
        elif act.kind == "RETREAT_TO":
            # Show what we're retreating to
            target_idx = act.b
            if target_idx is not None and 0 <= target_idx < len(me_state.bench):
                bench_mon = me_state.bench[target_idx]
                if bench_mon:
                    action_card = bench_mon.name
        
        replay.frames.append(GameFrame(
            frame_id=frame_id,
            turn_number=env._gs.turn_number,
            turn_player=tp,
            phase="action",
            action_kind=act.kind,
            action_card=action_card,  # Now properly set!
            action_description=desc_str,
            action_result=result_str,
            is_ability=is_abil,
            p0=p0_after,
            p1=p1_after
        ))
        frame_id += 1

    # End Frame
    winner = env._gs.winner
    replay.winner = winner
    replay.total_frames = len(replay.frames)
    return replay

def load_model(path: str, device: torch.device, n_actions: int):
    """Load either PolicyNet, PolicyValueNet, or AdvancedPolicyValueNet from checkpoint."""
    from tcg.mcts import PolicyValueNet
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Detect model type by checking state dict keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Infer obs_dim from input layer shape
    if 'input_embed.0.weight' in state_dict:
        obs_dim = state_dict['input_embed.0.weight'].shape[1]  # [hidden, obs_dim]
    else:
        obs_dim = checkpoint.get('obs_dim', 156)
    
    if any("transformer" in k or "input_embed" in k for k in state_dict.keys()):
        # AdvancedPolicyValueNet from train_advanced.py
        print(f"Detected AdvancedPolicyValueNet architecture")
        from train_advanced import AdvancedPolicyValueNet
        model = AdvancedPolicyValueNet(obs_dim, n_actions).to(device)
    elif any("shared" in k or "value_head" in k for k in state_dict.keys()):
        # PolicyValueNet
        print(f"Detected PolicyValueNet architecture")
        model = PolicyValueNet(obs_dim, n_actions).to(device)
    else:
        # PolicyNet
        print(f"Detected PolicyNet architecture")
        model = PolicyNet(obs_dim, n_actions).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Pokemon TCG replays with trained model")
    parser.add_argument("--policy", type=str, default="advanced_policy.pt", help="Path to trained policy")
    parser.add_argument("--count", type=int, default=5, help="Number of games to record")
    parser.add_argument("--out", type=str, default="recorded_replays.json", help="Output file")
    parser.add_argument("--deck0", type=str, default="alakazam", 
                        choices=["alakazam", "charizard", "gholdengo", "random"],
                        help="Deck for Player 0 (or 'random')")
    parser.add_argument("--deck1", type=str, default="charizard", 
                        choices=["alakazam", "charizard", "gholdengo", "random"],
                        help="Deck for Player 1 (or 'random')")
    args = parser.parse_args()

    import random as rng
    deck_choices = ["alakazam", "charizard", "gholdengo"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model (auto-detects model type)
    n_actions = len(ACTION_TABLE)
    
    if os.path.exists(args.policy):
        model = load_model(args.policy, device, n_actions)
        print(f"Loaded policy: {args.policy}")
    else:
        print(f"Warning: policy {args.policy} not found, using random PolicyNet weights.")
        obs_dim = 156
        model = PolicyNet(obs_dim, n_actions).to(device)
    
    all_replays = []
    wins = {0: 0, 1: 0, None: 0}
    deck_wins = {}
    
    for i in range(args.count):
        # Select decks (random if specified)
        d0 = rng.choice(deck_choices) if args.deck0 == "random" else args.deck0
        d1 = rng.choice(deck_choices) if args.deck1 == "random" else args.deck1
        
        print(f"Recording Game {i+1}/{args.count}... ({DECK_NAMES[d0]} vs {DECK_NAMES[d1]})")
        replay = record_visual_replay(model, device, game_id=i+1, deck0=d0, deck1=d1)
        all_replays.append(replay.to_dict())
        
        p0_name = f"{DECK_NAMES[d0]} (P0)"
        p1_name = f"{DECK_NAMES[d1]} (P1)"
        
        winner_name = "DRAW/TIMEOUT"
        if replay.winner == 0: 
            winner_name = p0_name
            wins[0] += 1
            deck_wins[DECK_NAMES[d0]] = deck_wins.get(DECK_NAMES[d0], 0) + 1
        elif replay.winner == 1: 
            winner_name = p1_name
            wins[1] += 1
            deck_wins[DECK_NAMES[d1]] = deck_wins.get(DECK_NAMES[d1], 0) + 1
        else:
            wins[None] += 1
            
        print(f"  -> Winner: \033[92m{winner_name}\033[0m")

    # Save
    with open(args.out, "w") as f:
        json.dump({"total_games": args.count, "replays": all_replays}, f, indent=2)
    
    print(f"\nSummary of {args.count} games:")
    print(f"  P0 wins: {wins[0]}")
    print(f"  P1 wins: {wins[1]}")
    print(f"  Draws/Timeouts: {wins[None]}")
    if deck_wins:
        print(f"\n  Wins by deck:")
        for deck, w in sorted(deck_wins.items(), key=lambda x: -x[1]):
            print(f"    {deck}: {w}")
    print(f"\nSaved all replays to {args.out}")