from __future__ import annotations
import json
import torch
import numpy as np
import time
from tcg.env import PTCGEnv
from tcg.train_bc import PolicyNet
from tcg.actions import ACTION_TABLE, Action
from tcg.state import GameState, featurize, PokemonSlot, PlayerState

def ascii_plot(values, height=10, width=50):
    if not values:
        return
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1
    
    # Normalize to 0..height-1
    normalized = [int((v - min_val) / range_val * (height - 1)) for v in values]
    
    # Resample to width
    step = len(values) / width
    sampled = []
    for i in range(width):
        idx = int(i * step)
        if idx < len(normalized):
            sampled.append(normalized[idx])
        else:
            sampled.append(normalized[-1])
            
    print(f"\nTraining Loss Visualization (Min: {min_val:.4f}, Max: {max_val:.4f})")
    print("-" * (width + 5))
    for h in range(height - 1, -1, -1):
        row = f"{str(h).rjust(3)} | "
        for v in sampled:
            row += "*" if v >= h else " "
        print(row)
    print("-" * (width + 5))
    print("Epochs ->")

def print_game_state(gs: GameState, turn: int):
    me = gs.players[0]
    op = gs.players[1]
    
    print(f"\n=== Turn {turn} (Player {gs.turn_player}) ===")
    print(f"Me (P0) - Prizes: {me.prizes_taken}/6 | Deck: {me.deck_count} | Hand: {len(me.hand)}")
    print(f"  Hand: {me.hand}") # Show hand content
    active_name = me.active.name if me.active.name else "None"
    print(f"  Active: {active_name} (HP: {get_hp(active_name) - me.active.damage}/{get_hp(active_name)} Energy: {me.active.energy})")
    bench_str = ", ".join([f"{s.name}({s.damage})" for s in me.bench if s.name])
    print(f"  Bench: {bench_str}")
    
    print(f"Op (P1) - Prizes: {op.prizes_taken}/6")
    opt_active = op.active.name if op.active.name else "None"
    print(f"  Active: {opt_active} (Dmg: {op.active.damage})")
    
def get_hp(name):
    from tcg.cards import CARD_REGISTRY
    if name in CARD_REGISTRY:
        return CARD_REGISTRY[name].hp
    return 100

def run_visual_game():
    # 1. Plot Loss
    try:
        with open("loss_history.json", "r") as f:
            loss_history = json.load(f)
        ascii_plot(loss_history)
    except FileNotFoundError:
        print("No loss history found.")

    # 2. Load Policy
    print("\nLoading Policy...")
    state = torch.load("bc_policy.pt")
    obs_dim = state["obs_dim"]
    n_actions = state["n_actions"]
    model = PolicyNet(obs_dim, n_actions)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # 3. Simulate Game
    print("\nStarting Simulation...")
    env = PTCGEnv(scripted_opponent=True, max_turns=120)
    
    # User-provided Alakazam Deck
    my_deck = []
    # Pokemon
    my_deck.extend(["Abra"] * 4)
    my_deck.extend(["Kadabra"] * 3)
    my_deck.extend(["Alakazam"] * 4)
    my_deck.extend(["Dunsparce"] * 3)
    my_deck.extend(["Dunsparce"] * 1) # Total 4
    my_deck.extend(["Dudunsparce"] * 4)
    my_deck.extend(["Fan Rotom"] * 2)
    my_deck.extend(["Psyduck"] * 1) # Need to add to registry if missing
    my_deck.extend(["Fezandipiti ex"] * 1)
    
    # Trainers
    my_deck.extend(["Hilda"] * 4)
    my_deck.extend(["Dawn"] * 4)
    my_deck.extend(["Boss's Orders"] * 3)
    my_deck.extend(["Lillie's Determination"] * 2)
    my_deck.extend(["Tulip"] * 1)
    my_deck.extend(["Buddy-Buddy Poffin"] * 4)
    my_deck.extend(["Rare Candy"] * 3)
    my_deck.extend(["Nest Ball"] * 2)
    my_deck.extend(["Night Stretcher"] * 2)
    my_deck.extend(["Wondrous Patch"] * 2)
    my_deck.extend(["Enhanced Hammer"] * 2)
    my_deck.extend(["Battle Cage"] * 3)
    
    # Energy
    my_deck.extend(["Basic Psychic Energy"] * 3)
    my_deck.extend(["Enriching Energy"] * 1)
    my_deck.extend(["Jet Energy"] * 1)
    
    # Fill remaining to 60 if count is off (it should be exactly 60)
    
    # Opponent Deck (Generic)
    # Opponent Deck (Charizard)
    op_deck = []
    # Pokemon (19)
    op_deck.extend(["Charmander"] * 4)
    op_deck.extend(["Charmeleon"] * 3)
    op_deck.extend(["Charizard ex"] * 3)
    op_deck.extend(["Pidgey"] * 2)
    op_deck.extend(["Pidgeotto"] * 1)
    op_deck.extend(["Pidgeot ex"] * 2)
    op_deck.extend(["Chi-Yu"] * 1)
    op_deck.extend(["Fezandipiti ex"] * 1)
    op_deck.extend(["Psyduck"] * 1)
    op_deck.extend(["Tatsugiri"] * 1)
    # Trainers (31)
    op_deck.extend(["Arven"] * 4)
    op_deck.extend(["Dawn"] * 3)
    op_deck.extend(["Boss's Orders"] * 3)
    op_deck.extend(["Iono"] * 2)
    op_deck.extend(["Lillie's Determination"] * 1)
    op_deck.extend(["Professor Turo's Scenario"] * 1)
    op_deck.extend(["Buddy-Buddy Poffin"] * 4)
    op_deck.extend(["Ultra Ball"] * 3)
    op_deck.extend(["Rare Candy"] * 2)
    op_deck.extend(["Super Rod"] * 2)
    op_deck.extend(["Counter Catcher"] * 1)
    op_deck.extend(["Technical Machine: Evolution"] * 2)
    op_deck.extend(["Maximum Belt"] * 1)
    op_deck.extend(["Battle Cage"] * 2)
    # Energy (10)
    op_deck.extend(["Fire Energy"] * 7)
    op_deck.extend(["Jet Energy"] * 2)
    op_deck.extend(["Mist Energy"] * 1)

    obs, info = env.reset(options={"decks": [my_deck, op_deck]})
    done = False
    
    last_turn = -1
    
    while not done:
        gs = env._gs
        # Print State Header only if turn changed
        if gs.turn_number != last_turn:
            print_game_state(gs, gs.turn_number)
            last_turn = gs.turn_number
        
        # Get action from model
        with torch.no_grad():
            logits = model(torch.from_numpy(obs).float().unsqueeze(0))
            # Masked argmax
            mask = info["action_mask"]
            logit_vals = logits.numpy()[0]
            logit_vals[mask == 0] = -1e9
            act_idx = np.argmax(logit_vals)
            
        action = ACTION_TABLE[act_idx]
        
        # --- Heuristic Fallback for Untrained Decks ---
        if action.kind == "PASS":
            # Check for constructive actions in the mask
            constructive_indices = []
            for i, val in enumerate(mask):
                if val == 1:
                    a = ACTION_TABLE[i]
                    if a.kind in ("PLAY_BASIC_TO_BENCH", "EVOLVE_ACTIVE", "EVOLVE_BENCH", 
                                  "ATTACH_ACTIVE", "ATTACH_BENCH", "PLAY_TRAINER", 
                                  "ATTACK", "USE_ACTIVE_ABILITY"):
                         constructive_indices.append(i)
            
            if constructive_indices:
                import random
                act_idx = random.choice(constructive_indices)
                action = ACTION_TABLE[act_idx]
                print(f"  [Fallback] Override PASS -> {action.kind} {action.a}")
        # -----------------------------------------------

        # Color coding: Green for Evolve, Orange for Benching
        cx = ""
        rst = ""
        kind = action.kind
        if "EVOLVE" in kind: cx = "\033[92m"      # Green
        elif "PLAY_BASIC" in kind: cx = "\033[33m" # Orange
        elif "PLAY_TRAINER" in kind: cx = "\033[94m" # Blue
        elif "ATTACH" in kind: cx = "\033[95m"     # Magenta
        elif "ATTACK" in kind: cx = "\033[91m"     # Red
        elif "ABILITY" in kind: cx = "\033[96m"    # Cyan
        elif "RETREAT" in kind: cx = "\033[97m"    # White
        elif "PASS" in kind: cx = "\033[90m"       # Grey
        
        if cx: rst = "\033[0m"
        
        print(f">> Agent Action: {cx}{action.kind} {action.a} {action.b if action.b is not None else ''}{rst}")
        
        obs, reward, done, _, info = env.step(act_idx)
        
        # Optional: Print simple state update if still same turn?
        # For now, let's keep it clean. Just actions until next turn header.
        
        time.sleep(0.1)

    if env._gs.winner == 0:
        print("\n*** PLAYER WINS! ***")
    else:
        print("\n*** OPPONENT WINS! ***")

if __name__ == "__main__":
    run_visual_game()
