from __future__ import annotations
import torch
import numpy as np
import time
import random
from tcg.env import PTCGEnv
from tcg.train_bc import PolicyNet
from tcg.actions import ACTION_TABLE
from tcg.state import GameState

def print_board(gs: GameState):
    p0 = gs.players[0]
    p1 = gs.players[1]
    
    print(f"\n============================================================")
    print(f"Time: Turn {gs.turn_number} (Turn Player: P{gs.turn_player})")
    print(f"============================================================")
    
    # P1 (Opponent)
    print(f"\033[93mOPPONENT (P1)\033[0m Prizes: {p1.prizes_taken}/6 | Deck: {p1.deck_count}")
    print(f"  Hand ({len(p1.hand)}): {p1.hand}")
    print(f"  Active: {p1.active.name} (HP: {get_hp(p1.active.name) - p1.active.damage}/{get_hp(p1.active.name)} | Energy: {p1.active.energy})")
    bench_str_p1 = ", ".join([f"{s.name}({s.damage})" for s in p1.bench if s.name])
    print(f"  Bench: [{bench_str_p1}]")
    
    print("-" * 60)
    
    # P0 (Player/Agent)
    print(f"\033[92mPLAYER (P0)\033[0m   Prizes: {p0.prizes_taken}/6 | Deck: {p0.deck_count}")
    print(f"  Hand ({len(p0.hand)}): {p0.hand}")
    print(f"  Active: {p0.active.name} (HP: {get_hp(p0.active.name) - p0.active.damage}/{get_hp(p0.active.name)} | Energy: {p0.active.energy})")
    bench_str_p0 = ", ".join([f"{s.name}({s.damage})" for s in p0.bench if s.name])
    print(f"  Bench: [{bench_str_p0}]")
    print(f"============================================================")

def get_hp(name):
    # Simplified mock for visualizer if card_def isn't imported or easy to reach
    # Ideally should use tcg.cards.card_def but we need to import it properly or mock it.
    # visualize_agent had get_hp helper.
    from tcg.cards import CARD_REGISTRY
    if name in CARD_REGISTRY:
        return CARD_REGISTRY[name].hp
    return 100

def run_visual_match():
    # Load Policy
    try:
        fname = "rl_policy.pt"
        print(f"Loading {fname}...")
        checkpoint = torch.load(fname)
        obs_dim = checkpoint["obs_dim"]
        n_actions = checkpoint["n_actions"]
        model = PolicyNet(obs_dim, n_actions)
        model.load_state_dict(checkpoint["state_dict"])
    except:
        print("rl_policy.pt not found, loading bc_policy.pt...")
        fname = "bc_policy.pt"
        checkpoint = torch.load(fname)
        obs_dim = checkpoint["obs_dim"] 
        n_actions = checkpoint["n_actions"]
        model = PolicyNet(obs_dim, n_actions)
        model.load_state_dict(checkpoint["state_dict"])
    
    model.eval()
    
    # Init Env (Scripted False for Self-Play)
    env = PTCGEnv(scripted_opponent=False, max_turns=200)

    # Decks
    deck_p0 = []
    deck_p0.extend(["Abra"] * 4)
    deck_p0.extend(["Kadabra"] * 3)
    deck_p0.extend(["Alakazam"] * 4)
    deck_p0.extend(["Dunsparce"] * 4)
    deck_p0.extend(["Dudunsparce"] * 4)
    deck_p0.extend(["Fan Rotom"] * 2)
    deck_p0.extend(["Psyduck"] * 1)
    deck_p0.extend(["Fezandipiti ex"] * 1)
    deck_p0.extend(["Hilda"] * 4)
    deck_p0.extend(["Dawn"] * 4)
    deck_p0.extend(["Boss's Orders"] * 3)
    deck_p0.extend(["Lillie's Determination"] * 2)
    deck_p0.extend(["Tulip"] * 1)
    deck_p0.extend(["Buddy-Buddy Poffin"] * 4)
    deck_p0.extend(["Rare Candy"] * 3)
    deck_p0.extend(["Nest Ball"] * 2)
    deck_p0.extend(["Night Stretcher"] * 2)
    deck_p0.extend(["Wondrous Patch"] * 2)
    deck_p0.extend(["Enhanced Hammer"] * 2)
    deck_p0.extend(["Battle Cage"] * 3)
    deck_p0.extend(["Basic Psychic Energy"] * 3)
    deck_p0.extend(["Enriching Energy"] * 1)
    deck_p0.extend(["Jet Energy"] * 1)

    deck_p1 = []
    deck_p1.extend(["Charmander"] * 4)
    deck_p1.extend(["Charmeleon"] * 3)
    deck_p1.extend(["Charizard ex"] * 3)
    deck_p1.extend(["Pidgey"] * 2)
    deck_p1.extend(["Pidgeotto"] * 1)
    deck_p1.extend(["Pidgeot ex"] * 2)
    deck_p1.extend(["Chi-Yu"] * 1)
    deck_p1.extend(["Fezandipiti ex"] * 1)
    deck_p1.extend(["Psyduck"] * 1)
    deck_p1.extend(["Tatsugiri"] * 1)
    deck_p1.extend(["Arven"] * 4)
    deck_p1.extend(["Dawn"] * 3)
    deck_p1.extend(["Boss's Orders"] * 3)
    deck_p1.extend(["Iono"] * 2)
    deck_p1.extend(["Lillie's Determination"] * 1)
    deck_p1.extend(["Professor Turo's Scenario"] * 1)
    deck_p1.extend(["Buddy-Buddy Poffin"] * 4)
    deck_p1.extend(["Ultra Ball"] * 3)
    deck_p1.extend(["Rare Candy"] * 2)
    deck_p1.extend(["Super Rod"] * 2)
    deck_p1.extend(["Counter Catcher"] * 1)
    deck_p1.extend(["Technical Machine: Evolution"] * 2)
    deck_p1.extend(["Maximum Belt"] * 1)
    deck_p1.extend(["Battle Cage"] * 2)
    deck_p1.extend(["Fire Energy"] * 7)
    deck_p1.extend(["Jet Energy"] * 2)
    deck_p1.extend(["Mist Energy"] * 1)

    obs, info = env.reset(options={"decks": [deck_p0, deck_p1]})
    done = False
    
    print("\nStarting Visual Match (Randomized Sampling)...")
    last_turn_printed = -1
    
    while not done:
        gs = env._gs
        turn_p = gs.turn_player
        
        # Determine sampling strategy (random game?)
        # We'll use stochastic sampling (categorical) instead of argmax
        # to show variety.
        
        mask = info["action_mask"]
        
        with torch.no_grad():
            logits = model(torch.from_numpy(obs).float().unsqueeze(0))
            # Apply Mask
            logits_np = logits.numpy()[0]
            logits_np[mask == 0] = -1e9
            
            # Probabilities
            probs = torch.softmax(torch.tensor(logits_np), dim=0)
            
            # Heuristic Fallback logic (prevents getting stuck passing if untrained/confused)
            constructive = []
            if mask[0] == 1: 
                 for i, valid in enumerate(mask):
                     if valid and i != 0: 
                        a = ACTION_TABLE[i]
                        # Broad check for anything useful
                        if a.kind != "PASS":
                            constructive.append(i)
            
            # Sample from Distribution
            dist = torch.distributions.Categorical(probs)
            act_idx = dist.sample().item()
            action = ACTION_TABLE[act_idx]

            # Aggressive Heuristic: If PASS is chosen but constructive moves exist, force one.
            if action.kind == "PASS" and constructive:
                 act_idx = random.choice(constructive)
                 action = ACTION_TABLE[act_idx]
                 print(f"  \033[90m[Fallback] Override PASS -> {action.kind}\033[0m")
        
        # Output formatting
        if gs.turn_number != last_turn_printed:
            print_board(gs)
            last_turn_printed = gs.turn_number
            
        role = "P0 (You)" if turn_p == 0 else "P1 (Opp)"
        
        # Color coding
        cx = ""
        rst = ""
        kind = action.kind
        if "EVOLVE" in kind: cx = "\033[92m"      # Green
        elif "PLAY_BASIC" in kind: cx = "\033[33m" # Orange
        elif "PLAY_TRAINER" in kind: cx = "\033[94m" # Blue
        elif "ATTACH" in kind: cx = "\033[95m"     # Magenta
        elif "ATTACK" in kind: cx = "\033[91m"     # Red
        elif "ABILITY" in kind: cx = "\033[96m"    # Cyan
        elif "PASS" in kind: cx = "\033[90m"       # Grey
        if cx: rst = "\033[0m"
        
        print(f">> {role} Action: {cx}{action.kind} {action.a} {action.b if action.b is not None else ''}{rst}")
        
        obs, reward, done, _, info = env.step(act_idx)
        time.sleep(0.1)

    print_board(env._gs)
    winner = env._gs.winner
    print(f"\n*** P{winner} WINS! ***")

if __name__ == "__main__":
    run_visual_match()
