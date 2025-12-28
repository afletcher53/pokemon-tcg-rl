import torch
import numpy as np
import random
from tcg.env import PTCGEnv
from train_advanced import AdvancedPolicyValueNet # Need the class definition
from tcg.actions import ACTION_TABLE

def run_validation():
    fname = "advanced_policy.pt"
    print(f"Loading {fname}...")
    try:
        checkpoint = torch.load(fname, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print("advanced_policy.pt not found. Is training running?")
        return

    obs_dim = checkpoint.get("obs_dim", 156)
    n_actions = checkpoint.get("n_actions", len(ACTION_TABLE))
    
    # Load model architecture
    model = AdvancedPolicyValueNet(obs_dim, n_actions)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    print(f"Model loaded. ELO: {checkpoint.get('elo', 'N/A')}")

    # Define Decks
    alakazam_deck = ["Abra"]*4 + ["Kadabra"]*3 + ["Alakazam"]*4 + ["Dunsparce"]*4 + ["Dudunsparce"]*4 + \
                    ["Fan Rotom"]*2 + ["Psyduck"]*1 + ["Fezandipiti ex"]*1 + ["Hilda"]*4 + ["Dawn"]*4 + \
                    ["Boss's Orders"]*3 + ["Lillie's Determination"]*2 + ["Tulip"]*1 + ["Buddy-Buddy Poffin"]*4 + \
                    ["Rare Candy"]*3 + ["Nest Ball"]*2 + ["Night Stretcher"]*2 + ["Wondrous Patch"]*2 + \
                    ["Enhanced Hammer"]*2 + ["Battle Cage"]*3 + ["Basic Psychic Energy"]*3 + ["Enriching Energy"]*1 + \
                    ["Jet Energy"]*1

    charizard_deck = ["Charmander"]*3 + ["Charmeleon"]*2 + ["Charizard ex"]*2 + ["Pidgey"]*2 + ["Pidgeotto"]*2 + \
                     ["Pidgeot ex"]*2 + ["Psyduck"]*1 + ["Shaymin"]*1 + ["Tatsugiri"]*1 + ["Munkidori"]*1 + \
                     ["Chi-Yu"]*1 + ["Gouging Fire ex"]*1 + ["Fezandipiti ex"]*1 + ["Lillie's Determination"]*4 + \
                     ["Arven"]*4 + ["Boss's Orders"]*3 + ["Iono"]*2 + ["Professor Turo's Scenario"]*1 + \
                     ["Buddy-Buddy Poffin"]*4 + ["Ultra Ball"]*3 + ["Rare Candy"]*2 + ["Super Rod"]*2 + \
                     ["Counter Catcher"]*1 + ["Energy Search"]*1 + ["Unfair Stamp"]*1 + \
                     ["Technical Machine: Evolution"]*2 + ["Artazon"]*1 + ["Fire Energy"]*5 + \
                     ["Mist Energy"]*2 + ["Darkness Energy"]*1 + ["Jet Energy"]*1

    env = PTCGEnv(scripted_opponent=False, max_turns=200)
    
    obs, info = env.reset(options={"decks": [alakazam_deck, charizard_deck]})
    
    done = False
    turn_count = 0
    alakazam_seen = 0
    
    print("\n--- Game Start ---")
    
    while not done:
        gs = env._gs
        turn_player = gs.turn_player
        mask = info["action_mask"]
        
        # Select action using model
        with torch.no_grad():
            policy_logits, _ = model(torch.from_numpy(obs).float().unsqueeze(0))
            # Mask invalid
            logits_np = policy_logits.numpy()[0]
            logits_np[mask == 0] = -1e9
            
            # Greedy for validation
            act_idx = np.argmax(logits_np)
            action = ACTION_TABLE[act_idx]
            
            # Check for Alakazam
            if action.kind == "EVOLVE" and action.a == "Alakazam":
                alakazam_seen += 1
                print(f"!!! P{turn_player} Evolved to Alakazam !!!")
        
        # Log first few turns
        if turn_count < 20:
             print(f"Turn {gs.turn_number} P{turn_player}: {action.kind} {action.a} {action.b or ''}")
             
        obs, reward, done, _, info = env.step(act_idx)
        if gs.turn_player == 0 and turn_player == 1: # New turn cycle roughly
             turn_count += 1
             
    print(f"\nGame Over. Winner: P{env._gs.winner}")
    print(f"Total Turns: {env._gs.turn_number}")
    print(f"Alakazam Evolutions: {alakazam_seen}")

if __name__ == "__main__":
    run_validation()
