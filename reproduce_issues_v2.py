
print("Importing...")
from tcg.env import PTCGEnv
print("Imported env")
from tcg.state import GameState, PokemonSlot, PlayerState
from tcg.cards import card_def
import numpy as np

def setup_test_env():
    print("Setting up env...")
    env = PTCGEnv()
    env.reset()
    return env

def test_fan_rotom_turn_3():
    print("--- Testing Fan Rotom Turn 3 ---")
    env = setup_test_env()
    gs = env._gs
    
    # We want to be on Turn 3 (Player 0's second turn)
    gs.turn_number = 3
    gs.turn_player = 0
    
    p0 = gs.players[0]
    p0.active.name = "Charmander" # Just something active
    
    # Put Fan Rotom on bench
    p0.bench[0].name = "Fan Rotom"
    
    # Check action mask
    print("Checking action mask...")
    mask = env.action_mask()
    
    from tcg.actions import ACTION_TABLE
    
    ability_idx = -1
    for i, act in enumerate(ACTION_TABLE):
        if act.kind == "USE_ACTIVE_ABILITY":
            ability_idx = i
            break
            
    if ability_idx == -1:
        print("Error: Could not find USE_ACTIVE_ABILITY action")
        return

    can_use = mask[ability_idx] == 1
    print(f"Turn {gs.turn_number}, Player {gs.turn_player}: Fan Rotom ability action masked? {can_use}")
    
    if can_use:
        print("FAILURE: Fan Rotom marked as usable on Turn 3!")
    else:
        print("SUCCESS: Fan Rotom correctly blocked on Turn 3.")

def test_night_stretcher_empty_discard():
    print("\n--- Testing Night Stretcher (Empty Discard) ---")
    env = setup_test_env()
    gs = env._gs
    p0 = gs.players[0]
    
    p0.hand = ["Night Stretcher"]
    p0.discard_pile = [] # Empty
    
    from tcg.actions import ACTION_TABLE
    ns_idx = -1
    for i, act in enumerate(ACTION_TABLE):
        if act.kind == "PLAY_TRAINER" and act.a == "Night Stretcher" and act.b == 6:
            ns_idx = i
            break
    
    mask = env.action_mask()
    can_play = mask[ns_idx] == 1
    print(f"Discard empty. Night Stretcher playable? {can_play}")
    
    if can_play:
        print("FAILURE: Night Stretcher allowed with empty discard.")
    else:
        print("SUCCESS: Night Stretcher blocked with empty discard.")

def test_night_stretcher_special_energy():
    print("\n--- Testing Night Stretcher (Special Energy Only) ---")
    env = setup_test_env()
    gs = env._gs
    p0 = gs.players[0]
    
    p0.hand = ["Night Stretcher"]
    p0.discard_pile = ["Mist Energy"] 
    # Check if Mist Energy is "Energy" supertype and "Special" subtype
    me_def = card_def("Mist Energy")
    print(f"Mist Energy: Supertype={me_def.supertype}, Subtype={me_def.subtype}")
    
    from tcg.actions import ACTION_TABLE
    ns_idx = -1
    for i, act in enumerate(ACTION_TABLE):
        if act.kind == "PLAY_TRAINER" and act.a == "Night Stretcher" and act.b == 6:
            ns_idx = i
            break
            
    mask = env.action_mask()
    can_play = mask[ns_idx] == 1
    print(f"Discard has only Mist Energy. Night Stretcher playable? {can_play}")
    
    if can_play:
        print("FAILURE: Night Stretcher allowed with only Special Energy (should probably be blocked if requires Basic).")
    else:
        print("SUCCESS: Night Stretcher blocked with Special Energy.")

if __name__ == "__main__":
    test_fan_rotom_turn_3()
    test_night_stretcher_empty_discard()
    test_night_stretcher_special_energy()
