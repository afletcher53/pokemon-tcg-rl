
import numpy as np
from tcg.env import PTCGEnv
from tcg.actions import Action, ACTION_INDEX, ACTION_TABLE

def test_rules():
    print("Initializing Environment...")
    env = PTCGEnv()
    obs, info = env.reset()
    
    # Helper to find action index
    def get_action_idx(kind, a=None, b=None):
        act = Action(kind, a=a, b=b)
        if act in ACTION_INDEX:
            return ACTION_INDEX[act]
        return None

    # Enable "Cheat Mode" for testing: Set hand manually
    env._gs.players[0].hand = ["Basic Fire Energy", "Basic Fire Energy", "Charmander", "Charmeleon", "Professor's Research", "Arven"]
    env._gs.players[0].active.name = "Charmander" # Starting active
    env._gs.players[0].active.turn_played = 0 # Played effectively at start of game (before turn 1)
    
    print("\n--- Test 1: Energy Attachment Limit ---")
    mask = env.action_mask()
    idx = get_action_idx("ATTACH_ACTIVE", a="Basic Fire Energy")
    print(f"Propsoed Action: ATTACH_ACTIVE (Basic Fire Energy). Mask value: {mask[idx]}")
    assert mask[idx] == 1, "Should be able to attach energy first time"
    
    # Perform attachment
    env.step(idx)
    mask = env.action_mask()
    print(f"After attachment, checking mask again: {mask[idx]}")
    assert mask[idx] == 0, "Should NOT be able to attach energy twice"
    
    print("\n--- Test 2: First Turn Supporter Rule ---")
    # Reset turn to 1, player 0
    env._gs.turn_number = 1
    env._gs.turn_player = 0
    env._gs.players[0].supporter_used = False # Reset
    
    res_idx = get_action_idx("PLAY_TRAINER", a="Professor's Research")
    mask = env.action_mask()
    print(f"Turn 1 Player 0. Try playing Supporter. Mask: {mask[res_idx]}")
    assert mask[res_idx] == 0, "Should NOT be able to play Supporter on Turn 1"
    
    print("\n--- Test 3: Evolution Sickness ---")
    # Play a new basic to bench
    play_idx = get_action_idx("PLAY_BASIC_TO_BENCH", a="Charmander", b=0)
    # Give us a Charmander
    env._gs.players[0].hand.append("Charmander")
    env.step(play_idx) # Played on Turn 1
    
    # Try to evolve bench Charmander immediately
    evolve_idx = get_action_idx("EVOLVE_BENCH", a="Charmeleon", b=0)
    mask = env.action_mask()
    print(f"Just played Charmander. Turn: {env._gs.turn_number}. Try Evolve to Charmeleon. Mask: {mask[evolve_idx]}")
    assert mask[evolve_idx] == 0, "Should NOT be able to evolve turn it was played"
    
    # Pass turn -> Opponent -> Back to Me
    pass_idx = get_action_idx("PASS")
    env.step(pass_idx) # End P0 Turn 1
    # Opponent Turn 1 (scripted passes)
    # Now P0 Turn 2
    
    print(f"\n--- Test 4: Can Evolve Next Turn ---")
    print(f"Current Turn: {env._gs.turn_number}. Player: {env._gs.turn_player}")
    mask = env.action_mask()
    print(f"Try Evolve Bench Charmander now. Mask: {mask[evolve_idx]}")
    assert mask[evolve_idx] == 1, "Should be able to evolve now"
    
    # Do Evolution
    env.step(evolve_idx)
    print("Evolved successfully.")
    assert env._gs.players[0].bench[0].name == "Charmeleon", "Bench should be Charmeleon"
    
    print("\n--- Test 5: Supporter Limit ---")
    # Turn 2, haven't used supporter yet
    res_idx = get_action_idx("PLAY_TRAINER", a="Arven")
    mask = env.action_mask()
    print(f"Turn 2. Try Support (Arven). Mask: {mask[res_idx]}")
    assert mask[res_idx] == 1, "Can play supporter Turn 2"
    
    env.step(res_idx)
    mask = env.action_mask()
    prof_idx = get_action_idx('PLAY_TRAINER', a="Professor's Research")
    print(f"Played Arven. Try play another Supporter (Professor's Research). Mask: {mask[prof_idx]}")
    assert mask[prof_idx] == 0, "Cannot play 2 supporters"
    
    print("\nAll Tests Passed!")

if __name__ == "__main__":
    test_rules()
