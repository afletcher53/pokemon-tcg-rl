
import torch
import sys
import os
import random
import numpy as np

# Add current directory to path so imports work
sys.path.append(os.getcwd())

from tcg.env import PTCGEnv
from evaluate_alpharank import load_model
from tcg.actions import ACTION_TABLE

def analyze_match():
    # Load Main Agent
    agent_path = "best_elo_policy.pt"
    
    print(f"Loading Main: {agent_path}")
    agent_model, _ = load_model(agent_path)
    
    print("\n=== Playing Match: Best ELO (P0) vs Scripted (P1) ===")
    env = PTCGEnv(scripted_opponent=True) # Let env handle P1 logic automatically
    
    obs, info = env.reset()
    
    # Ensure P0 starts for clear diagnostics, or handle P1 start
    if env._gs.turn_player == 1:
        print("P1 started. Making random move to pass to P0...")
        # Just random move to get to P0
        mask = env.action_mask()
        valid_indices = np.where(mask)[0]
        action = np.random.choice(valid_indices)
        obs, _, done, _, info = env.step(action)
        # Now should be P0 turn if P1 didn't win immediately
    
    done = False
    step_count = 0
    MAX_STEPS = 20
    
    while not done and step_count < MAX_STEPS:
        current_player_idx = env._gs.turn_player
        if current_player_idx != 0:
            print("Diff player?? Should be P0 if scripted.")
            break
            
        p0 = env._gs.players[0]
        print(f"\n--- Turn {env._gs.turn_number} (P0) ---")
        print(f"Hand ({len(p0.hand)}): {p0.hand}")
        if p0.active.name:
             print(f"Active: {p0.active.name} (Energy: {len(p0.active.energy)})")
        else:
             print(f"Active: None")
        print(f"Bench: {[s.name for s in p0.bench if s.name]}")
        
        # Action selection
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        mask = env.action_mask()
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = agent_model(obs_tensor)
            masked_logits = logits.masked_fill(~mask_tensor, float('-inf'))
            probs = torch.softmax(masked_logits, dim=-1)
            action = torch.argmax(probs, dim=1).item()
            
            # Print top 5 probabilities
            top_probs, top_indices = torch.topk(probs, 5)
            print("Top Actions:")
            for p, idx in zip(top_probs[0], top_indices[0]):
                act_str = str(ACTION_TABLE[idx.item()])
                print(f"  {p.item():.4f}: {act_str}")
                
            # Check if PASS is the only valid action
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 1 and valid_indices[0] == 0:
                print("Only valid action is PASS.")
            else:
                nb_valid = len(valid_indices)
                print(f"Valid actions count: {nb_valid}")
                if action == 0 and nb_valid > 1:
                     print("!!! Agent chose PASS despite having other options !!!")
                     # Show some other valid options
                     other_valid = [i for i in valid_indices if i != 0][:5]
                     for i in other_valid:
                         print(f"  Could have done: {ACTION_TABLE[i]}")

        # Execute
        print(f"Executing: {ACTION_TABLE[action]}")
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        
    print("\n=== Game Over ===")
    print(f"Winner: {info.get('winner')}")
    print(f"Reason: {info.get('win_reason')}")

if __name__ == "__main__":
    analyze_match()
