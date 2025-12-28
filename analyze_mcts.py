
import torch
import sys
import os
import random
import numpy as np

# Add current directory to path so imports work
sys.path.append(os.getcwd())

from tcg.env import PTCGEnv
from evaluate_alpharank import load_model, MCTS_AVAILABLE
from tcg.actions import ACTION_TABLE
from tcg.mcts import MCTS

def analyze_match_mcts():
    # Load Main Agent
    agent_path = "best_elo_policy.pt"
    
    print(f"Loading Main: {agent_path}")
    agent_model, _ = load_model(agent_path)
    # Move to CPU for analysis
    agent_model.to('cpu')
    
    print("\n=== Playing Match: Best ELO (P0) with MCTS (50 sims) vs Scripted (P1) ===")
    env = PTCGEnv(scripted_opponent=True)
    
    obs, info = env.reset()
    
    # Ensure P0 starts
    if env._gs.turn_player == 1:
        print("P1 started. Making random move to pass to P0...")
        mask = env.action_mask()
        valid_indices = np.where(mask)[0]
        action = np.random.choice(valid_indices)
        obs, _, done, _, info = env.step(action)
    
    done = False
    step_count = 0
    MAX_STEPS = 10
    
    # Initialize MCTS
    mcts = MCTS(agent_model, torch.device('cpu'), num_simulations=50, use_value_net=True)
    
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
        
        # Action selection via MCTS
        print("Thinking with MCTS...")
        action, probs = mcts.search(env, return_probs=True)
        
        act_str = str(ACTION_TABLE[action])
        print(f"MCTS Selected: {act_str}")
        
        # Check against raw policy
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        mask = env.action_mask()
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
        with torch.no_grad():
             logits, _ = agent_model(obs_tensor)
             logits = logits.masked_fill(~mask_tensor, float('-inf'))
             raw_probs = torch.softmax(logits, dim=-1)
             raw_best = torch.argmax(raw_probs).item()
             print(f"Raw Policy would have chosen: {ACTION_TABLE[raw_best]} (Prob: {raw_probs[0][raw_best]:.4f})")

        # Execute
        print(f"Executing: {act_str}")
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    analyze_match_mcts()
