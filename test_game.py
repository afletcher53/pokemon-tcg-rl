
import torch
import sys
import os
# Add current directory to path so imports work
sys.path.append(os.getcwd())

from tcg.env import PTCGEnv
from evaluate_alpharank import load_model

def test():
    # Load two different models
    m1_path = "league/agent_ep0.pt"
    m2_path = "best_elo_policy.pt"
    
    # Check if files exist
    if not os.path.exists(m1_path) or not os.path.exists(m2_path):
        print("Models not found, skipping test.")
        return

    print(f"Loading {m1_path}...")
    m1, _ = load_model(m1_path)
    print(f"Loading {m2_path}...")
    m2, _ = load_model(m2_path)
    
    print("Playing 1 game...")
    # env verbose is printed to stdout
    env = PTCGEnv()
    
    # Custom play loop
    obs, info = env.reset()
    
    done = False
    steps = 0
    while not done:
        current_player = env._gs.turn_player
        model = m1 if current_player == 0 else m2
        
        # Action selection (simplified from evaluate_alpharank.py)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        mask = env.action_mask()
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = model(obs_tensor)
            logits = logits.masked_fill(~mask_tensor, float('-inf'))
            probs = torch.softmax(logits, dim=-1)
            # greedy for test
            action = torch.argmax(probs, dim=1).item()
            
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        
        if steps % 100 == 0:
            print(f"Step {steps}, Turn {env._gs.turn_number}")
            
    print(f"Game Over!")
    print(f"Winner: {info.get('winner')}")
    print(f"Reason: {env._gs.win_reason if hasattr(env._gs, 'win_reason') else 'Unknown'}")
    print(f"Total Steps: {steps}")
    print(f"Total Turns: {env._gs.turn_number}")

if __name__ == "__main__":
    test()
