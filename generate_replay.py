#!/usr/bin/env python3
"""
Generate readable game replays from the latest checkpoint.
"""

import torch
import numpy as np
from tcg.env import PTCGEnv
from tcg.mcts import MCTS
from tcg.actions import ACTION_TABLE
from train_advanced import AdvancedPolicyValueNet

def generate_replay(checkpoint_path: str, num_games: int = 3, output_file: str = "replay.txt"):
    """Generate replays of games using the checkpoint."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_dim = 1452
    n_actions = len(ACTION_TABLE)
    
    # Load model
    model = AdvancedPolicyValueNet(obs_dim, n_actions).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Generating {num_games} replays...\n")
    
    with open(output_file, 'w') as f:
        for game_idx in range(num_games):
            f.write(f"\n{'='*60}\n")
            f.write(f"GAME {game_idx + 1}\n")
            f.write(f"{'='*60}\n\n")
            
            env = PTCGEnv()
            env.reset()
            
            mcts = MCTS(model, device, num_simulations=15)
            
            turn_count = 0
            last_turn = -1
            
            while not env._gs.done and turn_count < 200:
                gs = env._gs
                current_turn = gs.turn_number
                player = gs.turn_player
                me = gs.players[player]
                opp = gs.players[1 - player]
                
                # Log new turn
                if current_turn != last_turn:
                    f.write(f"\n--- Turn {current_turn} (Player {player}) ---\n")
                    f.write(f"Active: {me.active.name if me.active.name else 'None'}")
                    if me.active.name:
                        f.write(f" [{me.active.damage} dmg, {len(me.active.energy)} energy]\n")
                    else:
                        f.write("\n")
                    
                    bench_names = [s.name for s in me.bench if s.name]
                    if bench_names:
                        f.write(f"Bench: {', '.join(bench_names)}\n")
                    
                    f.write(f"Hand: {len(me.hand)} cards, Prizes: {len(me.prizes)} left\n")
                    last_turn = current_turn
                
                # Get action
                mask = env.action_mask()
                action = mcts.search(env)
                act = ACTION_TABLE[action]
                
                # Log action
                action_desc = f"  → {act.kind}"
                if act.a:
                    action_desc += f": {act.a}"
                f.write(action_desc + "\n")
                
                # Check for attack and log damage
                if act.kind == 'ATTACK':
                    target_name = opp.active.name if opp.active.name else "opponent"
                    f.write(f"    (attacking {target_name})\n")
                
                obs, reward, done, truncated, info = env.step(action)
                
                # Check for prize taken
                if 'prize_taken' in info:
                    f.write(f"    *** PRIZE TAKEN! ***\n")
                
                # Check for KO
                if not opp.active.name and gs.turn_number == current_turn:
                    f.write(f"    *** KO! ***\n")
                
                turn_count += 1
                if truncated:
                    break
            
            # Game result
            f.write(f"\n{'='*40}\n")
            if env._gs.winner == 0:
                f.write("RESULT: Player 0 WINS!\n")
            elif env._gs.winner == 1:
                f.write("RESULT: Player 1 WINS!\n")
            else:
                f.write("RESULT: DRAW (deck-out or max turns)\n")
            f.write(f"Total turns: {env._gs.turn_number}\n")
            f.write(f"P0 prizes taken: {6 - len(env._gs.players[0].prizes)}\n")
            f.write(f"P1 prizes taken: {6 - len(env._gs.players[1].prizes)}\n")
            f.write(f"{'='*40}\n")
            
            print(f"Game {game_idx + 1}: P{env._gs.winner} wins in {env._gs.turn_number} turns")
    
    print(f"\nReplays saved to: {output_file}")


if __name__ == "__main__":
    import sys
    import glob
    
    # Find latest checkpoint
    checkpoints = sorted(glob.glob("checkpoints/checkpoint_ep*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
    else:
        latest = "checkpoints/checkpoint_ep100.pt"
    
    if len(sys.argv) > 1:
        latest = sys.argv[1]
    
    generate_replay(latest, num_games=3, output_file="game_replays.txt")
