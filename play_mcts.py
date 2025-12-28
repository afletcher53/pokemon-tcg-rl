"""
Play a game using MCTS with either PolicyNet or PolicyValueNet.
Supports both the original RL training and AlphaZero-style models.
"""
import torch
import numpy as np
import argparse
import time
import json
from tcg.env import PTCGEnv
from tcg.train_bc import PolicyNet
from tcg.mcts import MCTS, PolicyValueNet
from tcg.actions import ACTION_TABLE
from tcg.state import featurize


def load_model(path: str, device: torch.device, n_actions: int):
    """Load either PolicyNet, PolicyValueNet, or AdvancedPolicyValueNet from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    obs_dim = checkpoint.get("obs_dim", 156)
    
    # Detect model type by checking state dict keys
    state_dict = checkpoint["state_dict"]
    
    if any("transformer" in k or "input_embed" in k for k in state_dict.keys()):
        # AdvancedPolicyValueNet from train_advanced.py
        print(f"Detected AdvancedPolicyValueNet architecture")
        # Import the model class
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


def main():
    parser = argparse.ArgumentParser(description="Play Pokemon TCG with MCTS")
    parser.add_argument("--policy", type=str, default="alphazero_policy.pt", 
                        help="Path to policy checkpoint (supports both PolicyNet and PolicyValueNet)")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--use_value_net", action="store_true", 
                        help="Use value network instead of rollouts (only for PolicyValueNet)")
    parser.add_argument("--temperature", type=float, default=0.1, 
                        help="MCTS temperature (0=greedy, higher=more random)")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    parser.add_argument("--record", type=str, default=None, 
                        help="Save game replay to JSON file")
    parser.add_argument("--max_turns", type=int, default=100, help="Maximum turns")
    args = parser.parse_args()

    print("=" * 60)
    print("MCTS Pokemon TCG Player")
    print("=" * 60)
    print(f"P0 (Alakazam) uses MCTS vs P1 (Charizard) Scripted")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    n_actions = len(ACTION_TABLE)
    
    # Load model
    print(f"\nLoading policy from {args.policy}...")
    try:
        model = load_model(args.policy, device, n_actions)
    except Exception as e:
        print(f"Error loading policy: {e}")
        print("Trying fallback to rl_policy.pt...")
        try:
            model = load_model("rl_policy.pt", device, n_actions)
        except:
            print("No valid policy found. Exiting.")
            return
    
    # Determine if model supports value function
    is_policy_value = isinstance(model, PolicyValueNet)
    use_value = args.use_value_net and is_policy_value
    
    print(f"Model type: {'PolicyValueNet' if is_policy_value else 'PolicyNet'}")
    print(f"Using value network: {use_value}")
    print(f"MCTS simulations: {args.sims}")
    print(f"Temperature: {args.temperature}")
    
    # Init Environment
    env = PTCGEnv(scripted_opponent=True, max_turns=args.max_turns)
    
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
    deck_p1.extend(["Charmander"] * 3)
    deck_p1.extend(["Charmeleon"] * 2)
    deck_p1.extend(["Charizard ex"] * 2)
    deck_p1.extend(["Pidgey"] * 2)
    deck_p1.extend(["Pidgeotto"] * 2)
    deck_p1.extend(["Pidgeot ex"] * 2)
    deck_p1.extend(["Psyduck"] * 1)
    deck_p1.extend(["Shaymin"] * 1)
    deck_p1.extend(["Tatsugiri"] * 1)
    deck_p1.extend(["Munkidori"] * 1)
    deck_p1.extend(["Chi-Yu"] * 1)
    deck_p1.extend(["Gouging Fire ex"] * 1)
    deck_p1.extend(["Fezandipiti ex"] * 1)
    deck_p1.extend(["Lillie's Determination"] * 4)
    deck_p1.extend(["Arven"] * 4)
    deck_p1.extend(["Boss's Orders"] * 3)
    deck_p1.extend(["Iono"] * 2)
    deck_p1.extend(["Professor Turo's Scenario"] * 1)
    deck_p1.extend(["Buddy-Buddy Poffin"] * 4)
    deck_p1.extend(["Ultra Ball"] * 3)
    deck_p1.extend(["Rare Candy"] * 2)
    deck_p1.extend(["Super Rod"] * 2)
    deck_p1.extend(["Counter Catcher"] * 1)
    deck_p1.extend(["Energy Search"] * 1)
    deck_p1.extend(["Unfair Stamp"] * 1)
    deck_p1.extend(["Technical Machine: Evolution"] * 2)
    deck_p1.extend(["Artazon"] * 1)
    deck_p1.extend(["Fire Energy"] * 5)
    deck_p1.extend(["Mist Energy"] * 2)
    deck_p1.extend(["Darkness Energy"] * 1)
    deck_p1.extend(["Jet Energy"] * 1)
    
    # Init MCTS
    mcts = MCTS(
        policy_net=model, 
        device=device, 
        num_simulations=args.sims, 
        max_rollout_steps=150,
        use_value_net=use_value,
        use_policy_rollouts=True,
        temperature=args.temperature
    )

    # Play Game
    obs, info = env.reset(options={"decks": [deck_p0, deck_p1]})
    done = False
    
    replay = {
        "moves": [],
        "winner": None,
        "total_turns": 0
    }
    
    print("\n" + "=" * 60)
    print("--- GAME START ---")
    print("=" * 60)
    
    move_count = 0
    total_think_time = 0
    
    while not done:
        turn_player = env._gs.turn_player
        
        if turn_player == 0:
            # AI (MCTS) Turn
            start_t = time.time()
            
            if args.debug:
                print(f"\nTurn {env._gs.turn_number} (P0): Thinking...")
                if env._gs.players[0].active:
                    print(f"  Active: {env._gs.players[0].active.name}")
            
            act_idx, mcts_probs = mcts.search(env, return_probs=True)
            
            dt = time.time() - start_t
            total_think_time += dt
            move_count += 1
            
            action = ACTION_TABLE[act_idx]
            
            # Pretty print action
            action_str = f"{action.kind}"
            if action.a:
                action_str += f" {action.a}"
            if action.b is not None:
                action_str += f" -> {action.b}"
            
            print(f"Turn {env._gs.turn_number:3d} [P0]: {action_str:40s} ({dt:.1f}s)")
            
            # Record for replay
            if args.record:
                top_actions = np.argsort(mcts_probs)[::-1][:3]
                replay["moves"].append({
                    "turn": env._gs.turn_number,
                    "player": 0,
                    "action": action_str,
                    "think_time": dt,
                    "top_probs": [(int(a), float(mcts_probs[a])) for a in top_actions if mcts_probs[a] > 0.01]
                })
            
            obs, reward, done, _, info = env.step(act_idx)
        else:
            # Opponent turn handled by scripted_opponent
            pass

    print("\n" + "=" * 60)
    print("--- GAME OVER ---")
    print("=" * 60)
    
    winner = env._gs.winner
    reason = env._gs.win_reason
    
    print(f"\nWinner: Player {winner}")
    print(f"Reason: {reason}")
    print(f"Total turns: {env._gs.turn_number}")
    print(f"Total moves by P0: {move_count}")
    print(f"Avg think time: {total_think_time/max(move_count,1):.2f}s")
    
    # Check final board state
    print("\nFinal Board State:")
    p0 = env._gs.players[0]
    p1 = env._gs.players[1]
    
    print(f"  P0 Prizes: {6 - len(p0.prizes)} taken, Active: {p0.active.name if p0.active else 'None'}")
    evolved_p0 = [s.name for s in [p0.active] + p0.bench if s and s.name and s.name in ["Alakazam", "Kadabra", "Dudunsparce"]]
    print(f"  P0 Evolved: {evolved_p0 if evolved_p0 else 'None'}")
    
    print(f"  P1 Prizes: {6 - len(p1.prizes)} taken, Active: {p1.active.name if p1.active else 'None'}")
    
    # Save replay
    if args.record:
        replay["winner"] = winner
        replay["win_reason"] = reason
        replay["total_turns"] = env._gs.turn_number
        replay["avg_think_time"] = total_think_time / max(move_count, 1)
        
        with open(args.record, 'w') as f:
            json.dump(replay, f, indent=2)
        print(f"\nReplay saved to {args.record}")


if __name__ == "__main__":
    main()
