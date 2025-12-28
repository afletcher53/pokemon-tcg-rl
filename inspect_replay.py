import json
import sys

def analyze_replay(file_path, target_game_id=None):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    replays = data.get('replays', [])
    if not replays:
        print("No replays found.")
        return

    replay = None
    if target_game_id is not None:
        for r in replays:
            if r.get('game_id') == target_game_id:
                replay = r
                break
        if replay is None:
            print(f"Game ID {target_game_id} not found in replays.")
            # Fallback to last for debugging
            # replay = replays[-1]
            return
    else:
        replay = replays[-1]
    
    print(f"=== Replay Analysis: Game {replay.get('game_id', 'Unknown')} ===")
    print(f"Winner: Player {replay.get('winner')}")
    print(f"Total Turns reported: {replay.get('total_turns')}")
    
    frames = replay.get('frames', [])
    print(f"Total Frames: {len(frames)}")
    
    print("\n--- Game Log ---")
    current_turn = -1
    
    for frame in frames:
        turn = frame.get('turn_number')
        player = frame.get('turn_player')
        action_kind = frame.get('action_kind')
        desc = frame.get('action_description')
        res = frame.get('action_result', '')
        
        if turn != current_turn:
            print(f"\n[Turn {turn}] Player {player}'s Turn")
            current_turn = turn
            
        print(f"  P{player}: {desc}")
        if res:
             print(f"      -> {res}")

if __name__ == "__main__":
    game_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    analyze_replay('advanced_replays.json', game_id)
