#!/usr/bin/env python3
"""Quick script to analyze specific games from recorded replays."""
import json
import sys

def analyze_game(game_id):
    with open('recorded_replays.json') as f:
        data = json.load(f)
    
    # Find game
    games = [r for r in data['replays'] if r['game_id'] == game_id]
    if not games:
        print(f"Game {game_id} not found")
        return
    
    g = games[0]
    print(f"=== Game {game_id} ===")
    print(f"Winner: P{g['winner']}")
    print(f"Total turns: {g['total_turns']}")
    print(f"Total frames: {g['total_frames']}")
    print()
    
    # Look for critical moments where bench is empty but player has options
    for frame in g['frames']:
        for pid in [0, 1]:
            p = frame[f'p{pid}']
            bench_empty = all(b is None for b in p['bench'])
            
            # Check if this player has basics in hand or draw supporters
            hand = p.get('hand', [])
            has_basic = any(c in ["Abra", "Charmander", "Pidgey", "Fan Rotom", "Dunsparce", 
                                   "Fezandipiti ex", "Tatsugiri", "Psyduck", "Charcadet"] 
                           for c in hand)
            has_draw = any(c in ["Lillie's Determination", "Hilda", "Dawn", "Iono", 
                                  "Buddy-Buddy Poffin", "Nest Ball", "Ultra Ball", "Artazon"]
                          for c in hand)
            
            # Print if bench empty AND (has options or is losing)
            if bench_empty and (has_basic or has_draw):
                tp = frame['turn_player']
                if tp == pid:  # Only print if it's their turn
                    print(f"Frame {frame['frame_id']}: Turn {frame['turn_number']} P{pid}")
                    print(f"  Action: {frame['action_kind']} - {frame['action_description']}")
                    active = p.get('active', {})
                    print(f"  Active: {active.get('name', 'None')} (HP: {active.get('hp_current', 0)}/{active.get('hp_max', 0)})")
                    print(f"  Bench: EMPTY")
                    print(f"  Hand: {hand}")
                    print(f"  Has Basic in hand: {has_basic}")
                    print(f"  Has Draw/Search: {has_draw}")
                    print()

    # Print game ending
    last_frame = g['frames'][-1]
    print(f"=== Game End ===")
    print(f"Win reason: {last_frame.get('win_reason', 'Unknown')}")

if __name__ == "__main__":
    game_id = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    analyze_game(game_id)
