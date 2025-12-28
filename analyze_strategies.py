"""
Strategy Analyzer for Pokemon TCG RL Agent
Analyzes game replays to understand learned strategies.
"""
from __future__ import annotations
import torch
import numpy as np
import random
import json
import os
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import argparse

from tcg.env import PTCGEnv
from tcg.train_bc import PolicyNet
from tcg.actions import ACTION_TABLE
from tcg.cards import card_def


@dataclass
class GameAction:
    """Single action in a game."""
    turn: int
    player: int
    action_kind: str
    card_a: Optional[str]
    card_b: Optional[int]
    active_pokemon: Optional[str]
    bench_pokemon: List[str]
    hand_size: int
    prizes_remaining: int
    opponent_active: Optional[str]
    opponent_prizes: int


@dataclass 
class GameReplay:
    """Full game replay."""
    game_id: int
    winner: Optional[int]
    total_turns: int
    total_steps: int
    p0_deck: str  # Deck name
    p1_deck: str
    actions: List[GameAction] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "game_id": self.game_id,
            "winner": self.winner,
            "total_turns": self.total_turns,
            "total_steps": self.total_steps,
            "p0_deck": self.p0_deck,
            "p1_deck": self.p1_deck,
            "actions": [asdict(a) for a in self.actions]
        }


def run_analysis_games(model, device, num_games=100, verbose=False):
    """Run games and collect detailed replays."""
    os.environ['PTCG_QUIET'] = '1'
    
    env = PTCGEnv(scripted_opponent=False, max_turns=25)
    
    # Decks (same as training)
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

    # P1: Charizard (60 cards)
    deck_p1 = []
    # Pokemon (20)
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
    # Trainers (31)
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
    # Energy (9)
    deck_p1.extend(["Fire Energy"] * 5)
    deck_p1.extend(["Mist Energy"] * 2)
    deck_p1.extend(["Darkness Energy"] * 1)
    deck_p1.extend(["Jet Energy"] * 1)
    
    replays = []
    
    for game_id in range(num_games):
        obs, info = env.reset(options={"decks": [deck_p0, deck_p1]})
        done = False
        
        replay = GameReplay(
            game_id=game_id,
            winner=None,
            total_turns=0,
            total_steps=0,
            p0_deck="Alakazam",
            p1_deck="Charizard"
        )
        
        max_steps = 500
        step_count = 0
        
        while not done and step_count < max_steps:
            turn_player = env._gs.turn_player
            turn_num = env._gs.turn_number
            mask = info["action_mask"]
            
            # Get model action
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(obs_t)
            
            mask_t = torch.from_numpy(mask).float().to(device)
            huge_neg = torch.ones_like(logits) * -1e9
            masked_logits = torch.where(mask_t.unsqueeze(0) > 0, logits, huge_neg)
            probs = torch.softmax(masked_logits, dim=1)
            
            # Greedy selection for analysis (deterministic)
            act_idx = probs.argmax(dim=1).item()
            
            act = ACTION_TABLE[act_idx]
            me = env._gs.players[turn_player]
            op = env._gs.players[1 - turn_player]
            
            # Record action
            game_action = GameAction(
                turn=turn_num,
                player=turn_player,
                action_kind=act.kind,
                card_a=act.a,
                card_b=act.b,
                active_pokemon=me.active.name,
                bench_pokemon=[s.name for s in me.bench if s.name],
                hand_size=len(me.hand),
                prizes_remaining=len(me.prizes),
                opponent_active=op.active.name,
                opponent_prizes=len(op.prizes)
            )
            replay.actions.append(game_action)
            
            obs, _, done, _, info = env.step(act_idx)
            step_count += 1
        
        replay.winner = env._gs.winner
        replay.total_turns = env._gs.turn_number
        replay.total_steps = step_count
        replays.append(replay)
        
        if verbose and (game_id + 1) % 20 == 0:
            print(f"Analyzed {game_id + 1}/{num_games} games...")
    
    return replays


def analyze_replays(replays: List[GameReplay]) -> Dict[str, Any]:
    """Analyze replays to extract strategy insights."""
    
    analysis = {
        "summary": {},
        "opening_moves": {"p0": [], "p1": []},
        "action_frequencies": {"p0": defaultdict(int), "p1": defaultdict(int)},
        "winning_patterns": {"p0": [], "p1": []},
        "losing_patterns": {"p0": [], "p1": []},
        "card_play_rates": {"p0": defaultdict(int), "p1": defaultdict(int)},
        "attack_usage": {"p0": defaultdict(int), "p1": defaultdict(int)},
        "turn_action_breakdown": defaultdict(lambda: defaultdict(int)),
        "win_conditions": defaultdict(int),
        "avg_game_length": 0,
        "first_attack_turn": {"p0": [], "p1": []},
    }
    
    wins = {0: 0, 1: 0, None: 0}
    total_steps = 0
    total_turns = 0
    
    for replay in replays:
        wins[replay.winner] = wins.get(replay.winner, 0) + 1
        total_steps += replay.total_steps
        total_turns += replay.total_turns
        
        # Opening moves (first 5 actions per player)
        p0_opens = [a for a in replay.actions[:20] if a.player == 0][:5]
        p1_opens = [a for a in replay.actions[:20] if a.player == 1][:5]
        
        if p0_opens:
            analysis["opening_moves"]["p0"].append([a.action_kind + (f":{a.card_a}" if a.card_a else "") for a in p0_opens])
        if p1_opens:
            analysis["opening_moves"]["p1"].append([a.action_kind + (f":{a.card_a}" if a.card_a else "") for a in p1_opens])
        
        # Action frequencies
        first_attack_p0 = None
        first_attack_p1 = None
        
        for action in replay.actions:
            p_key = f"p{action.player}"
            analysis["action_frequencies"][p_key][action.action_kind] += 1
            
            # Card play rates
            if action.card_a and action.action_kind in ("PLAY_BASIC_TO_BENCH", "EVOLVE_ACTIVE", "EVOLVE_BENCH", "PLAY_TRAINER", "ATTACH_ACTIVE", "ATTACH_BENCH"):
                analysis["card_play_rates"][p_key][action.card_a] += 1
            
            # Attack usage
            if action.action_kind == "ATTACK" and action.active_pokemon:
                analysis["attack_usage"][p_key][action.active_pokemon] += 1
                if action.player == 0 and first_attack_p0 is None:
                    first_attack_p0 = action.turn
                elif action.player == 1 and first_attack_p1 is None:
                    first_attack_p1 = action.turn
            
            # Turn breakdown
            analysis["turn_action_breakdown"][action.turn][action.action_kind] += 1
        
        if first_attack_p0:
            analysis["first_attack_turn"]["p0"].append(first_attack_p0)
        if first_attack_p1:
            analysis["first_attack_turn"]["p1"].append(first_attack_p1)
        
        # Winning/Losing patterns (last 10 actions)
        last_actions = replay.actions[-10:] if len(replay.actions) >= 10 else replay.actions
        pattern = [(a.player, a.action_kind, a.card_a) for a in last_actions]
        
        if replay.winner == 0:
            analysis["winning_patterns"]["p0"].append(pattern)
            analysis["losing_patterns"]["p1"].append(pattern)
        elif replay.winner == 1:
            analysis["winning_patterns"]["p1"].append(pattern)
            analysis["losing_patterns"]["p0"].append(pattern)
    
    # Summary stats
    analysis["summary"] = {
        "total_games": len(replays),
        "p0_wins": wins.get(0, 0),
        "p1_wins": wins.get(1, 0),
        "draws": wins.get(None, 0),
        "p0_win_rate": wins.get(0, 0) / len(replays) if replays else 0,
        "p1_win_rate": wins.get(1, 0) / len(replays) if replays else 0,
        "avg_game_steps": total_steps / len(replays) if replays else 0,
        "avg_game_turns": total_turns / len(replays) if replays else 0,
        "avg_first_attack_p0": np.mean(analysis["first_attack_turn"]["p0"]) if analysis["first_attack_turn"]["p0"] else 0,
        "avg_first_attack_p1": np.mean(analysis["first_attack_turn"]["p1"]) if analysis["first_attack_turn"]["p1"] else 0,
    }
    
    # Convert defaultdicts for JSON
    analysis["action_frequencies"]["p0"] = dict(analysis["action_frequencies"]["p0"])
    analysis["action_frequencies"]["p1"] = dict(analysis["action_frequencies"]["p1"])
    analysis["card_play_rates"]["p0"] = dict(analysis["card_play_rates"]["p0"])
    analysis["card_play_rates"]["p1"] = dict(analysis["card_play_rates"]["p1"])
    analysis["attack_usage"]["p0"] = dict(analysis["attack_usage"]["p0"])
    analysis["attack_usage"]["p1"] = dict(analysis["attack_usage"]["p1"])
    analysis["turn_action_breakdown"] = {k: dict(v) for k, v in analysis["turn_action_breakdown"].items()}
    
    # Find common opening patterns
    def find_common_sequences(sequences, top_n=5):
        if not sequences:
            return []
        # Convert to tuple for hashing
        seq_counts = Counter(tuple(s) if isinstance(s[0], str) else tuple(tuple(x) if isinstance(x, list) else x for x in s) for s in sequences)
        return seq_counts.most_common(top_n)
    
    analysis["top_openings"] = {
        "p0": find_common_sequences(analysis["opening_moves"]["p0"]),
        "p1": find_common_sequences(analysis["opening_moves"]["p1"]),
    }
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Pretty print the analysis results."""
    print("\n" + "="*70)
    print("🎮 POKEMON TCG STRATEGY ANALYSIS")
    print("="*70)
    
    s = analysis["summary"]
    print(f"\n📊 SUMMARY")
    print(f"   Total Games Analyzed: {s['total_games']}")
    print(f"   P0 (Alakazam) Wins: {s['p0_wins']} ({s['p0_win_rate']:.1%})")
    print(f"   P1 (Charizard) Wins: {s['p1_wins']} ({s['p1_win_rate']:.1%})")
    print(f"   Draws/Timeouts: {s['draws']}")
    print(f"   Avg Game Length: {s['avg_game_steps']:.1f} steps, {s['avg_game_turns']:.1f} turns")
    print(f"   Avg First Attack - P0: Turn {s['avg_first_attack_p0']:.1f}, P1: Turn {s['avg_first_attack_p1']:.1f}")
    
    print(f"\n🎯 ACTION FREQUENCIES")
    for player in ["p0", "p1"]:
        name = "Alakazam" if player == "p0" else "Charizard"
        print(f"\n   {name} ({player.upper()}):")
        sorted_actions = sorted(analysis["action_frequencies"][player].items(), key=lambda x: -x[1])
        for action, count in sorted_actions[:8]:
            print(f"      {action}: {count}")
    
    print(f"\n🃏 TOP CARD PLAYS")
    for player in ["p0", "p1"]:
        name = "Alakazam" if player == "p0" else "Charizard"
        print(f"\n   {name} ({player.upper()}):")
        sorted_cards = sorted(analysis["card_play_rates"][player].items(), key=lambda x: -x[1])
        for card, count in sorted_cards[:10]:
            print(f"      {card}: {count}")
    
    print(f"\n⚔️ ATTACK USAGE (by Pokemon)")
    for player in ["p0", "p1"]:
        name = "Alakazam" if player == "p0" else "Charizard"
        print(f"\n   {name} ({player.upper()}):")
        sorted_attacks = sorted(analysis["attack_usage"][player].items(), key=lambda x: -x[1])
        for pokemon, count in sorted_attacks[:5]:
            print(f"      {pokemon}: {count} attacks")
    
    print(f"\n🚀 TOP OPENING STRATEGIES")
    for player in ["p0", "p1"]:
        name = "Alakazam" if player == "p0" else "Charizard"
        print(f"\n   {name} ({player.upper()}):")
        for i, (pattern, count) in enumerate(analysis["top_openings"][player][:3], 1):
            pattern_str = " → ".join(pattern[:5])
            print(f"      #{i} ({count}x): {pattern_str}")
    
    print("\n" + "="*70)


def save_for_visualization(analysis: Dict[str, Any], replays: List[GameReplay], output_dir: str = "."):
    """Save analysis data for web visualization."""
    
    # Save analysis summary
    with open(os.path.join(output_dir, "strategy_analysis.json"), "w") as f:
        # Filter out non-serializable parts
        save_data = {
            "summary": analysis["summary"],
            "action_frequencies": analysis["action_frequencies"],
            "card_play_rates": analysis["card_play_rates"],
            "attack_usage": analysis["attack_usage"],
            "turn_action_breakdown": analysis["turn_action_breakdown"],
            "top_openings": {
                "p0": [(list(p), c) for p, c in analysis["top_openings"]["p0"]],
                "p1": [(list(p), c) for p, c in analysis["top_openings"]["p1"]],
            }
        }
        json.dump(save_data, f, indent=2)
    
    # Save sample replays (first 20 games)
    sample_replays = [r.to_dict() for r in replays[:20]]
    with open(os.path.join(output_dir, "sample_replays.json"), "w") as f:
        json.dump(sample_replays, f, indent=2)
    
    print(f"\n💾 Saved analysis to {output_dir}/strategy_analysis.json")
    print(f"💾 Saved sample replays to {output_dir}/sample_replays.json")


def main():
    parser = argparse.ArgumentParser(description='Analyze RL Agent Strategies')
    parser.add_argument('--games', type=int, default=100, help='Number of games to analyze')
    parser.add_argument('--policy', type=str, default='rl_policy.pt', help='Policy file to use')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print("🔍 Loading Policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(args.policy)
        model = PolicyNet(checkpoint["obs_dim"], checkpoint["n_actions"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"   Loaded {args.policy}")
    except Exception as e:
        print(f"❌ Could not load policy: {e}")
        return
    
    print(f"\n🎮 Running {args.games} analysis games...")
    replays = run_analysis_games(model, device, num_games=args.games, verbose=args.verbose)
    
    print("\n📈 Analyzing strategies...")
    analysis = analyze_replays(replays)
    
    print_analysis(analysis)
    save_for_visualization(analysis, replays)


if __name__ == "__main__":
    main()
