"""
Strategy Analyzer v2 for Pokemon TCG RL Agent
Enhanced analysis for Alakazam deck strategy validation.

Key Alakazam Strategy Elements:
1. Slow evolve Alakazam line to maximize Psychic Draw triggers
2. Build massive hand size using draw supporters
3. Attack with Alakazam's Mind Jack (20 damage per card in hand)
4. Use Enhanced Hammer to counter Mist Energy
"""
from __future__ import annotations
import torch
import numpy as np
import random
import json
import os
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import argparse

from tcg.env import PTCGEnv
from tcg.train_bc import PolicyNet
from tcg.actions import ACTION_TABLE
from tcg.cards import card_def


@dataclass
class GameAction:
    """Enhanced action record with full game state snapshot."""
    turn: int
    player: int
    action_kind: str
    card_a: Optional[str]
    card_b: Optional[int]
    
    # Player state
    active_pokemon: Optional[str]
    active_energy: int
    active_damage: int
    bench_pokemon: List[str]
    bench_energy: List[int]
    hand_size: int
    hand_pokemon_count: int
    hand_trainer_count: int
    hand_energy_count: int
    prizes_remaining: int
    deck_size: int
    
    # Opponent state
    opponent_active: Optional[str]
    opponent_active_damage: int
    opponent_prizes: int
    opponent_bench_count: int
    
    # Computed fields for Alakazam strategy
    potential_alakazam_damage: int = 0  # 20 * hand_size if attacking with Alakazam


@dataclass 
class GameReplay:
    """Full game replay with enhanced tracking."""
    game_id: int
    winner: Optional[int]
    total_turns: int
    total_steps: int
    p0_deck: str
    p1_deck: str
    actions: List[GameAction] = field(default_factory=list)
    
    # Alakazam-specific tracking
    p0_alakazam_attacks: List[Dict] = field(default_factory=list)
    p0_evolution_timings: List[Dict] = field(default_factory=list)
    p0_hand_size_history: List[int] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "game_id": self.game_id,
            "winner": self.winner,
            "total_turns": self.total_turns,
            "total_steps": self.total_steps,
            "p0_deck": self.p0_deck,
            "p1_deck": self.p1_deck,
            "actions": [asdict(a) for a in self.actions],
            "p0_alakazam_attacks": self.p0_alakazam_attacks,
            "p0_evolution_timings": self.p0_evolution_timings,
        }


def run_analysis_games(model, device, num_games=100, verbose=False):
    """Run games and collect detailed replays with strategy tracking."""
    os.environ['PTCG_QUIET'] = '1'
    
    env = PTCGEnv(scripted_opponent=False, max_turns=50)
    
    # Alakazam deck (P0)
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

    # Charizard deck (P1)
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
        
        # Track when basics were played for evolution timing
        basics_played = {}  # pokemon_name -> turn played
        
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
            
            # Greedy selection for analysis
            act_idx = probs.argmax(dim=1).item()
            
            act = ACTION_TABLE[act_idx]
            me = env._gs.players[turn_player]
            op = env._gs.players[1 - turn_player]
            
            # Count hand composition
            hand_pokemon = sum(1 for c in me.hand if card_def(c).supertype == "Pokemon")
            hand_trainer = sum(1 for c in me.hand if card_def(c).supertype == "Trainer")
            hand_energy = sum(1 for c in me.hand if card_def(c).supertype == "Energy")
            
            # Calculate potential Alakazam damage
            potential_dmg = 20 * len(me.hand) if me.active.name == "Alakazam" else 0
            
            # Record action with full state
            game_action = GameAction(
                turn=turn_num,
                player=turn_player,
                action_kind=act.kind,
                card_a=act.a,
                card_b=act.b,
                active_pokemon=me.active.name,
                active_energy=me.active.energy,
                active_damage=me.active.damage,
                bench_pokemon=[s.name for s in me.bench if s.name],
                bench_energy=[s.energy for s in me.bench if s.name],
                hand_size=len(me.hand),
                hand_pokemon_count=hand_pokemon,
                hand_trainer_count=hand_trainer,
                hand_energy_count=hand_energy,
                prizes_remaining=len(me.prizes),
                deck_size=len(me.deck),
                opponent_active=op.active.name,
                opponent_active_damage=op.active.damage,
                opponent_prizes=len(op.prizes),
                opponent_bench_count=sum(1 for s in op.bench if s.name),
                potential_alakazam_damage=potential_dmg
            )
            replay.actions.append(game_action)
            
            # Track P0 specific strategy elements
            if turn_player == 0:
                replay.p0_hand_size_history.append(len(me.hand))
                
                # Track Alakazam attacks
                if act.kind == "ATTACK" and me.active.name == "Alakazam":
                    attack_info = {
                        "turn": turn_num,
                        "hand_size": len(me.hand),
                        "damage": 20 * len(me.hand),
                        "opponent_active": op.active.name,
                        "opponent_hp": card_def(op.active.name).hp if op.active.name else 0,
                        "opponent_damage_before": op.active.damage
                    }
                    replay.p0_alakazam_attacks.append(attack_info)
                
                # Track evolution timing
                if act.kind == "PLAY_BASIC_TO_BENCH" and act.a in ["Abra", "Dunsparce"]:
                    basics_played[f"{act.a}_{act.b}"] = turn_num
                
                if act.kind in ("EVOLVE_ACTIVE", "EVOLVE_BENCH"):
                    evo_info = {
                        "turn": turn_num,
                        "evolved_to": act.a,
                        "evolved_from": card_def(act.a).evolves_from,
                        "hand_size_before": len(me.hand),
                    }
                    replay.p0_evolution_timings.append(evo_info)
            
            obs, _, done, _, info = env.step(act_idx)
            step_count += 1
        
        replay.winner = env._gs.winner
        replay.total_turns = env._gs.turn_number
        replay.total_steps = step_count
        replays.append(replay)
        
        if verbose and (game_id + 1) % 20 == 0:
            print(f"Analyzed {game_id + 1}/{num_games} games...")
    
    return replays


def analyze_alakazam_strategy(replays: List[GameReplay]) -> Dict[str, Any]:
    """Analyze how well the agent learned the Alakazam strategy."""
    
    alakazam_analysis = {
        "attack_statistics": {
            "total_alakazam_attacks": 0,
            "avg_hand_size_at_attack": 0,
            "avg_damage_per_attack": 0,
            "hand_size_distribution": defaultdict(int),
            "damage_distribution": defaultdict(int),
            "attacks_by_turn": defaultdict(int),
            "one_hit_ko_rate": 0,
        },
        "evolution_patterns": {
            "total_evolutions": 0,
            "kadabra_evolutions": 0,
            "alakazam_evolutions": 0,
            "avg_turn_kadabra_evolved": 0,
            "avg_turn_alakazam_evolved": 0,
            "hand_size_at_kadabra_evo": [],
            "hand_size_at_alakazam_evo": [],
        },
        "hand_management": {
            "avg_hand_size": 0,
            "max_hand_size_reached": 0,
            "hand_size_progression": defaultdict(list),  # by turn
        },
        "draw_supporter_usage": {
            "hilda": 0,
            "dawn": 0,
            "lillies_determination": 0,
            "tulip": 0,
            "total_draw_supporters": 0,
        },
        "tech_card_usage": {
            "enhanced_hammer": 0,
            "boss_orders": 0,
            "battle_cage": 0,
            "rare_candy": 0,
        },
        "win_correlation": {
            "wins_with_alakazam_attack": 0,
            "losses_with_alakazam_attack": 0,
            "avg_hand_size_in_wins": 0,
            "avg_hand_size_in_losses": 0,
        }
    }
    
    all_hand_sizes = []
    all_attack_hand_sizes = []
    all_attack_damages = []
    kadabra_turns = []
    alakazam_turns = []
    ko_count = 0
    total_attacks = 0
    
    wins_hand_sizes = []
    losses_hand_sizes = []
    
    for replay in replays:
        # Hand size tracking
        if replay.p0_hand_size_history:
            all_hand_sizes.extend(replay.p0_hand_size_history)
            alakazam_analysis["hand_management"]["max_hand_size_reached"] = max(
                alakazam_analysis["hand_management"]["max_hand_size_reached"],
                max(replay.p0_hand_size_history)
            )
        
        # Attack analysis
        for attack in replay.p0_alakazam_attacks:
            all_attack_hand_sizes.append(attack["hand_size"])
            all_attack_damages.append(attack["damage"])
            alakazam_analysis["attack_statistics"]["hand_size_distribution"][attack["hand_size"]] += 1
            alakazam_analysis["attack_statistics"]["damage_distribution"][attack["damage"]] += 1
            alakazam_analysis["attack_statistics"]["attacks_by_turn"][attack["turn"]] += 1
            total_attacks += 1
            
            # Check for OHKO
            if attack["opponent_hp"] > 0:
                remaining_hp = attack["opponent_hp"] - attack["opponent_damage_before"]
                if attack["damage"] >= remaining_hp:
                    ko_count += 1
        
        # Evolution timing
        for evo in replay.p0_evolution_timings:
            alakazam_analysis["evolution_patterns"]["total_evolutions"] += 1
            if evo["evolved_to"] == "Kadabra":
                alakazam_analysis["evolution_patterns"]["kadabra_evolutions"] += 1
                kadabra_turns.append(evo["turn"])
                alakazam_analysis["evolution_patterns"]["hand_size_at_kadabra_evo"].append(evo["hand_size_before"])
            elif evo["evolved_to"] == "Alakazam":
                alakazam_analysis["evolution_patterns"]["alakazam_evolutions"] += 1
                alakazam_turns.append(evo["turn"])
                alakazam_analysis["evolution_patterns"]["hand_size_at_alakazam_evo"].append(evo["hand_size_before"])
        
        # Action analysis for supporters and tech cards
        for action in replay.actions:
            if action.player == 0 and action.action_kind == "PLAY_TRAINER":
                card = action.card_a
                if card == "Hilda":
                    alakazam_analysis["draw_supporter_usage"]["hilda"] += 1
                    alakazam_analysis["draw_supporter_usage"]["total_draw_supporters"] += 1
                elif card == "Dawn":
                    alakazam_analysis["draw_supporter_usage"]["dawn"] += 1
                    alakazam_analysis["draw_supporter_usage"]["total_draw_supporters"] += 1
                elif card == "Lillie's Determination":
                    alakazam_analysis["draw_supporter_usage"]["lillies_determination"] += 1
                    alakazam_analysis["draw_supporter_usage"]["total_draw_supporters"] += 1
                elif card == "Tulip":
                    alakazam_analysis["draw_supporter_usage"]["tulip"] += 1
                    alakazam_analysis["draw_supporter_usage"]["total_draw_supporters"] += 1
                elif card == "Enhanced Hammer":
                    alakazam_analysis["tech_card_usage"]["enhanced_hammer"] += 1
                elif card == "Boss's Orders":
                    alakazam_analysis["tech_card_usage"]["boss_orders"] += 1
                elif card == "Battle Cage":
                    alakazam_analysis["tech_card_usage"]["battle_cage"] += 1
                elif card == "Rare Candy":
                    alakazam_analysis["tech_card_usage"]["rare_candy"] += 1
        
        # Win/loss correlation
        had_alakazam_attack = len(replay.p0_alakazam_attacks) > 0
        if replay.winner == 0:
            if had_alakazam_attack:
                alakazam_analysis["win_correlation"]["wins_with_alakazam_attack"] += 1
            if replay.p0_hand_size_history:
                wins_hand_sizes.extend(replay.p0_hand_size_history)
        elif replay.winner == 1:
            if had_alakazam_attack:
                alakazam_analysis["win_correlation"]["losses_with_alakazam_attack"] += 1
            if replay.p0_hand_size_history:
                losses_hand_sizes.extend(replay.p0_hand_size_history)
    
    # Compute averages
    alakazam_analysis["attack_statistics"]["total_alakazam_attacks"] = total_attacks
    if all_attack_hand_sizes:
        alakazam_analysis["attack_statistics"]["avg_hand_size_at_attack"] = np.mean(all_attack_hand_sizes)
        alakazam_analysis["attack_statistics"]["avg_damage_per_attack"] = np.mean(all_attack_damages)
    if total_attacks > 0:
        alakazam_analysis["attack_statistics"]["one_hit_ko_rate"] = ko_count / total_attacks
    
    if all_hand_sizes:
        alakazam_analysis["hand_management"]["avg_hand_size"] = np.mean(all_hand_sizes)
    
    if kadabra_turns:
        alakazam_analysis["evolution_patterns"]["avg_turn_kadabra_evolved"] = np.mean(kadabra_turns)
    if alakazam_turns:
        alakazam_analysis["evolution_patterns"]["avg_turn_alakazam_evolved"] = np.mean(alakazam_turns)
    
    if wins_hand_sizes:
        alakazam_analysis["win_correlation"]["avg_hand_size_in_wins"] = np.mean(wins_hand_sizes)
    if losses_hand_sizes:
        alakazam_analysis["win_correlation"]["avg_hand_size_in_losses"] = np.mean(losses_hand_sizes)
    
    # Convert defaultdicts
    alakazam_analysis["attack_statistics"]["hand_size_distribution"] = dict(alakazam_analysis["attack_statistics"]["hand_size_distribution"])
    alakazam_analysis["attack_statistics"]["damage_distribution"] = dict(alakazam_analysis["attack_statistics"]["damage_distribution"])
    alakazam_analysis["attack_statistics"]["attacks_by_turn"] = dict(alakazam_analysis["attack_statistics"]["attacks_by_turn"])
    
    return alakazam_analysis


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
        
        # Winning/Losing patterns
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
    
    # Convert defaultdicts
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
        seq_counts = Counter(tuple(s) if isinstance(s[0], str) else tuple(tuple(x) if isinstance(x, list) else x for x in s) for s in sequences)
        return seq_counts.most_common(top_n)
    
    analysis["top_openings"] = {
        "p0": find_common_sequences(analysis["opening_moves"]["p0"]),
        "p1": find_common_sequences(analysis["opening_moves"]["p1"]),
    }
    
    return analysis


def compute_strategy_score(alakazam_analysis: Dict[str, Any], general_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a strategy score to evaluate how well the agent learned the Alakazam strategy.
    
    Key indicators of good Alakazam play:
    1. High hand size at attack time (10+ cards is good, 15+ is excellent)
    2. Using Alakazam as primary attacker
    3. Proper evolution timing (not rushing, building hand first)
    4. Using draw supporters effectively
    5. High damage output when attacking
    """
    
    scores = {
        "hand_size_score": 0,  # 0-100
        "attack_efficiency_score": 0,  # 0-100
        "evolution_timing_score": 0,  # 0-100
        "draw_support_score": 0,  # 0-100
        "tech_usage_score": 0,  # 0-100
        "overall_score": 0,  # 0-100
        "strategy_assessment": "",
        "recommendations": []
    }
    
    atk_stats = alakazam_analysis["attack_statistics"]
    evo_stats = alakazam_analysis["evolution_patterns"]
    hand_stats = alakazam_analysis["hand_management"]
    draw_stats = alakazam_analysis["draw_supporter_usage"]
    tech_stats = alakazam_analysis["tech_card_usage"]
    
    # 1. Hand Size Score
    # Ideal: 12-20 cards in hand at attack time
    avg_hand = atk_stats.get("avg_hand_size_at_attack", 0)
    if avg_hand >= 15:
        scores["hand_size_score"] = 100
    elif avg_hand >= 12:
        scores["hand_size_score"] = 80
    elif avg_hand >= 8:
        scores["hand_size_score"] = 60
    elif avg_hand >= 5:
        scores["hand_size_score"] = 40
    else:
        scores["hand_size_score"] = 20
    
    # 2. Attack Efficiency Score
    # Based on damage output and KO rate
    avg_dmg = atk_stats.get("avg_damage_per_attack", 0)
    ko_rate = atk_stats.get("one_hit_ko_rate", 0)
    
    dmg_score = min(100, (avg_dmg / 300) * 100)  # 300 damage = perfect
    ko_score = ko_rate * 100
    scores["attack_efficiency_score"] = (dmg_score * 0.6 + ko_score * 0.4)
    
    # 3. Evolution Timing Score
    # Ideal: Kadabra turns 3-6, Alakazam turns 5-10
    # Use a gentler curve that still rewards early evolution
    avg_kadabra_turn = evo_stats.get("avg_turn_kadabra_evolved", 0)
    avg_alakazam_turn = evo_stats.get("avg_turn_alakazam_evolved", 0)
    
    # Kadabra: 100 if turn 3-6, then decay by 8 per turn after
    if avg_kadabra_turn <= 6:
        kadabra_timing_score = 100
    elif avg_kadabra_turn <= 10:
        kadabra_timing_score = 100 - (avg_kadabra_turn - 6) * 15
    else:
        kadabra_timing_score = max(0, 40 - (avg_kadabra_turn - 10) * 5)
    
    # Alakazam: 100 if turn 5-10, then decay
    if avg_alakazam_turn <= 10:
        alakazam_timing_score = 100
    elif avg_alakazam_turn <= 15:
        alakazam_timing_score = 100 - (avg_alakazam_turn - 10) * 12
    else:
        alakazam_timing_score = max(0, 40 - (avg_alakazam_turn - 15) * 5)
    
    scores["evolution_timing_score"] = (kadabra_timing_score + alakazam_timing_score) / 2
    
    # 4. Draw Support Usage Score
    total_games = general_analysis["summary"]["total_games"]
    draw_per_game = draw_stats.get("total_draw_supporters", 0) / max(1, total_games)
    
    # Ideal: 3-5 draw supporters per game
    if 3 <= draw_per_game <= 6:
        scores["draw_support_score"] = 100
    elif 2 <= draw_per_game < 3:
        scores["draw_support_score"] = 70
    elif 1 <= draw_per_game < 2:
        scores["draw_support_score"] = 40
    else:
        scores["draw_support_score"] = draw_per_game * 10
    
    # 5. Tech Card Usage Score
    hammer_usage = tech_stats.get("enhanced_hammer", 0) / max(1, total_games)
    boss_usage = tech_stats.get("boss_orders", 0) / max(1, total_games)
    
    # Enhanced Hammer is key tech for Mist Energy counter
    tech_score = min(100, hammer_usage * 30 + boss_usage * 20)
    scores["tech_usage_score"] = tech_score
    
    # Overall Score (weighted average)
    scores["overall_score"] = (
        scores["hand_size_score"] * 0.30 +
        scores["attack_efficiency_score"] * 0.25 +
        scores["evolution_timing_score"] * 0.20 +
        scores["draw_support_score"] * 0.15 +
        scores["tech_usage_score"] * 0.10
    )
    
    # Assessment
    if scores["overall_score"] >= 80:
        scores["strategy_assessment"] = "EXCELLENT - Agent has learned the Alakazam strategy well!"
    elif scores["overall_score"] >= 60:
        scores["strategy_assessment"] = "GOOD - Agent understands core concepts but has room for improvement"
    elif scores["overall_score"] >= 40:
        scores["strategy_assessment"] = "DEVELOPING - Agent shows some understanding but needs more training"
    else:
        scores["strategy_assessment"] = "NEEDS WORK - Agent has not grasped the core strategy yet"
    
    # Recommendations
    if scores["hand_size_score"] < 60:
        scores["recommendations"].append("Focus on building hand size before attacking with Alakazam")
    if scores["attack_efficiency_score"] < 60:
        scores["recommendations"].append("Wait for higher hand sizes to maximize damage output")
    if scores["evolution_timing_score"] < 60:
        scores["recommendations"].append("Optimize evolution timing - don't rush Alakazam evolution")
    if scores["draw_support_score"] < 60:
        scores["recommendations"].append("Use more draw supporters (Hilda, Dawn, Lillie's Determination)")
    if scores["tech_usage_score"] < 60:
        scores["recommendations"].append("Utilize tech cards like Enhanced Hammer more effectively")
    
    return scores


def print_alakazam_analysis(alakazam_analysis: Dict[str, Any], strategy_score: Dict[str, Any]):
    """Print detailed Alakazam strategy analysis."""
    
    print("\n" + "="*70)
    print("🔮 ALAKAZAM STRATEGY ANALYSIS")
    print("="*70)
    
    # Strategy Score Summary
    print(f"\n📊 STRATEGY SCORE: {strategy_score['overall_score']:.1f}/100")
    print(f"   Assessment: {strategy_score['strategy_assessment']}")
    
    print(f"\n   Component Scores:")
    print(f"      Hand Size Management: {strategy_score['hand_size_score']:.1f}/100")
    print(f"      Attack Efficiency:    {strategy_score['attack_efficiency_score']:.1f}/100")
    print(f"      Evolution Timing:     {strategy_score['evolution_timing_score']:.1f}/100")
    print(f"      Draw Support Usage:   {strategy_score['draw_support_score']:.1f}/100")
    print(f"      Tech Card Usage:      {strategy_score['tech_usage_score']:.1f}/100")
    
    if strategy_score["recommendations"]:
        print(f"\n   💡 Recommendations:")
        for rec in strategy_score["recommendations"]:
            print(f"      • {rec}")
    
    # Attack Statistics
    atk = alakazam_analysis["attack_statistics"]
    print(f"\n⚔️ ALAKAZAM ATTACK STATISTICS")
    print(f"   Total Alakazam Attacks: {atk['total_alakazam_attacks']}")
    print(f"   Avg Hand Size at Attack: {atk['avg_hand_size_at_attack']:.1f} cards")
    print(f"   Avg Damage per Attack: {atk['avg_damage_per_attack']:.0f} damage")
    print(f"   One-Hit KO Rate: {atk['one_hit_ko_rate']:.1%}")
    
    if atk["hand_size_distribution"]:
        print(f"\n   Hand Size Distribution at Attack:")
        for size in sorted(atk["hand_size_distribution"].keys()):
            count = atk["hand_size_distribution"][size]
            bar = "█" * min(20, count)
            damage = size * 20
            print(f"      {size:2d} cards ({damage:3d} dmg): {bar} ({count})")
    
    # Evolution Patterns
    evo = alakazam_analysis["evolution_patterns"]
    print(f"\n🔄 EVOLUTION PATTERNS")
    print(f"   Kadabra Evolutions: {evo['kadabra_evolutions']} (Avg Turn: {evo['avg_turn_kadabra_evolved']:.1f})")
    print(f"   Alakazam Evolutions: {evo['alakazam_evolutions']} (Avg Turn: {evo['avg_turn_alakazam_evolved']:.1f})")
    
    if evo["hand_size_at_kadabra_evo"]:
        print(f"   Avg Hand Size at Kadabra Evo: {np.mean(evo['hand_size_at_kadabra_evo']):.1f}")
    if evo["hand_size_at_alakazam_evo"]:
        print(f"   Avg Hand Size at Alakazam Evo: {np.mean(evo['hand_size_at_alakazam_evo']):.1f}")
    
    # Hand Management
    hand = alakazam_analysis["hand_management"]
    print(f"\n✋ HAND MANAGEMENT")
    print(f"   Average Hand Size: {hand['avg_hand_size']:.1f} cards")
    print(f"   Max Hand Size Reached: {hand['max_hand_size_reached']} cards")
    
    # Draw Supporter Usage
    draw = alakazam_analysis["draw_supporter_usage"]
    print(f"\n📚 DRAW SUPPORTER USAGE")
    print(f"   Hilda: {draw['hilda']}")
    print(f"   Dawn: {draw['dawn']}")
    print(f"   Lillie's Determination: {draw['lillies_determination']}")
    print(f"   Tulip: {draw['tulip']}")
    print(f"   Total: {draw['total_draw_supporters']}")
    
    # Tech Card Usage
    tech = alakazam_analysis["tech_card_usage"]
    print(f"\n🔧 TECH CARD USAGE")
    print(f"   Enhanced Hammer: {tech['enhanced_hammer']} (Counter Mist Energy)")
    print(f"   Boss's Orders: {tech['boss_orders']} (Target selection)")
    print(f"   Battle Cage: {tech['battle_cage']} (Bench protection)")
    print(f"   Rare Candy: {tech['rare_candy']} (Fast evolution)")
    
    # Win Correlation
    win = alakazam_analysis["win_correlation"]
    print(f"\n🏆 WIN CORRELATION")
    print(f"   Wins with Alakazam Attack: {win['wins_with_alakazam_attack']}")
    print(f"   Losses with Alakazam Attack: {win['losses_with_alakazam_attack']}")
    print(f"   Avg Hand Size in Wins: {win['avg_hand_size_in_wins']:.1f}")
    print(f"   Avg Hand Size in Losses: {win['avg_hand_size_in_losses']:.1f}")


def print_general_analysis(analysis: Dict[str, Any]):
    """Pretty print the general analysis results."""
    print("\n" + "="*70)
    print("🎮 GENERAL GAME ANALYSIS")
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


def save_analysis(analysis: Dict[str, Any], alakazam_analysis: Dict[str, Any], 
                  strategy_score: Dict[str, Any], replays: List[GameReplay], 
                  output_dir: str = "."):
    """Save all analysis data."""
    
    # Combined analysis
    full_analysis = {
        "general": {
            "summary": analysis["summary"],
            "action_frequencies": analysis["action_frequencies"],
            "card_play_rates": analysis["card_play_rates"],
            "attack_usage": analysis["attack_usage"],
        },
        "alakazam_strategy": alakazam_analysis,
        "strategy_score": strategy_score,
    }
    
    with open(os.path.join(output_dir, "strategy_analysis_v2.json"), "w") as f:
        json.dump(full_analysis, f, indent=2, default=str)
    
    # Sample replays
    sample_replays = [r.to_dict() for r in replays[:20]]
    with open(os.path.join(output_dir, "sample_replays_v2.json"), "w") as f:
        json.dump(sample_replays, f, indent=2)
    
    print(f"\n💾 Saved analysis to {output_dir}/strategy_analysis_v2.json")
    print(f"💾 Saved sample replays to {output_dir}/sample_replays_v2.json")


def main():
    parser = argparse.ArgumentParser(description='Analyze RL Agent Alakazam Strategy')
    parser.add_argument('--games', type=int, default=100, help='Number of games to analyze')
    parser.add_argument('--policy', type=str, default='rl_policy.pt', help='Policy file to use')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', type=str, default='.', help='Output directory')
    args = parser.parse_args()
    
    print("🔍 Loading Policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(args.policy, map_location=device)
        from tcg.train_bc import PolicyNet
        model = PolicyNet(checkpoint["obs_dim"], checkpoint["n_actions"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"   Loaded {args.policy}")
    except Exception as e:
        print(f"❌ Could not load policy: {e}")
        return
    
    print(f"\n🎮 Running {args.games} analysis games...")
    replays = run_analysis_games(model, device, num_games=args.games, verbose=args.verbose)
    
    print("\n📈 Analyzing general patterns...")
    general_analysis = analyze_replays(replays)
    
    print("🔮 Analyzing Alakazam strategy...")
    alakazam_analysis = analyze_alakazam_strategy(replays)
    
    print("📊 Computing strategy score...")
    strategy_score = compute_strategy_score(alakazam_analysis, general_analysis)
    
    print_general_analysis(general_analysis)
    print_alakazam_analysis(alakazam_analysis, strategy_score)
    
    save_analysis(general_analysis, alakazam_analysis, strategy_score, replays, args.output)


if __name__ == "__main__":
    main()