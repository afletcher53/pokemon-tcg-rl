"""
Strategy Visualization for Pokemon TCG RL Agent
Generates visual reports of Alakazam strategy learning.
"""
import json
import os
import argparse
from typing import Dict, Any


def create_ascii_bar(value: float, max_value: float = 100, width: int = 30) -> str:
    """Create an ASCII progress bar."""
    filled = int((value / max_value) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:.1f}%"


def create_histogram(data: Dict[int, int], title: str, max_width: int = 40) -> str:
    """Create an ASCII histogram."""
    if not data:
        return f"{title}\n   No data available"
    
    max_val = max(data.values())
    lines = [title]
    
    for key in sorted(data.keys()):
        count = data[key]
        bar_width = int((count / max_val) * max_width)
        bar = "█" * bar_width
        lines.append(f"   {key:3d}: {bar} ({count})")
    
    return "\n".join(lines)


def generate_strategy_report(analysis_path: str) -> str:
    """Generate a comprehensive strategy report."""
    
    with open(analysis_path, 'r') as f:
        data = json.load(f)
    
    general = data.get("general", {})
    alakazam = data.get("alakazam_strategy", {})
    score = data.get("strategy_score", {})
    
    report = []
    
    # Header
    report.append("╔" + "═" * 68 + "╗")
    report.append("║" + " ALAKAZAM STRATEGY LEARNING REPORT ".center(68) + "║")
    report.append("╚" + "═" * 68 + "╝")
    
    # Overall Score
    overall = score.get("overall_score", 0)
    report.append("\n┌─ OVERALL STRATEGY SCORE ─────────────────────────────────────────┐")
    report.append(f"│                                                                   │")
    report.append(f"│   {create_ascii_bar(overall, 100, 50)}    │")
    report.append(f"│                                                                   │")
    report.append(f"│   Assessment: {score.get('strategy_assessment', 'N/A'):<49} │")
    report.append("└───────────────────────────────────────────────────────────────────┘")
    
    # Component Scores
    report.append("\n┌─ COMPONENT SCORES ───────────────────────────────────────────────┐")
    components = [
        ("Hand Size Management", score.get("hand_size_score", 0)),
        ("Attack Efficiency", score.get("attack_efficiency_score", 0)),
        ("Evolution Timing", score.get("evolution_timing_score", 0)),
        ("Draw Support Usage", score.get("draw_support_score", 0)),
        ("Tech Card Usage", score.get("tech_usage_score", 0)),
    ]
    
    for name, value in components:
        bar = create_ascii_bar(value, 100, 35)
        report.append(f"│   {name:<22} {bar} │")
    report.append("└───────────────────────────────────────────────────────────────────┘")
    
    # Key Metrics
    atk = alakazam.get("attack_statistics", {})
    evo = alakazam.get("evolution_patterns", {})
    hand = alakazam.get("hand_management", {})
    
    report.append("\n┌─ KEY STRATEGY METRICS ────────────────────────────────────────────┐")
    report.append(f"│                                                                   │")
    report.append(f"│   Alakazam Attacks: {atk.get('total_alakazam_attacks', 0):<45} │")
    report.append(f"│   Avg Hand at Attack: {atk.get('avg_hand_size_at_attack', 0):.1f} cards {'(EXCELLENT!)' if atk.get('avg_hand_size_at_attack', 0) >= 12 else '(needs work)' if atk.get('avg_hand_size_at_attack', 0) < 8 else '(good)':>33} │")
    report.append(f"│   Avg Damage Output: {atk.get('avg_damage_per_attack', 0):.0f} damage{' ':>36} │")
    report.append(f"│   OHKO Rate: {atk.get('one_hit_ko_rate', 0):.1%}{' ':>48} │")
    report.append(f"│                                                                   │")
    report.append(f"│   Kadabra Evo Turn: {evo.get('avg_turn_kadabra_evolved', 0):.1f} (ideal: 3-5){' ':>32} │")
    report.append(f"│   Alakazam Evo Turn: {evo.get('avg_turn_alakazam_evolved', 0):.1f} (ideal: 5-8){' ':>31} │")
    report.append(f"│   Max Hand Size: {hand.get('max_hand_size_reached', 0)} cards{' ':>40} │")
    report.append(f"│                                                                   │")
    report.append("└───────────────────────────────────────────────────────────────────┘")
    
    # Win/Loss Summary
    summary = general.get("summary", {})
    report.append("\n┌─ GAME OUTCOMES ───────────────────────────────────────────────────┐")
    p0_wins = summary.get("p0_wins", 0)
    p1_wins = summary.get("p1_wins", 0)
    total = summary.get("total_games", 1)
    
    p0_bar = "█" * int((p0_wins / total) * 40)
    p1_bar = "█" * int((p1_wins / total) * 40)
    
    report.append(f"│   Alakazam (P0): {p0_bar:<40} {p0_wins:>3}/{total} │")
    report.append(f"│   Charizard (P1): {p1_bar:<40} {p1_wins:>3}/{total} │")
    report.append("└───────────────────────────────────────────────────────────────────┘")
    
    # Hand Size Distribution at Attack
    hand_dist = atk.get("hand_size_distribution", {})
    if hand_dist:
        # Convert string keys to int if needed
        hand_dist = {int(k): v for k, v in hand_dist.items()}
        
        report.append("\n┌─ HAND SIZE AT ALAKAZAM ATTACK ────────────────────────────────────┐")
        max_count = max(hand_dist.values()) if hand_dist else 1
        for size in sorted(hand_dist.keys()):
            count = hand_dist[size]
            damage = size * 20
            bar_width = int((count / max_count) * 30)
            bar = "█" * bar_width
            quality = "🔥" if size >= 12 else "✓" if size >= 8 else "⚠"
            report.append(f"│   {size:2d} cards ({damage:3d} dmg) {quality}: {bar:<30} {count:>3} │")
        report.append("└───────────────────────────────────────────────────────────────────┘")
    
    # Recommendations
    recs = score.get("recommendations", [])
    if recs:
        report.append("\n┌─ RECOMMENDATIONS FOR IMPROVEMENT ────────────────────────────────┐")
        for rec in recs:
            # Wrap long recommendations
            if len(rec) > 63:
                report.append(f"│   • {rec[:60]}...  │")
            else:
                report.append(f"│   • {rec:<63} │")
        report.append("└───────────────────────────────────────────────────────────────────┘")
    
    # Strategy Explanation
    report.append("\n┌─ OPTIMAL ALAKAZAM STRATEGY ──────────────────────────────────────┐")
    report.append("│                                                                   │")
    report.append("│   1. Setup Phase (Turns 1-3):                                     │")
    report.append("│      • Bench multiple Abra via Buddy-Buddy Poffin                 │")
    report.append("│      • Use Fan Rotom's Fan Call ability on Turn 1                 │")
    report.append("│      • Play draw supporters (Hilda, Dawn) to build hand           │")
    report.append("│                                                                   │")
    report.append("│   2. Evolution Phase (Turns 3-6):                                 │")
    report.append("│      • Evolve Abra → Kadabra (Psychic Draw: +2 cards)             │")
    report.append("│      • Continue using draw supporters                              │")
    report.append("│      • Set up Dudunsparce for Run Away Draw ability               │")
    report.append("│                                                                   │")
    report.append("│   3. Attack Phase (Turn 6+):                                      │")
    report.append("│      • Evolve Kadabra → Alakazam (Psychic Draw: +3 cards)         │")
    report.append("│      • Attack with Mind Jack: 20 damage × hand size               │")
    report.append("│      • Use Enhanced Hammer to remove Mist Energy                  │")
    report.append("│      • Use Boss's Orders to target key threats                    │")
    report.append("│                                                                   │")
    report.append("│   Target: 12-20 cards in hand = 240-400 damage per attack!        │")
    report.append("└───────────────────────────────────────────────────────────────────┘")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Visualize Strategy Analysis')
    parser.add_argument('--input', type=str, default='strategy_analysis_v2.json', 
                        help='Path to analysis JSON file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for report (default: print to console)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Analysis file '{args.input}' not found.")
        print("Run analyze_strategies_v2.py first to generate the analysis.")
        return
    
    report = generate_strategy_report(args.input)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()