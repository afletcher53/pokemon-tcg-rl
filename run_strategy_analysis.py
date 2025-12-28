#!/usr/bin/env python3
"""
Quick-start script for Pokemon TCG RL Strategy Analysis.

Usage:
    python run_strategy_analysis.py              # Run 100 games with default policy
    python run_strategy_analysis.py --games 500  # Run 500 games
    python run_strategy_analysis.py --quick      # Quick test with 20 games
"""
import subprocess
import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run Strategy Analysis')
    parser.add_argument('--games', type=int, default=100, help='Number of games')
    parser.add_argument('--policy', type=str, default='rl_policy.pt', help='Policy file')
    parser.add_argument('--quick', action='store_true', help='Quick test (20 games)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    num_games = 20 if args.quick else args.games
    
    print("=" * 60)
    print("🎮 POKEMON TCG RL STRATEGY ANALYZER")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  • Games to analyze: {num_games}")
    print(f"  • Policy file: {args.policy}")
    print(f"  • Mode: {'Quick test' if args.quick else 'Full analysis'}")
    
    if not os.path.exists(args.policy):
        print(f"\n❌ Policy file '{args.policy}' not found!")
        print("   Run training first: python -m tcg.train_rl")
        return 1
    
    print(f"\n🔄 Running strategy analysis...")
    print("-" * 60)
    
    # Run the analysis
    cmd = [
        sys.executable, 
        "analyze_strategies_v2.py",
        "--games", str(num_games),
        "--policy", args.policy
    ]
    if args.verbose:
        cmd.append("--verbose")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("\n❌ Analysis failed!")
        return 1
    
    # Generate visual report if analysis succeeded
    if os.path.exists("strategy_analysis_v2.json"):
        print("\n" + "-" * 60)
        print("📊 Generating visual report...")
        print("-" * 60)
        
        subprocess.run([
            sys.executable,
            "visualize_strategy.py",
            "--input", "strategy_analysis_v2.json"
        ])
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)
    print("\nOutput files:")
    print("  • strategy_analysis_v2.json - Full analysis data")
    print("  • sample_replays_v2.json - Sample game replays")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())