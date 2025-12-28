#!/usr/bin/env python3
"""
Advanced Training Metrics Visualizer for Pokemon TCG RL Agent

Features:
- Auto-detects which training metrics file to use
- Identifies learning breakthroughs (sudden improvements)
- Shows what was learned (strategy shifts)
- Live mode with real-time updates
- Beautiful visualizations with confidence intervals

Usage:
    python3 plot_training.py                         # Auto-detect and plot
    python3 plot_training.py --file alphazero_metrics.csv
    python3 plot_training.py --live                  # Live updates
    python3 plot_training.py --output graph.png     # Save to file
"""

import argparse
import time
import os
import sys
from datetime import datetime

def install_deps():
    """Install required packages if missing."""
    try:
        import matplotlib
        import pandas
        import numpy
    except ImportError:
        print("Installing required packages...")
        os.system(f"{sys.executable} -m pip install matplotlib pandas numpy")

install_deps()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np


# =============================================================================
# LEARNING EVENT DETECTION
# =============================================================================

def detect_learning_events(df, column='rolling_winrate', threshold=0.1, window=20):
    """
    Detect significant changes in a metric that indicate learning breakthroughs.
    
    Returns list of (episode, change_magnitude, direction) tuples.
    """
    if len(df) < window * 2:
        return []
    
    events = []
    values = df[column].values
    episodes = df['episode'].values
    
    for i in range(window, len(values) - window):
        before = values[i-window:i].mean()
        after = values[i:i+window].mean()
        change = after - before
        
        if abs(change) > threshold:
            direction = "improved" if change > 0 else "declined"
            events.append({
                'episode': episodes[i],
                'change': change,
                'direction': direction,
                'before': before,
                'after': after,
            })
            # Skip ahead to avoid detecting the same event multiple times
            i += window // 2
    
    return events


def analyze_strategy_shift(df, event_episode, window=50):
    """
    Analyze what changed around a learning event.
    """
    idx = df[df['episode'] == event_episode].index
    if len(idx) == 0:
        return {}
    
    idx = idx[0]
    start = max(0, idx - window)
    end = min(len(df), idx + window)
    
    before = df.iloc[start:idx]
    after = df.iloc[idx:end]
    
    analysis = {}
    
    # Check game length changes
    if 'avg_game_length' in df.columns:
        len_before = before['avg_game_length'].mean()
        len_after = after['avg_game_length'].mean()
        len_change = len_after - len_before
        if abs(len_change) > 5:
            if len_change < 0:
                analysis['game_speed'] = f"Games got faster ({len_before:.0f} → {len_after:.0f} steps)"
            else:
                analysis['game_speed'] = f"Games got longer ({len_before:.0f} → {len_after:.0f} steps)"
    
    # Check loss changes
    if 'total_loss' in df.columns:
        loss_before = before['total_loss'].mean()
        loss_after = after['total_loss'].mean()
        if loss_after < loss_before * 0.9:
            analysis['loss'] = f"Loss dropped significantly ({loss_before:.2f} → {loss_after:.2f})"
    elif 'policy_loss' in df.columns:
        loss_before = before['policy_loss'].mean()
        loss_after = after['policy_loss'].mean()
        if loss_after < loss_before * 0.9:
            analysis['loss'] = f"Policy loss dropped ({loss_before:.2f} → {loss_after:.2f})"
    
    # Check value loss (AlphaZero specific)
    if 'value_loss' in df.columns:
        val_before = before['value_loss'].mean()
        val_after = after['value_loss'].mean()
        if val_after < val_before * 0.85:
            analysis['value_prediction'] = f"Value predictions improved ({val_before:.2f} → {val_after:.2f})"
    
    return analysis


# =============================================================================
# VISUALIZATION
# =============================================================================

def get_latest_metrics_file():
    """Find the most recently modified metrics CSV."""
    candidates = [
        'alphazero_metrics.csv',
        'advanced_training_metrics.csv',
        'training_metrics.csv',
    ]
    
    files = []
    for f in candidates:
        if os.path.exists(f):
            files.append((f, os.path.getmtime(f)))
    
    if not files:
        return None
    
    # Return most recently modified
    files.sort(key=lambda x: x[1], reverse=True)
    return files[0][0]


def plot_alphazero_metrics(df, ax_dict, events=None):
    """Plot metrics for AlphaZero-style training."""
    
    # 1. Win Rate with events
    ax = ax_dict['winrate']
    ax.clear()
    
    episodes = df['episode'].values
    
    # Handle different column names for win rate
    win_rate = None
    for col in ['rolling_winrate', 'win_rate', 'rolling_wr_100', 'p0_winrate']:
        if col in df.columns:
            win_rate = df[col].values * 100
            break
    
    if win_rate is None:
        # Fallback - just show zeros
        win_rate = np.zeros(len(episodes))
    
    # Plot with gradient fill
    ax.plot(episodes, win_rate, 'cyan', linewidth=2, label='Win Rate')
    ax.fill_between(episodes, 0, win_rate, alpha=0.3, color='cyan')
    
    # Mark learning events
    if events:
        for event in events:
            color = 'lime' if event['direction'] == 'improved' else 'red'
            ax.axvline(x=event['episode'], color=color, linestyle='--', alpha=0.7, linewidth=2)
            ax.annotate(f"{event['change']*100:+.0f}%", 
                       xy=(event['episode'], min(95, event['after']*100 + 5)),
                       fontsize=9, color=color, fontweight='bold',
                       ha='center')
    
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Win Rate (%)', fontsize=10)
    ax.set_title('[TARGET] P0 Win Rate (with Learning Events)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random (50%)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Loss Components
    ax = ax_dict['loss']
    ax.clear()
    
    if 'policy_loss' in df.columns:
        ax.plot(episodes, df['policy_loss'], 'orange', linewidth=2, label='Policy Loss')
    if 'value_loss' in df.columns:
        ax.plot(episodes, df['value_loss'], 'purple', linewidth=2, label='Value Loss')
    if 'total_loss' in df.columns:
        ax.plot(episodes, df['total_loss'], 'white', linewidth=2, alpha=0.7, label='Total Loss')
    elif 'avg_loss' in df.columns:
        ax.plot(episodes, df['avg_loss'], 'white', linewidth=2, alpha=0.7, label='Total Loss')
    
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('[DOWN] Training Loss (Should Decrease)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Game Length
    ax = ax_dict['length']
    ax.clear()
    
    # Check for game length column
    length_col = None
    for col in ['avg_game_length', 'avg_length']:
        if col in df.columns:
            length_col = col
            break
    
    if length_col:
        length = df[length_col].values
        ax.plot(episodes, length, 'lime', linewidth=2)
        ax.fill_between(episodes, 0, length, alpha=0.2, color='lime')
        
        # Add trend line
        if len(episodes) > 10:
            z = np.polyfit(episodes, length, 1)
            p = np.poly1d(z)
            ax.plot(episodes, p(episodes), 'lime', linestyle='--', alpha=0.5, label=f'Trend: {"↓" if z[0] < 0 else "↑"}')
    
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Steps', fontsize=10)
    ax.set_title('[TIME] Average Game Length', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Strategy (Prizes, Evolutions, Checkpoint Progress)
    ax = ax_dict['strategy']
    ax.clear()
    
    has_strategy_data = False
    
    if 'avg_prizes' in df.columns:
        prizes = df['avg_prizes'].values
        ax.plot(episodes, prizes, 'cyan', linewidth=2, label='Avg Prizes')
        ax.fill_between(episodes, 0, prizes, alpha=0.2, color='cyan')
        has_strategy_data = True
    
    if 'avg_evolutions' in df.columns:
        evos = df['avg_evolutions'].values
        ax.plot(episodes, evos, 'orange', linewidth=2, linestyle='--', label='Avg Evolutions')
        has_strategy_data = True
    
    # Add checkpoint win rate if available
    if 'checkpoint_winrate' in df.columns:
        ckpt_wr = df['checkpoint_winrate'].values * 100  # Convert to percentage
        # Only plot non-0.5 values (actual evaluations)
        mask = ckpt_wr != 50.0
        if mask.any():
            ax2 = ax.twinx()
            ax2.scatter(np.array(episodes)[mask], ckpt_wr[mask], 
                       color='lime', s=80, marker='D', label='vs Prev Checkpoint', zorder=5)
            ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
            ax2.axhline(y=55, color='lime', linestyle=':', alpha=0.3)
            ax2.axhline(y=45, color='red', linestyle=':', alpha=0.3)
            ax2.set_ylabel('Checkpoint WR%', fontsize=10, color='lime')
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='y', labelcolor='lime')
            has_strategy_data = True

    if has_strategy_data:
        ax.set_title('[STRATEGY] Prizes, Evolutions & Progress', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
    else:
        # Fallback to Temperature if no prize data yet
        if 'temperature' in df.columns:
            temp = df['temperature'].values
            ax.plot(episodes, temp, 'red', linewidth=2, label='Temperature')
            ax.set_title('[TEMP] Exploration Decay', fontsize=12, fontweight='bold')
        elif 'buffer_size' in df.columns:
            buffer = df['buffer_size'].values
            ax.plot(episodes, buffer, 'yellow', linewidth=2, label='Buffer')
            ax.set_title('[BUFFER] Experience Replay', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Episode', fontsize=10)
    ax.grid(True, alpha=0.3)


def create_summary_panel(fig, df, events):
    """Create a text summary panel."""
    # Add text box with summary
    summary_lines = []
    
    if len(df) > 0:
        latest = df.iloc[-1]
        first = df.iloc[0]
        
        # Latest stats
        ep = int(float(latest['episode']))
        summary_lines.append(f"[STATS] Episode: {ep:,}")
        
        # Win rate - check multiple column names
        for col in ['rolling_winrate', 'win_rate', 'rolling_wr_100', 'p0_winrate']:
            if col in df.columns:
                wr = latest[col] * 100
                summary_lines.append(f"[TARGET] Win Rate: {wr:.1f}%")
                break
        
        # Game length - check multiple column names
        for col in ['avg_game_length', 'avg_length']:
            if col in df.columns:
                gl = latest[col]
                summary_lines.append(f"[TIME] Game Length: {gl:.0f} steps")
                break
        
        if 'total_loss' in df.columns:
            loss = latest['total_loss']
            summary_lines.append(f"[DOWN] Loss: {loss:.3f}")
        
        if 'temperature' in df.columns:
            temp = latest['temperature']
            summary_lines.append(f"[TEMP] Temperature: {temp:.3f}")
        
        if 'elo' in df.columns:
            elo = latest['elo']
            summary_lines.append(f"[TROPHY] ELO: {elo:.0f}")
        
        # Improvement calculation
        wr_col = None
        for col in ['rolling_winrate', 'win_rate', 'rolling_wr_100', 'p0_winrate']:
            if col in df.columns:
                wr_col = col
                break
        
        if wr_col and len(df) > 20:
            first_wr = df[wr_col].head(10).mean()
            last_wr = df[wr_col].tail(10).mean()
            improvement = (last_wr - first_wr) * 100
            if improvement > 0:
                summary_lines.append(f"\n[UP] Improved: +{improvement:.1f}%")
            else:
                summary_lines.append(f"\n[DOWN] Changed: {improvement:.1f}%")
    
    # Learning events summary
    if events:
        summary_lines.append(f"\n[!] Learning Events: {len(events)}")
        for i, event in enumerate(events[:3]):  # Show top 3
            summary_lines.append(f"   Ep {event['episode']}: {event['change']*100:+.1f}%")
    
    summary_text = '\n'.join(summary_lines)
    
    # Add text box
    props = dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='cyan')
    fig.text(0.02, 0.98, summary_text, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', fontfamily='monospace', color='white',
             bbox=props)


def plot_metrics(csv_path=None, output_file=None, live=False, detect_events=True):
    """Main plotting function."""
    
    # Auto-detect file if not specified
    if csv_path is None:
        csv_path = get_latest_metrics_file()
        if csv_path is None:
            print("[X] No metrics file found. Run training first!")
            return
        print(f"[FILE] Using: {csv_path}")
    
    plt.style.use('dark_background')
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('[GAME] Pokemon TCG AI Training Dashboard', fontsize=18, fontweight='bold', color='cyan')
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, left=0.18, right=0.98, top=0.92, bottom=0.08)
    
    ax_dict = {
        'winrate': fig.add_subplot(gs[0, 0]),
        'loss': fig.add_subplot(gs[0, 1]),
        'length': fig.add_subplot(gs[1, 0]),
        'strategy': fig.add_subplot(gs[1, 1]),
    }
    
    def update_plot():
        if not os.path.exists(csv_path):
            return False
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False
        
        if len(df) < 2:
            return False
        
        # Detect learning events
        events = []
        if detect_events:
            # Find the win rate column
            wr_col = None
            for col in ['rolling_winrate', 'win_rate', 'rolling_wr_100', 'p0_winrate']:
                if col in df.columns:
                    wr_col = col
                    break
            if wr_col:
                events = detect_learning_events(df, wr_col, threshold=0.08)
        
        # Plot
        plot_alphazero_metrics(df, ax_dict, events)
        create_summary_panel(fig, df, events)
        
        # Print learning events to console
        if events and not live:
            print("\n[!] LEARNING EVENTS DETECTED:")
            print("-" * 50)
            for event in events:
                analysis = analyze_strategy_shift(df, event['episode'])
                print(f"\n[*] Episode {event['episode']}: {event['direction'].upper()}")
                print(f"   Win rate: {event['before']*100:.1f}% → {event['after']*100:.1f}%")
                print(f"   Change: {event['change']*100:+.1f}%")
                if analysis:
                    for key, desc in analysis.items():
                        print(f"   • {desc}")
        
        return True
    
    if live:
        print("[LIVE] Live mode: Updates every 5 seconds. Press Ctrl+C to stop.")
        plt.ion()
        
        while True:
            try:
                success = update_plot()
                if success:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(5)
                else:
                    print(f"Waiting for {csv_path}...")
                    plt.pause(2)
            except KeyboardInterrupt:
                print("\n[BYE] Stopped live plotting.")
                break
        
        plt.ioff()
    else:
        if update_plot():
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='black')
                print(f"[OK] Saved plot to {output_file}")
            else:
                plt.show()
        else:
            print(f"[X] Could not read metrics from {csv_path}")


def print_learning_summary(csv_path=None):
    """Print a text-based learning summary."""
    if csv_path is None:
        csv_path = get_latest_metrics_file()
        if csv_path is None:
            print("No metrics file found.")
            return
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "=" * 60)
    print("[STATS] TRAINING SUMMARY")
    print("=" * 60)
    
    print(f"\nFile: {csv_path}")
    print(f"Episodes: {df['episode'].min()} to {df['episode'].max()}")
    
    # Detect events
    if 'rolling_winrate' in df.columns:
        events = detect_learning_events(df, 'rolling_winrate', threshold=0.08)
    else:
        events = []
    
    if events:
        print(f"\n[!] LEARNING BREAKTHROUGHS: {len(events)}")
        print("-" * 40)
        
        for i, event in enumerate(events, 1):
            print(f"\n{i}. Episode {event['episode']}")
            print(f"   Direction: {event['direction'].upper()}")
            print(f"   Win Rate: {event['before']*100:.1f}% → {event['after']*100:.1f}%")
            
            analysis = analyze_strategy_shift(df, event['episode'])
            if analysis:
                print("   What changed:")
                for desc in analysis.values():
                    print(f"     • {desc}")
    else:
        print("\n(No significant learning breakthroughs detected yet)")
    
    # Current status
    latest = df.iloc[-1]
    print(f"\n[UP] CURRENT STATUS:")
    print("-" * 40)
    print(f"   Episode: {int(latest['episode'])}")
    if 'rolling_winrate' in df.columns:
        print(f"   Win Rate: {latest['rolling_winrate']*100:.1f}%")
    if 'total_loss' in df.columns:
        print(f"   Loss: {latest['total_loss']:.3f}")
    if 'avg_game_length' in df.columns:
        print(f"   Game Length: {latest['avg_game_length']:.0f} steps")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training metrics with learning detection")
    parser.add_argument("--file", "-f", help="CSV file to read (auto-detects if not specified)")
    parser.add_argument("--output", "-o", help="Save plot to file instead of showing")
    parser.add_argument("--live", "-l", action="store_true", help="Live update mode")
    parser.add_argument("--summary", "-s", action="store_true", help="Print text summary only")
    parser.add_argument("--no-events", action="store_true", help="Disable learning event detection")
    args = parser.parse_args()
    
    if args.summary:
        print_learning_summary(args.file)
    else:
        plot_metrics(args.file, args.output, args.live, detect_events=not args.no_events)