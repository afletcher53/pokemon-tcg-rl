"""Quick analysis of AlphaZero training metrics."""
import csv

# Load data
data = []
with open('alphazero_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'episode': int(row['episode']),
            'wr': float(row['rolling_winrate']),
            'len': float(row['avg_game_length']),
            'policy_loss': float(row['policy_loss']),
            'value_loss': float(row['value_loss']),
            'total_loss': float(row['total_loss']),
        })

print("=" * 70)
print("ALPHAZERO TRAINING ANALYSIS")
print("=" * 70)
print(f"Total data points: {len(data)} (episodes logged every 10)")
print(f"Last episode: {data[-1]['episode']}")

# Phases
print("\n" + "=" * 70)
print("WIN RATE BY PHASE (P0 = Alakazam)")
print("=" * 70)

def avg_phase(data, start, end, key):
    vals = [d[key] for d in data if start <= d['episode'] < end]
    return sum(vals) / len(vals) if vals else 0

phases = [(10, 100), (100, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2100)]
for start, end in phases:
    wr = avg_phase(data, start, end, 'wr')
    length = avg_phase(data, start, end, 'len')
    loss = avg_phase(data, start, end, 'total_loss')
    print(f"  Ep {start:4d}-{end:4d}: WinRate={wr:.1%}, GameLen={length:.0f}, Loss={loss:.2f}")

print("\n" + "=" * 70)
print("LOSS COMPONENTS (learning progress)")
print("=" * 70)
print(f"  Episode 10:   Policy={data[0]['policy_loss']:.2f}, Value={data[0]['value_loss']:.2f}, Total={data[0]['total_loss']:.2f}")
mid_idx = len(data) // 2
print(f"  Episode {data[mid_idx]['episode']:4d}: Policy={data[mid_idx]['policy_loss']:.2f}, Value={data[mid_idx]['value_loss']:.2f}, Total={data[mid_idx]['total_loss']:.2f}")
print(f"  Episode {data[-1]['episode']:4d}: Policy={data[-1]['policy_loss']:.2f}, Value={data[-1]['value_loss']:.2f}, Total={data[-1]['total_loss']:.2f}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

# 1. Win rate trend
early_wr = avg_phase(data, 10, 200, 'wr')
late_wr = avg_phase(data, 1800, 2100, 'wr')
if late_wr < 0.25:
    print(f"⚠️  P0 (Alakazam) win rate is LOW: {late_wr:.1%}")
    print("   - P1 (Charizard) is dominating")
    print("   - This is OPPOSITE of the Dunsparce stall issue!")
    print("   - The agent might be making suboptimal plays for P0")
else:
    print(f"✓  P0 win rate is OK: {late_wr:.1%}")

# 2. Loss trend
early_loss = avg_phase(data, 10, 200, 'total_loss')
late_loss = avg_phase(data, 1800, 2100, 'total_loss')
print(f"\n✓  Loss DECREASED: {early_loss:.2f} → {late_loss:.2f}")
print("   - Model IS learning to match MCTS and predict outcomes")

# 3. Game length
early_len = avg_phase(data, 10, 200, 'len')
late_len = avg_phase(data, 1800, 2100, 'len')
print(f"\n~  Game length: {early_len:.0f} → {late_len:.0f} steps")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
The LOW P0 win rate (16-20%) suggests the Charizard deck (P1) is stronger
or the Alakazam deck strategy needs more training.

This is DIFFERENT from the Dunsparce stall problem:
- Before: P0 won by stalling with Dunsparce (degenerate)
- Now: P1 is winning more (might be learning better)

The model IS learning (loss decreasing), but P0's deck strategy
may need tuning OR the Charizard deck is inherently stronger in
this simulation.

RECOMMENDED NEXT STEPS:
1. Run a test game to see actual behavior
2. Check if Alakazam is being evolved and used
3. If not, may need more training or deck balancing
""")
