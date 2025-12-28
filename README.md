# Pokémon TCG Reinforcement Learning Agent 🧠🃏

A state-of-the-art Reinforcement Learning agent for the Pokémon Trading Card Game, built with **AlphaZero-style MCTS**, **Population-Based Training (PBT)**, and **Hierarchical Strategic Combos**.

![Status](https://img.shields.io/badge/Status-Training-green) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

## 🚀 Key Features

### 1. Advanced Architecture (V3.0)
- **1584-Dimensional Observation Space**: Fully observable game state including:
  - **11 Energy Types** per Pokémon slot (blindness fixed)
  - Detailed hand tracking (Active + Bench + Opponent)
  - Discard pile composition
- **AlphaZero MCTS**: Monte Carlo Tree Search with neural network guidance for policy and value estimation.
- **Hierarchical RL**: Strategic "Combos" injected into MCTS to bridge temporal gaps (e.g., *Rare Candy → Charizard ex*).

### 2. Robust Game Engine
- **Full Rules Implementation**: Supports evolution, energy attachment, retreat, status conditions, and prize taking.
- **Accurate Ability System**:
  - **Per-Pokémon Tracking**: Abilities tracked individually (e.g., multiple *Alakazam* draws).
  - **Bench Abilities**: *Fezandipiti ex*, *Munkidori*, *Dudunsparce*, etc., work correctly from the bench.
  - **On-Evolution Triggers**: *Alakazam* and *Kadabra* properly trigger when evolved.
  - **Strategic Targeting**: *Munkidori* can target any opponent Pokémon (Battle Cage properly blocks bench only).

### 3. Training Pipeline
- **Population-Based Training (PBT)**: Evolves a population of agents with genetic algorithms (Mutation/Crossover/Elitism).
- **Curriculum Learning**: Phases training from "Force Attack" -> "Survival" -> "Full Strategy".
- **League System**: Maintains a history of past agents to prevent cyclic learning.
- **Combo Discovery**: Automatically identifies and logs successful action sequences to `agent_combos.txt`.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/afletcher53/pokemon-tcg-rl.git
cd pokemon-tcg-rl

# Install dependencies (requires Python 3.10+)
pip install -r requirements.txt
# OR if using uv/pkm
source pkm/bin/activate
```

## 🎮 Usage

### Train the Agent
Start the advanced training loop with MCTS and PBT:

```bash
# Full training run
python train_advanced.py --episodes 5000 --num_workers 4 --mcts_sims 50

# Optimized for aggressive play (Anti-Stall)
python train_advanced.py --episodes 5000 --max_turns 30 --scripted_ratio 0.6 --curriculum 1000
```

### Visualize Matches
Watch the agent play in real-time or review replays:

1. Open `replay_viewer.html` in your browser.
2. Load a JSON replay file (generated in `replays/`).

### Run Strategy Analysis
Analyze the agent's deck consistency and strategy:

```bash
python scripts/analyze_strategies.py
```

---

## 🧩 Project Structure

- **`tcg/`**: Core game engine.
  - `env.py`: Main Gym-compatible environment.
  - `mcts.py`: Monte Carlo Tree Search implementation.
  - `cards.py`: Card definitions and logic.
  - `effects.py`: Card effects (Draw, Search, Heal, Damage).
- **`train_advanced.py`**: Main training script with PBT and Genetic Algorithms.
- **`standalone_full.py`**: Single-file version of the entire codebase for portability.
- **`scripts/`**: Analysis and utility tools.

## 📝 Recent Fixes & Improvements

| Feature | Status | Description |
|---------|--------|-------------|
| **Energy Blindness** | ✅ Fixed | Expanded observation space to 1584 dimensions to track all energy types. |
| **Bench Abilities** | ✅ Fixed | *Fezandipiti ex*, *Genesect ex*, *Munkidori* now work from Bench. |
| **Combo System** | ✅ Added | 8 Strategic Combos (e.g., *Rare Candy -> Charizard*) injected into search. |
| **Charizard ex** | ✅ Optimized | *Infernal Reign* now distributes energy randomly to learn optimal placement. |
| **Logic Bugs** | ✅ Fixed | Fixed *Genesect ex* unlimited use, *Munkidori* targeting, and *Alakazam* triggers. |

## 🤝 Contributing
Pull requests are welcome! Please ensure:
1. New cards are added to `tcg/cards.py` and `tcg/effects.py`.
2. Tests pass (`python test_game.py`).
3. Syntax commands (`py_compile`) run clean.

## 📜 License
MIT License. Pokémon is a trademark of Nintendo/Creatures/GAME FREAK. This project is a research implementation and not affiliated with the Pokémon Company.
