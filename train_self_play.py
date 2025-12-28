from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
import argparse
from tqdm import tqdm
from tcg.env import PTCGEnv
from tcg.train_bc import PolicyNet
from tcg.actions import ACTION_TABLE
from tcg.mcts import MCTS

def run_self_play(verbose=False, use_mcts=False, mcts_sims=10):
    # Set environment variable for effects.py to check
    if not verbose:
        os.environ['PTCG_QUIET'] = '1'
    
    print("Initializing Self-Play RL Training...")
    if use_mcts:
        print(f"⚠️  MCTS Enabled (sims={mcts_sims}). Training will be significantly slower but higher quality.")
    
    # Setup device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Try to load existing policies
    # Observation size: 5 (glob) + 4 (hand) + 1 (op_count) + 12 slots * 10 + 8 (opp_model) + 18 (discard) = 156
    obs_dim = 156
    n_actions = len(ACTION_TABLE)
    model = PolicyNet(obs_dim, n_actions).to(device)
    
    loaded = False
    for path in ["rl_policy.pt", "bc_policy.pt"]:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=device)
                cp_n_actions = checkpoint.get("n_actions", 0)
                
                if cp_n_actions == n_actions:
                    model.load_state_dict(checkpoint["state_dict"])
                    print(f"Loaded {path} perfectly - continuing training")
                    loaded = True
                    break
                else:
                    print(f"Action space mismatch in {path}: {cp_n_actions} vs {n_actions}. Loading backbone only.")
                    # Partial load: only load layers that match in size
                    state_dict = checkpoint["state_dict"]
                    model_dict = model.state_dict()
                    # Filter out keys with different shapes
                    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    print(f"Loaded {len(pretrained_dict)} matching tensors from {path}")
                    loaded = True
                    break
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    if not loaded:
        print("Starting with a fresh policy.")

    optimizer = optim.Adam(model.parameters(), lr=3e-4)  # Higher LR for faster initial learning
    env = PTCGEnv(scripted_opponent=False, max_turns=200)  # More turns for decisive games
    
    # Entropy coefficient for exploration bonus (decays over training)
    entropy_coef_start = 0.05
    entropy_coef_end = 0.01

    # Decks
    # P0: Alakazam
    deck_p0 = []
    deck_p0.extend(["Abra"] * 4)
    deck_p0.extend(["Kadabra"] * 3)
    deck_p0.extend(["Alakazam"] * 4)
    deck_p0.extend(["Dunsparce"] * 4) # 3+1
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
    deck_p1.extend(["Pidgeotto"] * 2)  # 1 MEW + 1 OBF (treating as same)
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

    # Extended training for better learning
    episodes = 200000 if not use_mcts else 50000  # Reduced episodes for MCTS due to speed
    print(f"Starting {episodes} episodes of Self-Play with strategic reward shaping...")
    print("Key improvements: discount factor, return normalization, anti-pass heuristic, higher exploration")

    # Init MCTS if needed - using improved settings
    mcts_agent = None
    if use_mcts:
        mcts_agent = MCTS(
            policy_net=model, 
            device=device, 
            num_simulations=mcts_sims, 
            max_rollout_steps=150,  # Longer rollouts for better evaluation
            use_value_net=False,    # Use rollouts (not value net for PolicyNet)
            use_policy_rollouts=True,  # Use policy network for smarter rollouts
            temperature=1.0,  # Will be annealed
            c_puct=1.5
        )

    wins_p0 = 0
    wins_p1 = 0
    draws = 0
    
    # --- METRICS TRACKING ---
    import csv
    from collections import deque
    
    metrics_file = open("training_metrics.csv", "w", newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["episode", "win_rate_p0", "win_rate_p1", "draw_rate", 
                             "avg_game_length", "avg_loss", "avg_entropy", "rolling_wr_100"])
    
    # Rolling windows for tracking convergence
    recent_p0_wins = deque(maxlen=100)  # Last 100 games
    recent_game_lengths = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    recent_entropies = deque(maxlen=100)
    
    # For plateau detection (early stopping indicator)
    best_rolling_wr = 0.0
    plateau_episodes = 0
    plateau_threshold = 2000  # How many episodes without improvement

    # Use tqdm progress bar if not verbose
    episode_iterator = tqdm(range(episodes), desc="Training", disable=verbose, mininterval=0.5, ncols=120) if not verbose else range(episodes)
    
    for ep in episode_iterator:
        # Anneal shaping rewards: 1.0 -> 0.0 over first 75% of episodes
        # This allows "training wheels" early, but forces pure winning later
        shaping_scale = max(0.0, 1.0 - (ep / (episodes * 0.75)))
        
        obs, info = env.reset(options={"decks": [deck_p0, deck_p1]})
        done = False
        
        traj_p0 = []
        traj_p1 = []
        
        # Max steps per episode to prevent infinite loops (Increased from 500 to 1500 to allow Alakazam to play out)
        max_steps_per_episode = 2000
        step_count = 0
        
        while not done and step_count < max_steps_per_episode:
            turn_player = env._gs.turn_player
            mask = info["action_mask"]
            
            # Select Action
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            logits = model(obs_t) # [1, n_actions]
            
            # Apply Mask
            mask_t = torch.from_numpy(mask).float().to(logits.device)
            huge_neg = torch.ones_like(logits) * -1e9
            masked_logits = torch.where(mask_t.unsqueeze(0) > 0, logits, huge_neg)
            
            probs = torch.softmax(masked_logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            
            if use_mcts and turn_player == 0:
                 # Use MCTS for P0 (Agent)
                 act_idx = mcts_agent.search(env)
                 # We still need log_prob relative to the NETWORK to update the network
                 # This is effectively "Off-Policy" if we blindly use log_prob(act) step?
                 # No, we treat MCTS as "Selecting" the action, and we update the Model to agree with it?
                 # Standard RL: log_prob(a) * R.
                 # If 'a' comes from MCTS, and R is positive, we encourage Model to output 'a'.
                 log_prob = dist.log_prob(torch.tensor(act_idx).to(device))
            else:
                 act_idx = dist.sample().item()
                 log_prob = dist.log_prob(torch.tensor(act_idx).to(device))
            
            # Execute
            obs, step_reward, done, truncated, info = env.step(act_idx)
            
            # Store trajectory
            # Note: We skipped the "Categorize valid actions" heuristic block during replacement but it was largely display/unused logic in critical path
            step_data = {
                "log_prob": log_prob,
                "reward": 0.0, # Will be filled later
                "val": 0.0, # Critic not used here
                "entropy": dist.entropy().item()
            }
            if turn_player == 0:
                traj_p0.append(step_data)
            else:
                traj_p1.append(step_data)
                
            step_count += 1
            
            # Capture state before step for delta rewards
            prizes_before = env._gs.players[turn_player].prizes_taken
            
            # Update state references after step
            gs = env._gs
            me = gs.players[turn_player] # Note: turn_player might have changed if turn ended? 
            # Actually, turn_player index in the loop `turn_player = env._gs.turn_player` 
            # refers to the player WHO ACCTED.
            # If `env.step` caused turn end, `env._gs.turn_player` changes.
            # We should use the `turn_player` variable captured at start of loop (line 153).
            # So `me = gs.players[turn_player]` is correct for the player who acted.
            
            prizes_after = gs.players[turn_player].prizes_taken
            
            # ============================================================
            # SOPHISTICATED REWARD SHAPING (ANNEALED)
            # Teaches strategic concepts specific to each deck
            # ============================================================
            act_obj = ACTION_TABLE[act_idx]
            
            # Heuristic accumulator - will be scaled by shaping_scale
            shaping_reward = 0.0
            
            # --- PRIZE TAKING BOUNTY (The Ultimate Goal - UNSCALED) ---
            if prizes_after > prizes_before:
                diff = prizes_after - prizes_before
                step_reward += (5.0 * diff) # HUGE reward for taking prizes
            
            # --- Base Action Rewards ---
            # --- Base Action Rewards ---
            if act_obj.kind == "PASS":
                # Penalize passing HEAVILY to force action (Scaled)
                shaping_reward -= 0.5
            elif act_obj.kind == "ATTACK":
                # Check for Dunsparce stalling vs Alakazam aggression
                # Note: 'me' is the player state after the move, but active shouldn't change unless KO'd/switched
                # Better to trust the name of the active from before if we had it, but checking current is usually fine for 'who attacked'
                attacker_name = me.active.name if me.active else ""
                
                # Penalize Dunsparce stalling late game (Turn > 6)
                if attacker_name in ["Dunsparce", "Dudunsparce"] and gs.turn_number > 6:
                     shaping_reward -= 0.2
                
                # Reward Alakazam attacking
                if attacker_name == "Alakazam":
                     shaping_reward += 0.2
                
                # Evolution Reward
            elif act_obj.kind in ("EVOLVE_ACTIVE", "EVOLVE_BENCH"):
                 shaping_reward += 0.1
            else:
                step_reward -= 0.01  # Time penalty (Unscaled - encourage efficiency)
            
            # --- Energy Attachment (CRITICAL for attacking - prioritize Alakazam) ---
            if act_obj.kind == "ATTACH_ACTIVE":
                active_name = me.active.name if me.active else None
                if active_name == "Alakazam":
                    # HUGE reward for attaching to Alakazam - enables Powerful Hand!
                    shaping_reward += 2.0
                elif active_name in ("Charizard ex", "Pidgeot ex"):
                    shaping_reward += 1.5
                else:
                    shaping_reward += 0.5  # Smaller reward for basic/support Pokemon
            elif act_obj.kind == "ATTACH_BENCH":
                # Check what we're attaching to
                target_idx = act_obj.b
                if 0 <= target_idx < len(me.bench):
                    target = me.bench[target_idx]
                    if target.name == "Alakazam":
                        # Great - powering up Alakazam on bench!
                        shaping_reward += 1.5
                    else:
                        shaping_reward += 0.2

            # --- Evolution Timing Rewards (CRITICAL for Alakazam) ---
            if act_obj.kind in ("EVOLVE_ACTIVE", "EVOLVE_BENCH"):
                evo_card = act_obj.a
                turn = gs.turn_number
                
                # Alakazam line - reward early evolution HEAVILY
                if evo_card == "Kadabra":
                    if turn <= 4:
                        shaping_reward += 1.0  # Excellent - early Kadabra (triggers Psychic Draw!)
                    elif turn <= 6:
                        shaping_reward += 0.5  # Good
                    elif turn <= 8:
                        shaping_reward += 0.2  # Okay
                    # Late evolution gets no bonus - implicit penalty
                        
                elif evo_card == "Alakazam":
                    if turn <= 6:
                        shaping_reward += 1.5  # Excellent - early Alakazam ready to attack
                    elif turn <= 8:
                        shaping_reward += 0.8  # Good
                    elif turn <= 10:
                        shaping_reward += 0.3  # Okay
                    # Late evolution gets no bonus
                
                # Charizard line
                elif evo_card == "Charmeleon":
                    if turn <= 4:
                        shaping_reward += 0.4
                    else:
                        shaping_reward += 0.1
                        
                elif evo_card == "Charizard ex":
                    if turn <= 6:
                        shaping_reward += 0.7
                    else:
                        shaping_reward += 0.3
                
                # Pidgeot line (consistency engine)
                elif evo_card in ("Pidgeotto", "Pidgeot ex"):
                    if turn <= 5:
                        shaping_reward += 0.5  # Early Pidgeot is key
                    else:
                        shaping_reward += 0.2
                        
                # Dudunsparce for draw
                elif evo_card == "Dudunsparce":
                    if turn <= 1:
                        shaping_reward += 0.2
                    else:
                        shaping_reward += 0.0
            
            # --- Attack Rewards (DAMAGE-BASED - encourages learning to use best attackers) ---
            if act_obj.kind == "ATTACK":
                hand_size = len(me.hand)
                active = me.active.name if me.active else None
                
                # Calculate actual damage this attack would deal
                actual_damage = 0
                if active == "Alakazam":
                    # Powerful Hand: 20 * Hand Size
                    actual_damage = 20 * hand_size
                elif active == "Kadabra":
                    actual_damage = 30  # Super Psy Bolt
                elif active == "Fan Rotom":
                    actual_damage = 70  # Assault Landing (if stadium)
                elif active == "Dunsparce":
                    actual_damage = 20  # Gnaw
                elif active == "Dudunsparce":
                    actual_damage = 90  # Land Crush
                elif active == "Charizard ex":
                    actual_damage = 180 + (30 * gs.players[1-turn_player].prizes_taken)
                elif active == "Pidgeot ex":
                    actual_damage = 120
                else:
                    actual_damage = 30  # Default assumption for other attackers
                
                # DAMAGE-SCALED REWARD: Higher damage = much higher reward
                # This naturally teaches the model to prefer Alakazam attacks
                # Formula: base 0.5 + (damage / 40) → gives nice scaling
                # 20 damage (Dunsparce) → 0.5 + 0.5 = 1.0
                # 70 damage (Fan Rotom) → 0.5 + 1.75 = 2.25
                # 300 damage (Alakazam w/ 15 cards) → 0.5 + 7.5 = 8.0
                damage_reward = 0.5 + (actual_damage / 40.0)
                shaping_reward += damage_reward
                
                # PENALTY: Attacking with weak Pokemon when Alakazam is ready on bench
                # Check if Alakazam on bench has energy attached
                alakazam_ready_on_bench = False
                for slot in me.bench:
                    if slot.name == "Alakazam" and len(slot.energy) >= 1:
                        alakazam_ready_on_bench = True
                        break
                
                if alakazam_ready_on_bench and active not in ("Alakazam", "Charizard ex"):
                    # Penalize attacking with weak mons when Alakazam is ready
                    shaping_reward -= 2.0
            
            # --- Draw/Setup Actions ---
            if act_obj.kind == "USE_ACTIVE_ABILITY":
                active = me.active.name if me.active else None
                if active in ("Kadabra", "Alakazam"):
                    # Psychic Draw is On-Evolve, not Active Ability. Removed this reward.
                    pass
                elif active == "Dudunsparce":
                    # Punish Stall Loops!
                    shaping_reward -= 0.5 
                elif active == "Pidgeot ex":
                    # Quick Search - consistency
                    shaping_reward += 0.1 # Reduced from 0.5
                elif active == "Tatsugiri":
                    shaping_reward += 0.1
                elif active == "Fezandipiti ex":
                    # Flip the Script - VERY STRONG after a knockout!
                    # Draw 3 cards from deck is huge value
                    shaping_reward += 1.5  # Large reward to encourage usage
                
                # Check for Fan Rotom (may have been used from bench, so check flag)
                if getattr(me, "fan_call_used", False) and gs.turn_number <= 2:
                    # Give a large reward for using Fan Call early
                    shaping_reward += 0.8
            
            # --- Trainer Card Strategy ---
            if act_obj.kind == "PLAY_TRAINER":
                card = act_obj.a
                op = gs.players[1 - turn_player]
                
                # --- OPPONENT MODEL: Play important cards before suspected hand disruption ---
                # If opponent recently played supporter, they might have disruption saved (Iono, Judge, etc.)
                suspected_hand_disruption = (hasattr(op, 'turns_since_supporter') and 
                                  op.turns_since_supporter == 0 and 
                                  hasattr(op, 'total_supporters_played') and
                                  op.total_supporters_played > 0)
                
                # --- SURVIVAL MODE: Empty bench is DANGEROUS! ---
                # If bench is empty, we risk losing instantly on a KO
                # Drawing/searching for basics becomes CRITICAL
                bench_count = sum(1 for s in me.bench if s.name)
                in_survival_mode = (bench_count == 0)
                
                # Check if hand has basics we could bench
                basics_in_hand = ["Abra", "Charmander", "Pidgey", "Fan Rotom", "Dunsparce", 
                                  "Fezandipiti ex", "Tatsugiri", "Psyduck", "Charcadet"]
                has_basic_in_hand = any(c in basics_in_hand for c in me.hand)
                
                # Draw supporters - always valuable, CRITICAL when bench empty
                if card in ("Hilda", "Dawn", "Lillie's Determination", "Iono"):
                    if in_survival_mode and not has_basic_in_hand:
                        shaping_reward += 1.5  # HUGE reward - this might save us!
                    else:
                        shaping_reward += 0.2
                    
                # --- STADIUM BUMPING REWARDS ---
                # Stadiums: Reward for bumping opponent's stadium, bonus for specific effects
                elif card in ("Artazon", "Battle Cage"):
                    stadium_reward = 0.1  # Base reward for playing any stadium
                    
                    # Check if we're bumping opponent's stadium
                    opponent_had_stadium = getattr(op, 'stadium', None) is not None
                    if opponent_had_stadium:
                        stadium_reward += 0.3  # Big bonus for removing opponent's stadium!
                        # Extra bonus if we bumped Battle Cage (protects their bench)
                        if getattr(op, 'stadium', None) == "Battle Cage":
                            stadium_reward += 0.2  # They lose bench protection
                        # Extra if we bumped Artazon (search advantage)
                        elif getattr(op, 'stadium', None) == "Artazon":
                            stadium_reward += 0.1  # They lose search option
                    
                    # Card-specific bonuses
                    if card == "Artazon":
                        # SURVIVAL MODE: Artazon directly benches a basic - critical!
                        if in_survival_mode:
                            stadium_reward += 2.0  # CRITICAL - prevents loss!
                        elif gs.turn_number <= 3:
                            stadium_reward += 0.3  # Early Artazon = bench development
                        else:
                            stadium_reward += 0.1
                    elif card == "Battle Cage":
                        # Battle Cage is good when we have bench to protect
                        bc_bench_count = sum(1 for s in me.bench if s.name)
                        if bc_bench_count >= 2:
                            stadium_reward += 0.2  # Good for protecting bench
                    
                    shaping_reward += stadium_reward
                    
                # Setup cards
                elif card == "Buddy-Buddy Poffin":
                    if in_survival_mode:
                        shaping_reward += 2.0  # CRITICAL - this benches basics and prevents loss!
                    elif gs.turn_number <= 3:
                        shaping_reward += 0.4  # Early bench is crucial
                    else:
                        shaping_reward += 0.1
                elif card == "Rare Candy":
                    base_reward = 0.3  # Skip stage = tempo
                    # Extra bonus if opponent might disrupt our hand
                    if suspected_hand_disruption:
                        base_reward += 0.3  # Use it now before they shuffle our hand!
                    shaping_reward += base_reward
                elif card == "Boss's Orders":
                    # Smart Boss's Orders evaluation - based on OUTCOME potential
                    # Good Boss's Orders: enables KO, pulls weak target, disrupts setup
                    target_idx = act_obj.b if act_obj.b is not None else 0
                    boss_reward = 0.1  # Base minimal reward
                    op = gs.players[1 - turn_player]  # Get opponent
                    
                    if 0 <= target_idx < len(op.bench):
                        target_mon = op.bench[target_idx]
                        if target_mon:
                            # Get target's remaining HP
                            from tcg.cards import card_def
                            try:
                                target_hp = card_def(target_mon.name).hp - (target_mon.damage or 0)
                            except:
                                target_hp = 60  # Default assumption
                            
                            # Estimate our damage potential (simplified)
                            my_active = me.active.name if me.active else None
                            hand_size = len(me.hand)
                            my_damage = 30  # Default
                            if my_active == "Alakazam":
                                my_damage = 20 * hand_size
                            elif my_active == "Dudunsparce":
                                my_damage = 90
                            elif my_active == "Charizard ex":
                                my_damage = 180 + (30 * me.prizes_taken if hasattr(me, 'prizes_taken') else 0)
                            elif my_active == "Fan Rotom":
                                my_damage = 70
                            
                            # REWARD: Can we KO the target?
                            if my_damage >= target_hp:
                                boss_reward = 1.5  # Great Boss! Securing a KO!
                            # REWARD: Pulling a key evolution (easier to KO basics)
                            # REWARD: Pulling unevolved basics (evolution starters)
                            target_cd = card_def(target_mon.name)
                            is_evo_starter = target_cd.subtype == "Basic" and target_cd.hp <= 70
                            if is_evo_starter:
                                boss_reward = 0.8  # Good - kill before they evolve
                            # REWARD: Pulling damaged Pokemon for cleanup
                            elif target_mon.damage and target_mon.damage > 0:
                                boss_reward = 0.6  # Decent - already damaged
                            # PENALTY: Pulling a tank that we can't handle
                            elif target_hp > my_damage * 2:
                                boss_reward = -0.3  # Bad target selection
                            
                            # --- OPPONENT MODEL BONUS (DECK-AGNOSTIC) ---
                            # Target unevolved evolution starters if opponent just searched for evolutions
                            if hasattr(op, 'last_searched_type') and op.last_searched_type in ("Stage1", "Stage2", "Evolution"):
                                if is_evo_starter:
                                    boss_reward += 0.5  # Good timing - kill the base before they evolve
                    
                    shaping_reward += boss_reward
                elif card == "Enhanced Hammer":
                    shaping_reward += 0.2  # Disruption
                elif card == "Ultra Ball":
                    # Reward smart search choices based on target
                    target = act_obj.b
                    if target == 6:  # Intentional fail (hand thinning)
                        if "Iono" in me.hand:
                            shaping_reward += 0.3  # Good - setting up for Iono
                        else:
                            shaping_reward -= 0.2  # Usually suboptimal
                    elif target == 5:  # Key attacker
                        shaping_reward += 0.5  # Very good - getting main win condition
                    elif target == 3:  # Evolution of active
                        shaping_reward += 0.4  # Good - advancing board state
                    elif target == 4:  # Evolution of bench
                        shaping_reward += 0.3  # Decent - setting up future
                    elif target == 2:  # Stage 2
                        if "Rare Candy" in me.hand:
                            shaping_reward += 0.4  # Can use Rare Candy!
                        else:
                            shaping_reward += 0.2
                    elif target == 0:  # Basic
                        bench_count = sum(1 for s in me.bench if s.name)
                        if bench_count == 0:
                            shaping_reward += 0.5  # Critical - need bench presence!
                        else:
                            shaping_reward += 0.1
                    # Target 1 (Stage 1) is usually suboptimal
                elif card == "Nest Ball":
                    # Reward smart bench search
                    # SURVIVAL MODE: Nest Ball directly benches a basic - critical!
                    if in_survival_mode:
                        shaping_reward += 2.0  # CRITICAL - prevents loss!
                    else:
                        target = act_obj.b
                        if target == 0:  # Evolution starters
                            shaping_reward += 0.4  # Good - enables future evolutions
                        elif target == 1:  # Support Pokemon
                            if gs.turn_number <= 2:
                                shaping_reward += 0.3  # Early support is good
                            else:
                                shaping_reward += 0.1
                        elif target == 2:  # Tech Pokemon
                            shaping_reward += 0.2  # Situational value
                        # Target 3 (any) is neutral
                elif card == "Night Stretcher":
                    # Reward smart recovery
                    target = act_obj.b
                    if target == 3:  # Key attacker recovery
                        shaping_reward += 0.5  # Very good - getting back win condition
                    elif target == 2:  # Evolution of in-play
                        shaping_reward += 0.4  # Good - can evolve immediately
                    elif target == 1:  # Energy
                        shaping_reward += 0.2  # Energy acceleration
                    # Target 0 (any Pokemon) is neutral
                elif card == "Buddy-Buddy Poffin":
                    # Reward based on how many Pokemon benched
                    target = act_obj.b
                    if target == 2:  # Bench 2 Pokemon
                        shaping_reward += 0.4  # Best value from the card
                    elif target == 1:  # Bench 1 Pokemon
                        shaping_reward += 0.2  # Some value
                    elif target == 0:  # Intentional fail
                        # Only good if you have Iono coming or want to thin hand
                        if "Iono" in me.hand:
                            shaping_reward += 0.2  # Good hand manipulation
                        else:
                            shaping_reward -= 0.3  # Wasted card
            
            # --- Board Development ---
            if act_obj.kind == "PLAY_BASIC_TO_BENCH":
                benched = act_obj.a
                turn = gs.turn_number
                
                # SURVIVAL MODE: Any basic benched when bench is empty is CRITICAL!
                bench_was_empty = sum(1 for s in me.bench if s.name) <= 1  # Just benched = 1
                if bench_was_empty:
                    shaping_reward += 2.5  # HUGE reward - this prevents instant loss on KO!
                
                # Key evolution starters - reward early benching
                if benched == "Abra":
                    if turn <= 2:
                        shaping_reward += 0.8  # Critical - early Abra enables fast Alakazam
                    elif turn <= 4:
                        shaping_reward += 0.4
                    else:
                        shaping_reward += 0.1
                elif benched == "Charmander":
                    if turn <= 2:
                        shaping_reward += 0.6
                    elif turn <= 4:
                        shaping_reward += 0.3
                    else:
                        shaping_reward += 0.1
                elif benched == "Pidgey":
                    if turn <= 2:
                        shaping_reward += 0.5
                    else:
                        shaping_reward += 0.2
                elif turn <= 3:
                    shaping_reward += 0.3  # Other basics early
                else:
                    shaping_reward += 0.1
            
            # --- Retreat to Best Attacker (DAMAGE-POTENTIAL based) ---
            if act_obj.kind == "RETREAT_TO":
                target_idx = act_obj.b
                if 0 <= target_idx < len(me.bench):
                    target = me.bench[target_idx]
                    current_active = me.active.name if me.active else None
                    hand_size = len(me.hand)
                    
                    # Calculate damage potential for current active
                    current_damage = 0
                    if current_active == "Alakazam" and len(me.active.energy) >= 1:
                        current_damage = 20 * hand_size
                    elif current_active == "Dudunsparce":
                        current_damage = 90
                    elif current_active == "Charizard ex":
                        current_damage = 180
                    elif current_active == "Fan Rotom":
                        current_damage = 70
                    elif current_active in ("Kadabra", "Abra", "Dunsparce"):
                        current_damage = 30
                    
                    # Calculate damage potential for target
                    target_damage = 0
                    can_attack = len(target.energy) >= 1
                    if target.name == "Alakazam" and can_attack:
                        target_damage = 20 * hand_size
                    elif target.name == "Dudunsparce" and can_attack:
                        target_damage = 90
                    elif target.name == "Charizard ex" and can_attack:
                        target_damage = 180
                    elif target.name == "Fan Rotom" and can_attack:
                        target_damage = 70
                    elif can_attack:
                        target_damage = 30
                    
                    # REWARD based on improvement in damage potential
                    damage_improvement = target_damage - current_damage
                    
                    if damage_improvement > 100:
                        shaping_reward += 2.0  # Huge improvement (e.g., weak -> Alakazam)
                    elif damage_improvement > 40:
                        shaping_reward += 1.0  # Good improvement
                    elif damage_improvement > 0:
                        shaping_reward += 0.3  # Slight improvement
                    elif damage_improvement == 0 and target.name == current_active:
                        shaping_reward -= 0.5  # Same Pokemon name, no point
                    elif damage_improvement < -50:
                        shaping_reward -= 1.0  # Big downgrade (e.g., Alakazam -> Dunsparce)
            
            # --- Prize-Taking Bonus (from KOs - UNSCALED) ---
            # This is tracked via the base reward from env, but let's add emphasis
            if done and gs.winner == turn_player:
                step_reward += 0.5  # Bonus for game-winning action (True Reward)
            
            # --- BENCH SAFETY: Critical for survival (SCALED) ---
            # If active is KO'd and no bench = AUTO-LOSE
            # But FULL bench = can't search for new solutions!
            bench_count = sum(1 for slot in me.bench if slot.name)
            
            if bench_count == 0:
                # VERY DANGEROUS - one KO away from losing!
                shaping_reward -= 1.0  # Strong penalty for empty bench
                # Extra penalty if active is damaged (close to death)
                if me.active and me.active.damage > 0:
                    shaping_reward -= 0.5  # Critical danger!
            elif bench_count == 1:
                # Still risky - one backup
                if me.active and me.active.damage > 50:
                    shaping_reward -= 0.3  # Low safety margin
            elif bench_count == 5:
                # FULL bench = can't search for new basics (bad flexibility)
                shaping_reward -= 0.2  # Mild penalty for lost flexibility
            # 2-4 bench is ideal - no reward/penalty needed
            
            # Apply annealed shaping reward
            step_reward += (shaping_reward * shaping_scale)

            # Record
            if turn_player == 0:
                traj_p0[-1]["reward"] = step_reward
            else:
                traj_p1[-1]["reward"] = step_reward
        
        # Handle timeout (max steps reached without game ending)
        if not done and step_count >= max_steps_per_episode:
            done = True
            env._gs.done = True
        
        # Game Over
        winner = env._gs.winner
        if winner == 0: wins_p0 += 1
        elif winner == 1: wins_p1 += 1
        else: draws += 1
        
        # Calculate Returns
        # Win: +1.0, Loss: -1.0, Draw/Timeout: -0.3 (penalize draws, but less than losses)
        # In tournaments, wins matter - draws and losses both hurt standings
        # Win: +2.0 (Incentivize Winning strongly)
        # Loss: -1.0
        # Draw/Timeout: -1.0 (Same as loss - NEVER DRAW!)
        
        # Win Reason check
        reason = env._gs.win_reason
        
        # EARLY WIN BONUS: Winning faster is better for Bo3 tournaments
        # Turn 4 win = 2.5x multiplier, Turn 15 win = 1.0x multiplier
        # This encourages aggressive, efficient play
        game_turn = gs.turn_number
        early_win_multiplier = max(1.0, 2.5 - (game_turn - 4) * 0.15)  # Caps at 1.0 for turn 14+
        early_win_multiplier = min(2.5, early_win_multiplier)  # Caps at 2.5 for turn 4 or earlier
        
        if winner == 0:
            if "Prize" in reason:
                base_reward = 10.0 * early_win_multiplier  # Up to 25.0 for early prize wins!
                R0, R1 = base_reward, -2.0
            else:
                R0, R1 = 1.0, -1.0 # Deck out win (boring, no multiplier)
                
            if "Deck Out" in reason and winner == 0: R1 = -5.0 # P1 Decked out
            
        elif winner == 1:
            if "Prize" in reason:
                base_reward = 10.0 * early_win_multiplier
                R0, R1 = -2.0, base_reward
            else:
                R0, R1 = -1.0, 1.0
                
            if "Deck Out" in reason and winner == 1: R0 = -5.0 # P0 Decked out
            
        else:  # Draw/Timeout
            R0, R1 = -2.0, -2.0 # Both Lose
        
        loss = []
        
        def compute_returns(traj, final_reward, gamma=0.99):
            R = final_reward
            returns = []
            for item in reversed(traj):
                R = item["reward"] + gamma * R 
                returns.insert(0, R)
            return returns

        returns_p0 = compute_returns(traj_p0, R0)
        returns_p1 = compute_returns(traj_p1, R1)
        
        # Combine all returns for normalization (reduces variance significantly)
        all_returns = returns_p0 + returns_p1
        if len(all_returns) > 1:
            mean_ret = sum(all_returns) / len(all_returns)
            std_ret = (sum((r - mean_ret)**2 for r in all_returns) / len(all_returns)) ** 0.5
            std_ret = max(std_ret, 1e-8)  # Prevent division by zero
            returns_p0 = [(r - mean_ret) / std_ret for r in returns_p0]
            returns_p1 = [(r - mean_ret) / std_ret for r in returns_p1]
        
        # Entropy coefficient decays from entropy_coef_start to entropy_coef_end
        entropy_coef = entropy_coef_start + (entropy_coef_end - entropy_coef_start) * (ep / episodes)
        
        entropy_sum = torch.tensor(0.0, device=device)
        for i, item in enumerate(traj_p0):
            loss.append(-item["log_prob"] * returns_p0[i])
            entropy_sum = entropy_sum + item["entropy"]
            
        for i, item in enumerate(traj_p1):
            loss.append(-item["log_prob"] * returns_p1[i])
            entropy_sum = entropy_sum + item["entropy"]
            
        if loss:
            optimizer.zero_grad()
            policy_loss = torch.stack(loss).mean()  
            # Entropy bonus (subtract because we want to maximize entropy, minimize loss)
            total_loss = policy_loss - entropy_coef * entropy_sum / max(len(loss), 1)
            total_loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            recent_losses.append(policy_loss.item())
            recent_entropies.append((entropy_sum / max(len(loss), 1)).item())
        
        # Track game length and winner
        recent_game_lengths.append(step_count)
        recent_p0_wins.append(1 if winner == 0 else 0)
        
        # Update progress bar with more info
        if (ep + 1) % 1 == 0: # Update progress bar every episode
            rolling_wr = sum(recent_p0_wins) / len(recent_p0_wins) if recent_p0_wins else 0
            avg_length = sum(recent_game_lengths) / len(recent_game_lengths) if recent_game_lengths else 0
            
            if verbose:
                print(f"Episode {ep+1}/{episodes} | P0: {wins_p0} | P1: {wins_p1} | Draw: {draws} | Rolling WR: {rolling_wr:.1%}")
            elif hasattr(episode_iterator, 'set_postfix'):
                episode_iterator.set_postfix({
                    'P0': wins_p0, 
                    'P1': wins_p1,
                    'Draw': draws,
                    'RollingWR': f'{rolling_wr:.0%}',
                    'AvgLen': f'{avg_length:.0f}'
                })
        
        # Log to CSV every 10 episodes
        if (ep + 1) % 10 == 0:
            rolling_wr = sum(recent_p0_wins) / len(recent_p0_wins) if recent_p0_wins else 0
            avg_length = sum(recent_game_lengths) / len(recent_game_lengths) if recent_game_lengths else 0
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            avg_entropy = sum(recent_entropies) / len(recent_entropies) if recent_entropies else 0
            
            metrics_writer.writerow([
                ep + 1,
                wins_p0 / (ep + 1),
                wins_p1 / (ep + 1),
                draws / (ep + 1),
                avg_length,
                avg_loss,
                avg_entropy,
                rolling_wr
            ])
            metrics_file.flush()  # Ensure data is written
            
            # Plateau detection
            if rolling_wr > best_rolling_wr + 0.01:  # 1% improvement
                best_rolling_wr = rolling_wr
                plateau_episodes = 0
            else:
                plateau_episodes += 100
            
            if plateau_episodes >= plateau_threshold and ep > 5000:
                print(f"\n⚠️ Potential convergence: No improvement in {plateau_threshold} episodes. Rolling WR: {rolling_wr:.1%}")
            
        # Periodic Save
        if (ep + 1) % 25 == 0:  # Save frequently for testing
             torch.save({
                "obs_dim": obs_dim,
                "n_actions": n_actions,
                "state_dict": model.state_dict()
            }, "rl_policy.pt")
             if verbose:
                 print("Saved rl_policy.pt")

    # Close metrics file
    metrics_file.close()
    print(f"Training metrics saved to training_metrics.csv")

    # Save Final
    torch.save({
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "state_dict": model.state_dict()
    }, "rl_policy.pt")
    print("Saved rl_policy.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pokemon TCG RL Agent')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose debug output (default: progress bar mode)')
    parser.add_argument("--mcts", action="store_true", help="Use MCTS for action selection (Slow)")
    parser.add_argument("--mcts_sims", type=int, default=10, help="Number of MCTS simulations")
    args = parser.parse_args()
    
    run_self_play(verbose=args.verbose, use_mcts=args.mcts, mcts_sims=args.mcts_sims)

