"""
Scripted (Heuristic-based) Agents for Pokemon TCG.

These agents provide external pressure during self-play training to prevent
overfitting to the neural network's own quirks. By mixing in opponents with
known-good strategies, the trained model must learn general game skills.
"""

import random
from typing import Optional, Tuple, List
import numpy as np

from tcg.actions import ACTION_TABLE, Action
from tcg.cards import card_def


class ScriptedAgent:
    """
    Rule-based opponent with configurable strategy.
    
    Strategies:
    - "aggressive": Prioritize attacking and taking prizes
    - "evolution_rush": Prioritize evolution, then attack
    - "defensive": Prioritize benching and building board
    - "energy_first": Prioritize energy attachment
    - "random": Random legal actions (baseline)
    """
    
    def __init__(self, strategy: str = "aggressive"):
        self.strategy = strategy
        self.name = f"scripted_{strategy}"
    
    def select_action(self, env, obs: np.ndarray, mask: np.ndarray) -> int:
        """
        Select an action based on heuristics.
        
        Args:
            env: The PTCGEnv environment
            obs: Current observation (not used by most strategies)
            mask: Boolean mask of legal actions
            
        Returns:
            Action index
        """
        legal_actions = np.where(mask)[0]
        
        if len(legal_actions) == 0:
            return 0  # PASS if nothing legal (shouldn't happen)
        
        if len(legal_actions) == 1:
            return legal_actions[0]  # Only one choice
        
        if self.strategy == "aggressive":
            return self._aggressive_action(env, mask, legal_actions)
        elif self.strategy == "evolution_rush":
            return self._evolution_rush_action(env, mask, legal_actions)
        elif self.strategy == "defensive":
            return self._defensive_action(env, mask, legal_actions)
        elif self.strategy == "energy_first":
            return self._energy_first_action(env, mask, legal_actions)
        elif self.strategy == "control":
            return self._control_action(env, mask, legal_actions)
        elif self.strategy == "combo":
            return self._combo_action(env, mask, legal_actions)
        elif self.strategy == "random":
            return self._random_action(legal_actions)
        else:
            return self._random_action(legal_actions)
    
    def _get_actions_by_kind(self, legal_actions: np.ndarray, kind: str) -> List[int]:
        """Get legal action indices that match the given kind."""
        return [i for i in legal_actions if ACTION_TABLE[i].kind == kind]
    
    def _get_actions_by_kinds(self, legal_actions: np.ndarray, kinds: List[str]) -> List[int]:
        """Get legal action indices that match any of the given kinds."""
        return [i for i in legal_actions if ACTION_TABLE[i].kind in kinds]
    
    def _random_action(self, legal_actions: np.ndarray) -> int:
        """Random legal action (excluding PASS if other options exist)."""
        non_pass = [a for a in legal_actions if a != 0]
        if non_pass:
            return random.choice(non_pass)
        return legal_actions[0]
    
    def _aggressive_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Aggressive strategy:
        1. Attack if possible (prioritize high damage)
        2. Attach energy to active
        3. Evolve active
        4. Use abilities
        5. Play trainers
        6. Bench basics
        7. Random non-pass
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. ATTACK - highest priority
        attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
        if attacks:
            return random.choice(attacks)
        
        # 2. Attach energy to ACTIVE
        attach_active = self._get_actions_by_kind(legal_actions, "ATTACH_ACTIVE")
        if attach_active:
            return random.choice(attach_active)
        
        # 3. Evolve active
        evolve_active = self._get_actions_by_kind(legal_actions, "EVOLVE_ACTIVE")
        if evolve_active:
            # Prioritize Stage 2 evolutions
            stage2_evos = [a for a in evolve_active 
                         if card_def(ACTION_TABLE[a].a).subtype == "Stage2"]
            if stage2_evos:
                return random.choice(stage2_evos)
            return random.choice(evolve_active)
        
        # 4. Use abilities
        abilities = self._get_actions_by_kind(legal_actions, "USE_ACTIVE_ABILITY")
        if abilities:
            return random.choice(abilities)
        
        # 5. Play trainers (supporters and items)
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        if trainers:
            # Prioritize draw supporters
            for a in trainers:
                card = ACTION_TABLE[a].a
                if card in ("Hilda", "Dawn", "Lillie's Determination", "Iono", "Arven"):
                    return a
            return random.choice(trainers)
        
        # 6. Bench basics
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        if bench:
            return random.choice(bench)
        
        # 7. Attach energy to bench (if active doesn't need it)
        attach_bench = self._get_actions_by_kinds(legal_actions, 
            ["ATTACH_BENCH_0", "ATTACH_BENCH_1", "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"])
        if attach_bench:
            return random.choice(attach_bench)
        
        # 8. Evolve bench
        evolve_bench = self._get_actions_by_kinds(legal_actions,
            ["EVOLVE_BENCH_0", "EVOLVE_BENCH_1", "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"])
        if evolve_bench:
            return random.choice(evolve_bench)
        
        # Default: random non-pass
        return self._random_action(legal_actions)
    
    def _evolution_rush_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Evolution Rush strategy:
        1. Evolve (prioritize Stage 2 > Stage 1)
        2. Play Rare Candy if possible
        3. Bench evolution basics (Abra, Charmander, etc.)
        4. Use search/draw trainers
        5. Attack only when evolved
        6. Attach energy
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Evolve - HIGHEST priority
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        if all_evolves:
            # Prioritize Stage 2
            stage2 = [a for a in all_evolves 
                     if card_def(ACTION_TABLE[a].a).subtype == "Stage2"]
            if stage2:
                return random.choice(stage2)
            return random.choice(all_evolves)
        
        # 2. Play Rare Candy
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        rare_candy = [a for a in trainers if ACTION_TABLE[a].a == "Rare Candy"]
        if rare_candy:
            return random.choice(rare_candy)
        
        # 3. Bench evolution starters
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        evo_starters = [a for a in bench 
                       if ACTION_TABLE[a].a in ("Abra", "Charmander", "Pidgey", "Duskull", "Gimmighoul")]
        if evo_starters:
            return random.choice(evo_starters)
        
        # 4. Search/Draw trainers
        search_cards = [a for a in trainers 
                       if ACTION_TABLE[a].a in ("Hilda", "Dawn", "Arven", "Ultra Ball", 
                                                "Nest Ball", "Buddy-Buddy Poffin")]
        if search_cards:
            return random.choice(search_cards)
        
        # 5. Attack only if active is evolved
        if me.active.name:
            active_def = card_def(me.active.name)
            if active_def.subtype in ("Stage1", "Stage2"):
                attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
                if attacks:
                    return random.choice(attacks)
        
        # 6. Attach energy
        attach = self._get_actions_by_kinds(legal_actions, [
            "ATTACH_ACTIVE", "ATTACH_BENCH_0", "ATTACH_BENCH_1", 
            "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"
        ])
        if attach:
            return random.choice(attach)
        
        # 7. Any bench action
        if bench:
            return random.choice(bench)
        
        # Default
        return self._random_action(legal_actions)
    
    def _defensive_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Defensive strategy:
        1. Bench as many Pokemon as possible
        2. Evolve to increase HP
        3. Retreat damaged Pokemon
        4. Attach energy to bench
        5. Use draw supporters
        6. Attack only as last resort
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Bench basics - build the board
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        if bench:
            return random.choice(bench)
        
        # 2. Evolve for HP
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        if all_evolves:
            return random.choice(all_evolves)
        
        # 3. Retreat if damaged
        if me.active.name and me.active.damage > 0:
            retreat = self._get_actions_by_kind(legal_actions, "RETREAT_TO")
            if retreat:
                return random.choice(retreat)
        
        # 4. Attach energy to bench (save active for later)
        attach_bench = self._get_actions_by_kinds(legal_actions, 
            ["ATTACH_BENCH_0", "ATTACH_BENCH_1", "ATTACH_BENCH_2", 
             "ATTACH_BENCH_3", "ATTACH_BENCH_4"])
        if attach_bench:
            return random.choice(attach_bench)
        
        # 5. Draw supporters
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        draw_supporters = [a for a in trainers 
                          if ACTION_TABLE[a].a in ("Hilda", "Dawn", "Iono", 
                                                   "Lillie's Determination")]
        if draw_supporters:
            return random.choice(draw_supporters)
        
        # 6. Use abilities
        abilities = self._get_actions_by_kind(legal_actions, "USE_ACTIVE_ABILITY")
        if abilities:
            return random.choice(abilities)
        
        # 7. Attach to active if nothing else
        attach_active = self._get_actions_by_kind(legal_actions, "ATTACH_ACTIVE")
        if attach_active:
            return random.choice(attach_active)
        
        # 8. Attack as last resort
        attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
        if attacks:
            return random.choice(attacks)
        
        return self._random_action(legal_actions)
    
    def _energy_first_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Energy First strategy:
        1. Always attach energy if possible
        2. Use energy search trainers
        3. Build up for big attacks
        4. Attack when fully powered
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Attach energy - always
        attach = self._get_actions_by_kinds(legal_actions, [
            "ATTACH_ACTIVE", "ATTACH_BENCH_0", "ATTACH_BENCH_1", 
            "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"
        ])
        if attach:
            # Prefer active
            attach_active = self._get_actions_by_kind(legal_actions, "ATTACH_ACTIVE")
            if attach_active:
                return random.choice(attach_active)
            return random.choice(attach)
        
        # 2. Energy search trainers
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        energy_trainers = [a for a in trainers 
                         if ACTION_TABLE[a].a in ("Arven", "Energy Search", 
                                                  "Earthen Vessel", "Superior Energy Retrieval")]
        if energy_trainers:
            return random.choice(energy_trainers)
        
        # 3. Attack if active has enough energy
        if me.active.name:
            active_def = card_def(me.active.name)
            if active_def.attacks:
                cost = len(active_def.attacks[0].cost) if active_def.attacks[0].cost else 0
                if len(me.active.energy) >= cost:
                    attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
                    if attacks:
                        return random.choice(attacks)
        
        # 4. Evolve
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        if all_evolves:
            return random.choice(all_evolves)
        
        # 5. Bench
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        if bench:
            return random.choice(bench)
        
        # 6. Other trainers
        if trainers:
            return random.choice(trainers)
        
        return self._random_action(legal_actions)
    
    def _control_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Control strategy (Hand Disruption):
        1. Prioritize Iono, Unfair Stamp (disrupt opponent's hand)
        2. Use Boss's Orders to target key threats
        3. Slow evolution, focus on disruption
        4. Attack weak targets
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Hand disruption supporters
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        disruption = [a for a in trainers 
                     if ACTION_TABLE[a].a in ("Iono", "Unfair Stamp")]
        if disruption:
            return random.choice(disruption)
        
        # 2. Boss's Orders - target weak bench Pokemon
        boss = [a for a in trainers if ACTION_TABLE[a].a == "Boss's Orders"]
        if boss:
            return random.choice(boss)
        
        # 3. Bench basics
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        if bench:
            return random.choice(bench)
        
        # 4. Evolve
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        if all_evolves:
            return random.choice(all_evolves)
        
        # 5. Attach energy
        attach = self._get_actions_by_kinds(legal_actions, [
            "ATTACH_ACTIVE", "ATTACH_BENCH_0", "ATTACH_BENCH_1", 
            "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"
        ])
        if attach:
            return random.choice(attach)
        
        # 6. Attack
        attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
        if attacks:
            return random.choice(attacks)
        
        return self._random_action(legal_actions)
    
    def _combo_action(self, env, mask: np.ndarray, legal_actions: np.ndarray) -> int:
        """
        Combo strategy (Alakazam Mind Ruler):
        1. Build large hand (draw supporters)
        2. Evolve to Alakazam quickly (Rare Candy priority)
        3. Get Pidgeot ex for consistency
        4. Attack with Mind Ruler when hand is big (7+)
        """
        gs = env._gs
        me = gs.players[gs.turn_player]
        
        # 1. Draw supporters to build hand
        trainers = self._get_actions_by_kind(legal_actions, "PLAY_TRAINER")
        draw_cards = [a for a in trainers 
                     if ACTION_TABLE[a].a in ("Hilda", "Dawn", "Lillie's Determination")]
        
        # Only draw if hand is small
        if len(me.hand) < 6 and draw_cards:
            return random.choice(draw_cards)
        
        # 2. Rare Candy to Alakazam
        rare_candy = [a for a in trainers if ACTION_TABLE[a].a == "Rare Candy"]
        if rare_candy:
            return random.choice(rare_candy)
        
        # 3. Evolve (priority: Alakazam > Pidgeot)
        all_evolves = self._get_actions_by_kinds(legal_actions, [
            "EVOLVE_ACTIVE", "EVOLVE_BENCH_0", "EVOLVE_BENCH_1", 
            "EVOLVE_BENCH_2", "EVOLVE_BENCH_3", "EVOLVE_BENCH_4"
        ])
        alakazam_evolve = [a for a in all_evolves 
                         if ACTION_TABLE[a].a in ("Alakazam", "Alakazam ex")]
        if alakazam_evolve:
            return random.choice(alakazam_evolve)
        if all_evolves:
            return random.choice(all_evolves)
        
        # 4. Use Pidgeot ex ability
        abilities = self._get_actions_by_kind(legal_actions, "USE_ACTIVE_ABILITY")
        if abilities:
            return random.choice(abilities)
        
        # 5. Bench Abra/Pidgey
        bench = self._get_actions_by_kind(legal_actions, "PLAY_BASIC_TO_BENCH")
        evo_starters = [a for a in bench 
                       if ACTION_TABLE[a].a in ("Abra", "Pidgey")]
        if evo_starters:
            return random.choice(evo_starters)
        if bench:
            return random.choice(bench)
        
        # 6. Search trainers
        search = [a for a in trainers 
                 if ACTION_TABLE[a].a in ("Ultra Ball", "Nest Ball", "Arven")]
        if search:
            return random.choice(search)
        
        # 7. Attach energy
        attach = self._get_actions_by_kinds(legal_actions, [
            "ATTACH_ACTIVE", "ATTACH_BENCH_0", "ATTACH_BENCH_1", 
            "ATTACH_BENCH_2", "ATTACH_BENCH_3", "ATTACH_BENCH_4"
        ])
        if attach:
            return random.choice(attach)
        
        # 8. Attack if hand is big (Mind Ruler scales with hand size)
        if len(me.hand) >= 5:
            attacks = self._get_actions_by_kind(legal_actions, "ATTACK")
            if attacks:
                return random.choice(attacks)
        
        return self._random_action(legal_actions)


# Pre-built agents for easy access
SCRIPTED_AGENTS = {
    "aggressive": ScriptedAgent("aggressive"),
    "evolution_rush": ScriptedAgent("evolution_rush"),
    "defensive": ScriptedAgent("defensive"),
    "energy_first": ScriptedAgent("energy_first"),
    "control": ScriptedAgent("control"),
    "combo": ScriptedAgent("combo"),
    "random": ScriptedAgent("random"),
}


def get_scripted_agent(strategy: str = "aggressive") -> ScriptedAgent:
    """Get a scripted agent by strategy name."""
    if strategy in SCRIPTED_AGENTS:
        return SCRIPTED_AGENTS[strategy]
    return ScriptedAgent(strategy)
