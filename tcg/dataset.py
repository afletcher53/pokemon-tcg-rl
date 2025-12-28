# tcg/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from tcg.actions import ACTION_INDEX, ACTION_TABLE, Action
from tcg.parse_log import Event, parse_log
from tcg.state import GameState, featurize


@dataclass
class Transition:
    obs: np.ndarray
    act_idx: int


def _match_action_from_text(text: str) -> Optional[int]:
    """
    Heuristic mapping from log line -> Action index.
    Improve this over time. Start simple: PASS, PLAY_TRAINER, PLAY_BASIC_TO_BENCH, ATTACH_ACTIVE/BENCH, EVOLVE.
    """
    # PASS / END
    if "ended their turn" in text:
        return ACTION_INDEX[Action("PASS")]

    # PLAY something
    if " played " in text:
        # "X played Arven."
        # "X played Duskull to the Bench."
        # "X played Pidgey to the Active Spot." (ignore Active for now if no action exists)
        
        parts = text.split(" played ", 1)
        if len(parts) == 2:
            remainder = parts[1].strip().rstrip(".")
            
            # 1. Play to Bench
            if " to the Bench" in remainder:
                card = remainder.split(" to the Bench")[0].strip()
                # Default to slot 0
                a = Action("PLAY_BASIC_TO_BENCH", a=card, b=0)
                if a in ACTION_INDEX:
                    return ACTION_INDEX[a]
                # Fallback: maybe it's something else?
            
            # 2. Play Trainer (default case if no specific location, or location stripped?)
            # Usually Trainers are just "played Arven."
            # Stadiums: "played Artazon to the Stadium spot."
            if " to the Stadium spot" in remainder:
                card = remainder.split(" to the Stadium spot")[0].strip()
                a = Action("PLAY_TRAINER", a=card)
                if a in ACTION_INDEX:
                    return ACTION_INDEX[a]

            # Standard Trainer or fallback
            # If we failed above, try treating whole thing as card name (e.g. "Arven")
            a_trainer = Action("PLAY_TRAINER", a=remainder)
            if a_trainer in ACTION_INDEX:
                return ACTION_INDEX[a_trainer]
                
        return None

    # ATTACH energy
    if " attached " in text and " to " in text:
        # "X attached Basic Fire Energy to Tatsugiri in the Active Spot."
        # We'll treat as ATTACH_ACTIVE if "Active", else bench.
        try:
            after = text.split(" attached ", 1)[1]
            energy = after.split(" to ", 1)[0].strip()
            if "Active" in text:
                a = Action("ATTACH_ACTIVE", a=energy)
            else:
                # unknown bench target -> map to generic attach active if can't parse
                a = Action("ATTACH_ACTIVE", a=energy)
            if a in ACTION_INDEX:
                return ACTION_INDEX[a]

            # Fallback: attaching a Tool is a PLAY_TRAINER action
            a_tool = Action("PLAY_TRAINER", a=energy)
            if a_tool in ACTION_INDEX:
                return ACTION_INDEX[a_tool]
        except Exception:
            return None

    # EVOLVE
    if " evolved " in text and " to " in text:
        # "X evolved Charmander to Charmeleon on the Bench."
        try:
            evo = text.split(" to ", 1)[1].split(" on ", 1)[0].strip()
            # default to EVOLVE_BENCH slot 0 (toy). Improve by parsing which.
            a = Action("EVOLVE_BENCH", a=evo, b=0)
            if a in ACTION_INDEX:
                return ACTION_INDEX[a]
            a2 = Action("EVOLVE_ACTIVE", a=evo)
            if a2 in ACTION_INDEX:
                return ACTION_INDEX[a2]
        except Exception:
            return None

    # RETREAT
    if "retreated" in text:
        # map to retreat bench 0 (toy)
        a = Action("RETREAT_TO", b=0)
        if a in ACTION_INDEX:
            return ACTION_INDEX[a]

    # USE ability/attack
    if " used " in text:
        # Attack vs ability is hard; map to USE_ACTIVE_ABILITY unless " on " or " for " indicates attack
        if " on " in text or " for " in text:
            return ACTION_INDEX[Action("ATTACK")]
        return ACTION_INDEX[Action("USE_ACTIVE_ABILITY")]

    return None


def build_bc_dataset(raw_log: str, agent_player_name: str) -> List[Transition]:
    events = parse_log(raw_log)

    gs = GameState()
    # Initialize dummy state to match standard game start for feature consistnecy
    for p in gs.players:
        p.deck = ["Unknown"] * 60
        p.prizes = ["Unknown"] * 6
        p.hand = ["Unknown"] * 7
        
    data: List[Transition] = []

    # We only learn from agent's events (by player name)
    for ev in events:
        if ev.kind == "TURN_START":
            continue

        if ev.player != agent_player_name:
            continue

        act_idx = _match_action_from_text(ev.text)
        if act_idx is None:
            continue

        obs = featurize(gs).astype(np.float32)
        data.append(Transition(obs=obs, act_idx=act_idx))

        # NOTE: We are NOT replaying the log into gs here.
        # For a first pass BC, this still learns card/phase priors.
        # Once your engine exists, you will "replay" and update gs precisely.
    return data
