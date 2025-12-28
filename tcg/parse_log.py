from dataclasses import dataclass
from typing import List
import re

@dataclass
class Event:
    kind: str  # "TURN_START" or "ACTION"
    player: str
    text: str

def parse_log(raw_log: str) -> List[Event]:
    events = []
    
    # Identify turns
    # Pattern: "Turn # <digit> - <Name>'s Turn"
    # Note: Example log shows "Turn # 1 - af109885472's Turn"
    turn_pattern = re.compile(r"(Turn # (\d+) - (.+?)'s Turn)")
    matches = list(turn_pattern.finditer(raw_log))
    
    # Handle Setup Phase (before first turn)
    if matches:
        setup_chunk = raw_log[:matches[0].start()].strip()
        if setup_chunk:
            events.extend(_parse_chunk(setup_chunk, default_player="Unknown"))
            
        for i, match in enumerate(matches):
            turn_number = match.group(2)
            player_name = match.group(3)
            
            # Emit TURN_START event
            events.append(Event(kind="TURN_START", player=player_name, text=match.group(0)))
            
            start_pos = match.end()
            end_pos = matches[i+1].start() if i+1 < len(matches) else len(raw_log)
            
            chunk = raw_log[start_pos:end_pos].strip()
            events.extend(_parse_chunk(chunk, default_player=player_name))
            
    else:
        # If no turns found, treat whole log as one block (e.g. only setup or flat log)
        events.extend(_parse_chunk(raw_log, default_player="Unknown"))
        
    return events

def _parse_chunk(text: str, default_player: str) -> List[Event]:
    events = []
    # Replace newlines with spaces to handle wrapped lines, assuming period-separation is primary
    cleaned = text.replace('\n', ' ')
    
    # Split by period followed by space
    parts = re.split(r'\.\s+', cleaned)
    
    for p in parts:
        p = p.strip()
        if not p:
            continue
            
        # Determine player for this action
        assigned_player = default_player
        
        words = p.split()
        if words:
            candidate = words[0]
            # Heuristic: if first word is not a symbol, treat it as player name.
            # This covers "Rudyodinson played..." -> "Rudyodinson"
            # And "Setup Rudyodinson..." -> "Setup" ? 
            # If "Setup", we probably want to look at second word, or let it fail if user filters by name.
            # But the filter in dataset.py is strict.
            
            if candidate == "Setup" and len(words) > 1:
                candidate = words[1]
                
            if candidate not in ["-", "•"] and not candidate.startswith("Turn"):
                 assigned_player = candidate
        
        events.append(Event(kind="ACTION", player=assigned_player, text=p))
        
    return events
