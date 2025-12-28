"""
Helper functions for intelligent card selection in trainer effects.
Enables full control over which specific cards to select from deck/discard.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Optional, Callable
import itertools

if TYPE_CHECKING:
    from tcg.cards import CardDef
    from tcg.state import PlayerState

from tcg.cards import card_def


def get_selection_candidates(
    pile: List[str], 
    filter_func: Callable[[CardDef], bool],
    max_candidates: int = 15,
    prioritize_recent: bool = True
) -> List[int]:
    """
    Get indices of valid candidates from a pile (deck or discard).
    
    Args:
        pile: List of card names
        filter_func: Function that returns True if card is valid
        max_candidates: Maximum number of candidates to return (for pruning)
        prioritize_recent: If True and pruning needed, keep most recent cards
    
    Returns:
        List of indices into pile that match the filter
    """
    candidates = [i for i, c in enumerate(pile) if filter_func(card_def(c))]
    
    if len(candidates) <= max_candidates:
        return candidates
    
    # Prune to max_candidates
    if prioritize_recent:
        # Keep most recent (highest indices in discard pile)
        return candidates[-max_candidates:]
    else:
        # Keep oldest (lowest indices)
        return candidates[:max_candidates]


def score_card_priority(
    card_name: str,
    player: PlayerState,
    context: str = "general"
) -> int:
    """
    Score a card's priority for recovery/search.
    Higher score = more important to get.
    
    Args:
        card_name: Name of the card to score
        player: Player state for context
        context: Type of effect ('recovery', 'search', 'general')
    
    Returns:
        Priority score (0-1000)
    """
    cd = card_def(card_name)
    score = 0
    
    # Pokemon scoring
    if cd.supertype == "Pokemon":
        # Evolution pieces for in-play Pokemon get highest priority
        if cd.evolves_from:
            # Check if we have the pre-evolution in play
            all_in_play = [player.active.name] + [s.name for s in player.bench if s.name]
            if cd.evolves_from in all_in_play:
                score += 500  # Very high priority
        
        # Key attackers
        key_attackers = ["Alakazam", "Charizard ex", "Pidgeot ex", "Gholdengo ex", 
                        "Fezandipiti ex", "Genesect ex"]
        if card_name in key_attackers:
            score += 300
        
        # Support Pokemon
        support_mons = ["Fan Rotom", "Dunsparce", "Tatsugiri", "Lunatone", "Solrock"]
        if card_name in support_mons:
            score += 200
        
        # Basic Pokemon (good for bench development)
        if cd.subtype == "Basic":
            score += 100
    
    # Energy scoring
    elif cd.supertype == "Energy":
        # Count total energy on board
        total_energy = sum(len(player.active.energy), 
                          sum(len(s.energy) for s in player.bench if s.name))
        
        # If low on energy, prioritize it
        if total_energy < 5:
            score += 250
        elif total_energy < 10:
            score += 150
        else:
            score += 50
        
        # Special energy slightly more valuable
        if cd.subtype == "Special":
            score += 20
    
    # Trainer scoring
    elif cd.supertype == "Trainer":
        # Search cards
        search_cards = ["Ultra Ball", "Nest Ball", "Rare Candy", "Buddy-Buddy Poffin"]
        if card_name in search_cards:
            score += 200
        
        # Draw supporters
        draw_supporters = ["Professor's Research", "Lillie's Determination", "Dawn"]
        if card_name in draw_supporters:
            score += 180
        
        # Recovery cards
        recovery_cards = ["Lana's Aid", "Super Rod", "Night Stretcher"]
        if card_name in recovery_cards:
            score += 150
    
    return score


def get_prioritized_candidates(
    pile: List[str],
    filter_func: Callable[[CardDef], bool],
    player: PlayerState,
    max_candidates: int = 15
) -> List[int]:
    """
    Get candidates sorted by priority score.
    
    Returns:
        List of indices, sorted by descending priority
    """
    # Get all valid candidates
    all_candidates = [(i, c) for i, c in enumerate(pile) if filter_func(card_def(c))]
    
    # Score each
    scored = [(score_card_priority(c, player), i, c) for i, c in all_candidates]
    
    # Sort by score (descending)
    scored.sort(reverse=True)
    
    # Return top max_candidates indices
    return [i for _, i, _ in scored[:max_candidates]]


def generate_selection_combinations(
    candidates: List[int],
    min_select: int = 1,
    max_select: int = 3,
    max_combinations: int = 100
) -> List[Tuple[Optional[int], Optional[int], Optional[int]]]:
    """
    Generate combinations of card selections.
    
    Args:
        candidates: List of valid indices to select from
        min_select: Minimum number to select
        max_select: Maximum number to select (1-3 supported)
        max_combinations: Maximum combinations to generate (for action space control)
    
    Returns:
        List of tuples representing selections (c, d, e)
        Each tuple is padded with None to length 3
    """
    if max_select > 3:
        raise ValueError("max_select > 3 not supported yet (expand Action fields)")
    
    combos = []
    
    # Generate selections for each count
    for num_select in range(min_select, min(max_select, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, num_select):
            # Pad with None to reach 3 elements (for c, d, e fields)
            padded = combo + (None,) * (3 - len(combo))
            combos.append(padded)
            
            # Stop if we hit max_combinations
            if len(combos) >= max_combinations:
                return combos
    
    # Include "select nothing" option if allowed
    if min_select == 0:
        combos.append((None, None, None))
    
    return combos


def parse_selections(
    selection_indices: List[Optional[int]],
    valid_candidates: List[int]
) -> List[int]:
    """
    Parse and validate selection indices.
    
    Args:
        selection_indices: List of indices from action (c, d, e fields)
        valid_candidates: List of valid indices that can be selected
    
    Returns:
        List of unique, valid selected indices
    """
    selected = []
    for idx in selection_indices:
        if idx is not None and idx in valid_candidates and idx not in selected:
            selected.append(idx)
    return selected


def validate_selection_action(
    pile: List[str],
    selections: Tuple[Optional[int], ...],
    filter_func: Callable[[CardDef], bool]
) -> bool:
    """
    Validate that a selection action is legal.
    
    Returns:
        True if all selections are valid, unique, and match filter
    """
    seen = set()
    for idx in selections:
        if idx is None:
            continue
        
        # Check bounds
        if idx < 0 or idx >= len(pile):
            return False
        
        # Check uniqueness
        if idx in seen:
            return False
        seen.add(idx)
        
        # Check filter
        if not filter_func(card_def(pile[idx])):
            return False
    
    return True


# Commonly used filters
def filter_basic_energy(cd: CardDef) -> bool:
    """Filter for basic energy cards"""
    return cd.supertype == "Energy" and cd.subtype == "Basic"


def filter_pokemon(cd: CardDef) -> bool:
    """Filter for any Pokemon"""
    return cd.supertype == "Pokemon"


def filter_non_rule_box_pokemon(cd: CardDef) -> bool:
    """Filter for Pokemon without Rule Boxes"""
    return cd.supertype == "Pokemon" and not cd.has_rule_box()


def filter_basic_pokemon(cd: CardDef) -> bool:
    """Filter for Basic Pokemon"""
    return cd.supertype == "Pokemon" and cd.subtype == "Basic"


def filter_evolution_pokemon(cd: CardDef) -> bool:
    """Filter for Evolution Pokemon"""
    return cd.supertype == "Pokemon" and cd.subtype in ("Stage1", "Stage2")


def filter_pokemon_or_basic_energy(cd: CardDef) -> bool:
    """Filter for Pokemon or Basic Energy (for Super Rod, etc.)"""
    return cd.supertype == "Pokemon" or (cd.supertype == "Energy" and cd.subtype == "Basic")


def filter_lanas_aid_targets(cd: CardDef) -> bool:
    """Filter for Lana's Aid targets (non-Rule Box Pokemon or Basic Energy)"""
    return (cd.supertype == "Energy" and cd.subtype == "Basic") or \
           (cd.supertype == "Pokemon" and not cd.has_rule_box())
