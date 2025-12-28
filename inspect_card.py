#!/usr/bin/env python3
"""
Card Inspector Tool

Shows detailed implementation information for any card, including:
- Card text from cards.csv
- Implementation code from effects.py
- Edge cases and special interactions
- Action space mapping

Usage: python inspect_card.py "Card Name"
       python inspect_card.py --list  # List all implemented cards
"""

import sys
import re
from tcg.cards import CARD_REGISTRY, card_def

# Color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'


def get_card_text_from_csv(card_name):
    """Extract card text from cards.csv."""
    try:
        with open('cards.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the card
        card_data = []
        capturing = False
        
        for line in lines:
            line = line.strip()
            
            # Check if this is the card header
            if line.startswith(card_name + ' - '):
                capturing = True
                card_data.append(line)
                continue
            
            # Stop when we hit another card or empty line after capturing
            if capturing:
                if line and line[0].isupper() and ' - ' in line and 'HP' in line:
                    # New card started
                    break
                if line:
                    card_data.append(line)
        
        return '\n'.join(card_data) if card_data else None
    except FileNotFoundError:
        return None


def extract_implementation(card_name, supertype):
    """Extract implementation code from effects.py."""
    try:
        with open('tcg/effects.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        implementations = []
        
        # Search patterns based on card type
        if supertype == "Trainer":
            patterns = [
                (f'card_name == "{card_name}"', 'apply_trainer_effect'),
                (f'if act.a == "{card_name}"', 'action masking (env.py)'),
            ]
        elif supertype == "Pokemon":
            patterns = [
                (f'pokemon_name == "{card_name}"', 'apply_ability_effect'),
                (f'pokemon_name == "{card_name}"', 'apply_attack_effect'),
            ]
        elif supertype == "Energy":
            patterns = [
                (f'== "{card_name}"', 'energy effect'),
            ]
        else:
            patterns = [(f'"{card_name}"', 'general')]
        
        for pattern, context in patterns:
            i = 0
            while i < len(lines):
                line = lines[i]
                
                if pattern in line:
                    # Found a match, extract the block
                    start = i
                    indent_level = len(line) - len(line.lstrip())
                    
                    # Extract the full block
                    code_block = [line.rstrip()]
                    i += 1
                    
                    while i < len(lines):
                        next_line = lines[i]
                        next_indent = len(next_line) - len(next_line.lstrip())
                        
                        # Stop if we hit same or lower indentation level
                        if next_line.strip() and next_indent <= indent_level:
                            break
                        
                        code_block.append(next_line.rstrip())
                        i += 1
                    
                    implementations.append({
                        'context': context,
                        'line_start': start + 1,
                        'code': '\n'.join(code_block)
                    })
                else:
                    i += 1
        
        return implementations
    except FileNotFoundError:
        return []


def find_edge_cases(card_name, code_blocks):
    """Identify edge cases and special handling in the implementation."""
    edge_cases = []
    
    for block in code_blocks:
        code = block['code']
        
        # Look for common edge case patterns
        patterns = {
            'Psyduck.*Damp': '🚫 Blocked by Psyduck\'s Damp ability',
            'Shaymin.*Flower Curtain': '🛡️ Protected by Shaymin\'s Flower Curtain',
            'Battle Cage': '🏟️ Interaction with Battle Cage stadium',
            'Mist Energy': '💨 Blocked by Mist Energy protection',
            'ignore_weakness_resistance': '⚔️ Bypasses Weakness/Resistance',
            'damage_reduction': '🛡️ Damage reduction mechanic',
            'if not': '⚠️ Conditional requirement',
            'continue': '❌ Validation/masking logic',
            'fallback': '🔄 Fallback behavior',
            'heuristic': '🧠 Heuristic decision',
            'smart': '🎯 Intelligent selection',
        }
        
        for pattern, description in patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                edge_cases.append(description)
    
    return list(set(edge_cases))  # Remove duplicates


def get_action_space_info(card_name, supertype):
    """Get information about how the card uses the action space."""
    try:
        with open('tcg/actions.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if card has custom action generation
        if f'"{card_name}"' in content:
            return f"✅ Custom action generation (see build_action_table in tcg/actions.py)"
        
        # Check common patterns
        if supertype == "Trainer":
            return "📋 Uses standard PLAY_TRAINER action with targeting"
        elif supertype == "Pokemon":
            return "🎮 Uses ATTACK or USE_ACTIVE_ABILITY actions"
        else:
            return "⚡ Passive effect (automatically applied)"
    except:
        return "❓ Unknown action mapping"


def inspect_card(card_name):
    """Show detailed information about a card's implementation."""
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{CYAN}📋 CARD INSPECTOR: {card_name}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    # Check if card exists in registry
    if card_name not in CARD_REGISTRY:
        print(f"{RED}❌ Card '{card_name}' not found in CARD_REGISTRY{RESET}\n")
        return
    
    # Get card definition
    card = card_def(card_name)
    
    # 1. Basic Info
    print(f"{YELLOW}📊 CARD DEFINITION:{RESET}")
    print(f"   Supertype: {card.supertype}")
    if hasattr(card, 'subtype'):
        print(f"   Subtype: {card.subtype}")
    if hasattr(card, 'type'):
        print(f"   Type: {card.type}")
    if hasattr(card, 'hp'):
        print(f"   HP: {card.hp}")
    if hasattr(card, 'attacks') and card.attacks:
        print(f"   Attacks: {len(card.attacks)}")
        for i, atk in enumerate(card.attacks):
            print(f"      [{i}] {atk.name}: {atk.damage} damage, Cost: {atk.cost}")
    if hasattr(card, 'ability') and card.ability:
        print(f"   Ability: {card.ability}")
    print()
    
    # 2. Card Text from CSV
    print(f"{YELLOW}📖 CARD TEXT (from cards.csv):{RESET}")
    card_text = get_card_text_from_csv(card_name)
    if card_text:
        for line in card_text.split('\n'):
            print(f"   {line}")
    else:
        print(f"   {RED}(Not found in cards.csv){RESET}")
    print()
    
    # 3. Implementation Code
    print(f"{YELLOW}💻 IMPLEMENTATION CODE:{RESET}")
    implementations = extract_implementation(card_name, card.supertype)
    
    if implementations:
        for impl in implementations:
            print(f"\n   {CYAN}[{impl['context']}] (Line {impl['line_start']}){RESET}")
            print(f"   {MAGENTA}{'─'*74}{RESET}")
            for line in impl['code'].split('\n'):
                print(f"   {line}")
            print(f"   {MAGENTA}{'─'*74}{RESET}")
    else:
        print(f"   {YELLOW}⚠️ No explicit implementation found{RESET}")
        print(f"   {YELLOW}   (Card may use default behavior or passive effect){RESET}")
    print()
    
    # 4. Edge Cases
    edge_cases = find_edge_cases(card_name, implementations)
    if edge_cases:
        print(f"{YELLOW}⚡ EDGE CASES & SPECIAL HANDLING:{RESET}")
        for case in edge_cases:
            print(f"   {case}")
        print()
    
    # 5. Action Space Info
    print(f"{YELLOW}🎮 ACTION SPACE:{RESET}")
    action_info = get_action_space_info(card_name, card.supertype)
    print(f"   {action_info}")
    print()
    
    # 6. Related Cards (interactions)
    print(f"{YELLOW}🔗 KNOWN INTERACTIONS:{RESET}")
    interactions = []
    
    # Check for common interactions in implementation
    all_code = '\n'.join([impl['code'] for impl in implementations])
    
    if 'Psyduck' in all_code:
        interactions.append("Psyduck (Damp ability blocks self-KO)")
    if 'Shaymin' in all_code:
        interactions.append("Shaymin (Flower Curtain protects bench)")
    if 'Battle Cage' in all_code:
        interactions.append("Battle Cage (stadium interaction)")
    if 'Mist Energy' in all_code:
        interactions.append("Mist Energy (effect protection)")
    if 'has_rule_box' in all_code:
        interactions.append("Rule Box Pokémon (special targeting)")
    
    if interactions:
        for interaction in interactions:
            print(f"   • {interaction}")
    else:
        print(f"   {YELLOW}(No special interactions detected){RESET}")
    print()
    
    print(f"{BLUE}{'='*80}{RESET}\n")


def list_all_cards():
    """List all cards in CARD_REGISTRY grouped by type."""
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{CYAN}📚 ALL IMPLEMENTED CARDS{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    # Group by supertype
    by_type = {}
    for name in sorted(CARD_REGISTRY.keys()):
        card = card_def(name)
        supertype = card.supertype
        if supertype not in by_type:
            by_type[supertype] = []
        by_type[supertype].append(name)
    
    # Display
    for supertype in sorted(by_type.keys()):
        cards = by_type[supertype]
        print(f"{YELLOW}{supertype} ({len(cards)}):{RESET}")
        
        for card_name in cards:
            card = card_def(card_name)
            
            # Add type/subtype info
            info = []
            if hasattr(card, 'subtype'):
                info.append(card.subtype)
            if hasattr(card, 'type'):
                info.append(card.type)
            if hasattr(card, 'ability') and card.ability:
                info.append(f"Ability: {card.ability}")
            
            info_str = f" [{', '.join(info)}]" if info else ""
            print(f"   • {card_name}{info_str}")
        
        print()
    
    total = sum(len(cards) for cards in by_type.values())
    print(f"{GREEN}Total: {total} cards{RESET}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspect_card.py \"Card Name\"")
        print("  python inspect_card.py --list")
        print()
        print("Examples:")
        print("  python inspect_card.py \"Ultra Ball\"")
        print("  python inspect_card.py \"Alakazam\"")
        print("  python inspect_card.py \"Genesect ex\"")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == '--list':
        list_all_cards()
    else:
        inspect_card(arg)


if __name__ == "__main__":
    main()
