#!/usr/bin/env python3
"""
Card Implementation Verification Tool

Compares cards.csv (official card text) against:
1. CARD_REGISTRY (card definitions)
2. effects.py (ability/attack/trainer implementations)

Usage:
    python verify_card_implementations.py              # Check all cards
    python verify_card_implementations.py CardName     # Check specific card
    python verify_card_implementations.py --missing    # Show only missing
    python verify_card_implementations.py --pokemon    # Pokemon only
    python verify_card_implementations.py --trainers   # Trainers only
"""

import re
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path


@dataclass
class CardCSV:
    """Parsed card from cards.csv"""
    name: str
    card_type: str  # Pokemon, Trainer, Energy
    subtype: str    # Basic, Stage1, Stage2, Item, Supporter, Stadium, Tool
    hp: int
    poke_type: str  # Fire, Water, etc.
    ability_name: Optional[str] = None
    ability_text: Optional[str] = None
    attacks: List[Tuple[str, str, str]] = None  # (name, cost, text)
    raw_text: str = ""
    
    def __post_init__(self):
        if self.attacks is None:
            self.attacks = []


def parse_cards_csv(filepath: str = "cards.csv") -> Dict[str, CardCSV]:
    """Parse the cards.csv file into structured card data."""
    cards = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by card headers (Name - Type - HP)
    # Pattern: "CardName - Type - XXX HP"
    card_pattern = r'^([^\n]+) - ([A-Za-z]+) - (\d+) HP'
    
    lines = content.split('\n')
    current_card = None
    current_lines = []
    
    for line in lines:
        match = re.match(card_pattern, line)
        if match:
            # Save previous card
            if current_card:
                cards[current_card.name] = current_card
                current_card.raw_text = '\n'.join(current_lines)
            
            name = match.group(1).strip()
            poke_type = match.group(2).strip()
            hp = int(match.group(3))
            
            current_card = CardCSV(
                name=name,
                card_type="Pokemon",
                subtype="Basic",
                hp=hp,
                poke_type=poke_type
            )
            current_lines = [line]
        elif current_card:
            current_lines.append(line)
            
            # Parse subtype
            if "Stage 1" in line:
                current_card.subtype = "Stage1"
            elif "Stage 2" in line:
                current_card.subtype = "Stage2"
            elif "Trainer" in line:
                current_card.card_type = "Trainer"
                if "Item" in line:
                    current_card.subtype = "Item"
                elif "Supporter" in line:
                    current_card.subtype = "Supporter"
                elif "Stadium" in line:
                    current_card.subtype = "Stadium"
                elif "Tool" in line:
                    current_card.subtype = "Tool"
            elif "Energy" in line and "Special" in line:
                current_card.card_type = "Energy"
                current_card.subtype = "Special"
            
            # Parse ability
            ability_match = re.match(r'^Ability:\s*(.+)', line)
            if ability_match:
                current_card.ability_name = ability_match.group(1).strip()
            
            # Parse ability text (quoted text after Ability line)
            if current_card.ability_name and line.startswith('"') and current_card.ability_text is None:
                current_card.ability_text = line.strip('"')
            
            # Parse attacks (format: "CC Attack Name 120")
            attack_match = re.match(r'^([RCFWLPMDNCY]+)\s+(.+?)\s+(\d+[×x+]?|\d*)\s*$', line)
            if attack_match:
                cost = attack_match.group(1)
                name = attack_match.group(2).strip()
                damage = attack_match.group(3) or "0"
                current_card.attacks.append((name, cost, damage))
    
    # Save last card
    if current_card:
        cards[current_card.name] = current_card
        current_card.raw_text = '\n'.join(current_lines)
    
    return cards


def get_registry_cards() -> Dict[str, dict]:
    """Get cards from CARD_REGISTRY."""
    try:
        from tcg.cards import CARD_REGISTRY, card_def
        registry = {}
        for name in CARD_REGISTRY:
            cd = card_def(name)
            registry[name] = {
                'name': name,
                'supertype': cd.supertype,
                'subtype': cd.subtype,
                'type': cd.type,
                'hp': cd.hp,
                'ability': cd.ability,
                'attacks': [(a.name, a.damage, a.cost) for a in cd.attacks] if cd.attacks else [],
                'has_rule_box': cd.has_rule_box,
            }
        return registry
    except Exception as e:
        print(f"Error loading CARD_REGISTRY: {e}")
        return {}


def get_effects_implementations() -> Dict[str, List[Tuple[str, int, str]]]:
    """Scan effects.py and env.py for card implementations."""
    implementations = {}
    
    files_to_scan = ['tcg/effects.py', 'tcg/env.py']
    
    for filepath in files_to_scan:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except:
            continue
        
        # Patterns to find card implementations (use double quotes to handle apostrophes in names)
        patterns = [
            (r'card_name\s*==\s*"([^"]+)"', 'trainer'),
            (r'pokemon_name\s*==\s*"([^"]+)"', 'pokemon'),
            (r'energy_name\s*==\s*"([^"]+)"', 'energy'),
            # Check for passive abilities and tools
            (r'\.name\s*==\s*"([^"]+)"', 'passive'),
            (r'slot\.name\s*==\s*"([^"]+)"', 'passive'),
            (r'\.tool\s*==\s*"([^"]+)"', 'tool'),
            (r'act\.a\s*==\s*"([^"]+)"', 'action'),
        ]
        
        # Special handling for "card_name in (...)" patterns
        in_pattern = r'card_name\s+in\s*\(([^)]+)\)'
        
        current_function = None
        
        for i, line in enumerate(lines):
            # Track current function
            func_match = re.match(r'^def\s+(\w+)', line)
            if func_match:
                current_function = func_match.group(1)
            
            # Find card references
            for pattern, card_type in patterns:
                matches = re.findall(pattern, line)
                for card_name in matches:
                    if card_name not in implementations:
                        implementations[card_name] = []
                    implementations[card_name].append((current_function, i + 1, line.strip()))
            
            # Special handling for "card_name in (...)" - extract all names
            in_matches = re.findall(in_pattern, line)
            for tuple_content in in_matches:
                # Extract all quoted strings from the tuple (double quotes only for apostrophe handling)
                names = re.findall(r'"([^"]+)"', tuple_content)
                for card_name in names:
                    if card_name not in implementations:
                        implementations[card_name] = []
                    implementations[card_name].append((current_function, i + 1, line.strip()))
    
    return implementations


def verify_card(card_name: str, csv_cards: Dict, registry: Dict, implementations: Dict) -> dict:
    """Verify a single card's implementation."""
    result = {
        'name': card_name,
        'in_csv': card_name in csv_cards,
        'in_registry': card_name in registry,
        'has_implementation': card_name in implementations,
        'issues': [],
        'warnings': [],
        'csv_data': csv_cards.get(card_name),
        'registry_data': registry.get(card_name),
        'implementation_locations': implementations.get(card_name, []),
    }
    
    csv_card = csv_cards.get(card_name)
    reg_card = registry.get(card_name)
    impl_locs = implementations.get(card_name, [])
    
    # Check 1: In CSV but not in registry
    if csv_card and not reg_card:
        result['issues'].append("❌ In CSV but NOT in CARD_REGISTRY")
    
    # Check 2: Has ability in CSV but no implementation
    if csv_card and csv_card.ability_name:
        if not impl_locs:
            result['issues'].append(f"❌ Ability '{csv_card.ability_name}' has no implementation")
        elif reg_card and not reg_card.get('ability'):
            result['issues'].append(f"❌ Ability '{csv_card.ability_name}' not registered in CardDef")
    
    # Check 3: Has special attacks but no implementation
    if csv_card and csv_card.attacks:
        for atk_name, cost, damage in csv_card.attacks:
            # Check if attack needs special handling (x, +, or 0 damage)
            if 'x' in damage.lower() or '+' in damage or damage == '' or damage == '0':
                if not impl_locs:
                    result['warnings'].append(f"⚠️ Attack '{atk_name}' ({damage}) may need implementation")
    
    # Check 4: Registry vs CSV consistency
    if csv_card and reg_card:
        # HP mismatch
        if csv_card.hp != reg_card.get('hp', 0):
            result['warnings'].append(f"⚠️ HP mismatch: CSV={csv_card.hp}, Registry={reg_card.get('hp')}")
        
        # Type mismatch
        csv_type = csv_card.poke_type
        reg_type = reg_card.get('type', '')
        if csv_type.lower() != reg_type.lower():
            result['warnings'].append(f"⚠️ Type mismatch: CSV={csv_type}, Registry={reg_type}")
    
    # Check 5: Trainer cards need implementations (only if actually a Trainer)
    # Prioritize registry data over CSV since CSV parsing may have errors
    is_trainer = False
    if reg_card and reg_card.get('supertype') == "Trainer":
        is_trainer = True
    elif not reg_card and csv_card and csv_card.card_type == "Trainer":
        is_trainer = True
    
    if is_trainer and not impl_locs:
        result['issues'].append("❌ Trainer card has no effect implementation")
    
    # Status
    if not result['issues']:
        if result['warnings']:
            result['status'] = '⚠️ WARNINGS'
        elif result['in_registry'] and result['has_implementation']:
            result['status'] = '✅ OK'
        elif result['in_registry']:
            result['status'] = '🔶 PASSIVE'  # No special effect needed
        else:
            result['status'] = '❓ UNKNOWN'
    else:
        result['status'] = '❌ ISSUES'
    
    return result


def print_card_report(result: dict, verbose: bool = False):
    """Print verification result for a card."""
    name = result['name']
    status = result['status']
    
    print(f"\n{status} {name}")
    
    if result['csv_data']:
        csv = result['csv_data']
        print(f"   Type: {csv.card_type}/{csv.subtype} | {csv.poke_type} | {csv.hp}HP")
        if csv.ability_name:
            print(f"   Ability: {csv.ability_name}")
        if csv.attacks:
            attacks_str = ", ".join([f"{a[0]}({a[2]})" for a in csv.attacks])
            print(f"   Attacks: {attacks_str}")
    
    if result['implementation_locations'] and verbose:
        print(f"   Implementations:")
        for func, line, code in result['implementation_locations'][:3]:
            print(f"      L{line} in {func}: {code[:60]}...")
    
    for issue in result['issues']:
        print(f"   {issue}")
    
    for warning in result['warnings']:
        print(f"   {warning}")


def main():
    parser = argparse.ArgumentParser(description="Verify card implementations against cards.csv")
    parser.add_argument('card', nargs='?', help="Specific card to check")
    parser.add_argument('--missing', action='store_true', help="Show only cards with issues")
    parser.add_argument('--pokemon', action='store_true', help="Pokemon only")
    parser.add_argument('--trainers', action='store_true', help="Trainers only")
    parser.add_argument('--energy', action='store_true', help="Energy only")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    parser.add_argument('--summary', action='store_true', help="Summary only")
    args = parser.parse_args()
    
    print("=" * 80)
    print("CARD IMPLEMENTATION VERIFICATION")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    csv_cards = parse_cards_csv()
    print(f"   Cards in CSV: {len(csv_cards)}")
    
    registry = get_registry_cards()
    print(f"   Cards in Registry: {len(registry)}")
    
    implementations = get_effects_implementations()
    print(f"   Cards with effects: {len(implementations)}")
    
    # Determine which cards to check
    if args.card:
        cards_to_check = [args.card]
    else:
        # Combine all known cards
        all_cards = set(csv_cards.keys()) | set(registry.keys())
        cards_to_check = sorted(all_cards)
    
    # Filter by type
    if args.pokemon:
        cards_to_check = [c for c in cards_to_check if csv_cards.get(c, CardCSV("", "Pokemon", "", 0, "")).card_type == "Pokemon"]
    elif args.trainers:
        cards_to_check = [c for c in cards_to_check if csv_cards.get(c, CardCSV("", "Trainer", "", 0, "")).card_type == "Trainer"]
    elif args.energy:
        cards_to_check = [c for c in cards_to_check if csv_cards.get(c, CardCSV("", "Energy", "", 0, "")).card_type == "Energy"]
    
    # Verify each card
    results = []
    for card in cards_to_check:
        result = verify_card(card, csv_cards, registry, implementations)
        results.append(result)
        
        if args.missing and result['status'] in ['✅ OK', '🔶 PASSIVE']:
            continue
        
        if not args.summary:
            print_card_report(result, args.verbose)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    ok_count = sum(1 for r in results if r['status'] == '✅ OK')
    passive_count = sum(1 for r in results if r['status'] == '🔶 PASSIVE')
    warning_count = sum(1 for r in results if r['status'] == '⚠️ WARNINGS')
    issue_count = sum(1 for r in results if r['status'] == '❌ ISSUES')
    unknown_count = sum(1 for r in results if r['status'] == '❓ UNKNOWN')
    
    print(f"\n✅ OK:        {ok_count}")
    print(f"🔶 PASSIVE:   {passive_count}")
    print(f"⚠️ WARNINGS:  {warning_count}")
    print(f"❌ ISSUES:    {issue_count}")
    print(f"❓ UNKNOWN:   {unknown_count}")
    print(f"\nTotal: {len(results)}")
    
    # List issues
    if issue_count > 0:
        print("\n❌ Cards with Issues:")
        for r in results:
            if r['status'] == '❌ ISSUES':
                print(f"   - {r['name']}: {', '.join(r['issues'])}")
    
    print()
    return 0 if issue_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
