#!/usr/bin/env python3
"""
Comprehensive card verification script.
Checks all cards from cards.csv against implementations in cards.py and effects.py
"""

import csv
import re

def read_cards_csv():
    """Read and parse cards.csv"""
    cards = []
    with open('cards.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cards.append(row)
    return cards

def check_card_registry():
    """Check which cards are in CARD_REGISTRY"""
    with open('tcg/cards.py', 'r') as f:
        content = f.read()
    
    # Find all CardDef entries
    pattern = r'"([^"]+)":\s*CardDef\('
    matches = re.findall(pattern, content)
    return set(matches)

def check_effects_implementation():
    """Check which cards have effect implementations"""
    with open('tcg/effects.py', 'r') as f:
        content = f.read()
    
    # Find card mentions in effects.py
    implemented = set()
    
    # Patterns for different types
    patterns = [
        r'card_name\s*==\s*"([^"]+)"',  # Trainer effects
        r'pokemon_name\s*==\s*"([^"]+)"',  # Ability/attack effects
        r'if\s+c\s*==\s*"([^"]+)"',  # Card name checks
        r'me\.active\.tool\s*==\s*"([^"]+)"',  # Tool checks
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        implemented.update(matches)
    
    return implemented

def main():
    print("=" * 80)
    print("COMPREHENSIVE CARD VERIFICATION")
    print("=" * 80)
    print()
    
    # Read cards.csv
    csv_cards = read_cards_csv()
    print(f"📊 Total cards in cards.csv: {len(csv_cards)}")
    
    # Get registry
    registry = check_card_registry()
    print(f"📋 Total cards in CARD_REGISTRY: {len(registry)}")
    
    # Get effects
    effects = check_effects_implementation()
    print(f"⚡ Total cards with effects: {len(effects)}")
    print()
    
    # Categorize cards from CSV
    pokemon_cards = []
    trainer_cards = []
    energy_cards = []
    
    for card in csv_cards:
        supertype = card.get('Supertype', '')
        if supertype == 'Pokémon':
            pokemon_cards.append(card)
        elif supertype == 'Trainer':
            trainer_cards.append(card)
        elif supertype == 'Energy':
            energy_cards.append(card)
    
    print(f"Pokémon: {len(pokemon_cards)}")
    print(f"Trainers: {len(trainer_cards)}")
    print(f"Energy: {len(energy_cards)}")
    print()
    
    # Verification
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print()
    
    # Check each card
    missing_from_registry = []
    missing_effects = []
    implemented_cards = []
    
    for card in csv_cards:
        name = card.get('Name', '')
        supertype = card.get('Supertype', '')
        text = card.get('Text', '')
        ability = card.get('Ability', '')
        
        # Check registry
        in_registry = name in registry
        
        # Determine if card needs effect implementation
        needs_effects = False
        if supertype == 'Trainer' and text:
            needs_effects = True
        elif supertype == 'Pokémon' and (ability or 'effect' in text.lower()):
            needs_effects = True
        elif supertype == 'Energy' and text:
            needs_effects = True
        
        # Check effects
        has_effects = name in effects
        
        # Record results
        if not in_registry:
            missing_from_registry.append(name)
        
        if needs_effects and not has_effects:
            missing_effects.append({
                'name': name,
                'type': supertype,
                'text': text[:100] if text else '',
                'ability': ability
            })
        
        if in_registry and (not needs_effects or has_effects):
            implemented_cards.append(name)
    
    # Report
    print(f"✅ Correctly Implemented: {len(implemented_cards)}/{len(csv_cards)} ({100*len(implemented_cards)//len(csv_cards)}%)")
    print()
    
    if missing_from_registry:
        print(f"❌ Missing from CARD_REGISTRY: {len(missing_from_registry)}")
        for name in missing_from_registry[:10]:
            print(f"   - {name}")
        if len(missing_from_registry) > 10:
            print(f"   ... and {len(missing_from_registry) - 10} more")
        print()
    else:
        print("✅ All cards are in CARD_REGISTRY!")
        print()
    
    if missing_effects:
        print(f"⚠️  Missing Effect Implementations: {len(missing_effects)}")
        for card in missing_effects[:10]:
            print(f"   - {card['name']} ({card['type']})")
            if card['ability']:
                print(f"     Ability: {card['ability'][:60]}...")
            if card['text']:
                print(f"     Text: {card['text'][:60]}...")
        if len(missing_effects) > 10:
            print(f"   ... and {len(missing_effects) - 10} more")
        print()
    else:
        print("✅ All cards with effects are implemented!")
        print()
    
    # Detailed breakdown
    print("=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)
    print()
    
    # Pokemon with special mechanics
    pokemon_with_abilities = [c for c in pokemon_cards if c.get('Ability')]
    pokemon_with_special_attacks = [c for c in pokemon_cards if 'effect' in c.get('Text', '').lower()]
    
    print(f"🔷 Pokémon Breakdown:")
    print(f"   Total: {len(pokemon_cards)}")
    print(f"   With Abilities: {len(pokemon_with_abilities)}")
    print(f"   With Special Attacks: {len(pokemon_with_special_attacks)}")
    
    # Check each ability
    ability_missing = []
    for p in pokemon_with_abilities:
        if p['Name'] not in effects:
            ability_missing.append(p['Name'])
    
    if ability_missing:
        print(f"   ❌ Missing Ability Implementations: {ability_missing}")
    else:
        print(f"   ✅ All abilities implemented")
    print()
    
    # Trainer breakdown
    items = [c for c in trainer_cards if c.get('Subtype') == 'Item']
    supporters = [c for c in trainer_cards if c.get('Subtype') == 'Supporter']
    stadiums = [c for c in trainer_cards if c.get('Subtype') == 'Stadium']
    tools = [c for c in trainer_cards if c.get('Subtype') == 'Pokémon Tool']
    
    print(f"🔶 Trainer Breakdown:")
    print(f"   Total: {len(trainer_cards)}")
    print(f"   Items: {len(items)}")
    print(f"   Supporters: {len(supporters)}")
    print(f"   Stadiums: {len(stadiums)}")
    print(f"   Tools: {len(tools)}")
    
    trainer_missing = []
    for t in trainer_cards:
        if t['Name'] not in effects:
            trainer_missing.append(t['Name'])
    
    if trainer_missing:
        print(f"   ❌ Missing: {trainer_missing}")
    else:
        print(f"   ✅ All trainers implemented")
    print()
    
    # Energy breakdown
    basic_energy = [c for c in energy_cards if c.get('Subtype') == 'Basic']
    special_energy = [c for c in energy_cards if c.get('Subtype') == 'Special']
    
    print(f"⚡ Energy Breakdown:")
    print(f"   Total: {len(energy_cards)}")
    print(f"   Basic: {len(basic_energy)}")
    print(f"   Special: {len(special_energy)}")
    
    special_missing = []
    for e in special_energy:
        if e['Name'] not in effects and e.get('Text'):
            special_missing.append(e['Name'])
    
    if special_missing:
        print(f"   ❌ Missing: {special_missing}")
    else:
        print(f"   ✅ All special energy implemented")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    total_cards = len(csv_cards)
    total_implemented = len(implemented_cards)
    total_missing_registry = len(missing_from_registry)
    total_missing_effects = len(missing_effects)
    
    print(f"Total Cards in CSV: {total_cards}")
    print(f"In CARD_REGISTRY: {total_cards - total_missing_registry} ({100*(total_cards-total_missing_registry)//total_cards}%)")
    print(f"With Correct Effects: {total_implemented} ({100*total_implemented//total_cards}%)")
    print()
    
    if total_missing_registry == 0 and total_missing_effects == 0:
        print("🎉 ALL CARDS ARE CORRECTLY IMPLEMENTED!")
    else:
        print(f"⚠️  Work Needed:")
        if total_missing_registry > 0:
            print(f"   - Add {total_missing_registry} cards to CARD_REGISTRY")
        if total_missing_effects > 0:
            print(f"   - Implement {total_missing_effects} card effects")
    
    return total_missing_registry == 0 and total_missing_effects == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
