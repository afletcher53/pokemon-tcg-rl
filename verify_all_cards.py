#!/usr/bin/env python3
"""
Batch card verification using inspect_card.py
Tests all recently implemented/fixed cards
"""

import subprocess
import sys

# Cards to verify (categorized by implementation session)
CARDS_TO_VERIFY = {
    "Recently Implemented/Fixed": [
        "Lana's Aid",
        "Fighting Gong",
        "Super Rod",
        "Night Stretcher",
        "Buddy-Buddy Poffin",
        "Mega Charizard X ex",
        "Maximum Belt",
        "Enhanced Hammer",
        "Tatsugiri",
    ],
    "Key Abilities": [
        "Gholdengo ex",
        "Alakazam",
        "Pidgeot ex",
        "Psyduck",
        "Fezandipiti ex",
    ],
    "Complex Trainers": [
        "Ultra Ball",
        "Rare Candy",
        "Arven",
        "Iono",
    ],
    "Special Energy": [
        "Enriching Energy",
        "Jet Energy",
        "Mist Energy",
    ]
}

def verify_card(card_name):
    """Run inspect_card.py for a single card and capture result."""
    try:
        result = subprocess.run(
            ["python", "inspect_card.py", card_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Check if implementation was found
        output = result.stdout
        
        if "❌ Card" in output and "not found" in output:
            return "NOT_IN_REGISTRY"
        elif "⚠️ No explicit implementation found" in output:
            return "NO_IMPLEMENTATION"
        elif "IMPLEMENTATION CODE:" in output and len(output) > 500:
            return "IMPLEMENTED"
        else:
            return "UNKNOWN"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    print("=" * 80)
    print("BATCH CARD VERIFICATION")
    print("=" * 80)
    print()
    
    total_cards = 0
    implemented = 0
    missing = 0
    errors = 0
    
    for category, cards in CARDS_TO_VERIFY.items():
        print(f"\n{category}:")
        print("-" * 60)
        
        for card_name in cards:
            total_cards += 1
            status = verify_card(card_name)
            
            if status == "IMPLEMENTED":
                print(f"  ✅ {card_name}")
                implemented += 1
            elif status == "NOT_IN_REGISTRY":
                print(f"  ❌ {card_name} - NOT IN REGISTRY")
                missing += 1
            elif status == "NO_IMPLEMENTATION":
                print(f"  ⚠️  {card_name} - NO IMPLEMENTATION")
                missing += 1
            else:
                print(f"  ❓ {card_name} - {status}")
                errors += 1
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Cards Tested: {total_cards}")
    print(f"✅ Implemented: {implemented} ({100*implemented//total_cards}%)")
    print(f"❌ Missing: {missing}")
    print(f"❓ Errors: {errors}")
    print()
    
    if missing == 0 and errors == 0:
        print("🎉 ALL TESTED CARDS ARE PROPERLY IMPLEMENTED!")
        return 0
    else:
        print(f"⚠️  {missing + errors} cards need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
