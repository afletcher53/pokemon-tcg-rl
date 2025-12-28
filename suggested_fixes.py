#!/usr/bin/env python3
"""
Quick fixes for identified card implementation issues.
Run this to see suggested code changes.
"""

print("=" * 80)
print("CARD IMPLEMENTATION FIXES")
print("=" * 80)
print()

print("FIX #1: Tatsugiri - Add Active Spot Check")
print("-" * 80)
print("Location: tcg/effects.py, line 651")
print()
print("REPLACE:")
print("""
    elif pokemon_name == "Tatsugiri":  # Attract Customers
        # Reveal top 6 cards, put a Supporter into hand, shuffle ONLY the other revealed cards back.
        # Implementation: Check top 6
        top_6 = me.deck[-6:]
""")
print()
print("WITH:")
print("""
    elif pokemon_name == "Tatsugiri":  # Attract Customers
        # Check if Tatsugiri is in the Active Spot
        if me.active.name != "Tatsugiri":
            if me == env._gs.players[0] and should_print():
                print(f"    -> Tatsugiri: \\033[90mMust be Active to use Attract Customers\\033[0m")
            return
        
        # Reveal top 6 cards, put a Supporter into hand, shuffle ONLY the other revealed cards back.
        # Implementation: Check top 6
        top_6 = me.deck[-6:]
""")
print()
print()

print("FIX #2: Enhanced Hammer - Check for Special Energy")
print("-" * 80)
print("Location: tcg/effects.py, line 341")
print()
print("REPLACE:")
print("""
    elif card_name == "Enhanced Hammer":
        # Discard Special Energy from Op active
        if len(op.active.energy) > 0: 
             op.discard_pile.append(op.active.energy.pop())
""")
print()
print("WITH:")
print("""
    elif card_name == "Enhanced Hammer":
        # Discard a Special Energy from opponent's active
        special_indices = [i for i, e in enumerate(op.active.energy) 
                          if card_def(e).subtype == "Special"]
        if special_indices:
            idx = special_indices[0]  # Take first special energy
            discarded = op.active.energy.pop(idx)
            op.discard_pile.append(discarded)
            if me == env._gs.players[0] and should_print():
                print(f"    -> Enhanced Hammer: Discarded \\033[96m{discarded}\\033[0m")
        else:
            if me == env._gs.players[0] and should_print():
                print(f"    -> Enhanced Hammer: \\033[90mNo Special Energy found\\033[0m")
""")
print()
print()

print("FIX #3: Add Shaymin's Flower Curtain Passive Protection")
print("-" * 80)
print("Location: tcg/env.py, in _perform_attack method around line 1026")
print()
print("ADD AFTER LINE 1026 (after apply_attack_effect):")
print("""
        # Check for Shaymin's Flower Curtain protection
        # Prevent damage to non-Rule Box benched Pokemon
        if tgt != 6 and tgt != 5:  # Targeting a bench slot (0-4)
            # Check if opponent has Shaymin with Flower Curtain in play
            has_flower_curtain = False
            for slot in [op.active] + op.bench:
                if slot.name == "Shaymin":
                    has_flower_curtain = True
                    break
            
            if has_flower_curtain:
                # Check if target has a Rule Box
                target_slot = op.bench[tgt] if 0 <= tgt < 5 else None
                if target_slot and target_slot.name:
                    target_def = card_def(target_slot.name)
                    if not target_def.has_rule_box():
                        # Protected by Flower Curtain!
                        if me == gs.players[0] and should_print():
                            print(f"    -> Flower Curtain: Prevented damage to {target_slot.name}")
                        final_dmg = 0
""")
print()
print()

print("FIX #4: Add Maximum Belt Tool")
print("-" * 80)
print("Location: tcg/effects.py, line 993 (after Vitality Band)")
print()
print("ADD:")
print("""
    # 3. Maximum Belt (+50 vs Pokemon ex)
    if me.active.tool == "Maximum Belt":
        if op.active.name and card_def(op.active.name).has_rule_box():
            damage_out += 50
""")
print()
print("ALSO ADD to cards.py registry:")
print("""
    "Maximum Belt": CardDef("Maximum Belt", "Trainer", "Tool"),
""")
print()
print()

print("FIX #5: Add Mega Charizard X ex Attack")
print("-" * 80)
print("Location: tcg/effects.py, around line 973 (after Gholdengo ex)")
print()
print("ADD:")
print("""
    if pokemon_name == "Mega Charizard X ex":
        # Inferno X: Discard X Fire Energy from your Pokemon. 90x damage.
        # Similar to Make It Rain but can discard from any Pokemon
        
        fire_count = 0
        # Count Fire energy on all Pokemon
        for slot in [me.active] + me.bench:
            if slot.name:
                fire_count += sum(1 for e in slot.energy if card_def(e).type == "Fire" and card_def(e).subtype == "Basic")
        
        count_to_discard = min(fire_count, discard_amount)
        dmg = 0
        
        # Discard from active first, then bench
        for slot in [me.active] + me.bench:
            if slot.name and count_to_discard > 0:
                fire_indices = [i for i, e in enumerate(slot.energy) 
                               if card_def(e).type == "Fire" and card_def(e).subtype == "Basic"]
                
                for idx in sorted(fire_indices, reverse=True):
                    if count_to_discard <= 0:
                        break
                    card = slot.energy.pop(idx)
                    me.discard_pile.append(card)
                    dmg += 90
                    count_to_discard -= 1
        
        if me == gs.players[0] and should_print():
            print(f"    -> Inferno X: Discarded for {dmg} damage!")
        return dmg
""")
print()
print("ALSO ADD to cards.py registry:")
print("""
    "Mega Charizard X ex": CardDef("Mega Charizard X ex", "Pokemon", "Stage2", 
                                   hp=360, type="Fire", evolves_from="Charmeleon", 
                                   tags=("ex",),
                                   attacks=(Attack("Inferno X", 90, ["Fire", "Fire"]),),
                                   weakness="Water", retreat_cost=2),
""")
print()
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("✅ Fix #1 (Tatsugiri): Add 4 lines")
print("✅ Fix #2 (Enhanced Hammer): Replace 3 lines with 10 lines")
print("✅ Fix #3 (Shaymin): Add ~20 lines in env.py")
print("✅ Fix #4 (Maximum Belt): Add 4 lines in effects.py + 1 line in cards.py")
print("✅ Fix #5 (Mega Charizard X ex): Add ~30 lines in effects.py + card definition")
print()
print("Total estimated time: ~60 minutes for all fixes")
print()
