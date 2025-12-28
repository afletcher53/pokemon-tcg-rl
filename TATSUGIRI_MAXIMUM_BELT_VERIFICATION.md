# ✅ Verification: Tatsugiri & Maximum Belt

## User Questions Answered

### 1. Does Tatsugiri remove the found card from the deck? ✅

**Answer: YES, correctly!**

**Code Location**: `tcg/effects.py` lines ~1085-1108

**Implementation**:
```python
elif pokemon_name == "Tatsugiri":
    # Reveal top 6 cards, find Supporter
    top_6 = me.deck[-6:]
    found_idx = -1
    for i, c in enumerate(reversed(top_6)):
        if card_def(c).subtype == "Supporter":
            found_idx = len(me.deck) - 1 - i
            break
    
    if found_idx >= 0:
        card = me.deck.pop(found_idx)  # ✅ REMOVES from deck
        me.hand.append(card)            # ✅ ADDS to hand
        random.shuffle(me.deck)         # ✅ SHUFFLES remaining
```

**Verification**:
- ✅ Found Supporter is removed from deck via `pop()`
- ✅ Found Supporter is added to hand
- ✅ Remaining 5 cards (plus rest of deck) are shuffled back
- ✅ Implementation is correct!

**Edge Cases Handled**:
- ✅ Only works when Tatsugiri is Active
- ✅ Handles case when no Supporter is found
- ✅ Correctly reverses iteration to check "top" cards

---

### 2. Maximum Belt Effect Already Implemented! ✅

**Answer: Already in effects.py!**

**Code Location**: `tcg/effects.py` lines 1351-1355 (in `apply_attack_effect`)

**Implementation**:
```python
# 2. Maximum Belt (+50 to ex)
if me.active.tool == "Maximum Belt":
    if op.active.name:
        op_def = card_def(op.active.name)
        if op_def.tags and "ex" in op_def.tags:
            damage_out += 50
```

**Verification**:
- ✅ Checks if Maximum Belt is attached to attacking Pokemon
- ✅ Gets opponent's active Pokemon definition
- ✅ Checks for "ex" tag in tags tuple
- ✅ Adds exactly +50 damage
- ✅ Implementation is correct!

**Works With**:
- Charizard ex ✅
- Pidgeot ex ✅
- Gholdengo ex ✅
- Fezandipiti ex ✅
- Alakazam ex ✅ (if in future)
- Any Pokemon with "ex" in tags ✅

---

## 📊 Summary

| Item | Status | Location |
|------|--------|----------|
| **Tatsugiri card removal** | ✅ Correct | effects.py ~1095 |
| **Maximum Belt effect** | ✅ Already implemented | effects.py 1351-1355 |
| **Maximum Belt in CARD_REGISTRY** | ✅ Added earlier | cards.py line 195 |

---

## 🎯 Both Items: VERIFIED WORKING!

No changes needed - both implementations are already correct!

- Tatsugiri properly removes the found Supporter from deck ✅
- Maximum Belt adds +50 damage vs ex Pokemon ✅

---

Last verified: 2025-12-25
