# Architecture Explanation: Why Special Energy in env.py?

## ✅ Maximum Belt Added!

Just added `Maximum Belt` to `tcg/cards.py` CARD_REGISTRY. Now all tools are registered!

---

## 🏗️ Current Architecture

Your codebase has a clean separation of concerns:

### `effects.py` - Card-Specific Logic
**Purpose**: Complex card effects that need dedicated functions
**Contains**: 
- `apply_trainer_effect()` - Trainer card effects
- `apply_ability_effect()` - Pokemon abilities
- `apply_attack_effect()` - Special attack damage/effects

**Examples**:
- Ultra Ball (search + discard)
- Lana's Aid (card selection from discard)
- Alakazam's Psychic Draw ability
- Gholdengo ex's Make It Rain attack

---

### `env.py` - Game Loop & Simple Effects
**Purpose**: Core game mechanics and immediate/simple effects
**Contains**:
- `step()` - Main game loop
- `action_mask()` - Valid action filtering
- `_perform_attack()` - Attack execution
- **Simple inline effects** (current location of special energies)

---

## 🤔 Why Are Special Energies in env.py?

### Current Implementation (Lines 914-926)

```python
# ATTACH_ACTIVE
if act.a == "Enriching Energy":
    self._draw_cards(me, 4)

# ATTACH_BENCH  
elif act.a == "Jet Energy":
    me.active, me.bench[act.b] = me.bench[act.b], me.active
elif act.a == "Enriching Energy":
    self._draw_cards(me, 4)
```

**Reasoning** (likely historical):
1. **Simple effects** - Just 1-2 lines of code each
2. **Immediate execution** - Happens right when energy is attached
3. **No complex logic** - No card selection, no conditions, no targeting

---

## ❓ Should They Be Moved to effects.py?

### Argument FOR Moving:
✅ **Consistency** - All card-specific logic in one place  
✅ **Maintainability** - Easier to find all card implementations  
✅ **Scalability** - As more special energy is added, env.py gets cluttered  
✅ **Testability** - Can test energy effects in isolation  

### Argument AGAINST Moving:
❌ **Performance** - Function call overhead (extremely minor)  
❌ **Simplicity** - These are literally 1-2 line effects  
❌ **It works** - No bugs, no issues currently  

---

## 📝 What About Psyduck (Damp)?

Let me search for it:
