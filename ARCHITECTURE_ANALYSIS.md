# ✅ Architecture Analysis: env.py vs effects.py

## Summary

**Maximum Belt**: ✅ Added to CARD_REGISTRY  
**Psyduck (Damp)**: ✅ Already fully implemented in effects.py  
**Special Energies**: Currently in env.py (simple inline effects)

---

## 🔍 Where Everything Is

### Psyduck's Damp Ability ✅ 
**Location**: `effects.py` lines 865-877
**Function**: `_has_psyduck_damp_active(env)`
**Usage**: Called during self-KO abilities (like Dusknoir's Cursed Blast)

```python
def _has_psyduck_damp_active(env: 'PTCGEnv') -> bool:
    # Checks if Psyduck is in play on either side
    # Returns True to block self-KO abilities
```

**This IS in effects.py!** ✅ Correct architecture

---

### Special Energies (Enriching, Jet) 
**Location**: `env.py` lines 914-926
**Inline in**: `ATTACH_ACTIVE` and `ATTACH_BENCH` action handlers

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

---

## 🤔 Should Special Energies Be Moved?

### Option 1: Keep in env.py (Current)
**Pros:**
- ✅ Simple 1-2 line effects
- ✅ Executes immediately with attachment
- ✅ No function call overhead
- ✅ Works perfectly, no bugs

**Cons:**
- ❌ Inconsistent with other card effects
- ❌ Hard to find (not in effects.py)
- ❌ env.py gets cluttered as more energy is added

### Option 2: Move to effects.py (Recommended)
**Pros:**
- ✅ **Consistency** - All card logic in one place
- ✅ **Discoverability** - inspect_card.py would find them
- ✅ **Maintainability** - Single source of truth for cards
- ✅ **Extensibility** - Easy to add more special energy

**Cons:**
- ❌ Requires creating `apply_energy_effect()` function
- ❌ Small refactor needed in env.py

---

## 📝 Recommendation: Move Special Energies to effects.py

### Proposed Implementation

#### 1. Add to effects.py:
```python
def apply_energy_effect(env: 'PTCGEnv', player_idx: int, energy_name: str, attached_to_active: bool, bench_slot: int = None):
    """Apply special energy attachment effects."""
    me = env._gs.players[player_idx]
    
    if energy_name == "Enriching Energy":
        env._draw_cards(me, 4)
        if me == env._gs.players[0] and should_print():
            print(f"    -> Enriching Energy: Drew 4 cards")
    
    elif energy_name == "Jet Energy":
        if not attached_to_active and bench_slot is not None:
            # Switch the Pokémon to active
            me.active, me.bench[bench_slot] = me.bench[bench_slot], me.active
            if me == env._gs.players[0] and should_print():
                print(f"    -> Jet Energy: Switched to active")
```

#### 2. Update env.py:
```python
# ATTACH_ACTIVE
elif act.kind == "ATTACH_ACTIVE":
    me.hand.remove(act.a)
    me.active.energy.append(act.a)
    me.energy_attached = True
    apply_energy_effect(self, gs.turn_player, act.a, attached_to_active=True)

# ATTACH_BENCH
elif act.kind == "ATTACH_BENCH":
    me.hand.remove(act.a)
    me.bench[act.b].energy.append(act.a)
    me.energy_attached = True
    apply_energy_effect(self, gs.turn_player, act.a, attached_to_active=False, bench_slot=act.b)
```

---

## ✅ Benefits of Moving

1. **inspect_card.py will find them**
2. **Consistent architecture** - all card effects in effects.py
3. **Easy to add new special energy** in the future
4. **Single source of truth** for card implementations
5. **Better testing** - can test energy effects in isolation

---

## 🎯 Action Items

### Option A: Keep As-Is (Minimal Work)
- ✅ Already working
- Update documentation to note special energies are in env.py
- Add comment in env.py explaining why

### Option B: Refactor (Recommended, ~15 minutes)
1. Create `apply_energy_effect()` in effects.py
2. Update env.py to call it
3. Test with a simple game
4. Run verification again

---

## Current Status

| Card/Effect | Location | Status |
|-------------|----------|--------|
| **Maximum Belt** | cards.py | ✅ ADDED |
| **Psyduck (Damp)** | effects.py | ✅ Correctly located |
| **Enriching Energy** | env.py (inline) | ⚠️ Works but inconsistent |
| **Jet Energy** | env.py (inline) | ⚠️ Works but inconsistent  |
| **Mist Energy** | effects.py | ✅ Correctly located |

---

## My Recommendation

**Move special energies to effects.py** for consistency, even though they work fine now. It's a 15-minute refactor that will make the codebase cleaner and more maintainable long-term.

Would you like me to do this refactor now?
