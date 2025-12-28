# tcg/cards.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class Attack:
    name: str
    damage: int
    cost: List[str] # e.g. ["Fire", "Fire", "Colorless"]
    text: str = ""

@dataclass(frozen=True)
class CardDef:
    name: str
    supertype: str  # "Pokemon" | "Trainer" | "Energy"
    subtype: str  # "Basic", "Stage1", "Stage2", "Item", "Supporter", "Stadium"
    hp: int = 0
    type: str = "Colorless" 
    evolves_from: Optional[str] = None
    tags: tuple[str, ...] = ()
    attacks: Tuple[Attack, ...] = ()
    weakness: Optional[str] = None
    resistance: Optional[str] = None
    retreat_cost: int = 0
    ability: Optional[str] = None # Name of ability

    @property
    def has_rule_box(self):
        return "ex" in self.tags or "V" in self.tags

# Minimal registry (expand per deck)
CARD_REGISTRY: Dict[str, CardDef] = {
    # Pokemon
    "Psyduck": CardDef("Psyduck", "Pokemon", "Basic", hp=70, type="Water",
                       attacks=(Attack("Ram", 20, ["Colorless", "Colorless"]),),
                       weakness="Lightning", retreat_cost=1, ability="Damp"),

    "Charmander": CardDef("Charmander", "Pokemon", "Basic", hp=70, type="Fire", 
                          attacks=(Attack("Ember", 30, ["Fire", "Colorless"]),),
                          weakness="Water", retreat_cost=1),
                          
    "Charmeleon": CardDef("Charmeleon", "Pokemon", "Stage1", hp=90, type="Fire", evolves_from="Charmander",
                          attacks=(Attack("Combustion", 50, ["Fire", "Fire", "Colorless"]),),
                          weakness="Water", retreat_cost=2),
                          
    "Charizard ex": CardDef("Charizard ex", "Pokemon", "Stage2", hp=330, type="Darkness", evolves_from="Charmeleon", tags=("ex",),
                            attacks=(Attack("Burning Darkness", 180, ["Fire", "Fire"]),),
                            weakness="Grass", retreat_cost=2, ability="Infernal Reign"),
    
    "Pidgey": CardDef("Pidgey", "Pokemon", "Basic", hp=50, type="Colorless",
                      attacks=(Attack("Call for Family", 0, ["Colorless"]),
                               Attack("Tackle", 20, ["Colorless", "Colorless"])),
                      weakness="Lightning", resistance="Fighting", retreat_cost=1),
                      
    "Pidgeotto": CardDef("Pidgeotto", "Pokemon", "Stage1", hp=90, type="Colorless", evolves_from="Pidgey",
                         attacks=(Attack("Wing Attack", 40, ["Colorless", "Colorless"]),),
                         weakness="Lightning", retreat_cost=1),
                         
    "Pidgeot ex": CardDef("Pidgeot ex", "Pokemon", "Stage2", hp=280, type="Colorless", evolves_from="Pidgeotto", tags=("ex",),
                          attacks=(Attack("Gust", 120, ["Colorless", "Colorless"]),),
                          weakness="Lightning", retreat_cost=0, ability="Quick Search"),
    
    "Duskull": CardDef("Duskull", "Pokemon", "Basic", hp=60, type="Psychic",
                       attacks=(Attack("Rain of Pain", 10, ["Psychic"]),),
                       weakness="Darkness", retreat_cost=1),
                       
    "Dusclops": CardDef("Dusclops", "Pokemon", "Stage1", hp=90, type="Psychic", evolves_from="Duskull",
                        attacks=(Attack("Will-O-Wisp", 30, ["Psychic", "Colorless"]),),
                        weakness="Darkness", retreat_cost=2),
                        
    "Dusknoir": CardDef("Dusknoir", "Pokemon", "Stage2", hp=160, type="Psychic", evolves_from="Dusclops",
                        attacks=(Attack("Shadow Bind", 150, ["Psychic", "Psychic", "Colorless"]),),
                        weakness="Darkness", retreat_cost=3),
    
    "Tatsugiri": CardDef("Tatsugiri", "Pokemon", "Basic", hp=70, type="Dragon",
                         attacks=(Attack("Surf", 50, ["Fire", "Water"]),),
                         retreat_cost=1, ability="Attract Customers"),
                         
    "Klefki": CardDef("Klefki", "Pokemon", "Basic", hp=70, type="Psychic",
                      attacks=(Attack("Mischievous Lock", 10, ["Colorless"]),),
                      weakness="Metal", retreat_cost=1),
                      
    "Fan Rotom": CardDef("Fan Rotom", "Pokemon", "Basic", hp=70, type="Colorless",
                         attacks=(Attack("Assault Landing", 70, ["Colorless"]),),
                         weakness="Lightning", retreat_cost=1, ability="Fan Call"),
    
    "Dunsparce": CardDef("Dunsparce", "Pokemon", "Basic", hp=70, type="Colorless",
                         attacks=(Attack("Gnaw", 20, ["Colorless"]),),
                         weakness="Fighting", retreat_cost=1),
                         
    "Dudunsparce": CardDef("Dudunsparce", "Pokemon", "Stage1", hp=140, type="Colorless", evolves_from="Dunsparce",
                           attacks=(Attack("Land Crush", 90, ["Colorless", "Colorless", "Colorless"]),),
                           weakness="Fighting", retreat_cost=3, ability="Run Away Draw"),
    
    "Abra": CardDef("Abra", "Pokemon", "Basic", hp=50, type="Psychic",
                    attacks=(Attack("Teleportation Attack", 10, ["Psychic"]),),
                    weakness="Darkness", resistance="Fighting", retreat_cost=1),
                    
    "Kadabra": CardDef("Kadabra", "Pokemon", "Stage1", hp=80, type="Psychic", evolves_from="Abra",
                       attacks=(Attack("Super Psy Bolt", 30, ["Psychic"]),),
                       weakness="Darkness", retreat_cost=1, ability="Psychic Draw"),
                       
    "Alakazam": CardDef("Alakazam", "Pokemon", "Stage2", hp=140, type="Psychic", evolves_from="Kadabra",
                        attacks=(Attack("Powerful Hand", 0, ["Psychic"]),),
                        weakness="Darkness", retreat_cost=1, ability="Psychic Draw"),
    
    "Fezandipiti ex": CardDef("Fezandipiti ex", "Pokemon", "Basic", hp=210, type="Darkness", tags=("ex",),
                              attacks=(Attack("Cruel Arrow", 0, ["Colorless", "Colorless", "Colorless"]),),
                              weakness="Fighting", retreat_cost=1, ability="Flip the Script"),
    
    # New Pokemon for Charizard deck
    "Shaymin": CardDef("Shaymin", "Pokemon", "Basic", hp=80, type="Grass",
                       attacks=(Attack("Smash Kick", 30, ["Colorless", "Colorless"]),),
                       weakness="Fire", retreat_cost=1, ability="Flower Curtain"),
    
    "Munkidori": CardDef("Munkidori", "Pokemon", "Basic", hp=110, type="Psychic",
                         attacks=(Attack("Mind Bend", 60, ["Psychic", "Colorless"]),),
                         weakness="Darkness", resistance="Fighting", retreat_cost=1, ability="Adrena-Brain"),
    
    "Chi-Yu": CardDef("Chi-Yu", "Pokemon", "Basic", hp=110, type="Fire",
                      attacks=(Attack("Megafire of Envy", 50, ["Fire", "Fire"]),),  # 50+90 if KO'd last turn
                      weakness="Water", retreat_cost=1),
    
    "Gouging Fire ex": CardDef("Gouging Fire ex", "Pokemon", "Basic", hp=230, type="Fire", tags=("ex",),
                               attacks=(Attack("Blaze Blitz", 260, ["Fire", "Fire", "Colorless"]),),
                               weakness="Water", retreat_cost=2),
                               
    "Mega Charizard X ex": CardDef("Mega Charizard X ex", "Pokemon", "Stage2", hp=360, type="Fire", evolves_from="Charmeleon", tags=("ex",),
                                   attacks=(Attack("Inferno X", 90, ["Fire", "Fire"]),), # 90x per discarded Fire Energy
                                   weakness="Water", retreat_cost=2),
                               
    # Gholdengo / Solrock / Lunatone Deck
    "Gimmighoul": CardDef("Gimmighoul", "Pokemon", "Basic", hp=70, type="Psychic",
                          attacks=(Attack("Minor Errand-Running", 0, ["Colorless"]), 
                                   Attack("Tackle", 50, ["Colorless", "Colorless", "Colorless"])),
                          weakness="Darkness", resistance="Fighting", retreat_cost=2),
                          
    "Gholdengo ex": CardDef("Gholdengo ex", "Pokemon", "Stage1", hp=260, type="Metal", evolves_from="Gimmighoul", tags=("ex",),
                            attacks=(Attack("Make It Rain", 50, ["Metal"]),), # 50x
                            weakness="Fire", resistance="Grass", retreat_cost=2, ability="Coin Bonus"),
                            
    "Solrock": CardDef("Solrock", "Pokemon", "Basic", hp=110, type="Fighting",
                       attacks=(Attack("Cosmic Beam", 70, ["Fighting"]),), # Condition: Lunatone on bench
                       weakness="Grass", retreat_cost=1),
                       
    "Lunatone": CardDef("Lunatone", "Pokemon", "Basic", hp=110, type="Fighting",
                        attacks=(Attack("Power Gem", 50, ["Fighting", "Fighting"]),),
                        weakness="Grass", retreat_cost=1, ability="Lunar Cycle"),
                        
    "Genesect ex": CardDef("Genesect ex", "Pokemon", "Basic", hp=220, type="Metal", tags=("ex",),
                           attacks=(Attack("Protect Charge", 150, ["Metal", "Metal", "Colorless"]),),
                           weakness="Fire", resistance="Grass", retreat_cost=2, ability="Metallic Signal"),
                           
    "Hop's Cramorant": CardDef("Hop's Cramorant", "Pokemon", "Basic", hp=110, type="Colorless",
                               attacks=(Attack("Fickle Spitting", 120, ["Colorless"]),),
                               weakness="Lightning", resistance="Fighting", retreat_cost=1),
    
    # Trainers
    "Buddy-Buddy Poffin": CardDef("Buddy-Buddy Poffin", "Trainer", "Item"),
    "Rare Candy": CardDef("Rare Candy", "Trainer", "Item"),
    "Ultra Ball": CardDef("Ultra Ball", "Trainer", "Item"),
    "Arven": CardDef("Arven", "Trainer", "Supporter"),
    "Iono": CardDef("Iono", "Trainer", "Supporter"),
    "Boss's Orders": CardDef("Boss's Orders", "Trainer", "Supporter"),
    "Artazon": CardDef("Artazon", "Trainer", "Stadium"),
    "Battle Cage": CardDef("Battle Cage", "Trainer", "Stadium"),
    "Technical Machine: Evolution": CardDef("Technical Machine: Evolution", "Trainer", "Tool",
                                            attacks=(Attack("Evolution", 0, ["Colorless"]),)),
    "Counter Catcher": CardDef("Counter Catcher", "Trainer", "Item"),
    "Super Rod": CardDef("Super Rod", "Trainer", "Item"),
    "Night Stretcher": CardDef("Night Stretcher", "Trainer", "Item"),
    "Hilda": CardDef("Hilda", "Trainer", "Supporter"),
    "Dawn": CardDef("Dawn", "Trainer", "Supporter"),
    "Lillie's Determination": CardDef("Lillie's Determination", "Trainer", "Supporter"),
    "Tulip": CardDef("Tulip", "Trainer", "Supporter"),
    "Enhanced Hammer": CardDef("Enhanced Hammer", "Trainer", "Item"),
    "Professor's Research": CardDef("Professor's Research", "Trainer", "Supporter"),
    "Nest Ball": CardDef("Nest Ball", "Trainer", "Item"),
    "Wondrous Patch": CardDef("Wondrous Patch", "Trainer", "Item"),
    "Bill": CardDef("Bill", "Trainer", "Supporter"),  # Debug
    "Energy Search": CardDef("Energy Search", "Trainer", "Item"),
    "Professor Turo's Scenario": CardDef("Professor Turo's Scenario", "Trainer", "Supporter"),
    "Unfair Stamp": CardDef("Unfair Stamp", "Trainer", "Item"),
    "Earthen Vessel": CardDef("Earthen Vessel", "Trainer", "Item"),
    "Superior Energy Retrieval": CardDef("Superior Energy Retrieval", "Trainer", "Item"),
    "Fighting Gong": CardDef("Fighting Gong", "Trainer", "Item"), 
    "Lana's Aid": CardDef("Lana's Aid", "Trainer", "Supporter"),
    "Premium Power Pro": CardDef("Premium Power Pro", "Trainer", "Item"),
    "Prime Catcher": CardDef("Prime Catcher", "Trainer", "Item"),
    "Air Balloon": CardDef("Air Balloon", "Trainer", "Tool"),
    "Vitality Band": CardDef("Vitality Band", "Trainer", "Tool"),
    "Maximum Belt": CardDef("Maximum Belt", "Trainer", "Tool"),
    "Switch": CardDef("Switch", "Trainer", "Item"),
    "Escape Rope": CardDef("Escape Rope", "Trainer", "Item"),
    
    # Energy
    "Basic Fire Energy": CardDef("Basic Fire Energy", "Energy", "Basic", type="Fire"),
    "Fire Energy": CardDef("Fire Energy", "Energy", "Basic", type="Fire"),  # Alias
    "Basic Psychic Energy": CardDef("Basic Psychic Energy", "Energy", "Basic", type="Psychic"),
    "Jet Energy": CardDef("Jet Energy", "Energy", "Special", type="Colorless"),
    "Enriching Energy": CardDef("Enriching Energy", "Energy", "Special", type="Colorless"),
    "Mist Energy": CardDef("Mist Energy", "Energy", "Special", type="Colorless"),
    "Darkness Energy": CardDef("Darkness Energy", "Energy", "Basic", type="Darkness"),
    "Basic Water Energy": CardDef("Basic Water Energy", "Energy", "Basic", type="Water"),
    "Basic Metal Energy": CardDef("Basic Metal Energy", "Energy", "Basic", type="Metal"),
    "Metal Energy": CardDef("Metal Energy", "Energy", "Basic", type="Metal"),
    "Basic Fighting Energy": CardDef("Basic Fighting Energy", "Energy", "Basic", type="Fighting"),
    "Fighting Energy": CardDef("Fighting Energy", "Energy", "Basic", type="Fighting"),
}


def card_def(name: str) -> CardDef:
    if name not in CARD_REGISTRY:
        # Unknown card, keep system alive. Add it later.
        CARD_REGISTRY[name] = CardDef(name, "Unknown", "Unknown")
    return CARD_REGISTRY[name]
