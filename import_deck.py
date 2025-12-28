#!/usr/bin/env python3
"""
Deck Importer & Validator for Pokemon TCG AI
--------------------------------------------
1. Scrapes decklists from Limitless TCG (or parses provided text).
2. Checks against implemented cards in `tcg/cards.py`.
3. Validates card counts and legality.
4. Generates Python code to add the deck to `train_advanced.py`.

Usage:
    python3 import_deck.py --url "https://limitlesstcg.com/decks/list/..."
    python3 import_deck.py --file decklist.txt
"""

import argparse
import re
import sys
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple, Dict

# Import existing card database to verify implementation
# (Assumes tcg/cards.py exists in path)
try:
    from tcg.cards import _card_db
except ImportError:
    print("Warning: Could not import tcg.cards. Verification will be limited.")
    _card_db = {}

def parse_limitless_decklist(url: str) -> List[str]:
    """Scrape decklist from a Limitless TCG URL."""
    print(f"[*] Scraping {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"[!] Error fetching URL: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Limitless mostly stores deck export text in a specific modal or copy-able area
    # Or simpler: parse the visual list
    # The "Copy to Clipboard" usually has the clean text. 
    # Let's look for the decklist-export class or simplify by parsing table rows.
    
    cards = []
    
    # Try finding the export text area first (easiest format)
    export_btn = soup.find('div', class_='decklist-export')
    if export_btn:
        # Usually requires JS interaction or hidden div.
        # Fallback to parsing table rows which is reliable HTML.
        pass

    # Parse rows
    # Look for .decklist-card
    card_rows = soup.find_all('tr') # Limitless uses tables often
    
    # Generic Limitless parsing logic for main deck table
    # <span class="card-count">4</span> <a href="...">Comfey</a> <span class="set-code">LOR 79</span>
    
    extracted_lines = []
    
    # Alternative: Use the "Export" button endpoint if predictable? No.
    # Let's try parsing the text content of the page line by line looking for "X Name Set Code" pattern
    
    text = soup.get_text()
    lines = text.split('\n')
    
    # Regex for standard PTCGL export: "4 Comfey LOR 79"
    # Or Limitless format: "4 Comfey"
    
    # Better approach: Find the container .decklist-column or .decklist
    decklist_div = soup.find('div', class_='decklist')
    if not decklist_div:
        # Fallback for play.limitless
        decklist_div = soup.find('div', class_='decklist-cards')
        
    if decklist_div:
        for line in decklist_div.stripped_strings:
            # Lines look like "4", "Comfey", "LOR 79" or combined.
            # This is messy.
            pass
            
    # For now, let's assume the user copies the "PTCGL Export" text to a file or command line 
    # because scraping dynamic JS sites is flaky.
    # BUT simplest robust scrape is looking for the "copy" text area.
    
    # Actually, let's allow "Copy Text" input primarily, or simple URL logic.
    # If URL contains "limitlesstcg.com":
    # Look for the hidden input that serves the copy button? 
    # <input type="hidden" id="export-text" value="...">
    # This is often present!
    export_input = soup.find('textarea', id='export-text') # Common pattern
    if not export_input:
        export_input = soup.find('input', class_='export-deck-text') 
        
    if export_input and export_input.get('value'):
        return export_input['value'].split('\n')
        
    print("[!] Could not auto-extract decklist text. Please copy/paste the PTCGL export text.")
    return []

def parse_ptcgl_text(lines: List[str]) -> List[Tuple[int, str]]:
    """Parse standard PTCGL export format."""
    deck = []
    # Regex: "3 Comfey LOR 79" or "3 Comfey"
    # We essentially want Count + Name. Set/Code is irrelevant for internal engine usually.
    
    # Pattern: Digit+ space Name space (Set Code)
    # We stop at the Set Code (usually 3 chars upper + digits)
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('Pokémon:') or line.startswith('Trainer:') or line.startswith('Energy:'):
            continue
        if line.startswith('Total Cards:'):
            continue
            
        parts = line.split(' ')
        if not parts[0].isdigit():
            continue
            
        count = int(parts[0])
        
        # Heuristic to find name:
        # Join words until we hit a set code pattern (e.g. LOR, PAF, MEW) or end of line.
        # Set codes are usually 2-4 uppercase letters followed by numbers.
        
        name_parts = []
        for p in parts[1:]:
            # Check if likely a set code (ALL CAPS, 2-4 chars)
            if p.isupper() and 2 <= len(p) <= 4:
                # Likely set code, stop
                break
            # Also check for trailing set numbers (e.g. "79")
            if p.isdigit():
                break
                
            name_parts.append(p)
            
        name = " ".join(name_parts)
        
        # Clean up specific weirdness
        name = name.replace(" ex", " ex") # Normalize spaces? usually fine.
        deck.append((count, name))
        
    return deck

def verify_deck(deck: List[Tuple[int, str]]) -> Tuple[List[str], List[str]]:
    """Verify if cards exist in DB. Returns (valid_cards, missing_cards)."""
    valid_deck = []
    missing_cards = []
    
    for count, name in deck:
        # Normalize name (database match)
        # 1. Exact match
        if name in _card_db:
            valid_deck.extend([name] * count)
            continue
            
        # 2. Try removing/adding 'ex'
        if name + " ex" in _card_db:
            valid_deck.extend([name + " ex"] * count)
            print(f"[*] Mapped '{name}' -> '{name} ex'")
            continue
            
        # 3. Fuzzy match (e.g. "Boss's Orders (Ghetsis)" -> "Boss's Orders")
        found = False
        for db_name in _card_db.keys():
            if db_name.startswith(name) or name.startswith(db_name):
                # Check for Trainers with sub-names
                if "Boss's Orders" in name and "Boss's Orders" in db_name:
                    valid_deck.extend([db_name] * count)
                    found = True
                    break
                if "Professor's Research" in name and "Professor's Research" in db_name:
                    valid_deck.extend([db_name] * count)
                    found = True
                    break
        
        if found:
            continue
            
        print(f"[X] MISSING IMPLEMENTATION: {name} (x{count})")
        missing_cards.append(name)
        
    return valid_deck, missing_cards

def generate_python_code(deck_name: str, valid_cards: List[str]):
    """Generate the array code for train_advanced.py."""
    print(f"\n[+] Generated Code for '{deck_name}':")
    print("-" * 50)
    print(f"{deck_name}_deck = []")
    
    # Compress list back to counts
    from collections import Counter
    counts = Counter(valid_cards)
    
    for name, count in counts.items():
        print(f'{deck_name}_deck.extend(["{name}"] * {count})')
        
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Import Decklist")
    parser.add_argument("--url", help="Limitless TCG Deck URL")
    parser.add_argument("--file", help="Text file with PTCGL export")
    args = parser.parse_args()
    
    lines = []
    if args.url:
        lines = parse_limitless_decklist(args.url)
    elif args.file:
        with open(args.file, 'r') as f:
            lines = f.readlines()
    else:
        # Interactive paste
        print("Paste PTCGL Export text below (Ctrl+D to finish):")
        lines = sys.stdin.readlines()
        
    if not lines:
        return
        
    parsed_deck = parse_ptcgl_text(lines)
    print(f"\n[*] Parsed {sum(c for c,n in parsed_deck)} cards.")
    
    valid_list, missing = verify_deck(parsed_deck)
    
    if len(valid_list) == 60:
        print("\n[OK] Deck is legal (60 cards) and fully implemented!")
        generate_python_code("new_meta", valid_list)
    else:
        print(f"\n[!] Deck incomplete. Has {len(valid_list)} valid cards.")
        print(f"Missing {len(missing)} unique cards.")
        if missing:
            print("To fix, implement these in tcg/cards.py:")
            for m in missing:
                print(f"  - {m}")
                
if __name__ == "__main__":
    main()
