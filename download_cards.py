import os
import requests
import time

# Cards used in your decks
CARD_LIST = [
    "Abra", "Kadabra", "Alakazam", "Dunsparce", "Dudunsparce", "Fan Rotom",
    "Psyduck", "Fezandipiti ex", "Hilda", "Dawn", "Boss's Orders",
    "Lillie's Determination", "Tulip", "Buddy-Buddy Poffin", "Rare Candy",
    "Nest Ball", "Night Stretcher", "Wondrous Patch", "Enhanced Hammer",
    "Battle Cage", "Basic Psychic Energy", "Enriching Energy", "Jet Energy",
    "Charmander", "Charmeleon", "Charizard ex", "Pidgey", "Pidgeotto",
    "Pidgeot ex", "Shaymin", "Tatsugiri", "Munkidori", "Chi-Yu",
    "Gouging Fire ex", "Arven", "Iono", "Professor Turo's Scenario",
    "Ultra Ball", "Super Rod", "Counter Catcher", "Energy Search",
    "Unfair Stamp", "Technical Machine: Evolution", "Artazon",
    "Fire Energy", "Mist Energy", "Darkness Energy"
]

# Standard User-Agent to prevent 403 Forbidden errors
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

import re
def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_')

def download_card(card_name):
    clean_name = sanitize_name(card_name)
    path = f"./cards/{clean_name}.png"
    
    if os.path.exists(path):
        print(f"✓ {card_name} exists")
        return

    print(f"⬇ Downloading {card_name} (Newest Version)...")
    
    # Clean the name for search
    search_query = card_name.replace(" ex", "").replace(" V", "")
    
    # 1. We quote the name for exact matching: name:"Abra"
    # 2. We sort by release date descending: orderBy=-set.releaseDate
    # This ensures we get the SV/151 version, not the 1999 version.
    url = f"https://api.pokemontcg.io/v2/cards"
    params = {
        'q': f'name:"{search_query}"',
        'pageSize': 1,
        'orderBy': '-set.releaseDate' # <--- THIS ensures it's the correct (newest) card
    }
    
    try:

        # print the url sent with params
        print(f"URL: {url}?{params}")
        r = requests.get(url, headers=HEADERS, params=params)

        # print the url sent wit
        
        # Check for HTTP errors (403, 429, 500)
        if r.status_code != 200:
            print(f"❌ HTTP Error {r.status_code} for {card_name}")
            return

        data = r.json()
        
        if data['data']:
            # Get high res image
            image_url = data['data'][0]['images']['large']
            
            # Download the actual image
            img_r = requests.get(image_url, headers=HEADERS)
            with open(path, 'wb') as handler:
                handler.write(img_r.content)
            
            # Be nice to the API
            time.sleep(0.5) 
        else:
            print(f"⚠ Not found: {card_name}")
            
    except Exception as e:
        print(f"⚠ Error downloading {card_name}: {e}")

if __name__ == "__main__":
    if not os.path.exists("./cards"):
        os.makedirs("./cards")
        
    print("--- Starting Download (With Fixes) ---")
    for card in CARD_LIST:
        download_card(card)
    print("--- Done! ---")