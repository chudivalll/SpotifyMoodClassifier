import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
from pathlib import Path
import urllib.parse
import re
import urllib3
import sys
import traceback

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))

input_file = "spotify_playlist_complete.xlsx"
output_file = "spotify_playlist_bpm_enhanced.xlsx"

df = pd.read_excel(input_file)
print(f"Loaded {len(df)} tracks")

if 'Actual_BPM' not in df.columns:
    df['Actual_BPM'] = None
if 'Music_Key' not in df.columns:
    df['Music_Key'] = None

TRACK_COL_NAMES = ['Track Name', 'track_name', 'Name', 'title', 'track', 'name']
ARTIST_COL_NAMES = ['Artist', 'artist_name', 'artist', 'artists']

track_col = next((v for v in TRACK_COL_NAMES if v in df.columns), None)
artist_col = next((v for v in ARTIST_COL_NAMES if v in df.columns), None)

if not track_col or not artist_col:
    print(f"Can't find track/artist columns. Available: {list(df.columns)}")
    sys.exit(1)

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/92.0.4515.107 Safari/537.36',
]

KNOWN_BPM = {
    ("Alive - Anyma Remix", "RÜFÜS DU SOL"): (125, "Eb Minor"),
    ("Stereophobia - Original Mix", "Dark Oscillators"): (140, "A Minor"),
    ("What's Left to Give", "THE WHATNAUTS"): (90, "A#/Bb"),
    ("tv off (feat. lefty gunplay)", "KENDRICK LAMAR"): (100, "F#/Gb"),
}


def get_bpm_from_songbpm(track_name, artist_name):
    """Scrape songbpm.com for BPM and key."""
    query = urllib.parse.quote(f"{artist_name} {track_name}")
    url = f"https://songbpm.com/?q={query}"

    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://songbpm.com/'
    }

    try:
        time.sleep(random.uniform(2.0, 3.0))
        resp = requests.get(url, headers=headers, verify=False, timeout=15)

        if resp.status_code != 200:
            return None, None

        soup = BeautifulSoup(resp.text, 'html.parser')
        bpm, key = None, None

        # check search results
        results = soup.select('.song-result') or soup.select('.card') or soup.select('article')
        if results:
            first = results[0]

            # look for BPM in the result card
            bpm_el = first.select_one('.bpm')
            if bpm_el:
                m = re.search(r'(\d+)', bpm_el.text)
                if m:
                    bpm = int(m.group(1))

            # follow detail link if available
            link = first.select_one('a')
            if link and 'href' in link.attrs:
                detail_url = link['href']
                if not detail_url.startswith('http'):
                    detail_url = f"https://songbpm.com{detail_url}"

                detail_resp = requests.get(detail_url, headers=headers, verify=False, timeout=15)
                if detail_resp.status_code == 200:
                    detail_soup = BeautifulSoup(detail_resp.text, 'html.parser')
                    detail_text = detail_soup.get_text()

                    if not bpm:
                        m = re.search(r'(\d+)\s*BPM|BPM\s*:?\s*(\d+)', detail_text, re.IGNORECASE)
                        if m:
                            bpm = int(m.group(1) or m.group(2))

                    m = re.search(r'([A-G][#♯♭b]?\s*(?:Major|Minor))', detail_text, re.IGNORECASE)
                    if m:
                        key = m.group(1)
        else:
            # no structured results, scan page text
            text = soup.get_text()
            m = re.search(r'(\d+)\s*BPM|BPM\s*:?\s*(\d+)', text, re.IGNORECASE)
            if m:
                bpm = int(m.group(1) or m.group(2))

        return bpm, key

    except requests.exceptions.Timeout:
        return None, None
    except requests.exceptions.ConnectionError:
        return None, None
    except Exception as e:
        print(f"    error: {e}")
        return None, None


def search_with_retry(track_name, artist_name, retries=3, base_delay=5):
    for attempt in range(retries):
        try:
            bpm, key = get_bpm_from_songbpm(track_name, artist_name)
            if bpm is not None or key is not None:
                return bpm, key
            if attempt < retries - 1:
                time.sleep(base_delay * (attempt + 1))
        except Exception:
            time.sleep(base_delay * (attempt + 1))
    return None, None


BATCH_SIZE = 2
BATCH_DELAY = 10
indices = df[df['Actual_BPM'].isnull()].index.tolist()
found = 0
processed = 0

print(f"Scraping BPM for {len(indices)} tracks from songbpm.com...")

try:
    for batch_start in range(0, len(indices), BATCH_SIZE):
        batch = indices[batch_start:batch_start + BATCH_SIZE]

        for idx in batch:
            name = df.loc[idx, track_col]
            artist = df.loc[idx, artist_col]
            print(f"  [{processed+1}/{len(indices)}] {name} — {artist}", end="")

            if (name, artist) in KNOWN_BPM:
                bpm, key = KNOWN_BPM[(name, artist)]
                df.loc[idx, 'Actual_BPM'] = bpm
                df.loc[idx, 'Music_Key'] = key
                found += 1
                print(f" -> {bpm} (known)")
            else:
                bpm, key = search_with_retry(name, artist)
                if bpm:
                    df.loc[idx, 'Actual_BPM'] = bpm
                    found += 1
                if key:
                    df.loc[idx, 'Music_Key'] = key
                print(f" -> {bpm or 'miss'}")

            processed += 1
            df.to_excel(output_file, index=False)

        if batch_start + BATCH_SIZE < len(indices):
            time.sleep(BATCH_DELAY)

except KeyboardInterrupt:
    print("\nInterrupted — saving progress...")
    df.to_excel(output_file, index=False)
except Exception as e:
    print(f"\nError: {e}")
    df.to_excel(output_file, index=False)

# update tempo
tempo_col = next((c for c in df.columns if c.lower() in ['estimated_tempo', 'tempo', 'bpm']), None)
if tempo_col:
    mask = df['Actual_BPM'].notna()
    df.loc[mask, tempo_col] = df.loc[mask, 'Actual_BPM']

ml_file = "spotify_playlist_ml_bpm_enhanced.xlsx"
ml_cols = [c for c in df.columns if 'preview' not in c.lower()]
df.to_excel(output_file, index=False)
df[ml_cols].to_excel(ml_file, index=False)

data_dir = spotify_dir / 'data' / 'processed'
data_dir.mkdir(parents=True, exist_ok=True)
df.to_excel(data_dir / output_file, index=False)
df[ml_cols].to_excel(data_dir / ml_file, index=False)

bpm_count = df['Actual_BPM'].notna().sum()
print(f"\nDone: {found} new, {bpm_count} total ({bpm_count/len(df)*100:.0f}%)")
