import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
import urllib.parse
import re

spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))

input_file = "spotify_playlist_complete.xlsx"
output_file = "spotify_playlist_bpm_enhanced.xlsx"

df = pd.read_excel(input_file)
print(f"Loaded {len(df)} tracks from {input_file}")

# figure out which columns hold the track/artist names
TRACK_COL_NAMES = ['Track Name', 'track_name', 'Name', 'title', 'track']
ARTIST_COL_NAMES = ['Artist', 'artist_name', 'artist', 'artists']

track_col = next((v for v in TRACK_COL_NAMES if v in df.columns), None)
artist_col = next((v for v in ARTIST_COL_NAMES if v in df.columns), None)

if not track_col or not artist_col:
    print(f"ERROR: couldn't find track/artist columns. Available: {list(df.columns)}")
    exit(1)

if 'Actual_BPM' not in df.columns:
    df['Actual_BPM'] = None
if 'Music_Key' not in df.columns:
    df['Music_Key'] = None


def get_bpm_for_track(track_name, artist_name):
    """Scrape songdata.io for BPM and key info."""
    query = urllib.parse.quote(f"{track_name} {artist_name}")
    url = f"https://songdata.io/search?q={query}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            return None, None

        soup = BeautifulSoup(resp.text, 'html.parser')
        bpm, key = None, None

        # try dt/dd structure first
        track_info = soup.select_one('#track-info-pane')
        if track_info:
            for dt in track_info.select('dt'):
                dd = dt.find_next('dd')
                if not dd:
                    continue
                if 'BPM' in dt.text:
                    m = re.search(r'(\d+)', dd.text)
                    if m:
                        bpm = int(m.group(1))
                elif 'Key' in dt.text:
                    key = dd.text.strip()

        # fallback: scan the whole page for "123 BPM" pattern
        if not bpm:
            text = soup.get_text()
            m = re.search(r'(\d+)\s*BPM', text, re.IGNORECASE)
            if m:
                bpm = int(m.group(1))

        # look for key pattern like "Ab Major"
        if not key:
            text = soup.get_text()
            m = re.search(r'([A-G][b#]?)\s+(Major|Minor)', text, re.IGNORECASE)
            if m:
                key = m.group(0)

        return bpm, key

    except Exception as e:
        print(f"  scrape failed for {track_name}: {e}")
        return None, None


def search_with_retry(track_name, artist_name, retries=3, delay=2):
    for attempt in range(retries):
        bpm, key = get_bpm_for_track(track_name, artist_name)
        if bpm is not None:
            return bpm, key
        if attempt < retries - 1:
            time.sleep(delay)
    return None, None


# process tracks that don't have BPM yet
BATCH_SIZE = 5
BATCH_DELAY = 5
indices = df[df['Actual_BPM'].isnull()].index.tolist()
found = 0

print(f"Fetching BPM for {len(indices)} tracks...")

for batch_start in range(0, len(indices), BATCH_SIZE):
    batch = indices[batch_start:batch_start + BATCH_SIZE]

    for idx in batch:
        name = df.loc[idx, track_col]
        artist = df.loc[idx, artist_col]
        print(f"  {name} — {artist}", end="")

        bpm, key = search_with_retry(name, artist)
        if bpm:
            df.loc[idx, 'Actual_BPM'] = bpm
            df.loc[idx, 'Music_Key'] = key
            found += 1
            print(f" -> {bpm} BPM, {key}")
        else:
            print(" -> not found")

    df.to_excel(output_file, index=False)

    if batch_start + BATCH_SIZE < len(indices):
        time.sleep(BATCH_DELAY)

# update tempo column with actual BPM where we found it
tempo_col = next((c for c in df.columns if c in ['Estimated_Tempo', 'Tempo', 'tempo', 'bpm', 'BPM']), None)
if tempo_col:
    mask = df['Actual_BPM'].notna()
    df.loc[mask, tempo_col] = df.loc[mask, 'Actual_BPM']

# save ML version (no preview URLs)
ml_file = "spotify_playlist_ml_bpm_enhanced.xlsx"
ml_cols = [c for c in df.columns if 'preview' not in c.lower()]
df[ml_cols].to_excel(ml_file, index=False)

data_dir = spotify_dir / 'data' / 'processed'
data_dir.mkdir(parents=True, exist_ok=True)
df.to_excel(data_dir / output_file, index=False)
df[ml_cols].to_excel(data_dir / ml_file, index=False)

bpm_count = df['Actual_BPM'].notna().sum()
print(f"\nDone: {found} new BPM values found, {bpm_count} total ({bpm_count/len(df)*100:.0f}% coverage)")
if bpm_count > 0:
    print(f"  BPM range: {df['Actual_BPM'].min():.0f} - {df['Actual_BPM'].max():.0f}, avg {df['Actual_BPM'].mean():.0f}")
