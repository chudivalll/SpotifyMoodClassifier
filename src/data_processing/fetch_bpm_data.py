import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
import urllib.parse
import re

# Change to the Spotify directory
spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))
print(f"Changed working directory to: {os.getcwd()}")

# Load the dataset
input_file = "spotify_playlist_complete.xlsx"
output_file = "spotify_playlist_bpm_enhanced.xlsx"

print(f"Reading data from {input_file}...")
df = pd.read_excel(input_file)
print(f"Found {len(df)} tracks in the dataset")

# Print column names to check
print("\nColumn names in the dataset:")
for col in df.columns:
    print(f"- {col}")

# Identify the track name and artist columns based on available columns
track_name_col = None
artist_col = None

# Common variations of column names
track_name_variations = ['Track Name', 'track_name', 'Name', 'title', 'track']
artist_variations = ['Artist', 'artist_name', 'artist', 'artists']

# Find the track name column
for variation in track_name_variations:
    if variation in df.columns:
        track_name_col = variation
        break

# Find the artist column
for variation in artist_variations:
    if variation in df.columns:
        artist_col = variation
        break

if not track_name_col:
    print("ERROR: Could not find a column for track names. Please check your Excel file.")
    os._exit(1)

if not artist_col:
    print("ERROR: Could not find a column for artist names. Please check your Excel file.")
    os._exit(1)

print(f"Using '{track_name_col}' for track names and '{artist_col}' for artist names")

# Create columns for actual BPM and Key if they don't exist
if 'Actual_BPM' not in df.columns:
    df['Actual_BPM'] = None
    
if 'Music_Key' not in df.columns:
    df['Music_Key'] = None


def get_bpm_for_track(track_name, artist_name):
    """
    Scrapes SongData.io to get the BPM and key for a track
    Updated to match the actual HTML structure from the screenshot
    """
    # Format the search query
    search_query = f"{track_name} {artist_name}"
    encoded_query = urllib.parse.quote(search_query)
    url = f"https://songdata.io/search?q={encoded_query}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Save the HTML for debugging
            # with open('songdata_response.html', 'w', encoding='utf-8') as f:
            #     f.write(response.text)
            
            # Find the track info container
            track_info = soup.select_one('#track-info-pane')
            
            bpm = None
            key = None
            
            if track_info:
                # Look for BPM element - based on the screenshot, BPM is in a dt/dd element
                bpm_elements = track_info.select('dt')
                for dt in bpm_elements:
                    if 'BPM' in dt.text:
                        # Get the next dd element which contains the BPM value
                        dd = dt.find_next('dd')
                        if dd:
                            bpm_text = dd.text.strip()
                            # Extract only the number
                            bpm_match = re.search(r'(\d+)', bpm_text)
                            if bpm_match:
                                bpm = int(bpm_match.group(1))
                
                # Look for Key element
                key_elements = track_info.select('dt')
                for dt in key_elements:
                    if 'Key' in dt.text:
                        # Get the next dd element which contains the key value
                        dd = dt.find_next('dd')
                        if dd:
                            key = dd.text.strip()
            
            # Alternative approach: direct search for elements with specific text content
            if not bpm:
                # Try a more direct approach looking at div elements
                bpm_container = soup.find(string=re.compile('BPM', re.IGNORECASE))
                if bpm_container and bpm_container.parent:
                    bpm_element = bpm_container.parent.find_next('div')
                    if bpm_element:
                        bpm_text = bpm_element.text.strip()
                        bpm_match = re.search(r'(\d+)', bpm_text)
                        if bpm_match:
                            bpm = int(bpm_match.group(1))
            
            # Another approach - looking for elements by class attributes that might contain "bpm"
            if not bpm:
                for element in soup.select('[class*=bpm]'):
                    bpm_text = element.text.strip()
                    bpm_match = re.search(r'(\d+)', bpm_text)
                    if bpm_match:
                        bpm = int(bpm_match.group(1))
                        break
            
            # Another approach - directly parse the HTML structure seen in the screenshot
            if not bpm:
                # Look for a specific structure where BPM is displayed
                # In the screenshot, it shows BPM in a div with 'dl' and 'dt' structure
                dl_elements = soup.select('dl.card.grid.my-1.py-1')
                for dl in dl_elements:
                    dt_elements = dl.select('dt')
                    for dt in dt_elements:
                        if 'BPM' in dt.text:
                            dd = dt.find_next('dd')
                            if dd:
                                bpm_text = dd.text.strip()
                                bpm_match = re.search(r'(\d+)', bpm_text)
                                if bpm_match:
                                    bpm = int(bpm_match.group(1))
            
            # Also try to find by looking for "BPM" text nearby a number
            if not bpm:
                for element in soup.find_all(text=re.compile('BPM', re.IGNORECASE)):
                    parent = element.parent
                    for sibling in list(parent.next_siblings)[:3]:  # Check next few siblings
                        if sibling and sibling.string and sibling.string.strip().isdigit():
                            bpm = int(sibling.string.strip())
                            break
                    if bpm:
                        break
            
            # If keys not found by the dt/dd approach, try other methods
            if not key:
                key_container = soup.find(string=re.compile('Key', re.IGNORECASE))
                if key_container and key_container.parent:
                    key_element = key_container.parent.find_next('div')
                    if key_element:
                        key = key_element.text.strip()
            
            # Extract from screenshot-like structure
            # Looking at the screenshot, there's a section with "Key" and then "Ab Major"
            if not key:
                key_sections = soup.find_all('div', text=re.compile('Key', re.IGNORECASE))
                for section in key_sections:
                    next_element = section.find_next()
                    if next_element:
                        key = next_element.text.strip()
                        break
            
            # If we still haven't found the key, look for common music key formats (like "Ab Major")
            if not key:
                key_pattern = re.compile(r'([A-G][b#]?)\s+(Major|Minor)', re.IGNORECASE)
                for element in soup.find_all(text=key_pattern):
                    key = element.strip()
                    break
            
            return bpm, key
            
        return None, None
    
    except Exception as e:
        print(f"Error fetching data for {track_name}: {str(e)}")
        return None, None


def search_song_with_retry(track_name, artist_name, max_retries=3, delay=2):
    """
    Attempts to search for a song with retries in case of failure
    """
    for attempt in range(max_retries):
        bpm, key = get_bpm_for_track(track_name, artist_name)
        
        if bpm is not None:
            return bpm, key
        
        if attempt < max_retries - 1:
            print(f"Retrying {track_name} by {artist_name} in {delay} seconds...")
            time.sleep(delay)
    
    return None, None


# Function to create batches
def batch_generator(iterable, batch_size=10):
    length = len(iterable)
    for i in range(0, length, batch_size):
        yield iterable[i:min(i + batch_size, length)]


# Process the tracks in batches with a delay between batches
batch_size = 5  # Small batch size to avoid rate limiting
delay_between_batches = 5  # Seconds to wait between batches
total_tracks = len(df)
tracks_processed = 0
tracks_with_bpm = 0

print(f"Starting to fetch BPM data for {total_tracks} tracks in batches of {batch_size}...")

# Get indices of rows that don't already have BPM data
indices_to_process = df[df['Actual_BPM'].isnull()].index.tolist()

for batch_indices in batch_generator(indices_to_process, batch_size):
    print(f"\nProcessing batch of {len(batch_indices)} tracks...")
    
    for idx in batch_indices:
        track_name = df.loc[idx, track_name_col]
        artist_name = df.loc[idx, artist_col]
        
        print(f"Searching for: {track_name} by {artist_name}")
        bpm, key = search_song_with_retry(track_name, artist_name)
        
        if bpm is not None:
            df.loc[idx, 'Actual_BPM'] = bpm
            df.loc[idx, 'Music_Key'] = key
            tracks_with_bpm += 1
            print(f"Found: BPM={bpm}, Key={key}")
        else:
            print(f"No BPM data found")
        
        tracks_processed += 1
        
        # Save progress after each track
        if tracks_processed % 10 == 0:
            df.to_excel(output_file, index=False)
            print(f"Progress saved: {tracks_processed}/{total_tracks} tracks processed")
    
    # Save after each batch
    df.to_excel(output_file, index=False)
    print(f"Batch complete. Progress: {tracks_processed}/{total_tracks} tracks processed, {tracks_with_bpm} with BPM data")
    
    if batch_indices != indices_to_process[-len(batch_indices):]:
        print(f"Waiting {delay_between_batches} seconds before next batch...")
        time.sleep(delay_between_batches)

# Find the tempo/BPM column in our data
tempo_col = None
for col in df.columns:
    if col in ['Estimated_Tempo', 'Tempo', 'tempo', 'bpm', 'BPM']:
        tempo_col = col
        break

# Update our tempo with actual BPM where available
if tempo_col:
    print(f"Using '{tempo_col}' as the tempo column to update with actual BPM values")
    for idx in df.index:
        if pd.notnull(df.loc[idx, 'Actual_BPM']):
            df.loc[idx, tempo_col] = df.loc[idx, 'Actual_BPM']
else:
    print("No existing tempo/BPM column found to update")

# Create ML-ready version (without preview URLs)
ml_output_file = "spotify_playlist_ml_bpm_enhanced.xlsx"
preview_cols = [col for col in df.columns if 'preview' in col.lower()]
ml_columns = [col for col in df.columns if col not in preview_cols]
df_ml = df[ml_columns]
df_ml.to_excel(ml_output_file, index=False)

# Save to data/processed directory as well
data_dir = spotify_dir / 'data' / 'processed'
data_dir.mkdir(parents=True, exist_ok=True)

df.to_excel(data_dir / output_file, index=False)
df_ml.to_excel(data_dir / ml_output_file, index=False)

print(f"\nProcessing complete!")
print(f"Successfully retrieved BPM data for {tracks_with_bpm} out of {total_tracks} tracks")
print(f"Full dataset saved to {output_file}")
print(f"ML-ready dataset saved to {ml_output_file}")

# Print summary of BPM data
actual_bpm_count = df['Actual_BPM'].notnull().sum()
print(f"\nBPM Data Summary:")
print(f"- Tracks with actual BPM data: {actual_bpm_count} ({actual_bpm_count/len(df)*100:.1f}%)")
if actual_bpm_count > 0:
    print(f"- Mean actual BPM: {df['Actual_BPM'].mean():.1f}")
    print(f"- Min actual BPM: {df['Actual_BPM'].min()}")
    print(f"- Max actual BPM: {df['Actual_BPM'].max()}")

print("\nNOTE: This script performs web scraping which may be against the terms of service.")
print("Please respect the website's robots.txt and usage policies. Consider purchasing an API if available.") 