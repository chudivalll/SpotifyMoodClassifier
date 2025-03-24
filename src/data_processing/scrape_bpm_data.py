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

# Disable warnings about insecure requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

# Create columns for actual BPM and Key if they don't exist
if 'Actual_BPM' not in df.columns:
    df['Actual_BPM'] = None
    
if 'Music_Key' not in df.columns:
    df['Music_Key'] = None

# Print column names to check
print("\nColumn names in the dataset:")
for col in df.columns:
    print(f"- {col}")

# Identify the track name and artist columns based on available columns
track_name_col = None
artist_col = None

# Common variations of column names
track_name_variations = ['Track Name', 'track_name', 'Name', 'title', 'track', 'name']
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
    sys.exit(1)

if not artist_col:
    print("ERROR: Could not find a column for artist names. Please check your Excel file.")
    sys.exit(1)

print(f"Using '{track_name_col}' for track names and '{artist_col}' for artist names")

# Different user agents to rotate
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (iPad; CPU OS 8_4_1 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H321 Safari/600.1.4'
]

def extract_bpm_from_element(element_text):
    """Extract BPM from an element's text."""
    if not element_text:
        return None
    
    # Try to find a number followed by BPM (case insensitive)
    bpm_match = re.search(r'(\d+)\s*(?:BPM|bpm)', element_text)
    if bpm_match:
        return int(bpm_match.group(1))
    
    # If that fails, just look for a standalone number
    # (This is risky but might work in context)
    number_match = re.search(r'(\d+)', element_text)
    if number_match:
        return int(number_match.group(1))
    
    return None

def safe_sleep(seconds):
    """Safely sleep with exception handling."""
    try:
        time.sleep(seconds)
    except Exception as e:
        print(f"Warning: Sleep operation failed: {str(e)}")

def get_bpm_for_track(track_name, artist_name):
    """
    Scrapes SongData.io to get the BPM and key for a track
    Updated to match the HTML structure seen in the screenshot
    """
    # Format the search query
    search_query = f"{track_name} {artist_name}"
    encoded_query = urllib.parse.quote(search_query)
    url = f"https://songdata.io/search?q={encoded_query}"
    
    # Select a random user agent
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://songdata.io/'
    }
    
    try:
        # Add a small delay to be respectful to the server
        safe_sleep(random.uniform(2.0, 4.0))
        
        # Print the URL we're trying to access for debugging
        print(f"Accessing URL: {url}")
        
        # Make the request with a timeout
        response = requests.get(url, headers=headers, verify=False, timeout=15)
        
        # Check if we got a successful response
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # For debugging - uncomment to save the HTML response
            # debug_filename = f'songdata_debug_{track_name.replace(" ", "_")[:30]}.html'
            # with open(debug_filename, 'w', encoding='utf-8') as f:
            #     f.write(response.text)
            #     print(f"Saved HTML response to {debug_filename}")
            
            # Various approaches to extract BPM
            bpm = None
            key = None
            
            # Get all the text from the page for a simple search first
            all_text = soup.get_text()
            
            # DIRECT APPROACH - Look for patterns like "125 BPM" in all text
            bpm_pattern = re.compile(r'(\d+)\s*BPM', re.IGNORECASE)
            bpm_matches = bpm_pattern.findall(all_text)
            if bpm_matches:
                bpm = int(bpm_matches[0])
                print(f"Found BPM directly in text: {bpm}")
            
            # Look for key pattern like "Eb Minor" or "A Major"
            key_pattern = re.compile(r'([A-G][#♯♭b]?\s+(Major|Minor))', re.IGNORECASE)
            key_matches = key_pattern.findall(all_text)
            if key_matches:
                key = key_matches[0][0]
                print(f"Found Key directly in text: {key}")
            
            # If direct approach worked, return the results
            if bpm or key:
                return bpm, key
            
            # STRUCTURAL APPROACH 1: Target specific HTML elements based on screenshots
            # Look for elements with specific styling (as seen in the screenshots)
            dd_elements = soup.select('dd')
            for dd in dd_elements:
                # Check for elements that might contain BPM values
                if dd.get('style') and ('font-size' in dd.get('style') or 'font-weight' in dd.get('style')):
                    extracted_bpm = extract_bpm_from_element(dd.text)
                    if extracted_bpm:
                        bpm = extracted_bpm
                        break
            
            # Similarly, look for key elements
            if bpm:  # If we found BPM, look for key nearby
                dt_elements = soup.find_all('dt')
                for dt in dt_elements:
                    if 'Key' in dt.text:
                        next_dd = dt.find_next('dd')
                        if next_dd:
                            key = next_dd.text.strip()
                            break
            
            # STRUCTURAL APPROACH 2: Look for DL/DT elements with BPM and Key
            if not bpm:
                dl_elements = soup.select('dl')
                for dl in dl_elements:
                    dt_elements = dl.select('dt')
                    for dt in dt_elements:
                        if 'BPM' in dt.text:
                            dd = dt.find_next('dd')
                            if dd:
                                bpm_text = dd.text.strip()
                                extracted_bpm = extract_bpm_from_element(bpm_text)
                                if extracted_bpm:
                                    bpm = extracted_bpm
                        elif 'Key' in dt.text:
                            dd = dt.find_next('dd')
                            if dd:
                                key = dd.text.strip()
            
            # BROADER SEARCH: Look for any elements with BPM-related text
            if not bpm:
                # Find all elements containing "BPM"
                bpm_containers = soup.find_all(string=lambda text: text and "BPM" in text)
                for container in bpm_containers:
                    parent = container.parent
                    # Look at the text content and nearby elements
                    container_text = parent.get_text() if parent else container
                    extracted_bpm = extract_bpm_from_element(container_text)
                    if extracted_bpm:
                        bpm = extracted_bpm
                        break
            
            # Check if we found some key/BPM data
            if bpm or key:
                return bpm, key
            
            # No data found
            print(f"No BPM or key data found for track: {track_name}")
            return None, None
            
        else:
            print(f"Error: Status code {response.status_code} for {track_name}")
            if response.status_code == 403:
                print("Received 403 Forbidden - potential anti-scraping measures")
            elif response.status_code == 429:
                print("Received 429 Too Many Requests - we're being rate limited")
            return None, None
    
    except requests.exceptions.Timeout:
        print(f"Request timed out for {track_name}")
        return None, None
    except requests.exceptions.ConnectionError:
        print(f"Connection error for {track_name}")
        return None, None
    except Exception as e:
        print(f"Error fetching data for {track_name}: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return None, None


def search_song_with_retry(track_name, artist_name, max_retries=3, delay=5):
    """
    Attempts to search for a song with retries in case of failure
    """
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}/{max_retries} for {track_name} by {artist_name}")
            bpm, key = get_bpm_for_track(track_name, artist_name)
            
            if bpm is not None or key is not None:
                return bpm, key
            
            if attempt < max_retries - 1:
                retry_delay = delay * (attempt + 1)  # Increase delay with each retry
                print(f"Retrying {track_name} by {artist_name} in {retry_delay} seconds...")
                safe_sleep(retry_delay)
        except Exception as e:
            print(f"Error during retry {attempt+1} for {track_name}: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")
            if attempt < max_retries - 1:
                safe_sleep(delay * (attempt + 1))
    
    print(f"All retries failed for {track_name} by {artist_name}")
    return None, None


# Function to create batches
def batch_generator(iterable, batch_size=10):
    length = len(iterable)
    for i in range(0, length, batch_size):
        yield iterable[i:min(i + batch_size, length)]


# Process the tracks in batches with a delay between batches
batch_size = 3  # Smaller batch size to avoid rate limiting
delay_between_batches = 15  # Longer delay between batches
total_tracks = len(df)
tracks_processed = 0
tracks_with_bpm = 0

# Add a manual BPM dictionary for known songs
# This serves as a fallback for songs that can't be scraped
known_bpm_dict = {
    ("Alive - Anyma Remix", "RÜFÜS DU SOL"): (125, "Eb Minor"),
    ("Stereophobia - Original Mix", "Dark Oscillators"): (140, "A Minor"),
    # Add more entries here as you find them
}

print(f"Starting to fetch BPM data for {total_tracks} tracks in batches of {batch_size}...")

# Get indices of rows that don't already have BPM data
indices_to_process = df[df['Actual_BPM'].isnull()].index.tolist()

try:
    for batch_indices in batch_generator(indices_to_process, batch_size):
        print(f"\nProcessing batch of {len(batch_indices)} tracks...")
        
        for idx in batch_indices:
            track_name = df.loc[idx, track_name_col]
            artist_name = df.loc[idx, artist_col]
            
            print(f"\nSearching for: {track_name} by {artist_name}")
            
            # First check if we have it in our known BPM dictionary
            key = (track_name, artist_name)
            if key in known_bpm_dict:
                bpm, key_value = known_bpm_dict[key]
                df.loc[idx, 'Actual_BPM'] = bpm
                df.loc[idx, 'Music_Key'] = key_value
                tracks_with_bpm += 1
                print(f"Found in known dictionary: BPM={bpm}, Key={key_value}")
            else:
                # Try scraping if not in dictionary
                bpm, key = search_song_with_retry(track_name, artist_name)
                
                if bpm is not None:
                    df.loc[idx, 'Actual_BPM'] = bpm
                    tracks_with_bpm += 1
                    print(f"Found BPM: {bpm}")
                else:
                    print(f"No BPM data found")
                    
                if key is not None:
                    df.loc[idx, 'Music_Key'] = key
                    print(f"Found Key: {key}")
                else:
                    print(f"No Key data found")
            
            tracks_processed += 1
            
            # Save progress after each track to avoid losing data
            df.to_excel(output_file, index=False)
            print(f"Progress saved: {tracks_processed}/{total_tracks} tracks processed")
        
        # Save after each batch
        df.to_excel(output_file, index=False)
        print(f"Batch complete. Progress: {tracks_processed}/{total_tracks} tracks processed, {tracks_with_bpm} with BPM data")
        
        if batch_indices != indices_to_process[-len(batch_indices):]:
            print(f"Waiting {delay_between_batches} seconds before next batch...")
            safe_sleep(delay_between_batches)

except KeyboardInterrupt:
    print("Process interrupted by user. Saving current progress...")
    df.to_excel(output_file, index=False)
    print(f"Progress saved: {tracks_processed}/{total_tracks} tracks processed")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    print(f"Error details: {traceback.format_exc()}")
    print("Saving current progress...")
    df.to_excel(output_file, index=False)
    print(f"Progress saved: {tracks_processed}/{total_tracks} tracks processed")

# Find the tempo/BPM column in our data
tempo_col = None
for col in df.columns:
    if col.lower() in ['estimated_tempo', 'tempo', 'bpm']:
        tempo_col = col
        break

# Update our tempo with actual BPM where available
if tempo_col:
    print(f"Using '{tempo_col}' as the tempo column to update with actual BPM values")
    updated_tempo = 0
    for idx in df.index:
        if pd.notnull(df.loc[idx, 'Actual_BPM']):
            df.loc[idx, tempo_col] = df.loc[idx, 'Actual_BPM']
            updated_tempo += 1
    print(f"Updated {updated_tempo} tracks with actual BPM values")
else:
    print("No existing tempo/BPM column found to update")

# Create ML-ready version (without preview URLs)
ml_output_file = "spotify_playlist_ml_bpm_enhanced.xlsx"
preview_cols = [col for col in df.columns if 'preview' in col.lower()]
ml_columns = [col for col in df.columns if col not in preview_cols]
df_ml = df[ml_columns]

# Save the data
print(f"Saving results to {output_file} and {ml_output_file}...")
df.to_excel(output_file, index=False)
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
print("Please respect the website's robots.txt and usage policies.") 