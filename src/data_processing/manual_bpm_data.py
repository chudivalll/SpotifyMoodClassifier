import os
import pandas as pd
from pathlib import Path

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

# Dictionary of known BPM values (lowercase keys for case-insensitive matching)
# Format: {(track_name_lowercase, artist_name_lowercase): (bpm, key)}
known_bpm_data = {
    # Song we already verified
    ("i'll erase away your pain", "the whatnauts"): (121, "Ab Major"),
    
    # Popular songs with known BPM values from SongData.io
    ("blinding lights", "the weeknd"): (171, "C# Minor"),
    ("don't start now", "dua lipa"): (124, "Bb Minor"),
    ("watermelon sugar", "harry styles"): (95, "C Major"),
    ("bad guy", "billie eilish"): (135, "G Minor"),
    ("stay", "the kid laroi"): (170, "Bb Minor"),
    ("levitating", "dua lipa"): (103, "B Minor"),
    ("save your tears", "the weeknd"): (118, "C Major"),
    ("montero (call me by your name)", "lil nas x"): (179, "G Major"),
    ("peaches", "justin bieber"): (90, "F# Major"),
    ("dynamite", "bts"): (114, "B Major"),
    ("heat waves", "glass animals"): (81, "E Minor"),
    ("good 4 u", "olivia rodrigo"): (166, "A Minor"),
    ("drivers license", "olivia rodrigo"): (144, "F# Major"),
    ("industry baby", "lil nas x"): (150, "F Minor"),
    ("positions", "ariana grande"): (144, "D Minor"),
    ("mood", "24kgoldn"): (91, "C# Minor"),
    ("without you", "the kid laroi"): (92, "F# Minor"),
    ("leave the door open", "silk sonic"): (148, "Bb Major"),
    ("we're good", "dua lipa"): (124, "Bb Major"),
    
    # Classic songs
    ("bohemian rhapsody", "queen"): (72, "Bb Major"),
    ("hotel california", "eagles"): (75, "B Minor"),
    ("billie jean", "michael jackson"): (117, "F# Minor"),
    ("sweet child o' mine", "guns n' roses"): (126, "D Major"),
    ("stairway to heaven", "led zeppelin"): (82, "A Minor"),
    ("imagine", "john lennon"): (75, "C Major"),
    ("purple haze", "jimi hendrix"): (108, "E Major"),
    ("smells like teen spirit", "nirvana"): (117, "F Minor"),
    ("hey jude", "the beatles"): (74, "F Major"),
    ("like a rolling stone", "bob dylan"): (96, "C Major"),
    
    # Hip hop & R&B
    ("in da club", "50 cent"): (90, "F# Minor"),
    ("hotline bling", "drake"): (135, "D Minor"),
    ("love the way you lie", "eminem"): (87, "G Minor"),
    ("empire state of mind", "jay-z"): (173, "F# Minor"),
    ("alright", "kendrick lamar"): (110, "F Minor"),
    ("god's plan", "drake"): (77, "A Minor"),
    ("humble", "kendrick lamar"): (150, "F# Minor"),
    ("sicko mode", "travis scott"): (155, "F Minor"),
    ("no role modelz", "j. cole"): (100, "F# Minor"),
    ("goosebumps", "travis scott"): (130, "C# Minor"),
    
    # Dance/Electronic
    ("levels", "avicii"): (126, "C# Minor"),
    ("one more time", "daft punk"): (123, "D Major"),
    ("strobe", "deadmau5"): (128, "Bb Minor"),
    ("animals", "martin garrix"): (128, "F# Minor"),
    ("titanium", "david guetta"): (126, "Eb Major"),
    ("dont you worry child", "swedish house mafia"): (129, "G Major"),
    ("lean on", "major lazer"): (98, "G Major"),
    ("closer", "the chainsmokers"): (95, "Ab Major"),
    ("scary monsters and nice sprites", "skrillex"): (140, "A Minor"),
    ("wake me up", "avicii"): (124, "B Minor"),
    
    # Alternative/Rock
    ("creep", "radiohead"): (92, "G Major"),
    ("seven nation army", "the white stripes"): (124, "E Minor"),
    ("somebody that i used to know", "gotye"): (129, "D Major"),
    ("take me to church", "hozier"): (129, "E Minor"),
    ("viva la vida", "coldplay"): (138, "Ab Major"),
    ("do i wanna know?", "arctic monkeys"): (85, "G Minor"),
    ("radioactive", "imagine dragons"): (136, "B Minor"),
    ("sex on fire", "kings of leon"): (153, "A Major"),
    ("the less i know the better", "tame impala"): (117, "Eb Major"),
    ("high and dry", "radiohead"): (92, "F# Major")
}

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
    exit(1)

if not artist_col:
    print("ERROR: Could not find a column for artist names. Please check your Excel file.")
    exit(1)

print(f"Using '{track_name_col}' for track names and '{artist_col}' for artist names")

# Add the known BPM data to the dataset
matched_tracks = 0

for idx, row in df.iterrows():
    track_name = str(row[track_name_col]).lower()
    artist_name = str(row[artist_col]).lower()
    
    # Try exact match
    key = (track_name, artist_name)
    if key in known_bpm_data:
        bpm, key_value = known_bpm_data[key]
        df.at[idx, 'Actual_BPM'] = bpm
        df.at[idx, 'Music_Key'] = key_value
        matched_tracks += 1
        print(f"Found exact match for: {row[track_name_col]} by {row[artist_col]} - BPM: {bpm}, Key: {key_value}")
    else:
        # Try partial matching for tracks where the name might be slightly different
        for known_track, known_artist in known_bpm_data.keys():
            # Check if the known track name is contained in the current track name or vice versa
            track_match = known_track in track_name or track_name in known_track
            # Check if the known artist is the same or similar
            artist_match = known_artist in artist_name or artist_name in known_artist
            
            if track_match and artist_match:
                bpm, key_value = known_bpm_data[(known_track, known_artist)]
                df.at[idx, 'Actual_BPM'] = bpm
                df.at[idx, 'Music_Key'] = key_value
                matched_tracks += 1
                print(f"Found fuzzy match for: {row[track_name_col]} by {row[artist_col]} - BPM: {bpm}, Key: {key_value}")
                break

# Find the tempo/BPM column in our data
tempo_col = None
for col in df.columns:
    if col in ['Estimated_Tempo', 'Tempo', 'tempo', 'bpm', 'BPM']:
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
print(f"Successfully added BPM data for {matched_tracks} out of {len(df)} tracks")

# Print summary of BPM data
actual_bpm_count = df['Actual_BPM'].notnull().sum()
print(f"\nBPM Data Summary:")
print(f"- Tracks with actual BPM data: {actual_bpm_count} ({actual_bpm_count/len(df)*100:.1f}%)")
if actual_bpm_count > 0:
    print(f"- Mean actual BPM: {df['Actual_BPM'].mean():.1f}")
    print(f"- Min actual BPM: {df['Actual_BPM'].min()}")
    print(f"- Max actual BPM: {df['Actual_BPM'].max()}")

print("\nYou can manually add more tracks to the 'known_bpm_data' dictionary in this script")
print("by searching for them on SongData.io and adding the entries.") 