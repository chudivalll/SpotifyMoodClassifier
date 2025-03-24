import os
import pandas as pd
from pathlib import Path

# Change to the Spotify directory
spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))
print(f"Changed working directory to: {os.getcwd()}")

# Read the existing data
input_file = "spotify_playlist_ml_enhanced_amix.xlsx"
output_file = "spotify_playlist_complete.xlsx"

print(f"Reading data from {input_file}...")
df = pd.read_excel(input_file)

print(f"Found {len(df)} tracks in the dataset")

# Create proxy/estimated columns for missing audio features
print("Adding estimated columns for missing audio features...")

# 1. Add BPM (Tempo) column - estimated from release year patterns
# Modern hip-hop tends to be 80-100 BPM, older songs might be different
def estimate_tempo(row):
    # This is a very rough estimate based on genre and era
    genre = row['Primary Genre']
    year = row['Release Year'] if not pd.isna(row['Release Year']) else 2020
    year = int(year) if year and str(year).isdigit() else 2020
    
    # Base BPM by genre (very approximate)
    if 'rap' in str(genre).lower() or 'hip hop' in str(genre).lower():
        base_bpm = 95
    elif 'r&b' in str(genre).lower():
        base_bpm = 90
    elif 'rock' in str(genre).lower():
        base_bpm = 120
    elif 'pop' in str(genre).lower():
        base_bpm = 110
    else:
        base_bpm = 100
    
    # Adjust by era (very approximate)
    if year < 1980:
        era_adjust = -5
    elif year < 1990:
        era_adjust = 0
    elif year < 2000:
        era_adjust = 5
    elif year < 2010:
        era_adjust = 8
    else:
        era_adjust = 10
        
    # Add some randomness based on popularity
    pop_factor = row['Popularity'] / 100
    random_factor = (pop_factor - 0.5) * 20
    
    # Combine all factors with some realistic constraints
    tempo = base_bpm + era_adjust + random_factor
    return max(60, min(180, tempo))  # Keep within reasonable BPM range

# 2. Add Danceability (0-1) - estimated from tempo and popularity
def estimate_danceability(row):
    tempo = row['Estimated_Tempo']
    # Songs around 100-130 BPM tend to be more danceable
    tempo_factor = 1 - abs((tempo - 110) / 100)
    
    # Popular songs tend to be more danceable
    pop_factor = row['Popularity'] / 100
    
    # Combine factors with reasonable constraints
    dance = (tempo_factor * 0.6) + (pop_factor * 0.4)
    return max(0.1, min(0.95, dance))

# 3. Add Energy (0-1) - based on popularity and loudness
def estimate_energy(row):
    # Popular songs often have higher energy
    pop_factor = row['Popularity'] / 100
    
    # Explicit songs might have more energy
    explicit_factor = 0.1 if row['Explicit'] else 0
    
    # Base on recency - newer songs tend to be more energetic
    year = row['Release Year'] if not pd.isna(row['Release Year']) else 2020
    year = int(year) if year and str(year).isdigit() else 2020
    year_factor = (year - 1950) / 100  # Normalize
    
    # Combine factors with reasonable constraints
    energy = (pop_factor * 0.4) + (year_factor * 0.4) + explicit_factor
    return max(0.1, min(0.95, energy))

# 4. Add Valence (0-1) - positivity/happiness
def estimate_valence(row):
    # This is highly subjective but we can make some guesses
    genre = row['Primary Genre']
    
    # Base valence by genre (very approximate)
    if 'rap' in str(genre).lower() or 'hip hop' in str(genre).lower():
        base_valence = 0.5  # Can be either positive or negative
    elif 'r&b' in str(genre).lower():
        base_valence = 0.6  # Often emotional but positive
    elif 'rock' in str(genre).lower():
        base_valence = 0.4  # Often more negative/intense
    elif 'pop' in str(genre).lower():
        base_valence = 0.7  # Pop tends to be more upbeat
    else:
        base_valence = 0.5  # Default
    
    # Add some randomness
    random_factor = (row['Popularity'] / 100 - 0.5) * 0.2
    
    # Combine with reasonable constraints
    valence = base_valence + random_factor
    return max(0.1, min(0.95, valence))

# 5. Add Acousticness (0-1) - inverse of energy for estimation
def estimate_acousticness(row):
    # Inverse of energy is a reasonable proxy (not perfect)
    energy = row['Estimated_Energy']
    return max(0.05, min(0.95, 1 - energy))

# 6. Add Speechiness (0-1) - higher for rap/hip-hop
def estimate_speechiness(row):
    genre = row['Primary Genre']
    
    # Base speechiness by genre
    if 'rap' in str(genre).lower() or 'hip hop' in str(genre).lower():
        base_speech = 0.7  # Rap has high speechiness
    elif 'r&b' in str(genre).lower():
        base_speech = 0.3  # Medium speechiness
    else:
        base_speech = 0.1  # Most music has low speechiness
    
    # Add some randomness
    random_factor = (row['Popularity'] / 100 - 0.5) * 0.1
    
    return max(0.01, min(0.95, base_speech + random_factor))

# 7. Add Instrumentalness (0-1) - inverse of speechiness
def estimate_instrumentalness(row):
    # Mostly inverse of speechiness
    speech = row['Estimated_Speechiness']
    return max(0.01, min(0.95, 1 - speech))

# Add the estimated columns
print("Adding estimated Tempo...")
df['Estimated_Tempo'] = df.apply(estimate_tempo, axis=1)

print("Adding estimated Danceability...")
df['Estimated_Danceability'] = df.apply(estimate_danceability, axis=1)

print("Adding estimated Energy...")
df['Estimated_Energy'] = df.apply(estimate_energy, axis=1)

print("Adding estimated Valence...")
df['Estimated_Valence'] = df.apply(estimate_valence, axis=1)

print("Adding estimated Acousticness...")
df['Estimated_Acousticness'] = df.apply(estimate_acousticness, axis=1)

print("Adding estimated Speechiness...")
df['Estimated_Speechiness'] = df.apply(estimate_speechiness, axis=1)

print("Adding estimated Instrumentalness...")
df['Estimated_Instrumentalness'] = df.apply(estimate_instrumentalness, axis=1)

# Save the updated dataset
print(f"Saving complete dataset to {output_file}...")
df.to_excel(output_file, index=False)

# Create ML-ready version (without preview URLs)
ml_output_file = "spotify_playlist_ml_complete.xlsx"
ml_columns = [col for col in df.columns if col != 'Preview URL']
df_ml = df[ml_columns]
df_ml.to_excel(ml_output_file, index=False)

# Also save to data/processed directory
data_dir = spotify_dir / 'data' / 'processed'
data_dir.mkdir(parents=True, exist_ok=True)

df.to_excel(data_dir / output_file, index=False)
df_ml.to_excel(data_dir / ml_output_file, index=False)

print(f"\nProcessing complete!")
print(f"Full dataset saved to {output_file}")
print(f"ML-ready dataset saved to {ml_output_file}")

# Print summary of added columns
print("\nSummary of estimated audio features:")
for col in ['Estimated_Tempo', 'Estimated_Danceability', 'Estimated_Energy', 
            'Estimated_Valence', 'Estimated_Acousticness', 
            'Estimated_Speechiness', 'Estimated_Instrumentalness']:
    print(f"- {col}: mean={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")

print("\nIMPORTANT NOTE: These audio features are ESTIMATES based on available metadata")
print("They are approximations and not actual Spotify audio feature values")
print("Use these for exploratory analysis only") 