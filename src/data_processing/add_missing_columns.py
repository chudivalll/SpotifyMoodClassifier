import os
import pandas as pd
from pathlib import Path

spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))

input_file = "spotify_playlist_ml_enhanced_amix.xlsx"
output_file = "spotify_playlist_complete.xlsx"

print(f"Reading {input_file}...")
df = pd.read_excel(input_file)
print(f"{len(df)} tracks loaded")


# rough heuristics to fill in missing audio features when the API won't give them to us

def estimate_tempo(row):
    genre = str(row['Primary Genre']).lower()
    year = int(row['Release Year']) if pd.notna(row['Release Year']) and str(row['Release Year']).isdigit() else 2020

    if 'rap' in genre or 'hip hop' in genre:
        base = 95
    elif 'r&b' in genre:
        base = 90
    elif 'rock' in genre:
        base = 120
    elif 'pop' in genre:
        base = 110
    else:
        base = 100

    # newer songs trend slightly faster
    era_adj = 0
    if year < 1980: era_adj = -5
    elif year < 2000: era_adj = 5
    elif year >= 2010: era_adj = 10

    pop_adj = (row['Popularity'] / 100 - 0.5) * 20
    return max(60, min(180, base + era_adj + pop_adj))


def estimate_danceability(row):
    # songs near 100-130 BPM tend to be more danceable
    tempo_factor = 1 - abs((row['Estimated_Tempo'] - 110) / 100)
    pop_factor = row['Popularity'] / 100
    return max(0.1, min(0.95, tempo_factor * 0.6 + pop_factor * 0.4))


def estimate_energy(row):
    pop_factor = row['Popularity'] / 100
    explicit_bump = 0.1 if row['Explicit'] else 0
    year = int(row['Release Year']) if pd.notna(row['Release Year']) and str(row['Release Year']).isdigit() else 2020
    year_factor = (year - 1950) / 100
    return max(0.1, min(0.95, pop_factor * 0.4 + year_factor * 0.4 + explicit_bump))


def estimate_valence(row):
    genre = str(row['Primary Genre']).lower()
    if 'pop' in genre:
        base = 0.7
    elif 'r&b' in genre:
        base = 0.6
    elif 'rock' in genre:
        base = 0.4
    else:
        base = 0.5
    adj = (row['Popularity'] / 100 - 0.5) * 0.2
    return max(0.1, min(0.95, base + adj))


def estimate_acousticness(row):
    return max(0.05, min(0.95, 1 - row['Estimated_Energy']))


def estimate_speechiness(row):
    genre = str(row['Primary Genre']).lower()
    if 'rap' in genre or 'hip hop' in genre:
        base = 0.7
    elif 'r&b' in genre:
        base = 0.3
    else:
        base = 0.1
    adj = (row['Popularity'] / 100 - 0.5) * 0.1
    return max(0.01, min(0.95, base + adj))


def estimate_instrumentalness(row):
    return max(0.01, min(0.95, 1 - row['Estimated_Speechiness']))


# apply estimates in dependency order
df['Estimated_Tempo'] = df.apply(estimate_tempo, axis=1)
df['Estimated_Danceability'] = df.apply(estimate_danceability, axis=1)
df['Estimated_Energy'] = df.apply(estimate_energy, axis=1)
df['Estimated_Valence'] = df.apply(estimate_valence, axis=1)
df['Estimated_Acousticness'] = df.apply(estimate_acousticness, axis=1)
df['Estimated_Speechiness'] = df.apply(estimate_speechiness, axis=1)
df['Estimated_Instrumentalness'] = df.apply(estimate_instrumentalness, axis=1)

df.to_excel(output_file, index=False)

ml_output = "spotify_playlist_ml_complete.xlsx"
df_ml = df[[c for c in df.columns if c != 'Preview URL']]
df_ml.to_excel(ml_output, index=False)

data_dir = spotify_dir / 'data' / 'processed'
data_dir.mkdir(parents=True, exist_ok=True)
df.to_excel(data_dir / output_file, index=False)
df_ml.to_excel(data_dir / ml_output, index=False)

print(f"Saved to {output_file} and {ml_output}")
print("\nEstimated feature ranges:")
for col in [c for c in df.columns if c.startswith('Estimated_')]:
    print(f"  {col}: {df[col].mean():.2f} avg, [{df[col].min():.2f} - {df[col].max():.2f}]")
print("\nNote: these are rough estimates, not actual Spotify audio features")
