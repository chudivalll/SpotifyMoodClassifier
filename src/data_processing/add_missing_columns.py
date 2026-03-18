"""
Estimate missing audio features from track metadata (genre, popularity, etc.)
when Spotify's audio features endpoint is unavailable.

Can be run standalone or imported as a module.
"""
import os
import pandas as pd
from pathlib import Path


def estimate_audio_features(df):
    """Add estimated audio feature columns to a DataFrame.

    Expects columns: popularity (0-100).
    Optional: primary_genre (or Primary Genre), release_date (or Release Date), explicit.
    Returns the DataFrame with tempo, danceability, energy, valence,
    acousticness, speechiness, instrumentalness columns added.
    """
    df = df.copy()

    # normalize column names — accept both formats
    col_map = {}
    for col in df.columns:
        col_map[col.lower().replace(' ', '_')] = col

    def _get_col(name, default=None):
        """Find a column by normalized name."""
        if name in df.columns:
            return name
        normalized = name.lower().replace(' ', '_')
        return col_map.get(normalized, default)

    genre_col = _get_col('primary_genre') or _get_col('Primary Genre')
    pop_col = _get_col('popularity') or _get_col('Popularity')
    date_col = _get_col('release_date') or _get_col('Release Date')
    explicit_col = _get_col('explicit') or _get_col('Explicit')

    # derive release year if we have a date column
    if date_col and 'release_year' not in [c.lower() for c in df.columns]:
        df['release_year'] = df[date_col].astype(str).str[:4]
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(2020).astype(int)
    year_col = _get_col('release_year') or _get_col('Release Year')

    def _genre(row):
        if genre_col and pd.notna(row.get(genre_col)):
            return str(row[genre_col]).lower()
        return ''

    def _pop(row):
        if pop_col:
            return row.get(pop_col, 50) or 50
        return 50

    def _year(row):
        if year_col:
            v = row.get(year_col, 2020)
            try:
                return int(v) if pd.notna(v) else 2020
            except (ValueError, TypeError):
                return 2020
        return 2020

    def _explicit(row):
        if explicit_col:
            return bool(row.get(explicit_col, False))
        return False

    # only estimate columns that don't already exist
    if 'tempo' not in df.columns:
        def est_tempo(row):
            genre = _genre(row)
            if 'rap' in genre or 'hip hop' in genre:
                base = 95
            elif 'r&b' in genre:
                base = 90
            elif 'rock' in genre:
                base = 120
            elif 'pop' in genre:
                base = 110
            elif 'electronic' in genre:
                base = 128
            else:
                base = 100

            year = _year(row)
            era_adj = -5 if year < 1980 else (5 if year < 2000 else 10 if year >= 2010 else 0)
            pop_adj = (_pop(row) / 100 - 0.5) * 20
            return max(60, min(180, base + era_adj + pop_adj))

        df['tempo'] = df.apply(est_tempo, axis=1)

    if 'danceability' not in df.columns:
        def est_dance(row):
            tempo_factor = 1 - abs((row['tempo'] - 110) / 100)
            pop_factor = _pop(row) / 100
            return max(0.1, min(0.95, tempo_factor * 0.6 + pop_factor * 0.4))

        df['danceability'] = df.apply(est_dance, axis=1)

    if 'energy' not in df.columns:
        def est_energy(row):
            pop_factor = _pop(row) / 100
            explicit_bump = 0.1 if _explicit(row) else 0
            year_factor = (_year(row) - 1950) / 100
            return max(0.1, min(0.95, pop_factor * 0.4 + year_factor * 0.4 + explicit_bump))

        df['energy'] = df.apply(est_energy, axis=1)

    if 'valence' not in df.columns:
        def est_valence(row):
            genre = _genre(row)
            if 'pop' in genre:
                base = 0.7
            elif 'r&b' in genre:
                base = 0.6
            elif 'rock' in genre:
                base = 0.4
            else:
                base = 0.5
            adj = (_pop(row) / 100 - 0.5) * 0.2
            return max(0.1, min(0.95, base + adj))

        df['valence'] = df.apply(est_valence, axis=1)

    if 'acousticness' not in df.columns:
        df['acousticness'] = df['energy'].apply(lambda e: max(0.05, min(0.95, 1 - e)))

    if 'speechiness' not in df.columns:
        def est_speech(row):
            genre = _genre(row)
            if 'rap' in genre or 'hip hop' in genre:
                base = 0.7
            elif 'r&b' in genre:
                base = 0.3
            else:
                base = 0.1
            adj = (_pop(row) / 100 - 0.5) * 0.1
            return max(0.01, min(0.95, base + adj))

        df['speechiness'] = df.apply(est_speech, axis=1)

    if 'instrumentalness' not in df.columns:
        df['instrumentalness'] = df['speechiness'].apply(lambda s: max(0.01, min(0.95, 1 - s)))

    if 'loudness' not in df.columns:
        df['loudness'] = df['energy'].apply(lambda e: -15 + e * 12)

    return df


# standalone usage
if __name__ == "__main__":
    import sys

    spotify_dir = Path(__file__).parent.parent.parent
    os.chdir(str(spotify_dir))

    input_file = sys.argv[1] if len(sys.argv) > 1 else "spotify_playlist_ml_enhanced_amix.xlsx"
    output_file = "spotify_playlist_complete.xlsx"

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        print("Usage: python add_missing_columns.py [input_file.xlsx]")
        sys.exit(1)

    df = pd.read_excel(input_file)
    print(f"Loaded {len(df)} tracks from {input_file}")

    df = estimate_audio_features(df)

    df.to_excel(output_file, index=False)
    print(f"Saved to {output_file}")

    for col in ['tempo', 'danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness']:
        if col in df.columns:
            print(f"  {col}: {df[col].mean():.2f} avg [{df[col].min():.2f} - {df[col].max():.2f}]")

    print("\nNote: these are heuristic estimates, not actual Spotify audio features")
