import pandas as pd
import os
import numpy as np

input_path = '/opt/ml/processing/input/'
output_path = '/opt/ml/processing/output/'

df = pd.read_excel(os.path.join(input_path, 'spotify_playlist_complete.xlsx'))
print(f"Loaded {len(df)} tracks")

# find track/artist columns
track_col = next((c for c in df.columns if c.lower() in ['track name', 'track_name', 'name', 'title', 'track']), None)
artist_col = next((c for c in df.columns if c.lower() in ['artist', 'artist_name', 'artists']), None)
print(f"Track col: {track_col}, Artist col: {artist_col}")

if 'Actual_BPM' not in df.columns:
    df['Actual_BPM'] = np.nan
if 'Music_Key' not in df.columns:
    df['Music_Key'] = ''

# fill missing BPM from tempo where available
if 'tempo' in df.columns:
    mask = df['Actual_BPM'].isna()
    df.loc[mask, 'Actual_BPM'] = df.loc[mask, 'tempo']

# quick stats
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    print(f"  {col}: [{df[col].min():.1f}, {df[col].max():.1f}], avg={df[col].mean():.2f}, nulls={df[col].isnull().sum()}")

# ML version without preview URLs
ml_df = df.drop(columns=[c for c in df.columns if 'preview' in c.lower()], errors='ignore')

df.to_csv(os.path.join(output_path, 'spotify_playlist_processed.csv'), index=False)
ml_df.to_csv(os.path.join(output_path, 'spotify_playlist_ml_ready.csv'), index=False)
print(f"Saved {len(df)} processed tracks")
