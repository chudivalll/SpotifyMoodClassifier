# spotify_data_processor.py
import pandas as pd
import os
import sys
import numpy as np

# SageMaker paths
input_path = '/opt/ml/processing/input/'
output_path = '/opt/ml/processing/output/'

# Load data
print("Loading data from", input_path)
df = pd.read_excel(os.path.join(input_path, 'spotify_playlist_complete.xlsx'))
print(f"Loaded {len(df)} tracks")

# Identify track and artist columns
track_column = None
artist_column = None

for col in df.columns:
    if col.lower() in ['track name', 'track_name', 'name', 'title', 'track']:
        track_column = col
    if col.lower() in ['artist', 'artist_name', 'artists']:
        artist_column = col

print(f"Identified track column: {track_column}, artist column: {artist_column}")

# Preprocess features
# Create Actual_BPM and Music_Key columns if they don't exist
if 'Actual_BPM' not in df.columns:
    df['Actual_BPM'] = np.nan
if 'Music_Key' not in df.columns:
    df['Music_Key'] = ''

print("Created BPM and Key columns if needed")

# Fill missing BPM values using Spotify's tempo feature
if 'tempo' in df.columns and 'Actual_BPM' in df.columns:
    df.loc[df['Actual_BPM'].isna(), 'Actual_BPM'] = df.loc[df['Actual_BPM'].isna(), 'tempo']
    print(f"Filled missing BPM values with tempo values")

# Print column statistics
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print("\nNumeric Column Statistics:")
for col in numeric_columns:
    print(f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}, null={df[col].isnull().sum()}")

# Prepare data for ML
ml_df = df.copy()
if 'preview_url' in ml_df.columns:
    ml_df.drop(columns=['preview_url'], inplace=True)
    print("Removed preview_url column for ML dataset")

# Save processed data
output_file = os.path.join(output_path, 'spotify_playlist_processed.csv')
ml_output_file = os.path.join(output_path, 'spotify_playlist_ml_ready.csv')

print(f"Saving processed data to {output_file}")
df.to_csv(output_file, index=False)

print(f"Saving ML-ready data to {ml_output_file}")
ml_df.to_csv(ml_output_file, index=False)

print(f"Processing complete. Processed {len(df)} tracks.") 