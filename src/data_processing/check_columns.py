import os
import pandas as pd
from pathlib import Path

# Change to the Spotify directory
spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))
print(f"Changed working directory to: {os.getcwd()}")

# Load the dataset
input_file = "spotify_playlist_complete.xlsx"

print(f"Reading data from {input_file}...")
df = pd.read_excel(input_file)
print(f"Found {len(df)} tracks in the dataset")

# Print column names
print("\nColumn names in the dataset:")
for i, column in enumerate(df.columns):
    print(f"{i+1}. {column}")

# Print a few rows to see the data structure
print("\nFirst two rows of data:")
print(df.head(2)) 