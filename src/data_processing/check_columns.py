import os
import pandas as pd
from pathlib import Path

spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))

df = pd.read_excel("spotify_playlist_complete.xlsx")
print(f"{len(df)} tracks\n")

print("Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nFirst 2 rows:")
print(df.head(2))
