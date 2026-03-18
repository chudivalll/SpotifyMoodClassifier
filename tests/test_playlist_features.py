"""Test client credentials flow: playlist tracks -> audio features."""
import sys
import os
from pathlib import Path
import time

os.chdir(str(Path(__file__).parent.parent))

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

PLAYLIST_ID = "6jgCEkpKSc7LQI8ZWdAVr6"


def test_playlist_to_features():
    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"

    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    results = sp.playlist_tracks(PLAYLIST_ID, limit=5)
    if not results or not results['items']:
        print("No tracks in playlist")
        return False

    for i, item in enumerate(results['items']):
        t = item['track']
        if t:
            print(f"  {i+1}. {t['name']} — {t['artists'][0]['name']}")

    track_ids = [item['track']['id'] for item in results['items'] if item['track']]
    time.sleep(1)

    features = sp.audio_features(track_ids)
    if features and any(features):
        valid = [f for f in features if f]
        print(f"\nGot features for {len(valid)}/{len(track_ids)} tracks")
        f = valid[0]
        print(f"  Sample: dance={f['danceability']}, energy={f['energy']}, tempo={f['tempo']}")
        return True

    print("No features returned")
    return False


if __name__ == "__main__":
    ok = test_playlist_to_features()
    print(f"\n{'Passed' if ok else 'Failed'}")
