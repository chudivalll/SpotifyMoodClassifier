"""
Progressive test: search -> single track features -> batch features -> debug logging.
Helps narrow down where API access breaks.
"""
import sys
import os
from pathlib import Path
import time

os.chdir(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')


def _make_client():
    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"
    return spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())


def test_search():
    sp = _make_client()
    try:
        results = sp.search(q='rock', type='track', limit=5)
        tracks = results['tracks']['items']
        print(f"Search: {len(tracks)} results")
        for i, t in enumerate(tracks):
            print(f"  {i+1}. {t['name']} — {t['artists'][0]['name']}")
        return True, tracks
    except Exception as e:
        print(f"Search failed: {e}")
        return False, None


def test_single_features(track):
    sp = _make_client()
    try:
        features = sp.audio_features(track['id'])
        if features and features[0]:
            f = features[0]
            print(f"Single track features: dance={f['danceability']}, energy={f['energy']}, tempo={f['tempo']}")
            return True
        print("No features returned")
        return False
    except Exception as e:
        print(f"Single features failed: {e}")
        return False


def test_batch_features(tracks):
    sp = _make_client()
    try:
        ids = [t['id'] for t in tracks]
        features = sp.audio_features(ids)
        valid = [f for f in features if f] if features else []
        print(f"Batch features: {len(valid)}/{len(ids)}")
        return len(valid) > 0
    except Exception as e:
        print(f"Batch features failed: {e}")
        return False


def test_debug():
    """Last resort: enable logging and try a hardcoded track."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    sp = _make_client()
    try:
        features = sp.audio_features("6rqhFgbbKwnb9MLmUQDhG6")
        if features and features[0]:
            print("Debug test passed")
            return True
        return False
    except Exception as e:
        print(f"Debug test failed: {e}")
        return False


if __name__ == "__main__":
    ok, tracks = test_search()
    if not ok or not tracks:
        sys.exit(1)

    if test_single_features(tracks[0]):
        test_batch_features(tracks[:3])
    else:
        print("\nSingle track failed, trying debug mode...")
        test_debug()
