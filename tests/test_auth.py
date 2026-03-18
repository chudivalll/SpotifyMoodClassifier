import sys
import os
from pathlib import Path
import time

os.chdir(str(Path(__file__).parent.parent))
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')


def test_basic_auth():
    """Verify client credentials auth works and can fetch audio features."""
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Missing credentials in .env")
        return False

    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"

    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    try:
        results = sp.search(q='rock classics', type='track', limit=20)
        print(f"Search: {len(results['tracks']['items'])} tracks found")

        track_ids = [t['id'] for t in results['tracks']['items']]
        features = sp.audio_features(track_ids)

        if features and any(features):
            valid = [f for f in features if f]
            print(f"Audio features: {len(valid)}/{len(track_ids)} tracks")

            f = features[0]
            t = results['tracks']['items'][0]
            print(f"  {t['name']} — dance={f['danceability']}, energy={f['energy']}, tempo={f['tempo']}")
            return True
        else:
            print("No audio features returned")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_batch_features():
    """Test fetching features in batches with rate limit handling."""
    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"

    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    results = sp.search(q='rock', type='track', limit=50)
    track_ids = [t['id'] for t in results['tracks']['items']]

    all_features = []
    batch_size = 20

    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        if i > 0:
            time.sleep(3)
        try:
            features = sp.audio_features(batch)
            if features:
                all_features.extend([f for f in features if f])
        except Exception as e:
            print(f"Batch error: {e}")
            time.sleep(5)

    print(f"Batch test: {len(all_features)}/{len(track_ids)} features retrieved")
    return len(all_features) > 0


if __name__ == "__main__":
    if test_basic_auth():
        print("\nBasic auth OK, testing batch retrieval...")
        test_batch_features()
    else:
        print("\nAuth failed — check .env credentials")
