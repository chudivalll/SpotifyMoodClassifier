"""Test OAuth flow -> playlist tracks -> audio features pipeline."""
import os
import sys
from pathlib import Path
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

os.chdir(str(Path(__file__).parent.parent))
load_dotenv()

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Missing credentials in .env")

PLAYLIST_ID = "6jgCEkpKSc7LQI8ZWdAVr6"


def test_playlist_features():
    scope = 'playlist-read-private playlist-read-collaborative user-library-read'

    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri='http://127.0.0.1:8888/callback',
        scope=scope,
        cache_path='.spotify_cache',
        open_browser=False
    )

    token_info = auth_manager.get_cached_token()
    if not token_info:
        print("No cached token. Authenticate:")
        auth_url = auth_manager.get_authorize_url()
        print(f"\n{auth_url}\n")
        redirect_url = input("Paste redirect URL: ")
        code = auth_manager.parse_response_code(redirect_url)
        token_info = auth_manager.get_access_token(code)

    sp = spotipy.Spotify(auth_manager=auth_manager)

    info = sp.playlist(PLAYLIST_ID, fields='name,tracks.total')
    print(f"Playlist: {info['name']} ({info['tracks']['total']} tracks)")

    results = sp.playlist_tracks(PLAYLIST_ID, limit=3,
                                  fields='items.track.name,items.track.artists,items.track.id')
    tracks = []
    for item in results['items']:
        t = item['track']
        if t:
            artist = ", ".join([a['name'] for a in t['artists']])
            print(f"  {t['name']} — {artist}")
            tracks.append(t)

    if not tracks:
        print("No tracks found")
        return False

    time.sleep(1)
    ids = [t['id'] for t in tracks]
    features = sp.audio_features(ids)

    if not features or not any(features):
        print("No audio features returned")
        return False

    for i, f in enumerate(features):
        if f:
            print(f"  -> dance={f['danceability']}, energy={f['energy']}, tempo={f['tempo']}, valence={f['valence']}")

    return True


if __name__ == "__main__":
    ok = test_playlist_features()
    print(f"\n{'PASSED' if ok else 'FAILED'}")
