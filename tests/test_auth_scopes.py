"""Test the full OAuth2 authorization code flow with manual browser redirect."""
import os
import sys
from pathlib import Path
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


def test_oauth_flow():
    scope = "user-read-private user-read-email playlist-read-private playlist-read-collaborative user-top-read user-library-read"
    cache_path = '.spotify_cache'

    if os.path.exists(cache_path):
        os.remove(cache_path)

    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri='http://127.0.0.1:8888/callback',
        scope=scope,
        cache_path=cache_path,
        show_dialog=True
    )

    auth_url = auth_manager.get_authorize_url()
    print(f"\nOpen this URL:\n{auth_url}\n")
    print("Paste the full redirect URL after authorizing:")
    redirect_url = input("> ")

    code = auth_manager.parse_response_code(redirect_url)
    token_info = auth_manager.get_access_token(code)

    if not token_info or 'access_token' not in token_info:
        print("Failed to get token")
        return False

    sp = spotipy.Spotify(auth_manager=auth_manager)

    # verify profile
    user = sp.current_user()
    print(f"Authenticated: {user['display_name']} ({user['id']})")

    # verify playlist access
    playlist = sp.playlist(PLAYLIST_ID, fields="name")
    print(f"Playlist: {playlist['name']}")

    # verify audio features
    track_id = "5HNCy40Ni5BZJFw1TKzRsC"
    features = sp.audio_features(track_id)

    if features and features[0]:
        f = features[0]
        print(f"Audio features OK: dance={f['danceability']}, energy={f['energy']}, tempo={f['tempo']}")

        # try playlist tracks too
        results = sp.playlist_tracks(PLAYLIST_ID, limit=2)
        ids = [item['track']['id'] for item in results['items'] if item['track']]
        playlist_features = sp.audio_features(ids)

        if playlist_features and any(playlist_features):
            print(f"Playlist audio features: {sum(1 for f in playlist_features if f)}/{len(ids)} OK")
            return True

    print("Audio features not available")
    return False


if __name__ == "__main__":
    ok = test_oauth_flow()
    print(f"\n{'PASSED' if ok else 'FAILED'}")
