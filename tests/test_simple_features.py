"""Quick smoke test: can we get audio features via client credentials?"""
import os
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

os.chdir(str(Path(__file__).parent.parent))
load_dotenv()

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID, client_secret=CLIENT_SECRET
))

PLAYLIST_ID = "6jgCEkpKSc7LQI8ZWdAVr6"


def test_popular_track():
    """Shape of You — should always have audio features."""
    track_id = "5HNCy40Ni5BZJFw1TKzRsC"
    track = sp.track(track_id)
    print(f"Track: {track['name']} — {track['artists'][0]['name']}")

    features = sp.audio_features(track_id)
    if features and features[0]:
        f = features[0]
        print(f"  dance={f['danceability']}, energy={f['energy']}, tempo={f['tempo']}, valence={f['valence']}")
        return True
    print("  No features returned")
    return False


def test_playlist_tracks():
    """Test with a couple tracks from our playlist."""
    playlist = sp.playlist(PLAYLIST_ID, fields="name")
    print(f"\nPlaylist: {playlist['name']}")

    results = sp.playlist_tracks(PLAYLIST_ID, limit=2)
    track_ids = []
    track_names = []

    for item in results['items']:
        t = item['track']
        if t:
            name = f"{t['name']} — {t['artists'][0]['name']}"
            print(f"  {name}")
            track_ids.append(t['id'])
            track_names.append(name)

    features = sp.audio_features(track_ids)
    if features and any(features):
        for i, f in enumerate(features):
            if f:
                print(f"  {track_names[i]}: dance={f['danceability']}, energy={f['energy']}, tempo={f['tempo']}")
        return True

    print("  No features returned")
    return False


if __name__ == "__main__":
    r1 = test_popular_track()
    r2 = test_playlist_tracks()

    if r1 and r2:
        print("\nAll good")
    elif r1:
        print("\nPopular track OK but playlist tracks failed — might be a permissions issue")
    else:
        print("\nAudio features not accessible — check credentials or API restrictions")
