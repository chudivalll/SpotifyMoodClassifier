import sys
from pathlib import Path
import time
from spotipy.exceptions import SpotifyException

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection.spotify_ML import create_spotify_client

PLAYLIST_ID = "6jgCEkpKSc7LQI8ZWdAVr6"


def test_connection():
    sp = create_spotify_client()
    time.sleep(2)

    try:
        playlist = sp.playlist(PLAYLIST_ID, fields='name,tracks.total')
        print(f"Connected: {playlist['name']} ({playlist['tracks']['total']} tracks)")

        time.sleep(3)

        tracks = sp.playlist_tracks(PLAYLIST_ID, limit=3, offset=0,
                                    fields='items.track.name,items.track.artists.name')

        print("Sample tracks:")
        for item in tracks['items']:
            t = item['track']
            artists = ", ".join([a['name'] for a in t['artists']])
            print(f"  {t['name']} — {artists}")

        return True

    except SpotifyException as e:
        print(f"API error: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    print(f"\n{'Passed' if success else 'Failed'}")
