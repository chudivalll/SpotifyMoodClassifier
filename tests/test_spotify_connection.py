import sys
import os
from pathlib import Path
import time
from spotipy.exceptions import SpotifyException

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_collection.spotify_ML import sp

def test_connection():
    try:
        # Your playlist ID
        playlist_id = "6jgCEkpKSc7LQI8ZWdAVr6"
        
        print("Starting basic connection test...")
        print("Waiting 2 seconds before first request...")
        time.sleep(2)  # Initial delay
        
        try:
            # Just get basic playlist info first
            playlist = sp.playlist(playlist_id, fields='name,tracks.total')
            print(f"Successfully connected!")
            print(f"Playlist name: {playlist['name']}")
            print(f"Total tracks: {playlist['tracks']['total']}")
            
            # Wait before next request
            print("\nWaiting 3 seconds before fetching sample tracks...")
            time.sleep(3)
            
            # Get just 3 tracks as a sample
            tracks = sp.playlist_tracks(
                playlist_id,
                limit=3,
                offset=0,
                fields='items.track.name,items.track.artists.name'
            )
            
            print("\nSuccessfully retrieved 3 sample tracks:")
            for item in tracks['items']:
                track = item['track']
                artists = ", ".join([artist['name'] for artist in track['artists']])
                print(f"- {track['name']} by {artists}")
            
            return True
            
        except SpotifyException as e:
            print(f"Spotify API error: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Spotify API connection test...\n")
    success = test_connection()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!") 