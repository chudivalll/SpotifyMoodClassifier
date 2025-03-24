import sys
import os
from pathlib import Path
import time

# First, change to the Spotify directory
os.chdir(str(Path(__file__).parent.parent))
print(f"Changed working directory to: {os.getcwd()}")

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

def test_playlist_tracks_to_features():
    """Test getting tracks from a playlist and then their audio features"""
    print("\n==== TESTING PLAYLIST TRACKS TO FEATURES ====")
    
    # Load environment variables
    env_path = Path('.env')
    print(f"Looking for .env file at: {env_path.absolute()}")
    load_dotenv(env_path)
    
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    print(f"Client ID present: {'Yes' if CLIENT_ID else 'No'}")
    print(f"Client Secret present: {'Yes' if CLIENT_SECRET else 'No'}")
    
    # Set environment variables
    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"
    
    # Create client
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Your playlist ID
    playlist_id = "6jgCEkpKSc7LQI8ZWdAVr6"
    
    try:
        # Step 1: Get tracks from the playlist
        print(f"\nFetching tracks from playlist ID: {playlist_id}")
        results = sp.playlist_tracks(playlist_id, limit=5)  # Just get first 5 tracks for testing
        
        if not results or not results['items']:
            print("ERROR: No tracks found in playlist")
            return False
            
        print(f"SUCCESS: Found {len(results['items'])} tracks in playlist")
        
        # Print track names for debugging
        for i, item in enumerate(results['items']):
            if item['track']:
                track = item['track']
                print(f"{i+1}. {track['name']} by {track['artists'][0]['name']} (ID: {track['id']})")
        
        # Step 2: Extract track IDs
        track_ids = []
        for item in results['items']:
            if item['track']:
                track_ids.append(item['track']['id'])
                
        print(f"\nExtracted {len(track_ids)} track IDs from playlist")
        
        # Step 3: Get audio features for these tracks
        print("\nFetching audio features for track IDs...")
        time.sleep(1)  # Brief pause before API call
        
        features = sp.audio_features(track_ids)
        
        if features and any(features):
            valid_features = [f for f in features if f]
            print(f"SUCCESS: Got audio features for {len(valid_features)}/{len(track_ids)} tracks")
            
            # Print sample features from first track
            if valid_features:
                print("\nSample features from first track:")
                print(f"- Danceability: {valid_features[0]['danceability']}")
                print(f"- Energy: {valid_features[0]['energy']}")
                print(f"- Tempo: {valid_features[0]['tempo']}")
                print(f"- Valence: {valid_features[0]['valence']}")
            
            return True
        else:
            print("ERROR: No features returned")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== TESTING SPOTIFY PLAYLIST TO FEATURES FLOW ===")
    success = test_playlist_tracks_to_features()
    print(f"\nTest {'successful' if success else 'failed'}") 