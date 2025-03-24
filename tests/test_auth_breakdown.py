import sys
import os
from pathlib import Path
import time

# First, change to the Spotify directory
os.chdir(str(Path(__file__).parent.parent))
print(f"Changed working directory to: {os.getcwd()}")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

def test_search_only():
    """Test just the search functionality"""
    print("\n==== TESTING SEARCH ONLY ====")
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
    
    try:
        # Search for tracks
        print("\nSearching for tracks...")
        results = sp.search(q='rock', type='track', limit=5)
        print(f"SUCCESS: Found {len(results['tracks']['items'])} tracks")
        
        # Print track names
        for i, track in enumerate(results['tracks']['items']):
            print(f"{i+1}. {track['name']} by {track['artists'][0]['name']}")
        
        return True, results['tracks']['items']
    except Exception as e:
        print(f"ERROR in search: {str(e)}")
        return False, None

def test_single_track_features(track):
    """Test getting audio features for just one track"""
    print("\n==== TESTING SINGLE TRACK FEATURES ====")
    
    env_path = Path('.env')
    load_dotenv(env_path)
    
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    # Set environment variables
    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"
    
    # Create fresh client
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    try:
        print(f"Getting features for track: {track['name']} (ID: {track['id']})")
        
        # Get audio features for one track
        features = sp.audio_features(track['id'])
        
        if features and features[0]:
            print("SUCCESS: Got audio features")
            print(f"- Danceability: {features[0]['danceability']}")
            print(f"- Energy: {features[0]['energy']}")
            print(f"- Tempo: {features[0]['tempo']}")
            return True
        else:
            print("ERROR: No features returned")
            return False
    except Exception as e:
        print(f"ERROR getting features: {str(e)}")
        return False

def test_small_batch_features(tracks):
    """Test getting audio features for a small batch of tracks"""
    print("\n==== TESTING SMALL BATCH FEATURES ====")
    
    env_path = Path('.env')
    load_dotenv(env_path)
    
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    # Set environment variables
    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"
    
    # Create fresh client
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    try:
        # Get track IDs
        track_ids = [track['id'] for track in tracks]
        print(f"Getting features for {len(track_ids)} tracks")
        
        # Print track IDs for debugging
        for i, track_id in enumerate(track_ids):
            print(f"  Track ID {i+1}: {track_id}")
        
        # Get audio features
        features = sp.audio_features(track_ids)
        
        if features and any(features):
            valid_features = [f for f in features if f]
            print(f"SUCCESS: Got audio features for {len(valid_features)}/{len(track_ids)} tracks")
            return True
        else:
            print("ERROR: No features returned")
            return False
    except Exception as e:
        print(f"ERROR getting batch features: {str(e)}")
        print(f"Exception type: {type(e)}")
        return False

def test_debug_audio_features_request():
    """Debug the actual request being made to the Spotify API"""
    print("\n==== DEBUGGING AUDIO FEATURES REQUEST ====")
    
    env_path = Path('.env')
    load_dotenv(env_path)
    
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    # Create a fresh client with logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Set environment variables
    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"
    
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, requests_timeout=10)
    
    try:
        # Just use a hard-coded popular track ID for testing
        test_track_id = "6rqhFgbbKwnb9MLmUQDhG6"  # Spotify popular track
        print(f"Testing with hard-coded track ID: {test_track_id}")
        
        # Print token info for debugging
        print("Access token info available:", sp._auth)
        
        # Try with debug logging
        print("\nMaking request to audio-features endpoint...")
        features = sp.audio_features(test_track_id)
        
        if features and features[0]:
            print("SUCCESS: Got audio features for test track")
            print(f"Features: {features[0]}")
            return True
        else:
            print("ERROR: No features returned for test track")
            return False
    except Exception as e:
        print(f"ERROR in debug request: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== BREAKING DOWN SPOTIFY API TESTS ===")
    
    # Test 1: Search functionality
    search_success, tracks = test_search_only()
    
    if search_success and tracks:
        # Test 2: Single track features
        single_success = test_single_track_features(tracks[0])
        
        # Test 3: Small batch features
        if single_success:
            batch_success = test_small_batch_features(tracks[:3])
        else:
            print("\nSkipping batch test since single track test failed")
            
            # Test 4: Debug request
            print("\nRunning debug test with hard-coded track ID...")
            debug_success = test_debug_audio_features_request()
    else:
        print("\nSkipping feature tests since search failed") 