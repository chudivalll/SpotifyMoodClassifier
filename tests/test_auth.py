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

def test_spotify_auth():
    # Load environment variables from .env in the Spotify directory
    env_path = Path('.env')
    print(f"Looking for .env file at: {env_path.absolute()}")
    load_dotenv(env_path)
    
    # Get credentials
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    print("\nCredential Check:")
    print(f"Client ID present: {'Yes' if CLIENT_ID else 'No'}")
    print(f"Client Secret present: {'Yes' if CLIENT_SECRET else 'No'}")
    
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Error: Missing credentials in .env file")
        return False
    
    try:
        # Set environment variables
        os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
        os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
        os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"
        
        # Test basic authentication
        client_credentials_manager = SpotifyClientCredentials()
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        
        # Try a simple API call
        print("\nSearching for tracks...")
        results = sp.search(q='rock classics', type='track', limit=20)
        print("Search completed successfully!")
        print(f"Found {len(results['tracks']['items'])} tracks")
        
        # Get track IDs
        track_ids = [track['id'] for track in results['tracks']['items']]
        print(f"\nRetrieved {len(track_ids)} track IDs")
        
        # Get audio features
        print("\nFetching audio features...")
        audio_features = sp.audio_features(track_ids)
        
        # Verify results
        if audio_features and any(audio_features):
            print(f"Successfully fetched audio features for {len([f for f in audio_features if f])} tracks")
            
            # Print sample of features for first track
            if audio_features[0]:
                print("\nSample audio features for first track:")
                sample_track = results['tracks']['items'][0]
                print(f"Track: {sample_track['name']} by {sample_track['artists'][0]['name']}")
                
                # Print key audio features
                features = audio_features[0]
                print(f"- Danceability: {features['danceability']}")
                print(f"- Energy: {features['energy']}")
                print(f"- Tempo: {features['tempo']}")
                print(f"- Valence: {features['valence']}")
            
            return True
        else:
            print("Failed to retrieve audio features")
            return False
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        return False

def test_rate_limited_features():
    """Test fetching audio features with rate limiting handling"""
    env_path = project_root / '.env'
    load_dotenv(env_path)
    
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    # Set environment variables
    os.environ["SPOTIPY_CLIENT_ID"] = CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8888"
    
    # Create client
    client_credentials_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Search for tracks
    print("\nSearching for tracks...")
    results = sp.search(q='rock', type='track', limit=50)
    track_ids = [track['id'] for track in results['tracks']['items']]
    
    # Get audio features in batches
    all_features = []
    batch_size = 20
    
    print(f"\nFetching audio features for {len(track_ids)} tracks in batches of {batch_size}...")
    
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        try:
            # Add delay between batches (except first)
            if i > 0:
                print("Waiting 3 seconds...")
                time.sleep(3)
                
            features = sp.audio_features(batch)
            if features and any(features):
                valid_features = [f for f in features if f is not None]
                all_features.extend(valid_features)
                print(f"Successfully processed {len(valid_features)} tracks")
            else:
                print("No features returned for this batch")
                
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            print("Continuing with next batch after 5 second delay...")
            time.sleep(5)
    
    print(f"\nCompleted! Retrieved features for {len(all_features)} tracks")
    
    if all_features:
        # Print sample features
        print("\nSample features for first track:")
        print(f"Danceability: {all_features[0]['danceability']}")
        print(f"Energy: {all_features[0]['energy']}")
        print(f"Tempo: {all_features[0]['tempo']}")
        print(f"Valence: {all_features[0]['valence']}")
    
    return len(all_features) > 0

if __name__ == "__main__":
    print("Testing Spotify Authentication...\n")
    success = test_spotify_auth()
    
    if success:
        print("\nBasic test successful! Testing batch feature retrieval...")
        batch_success = test_rate_limited_features()
        
        if batch_success:
            print("\nAll tests passed successfully!")
        else:
            print("\nBasic authentication works but batch feature retrieval failed")
    else:
        print("\nBasic authentication test failed. Please verify your credentials in the .env file") 