import os
import sys
from pathlib import Path
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Change to the Spotify directory
spotify_dir = Path(__file__).parent.parent
os.chdir(str(spotify_dir))
print(f"Changed working directory to: {os.getcwd()}")

# Load environment variables
load_dotenv()

# Get Spotify credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Missing Spotify credentials in .env file")

def test_audio_features_auth():
    """Test Spotify authentication and audio features access with correct scopes"""
    print("\n=== Testing Spotify Authorization Code Flow ===")
    
    # Define scopes - updated with all necessary scopes for audio features
    scope = "user-read-private user-read-email playlist-read-private playlist-read-collaborative user-top-read user-library-read"
    
    # Create cache path
    cache_path = '.spotify_cache'
    
    # Clear existing cache to force new authentication
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("Removed existing cache file to start fresh authentication")
    
    # Set redirect URI - MUST match exactly what's in your Spotify Dashboard
    redirect_uri = 'http://127.0.0.1:8888/callback'
    
    # Initialize the SpotifyOAuth object
    print("Setting up OAuth authentication...")
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=redirect_uri,
        scope=scope,
        cache_path=cache_path,
        show_dialog=True  # Force showing the auth dialog
    )
    
    # Generate and print authorization URL
    auth_url = auth_manager.get_authorize_url()
    print(f"\n⚠️ Please navigate to this URL in your browser to authorize access:\n{auth_url}\n")
    print("After authorizing, copy the FULL URL you are redirected to (including 'http://127.0.0.1:8888/callback?code=...')")
    
    # Get the redirect URL from user input
    redirect_url = input("\nEnter the FULL URL you were redirected to: ")
    
    # Extract code from redirect URL
    try:
        code = auth_manager.parse_response_code(redirect_url)
        print(f"Successfully parsed authorization code")
        
        # Exchange code for access token
        print("Exchanging code for access token...")
        token_info = auth_manager.get_access_token(code)
        
        if not token_info or 'access_token' not in token_info:
            print("❌ Failed to get access token")
            return False
            
        print("✅ Successfully obtained access token")
        
        # Create Spotify client with token
        sp = spotipy.Spotify(auth_manager=auth_manager)
        
        # Test basic profile access to verify authentication
        print("\nTesting profile access...")
        user_info = sp.current_user()
        print(f"Authenticated as: {user_info['display_name']} (ID: {user_info['id']})")
        
        # Test playlist access
        print("\nTesting playlist access...")
        playlist_id = "6jgCEkpKSc7LQI8ZWdAVr6"  # Your car classics playlist
        playlist = sp.playlist(playlist_id, fields="name,owner.display_name")
        print(f"Successfully accessed playlist: {playlist['name']}")
        
        # Get a single popular track with known audio features (for testing)
        print("\nFetching a popular track with known audio features...")
        popular_track_id = "5HNCy40Ni5BZJFw1TKzRsC"  # Shape of You by Ed Sheeran
        
        # Test audio features endpoint with popular track
        print("\nTesting audio features endpoint with popular track...")
        try:
            feature = sp.audio_features(popular_track_id)[0]
            
            if feature:
                print("✅ Successfully retrieved audio features for popular track!")
                print(f"- Danceability: {feature['danceability']}")
                print(f"- Energy: {feature['energy']}")
                print(f"- Tempo: {feature['tempo']} BPM")
                print(f"- Valence: {feature['valence']}")
                
                # Now try with playlist tracks
                print("\nNow testing with tracks from your playlist...")
                results = sp.playlist_tracks(playlist_id, limit=2)
                
                if not results or 'items' not in results or not results['items']:
                    print("❌ Failed to get tracks from playlist")
                    return False
                    
                # Extract track IDs
                track_ids = []
                for item in results['items']:
                    if item['track']:
                        track = item['track']
                        print(f"Found track: {track['name']} by {track['artists'][0]['name']} (ID: {track['id']})")
                        track_ids.append(track['id'])
                
                # Test audio features for playlist tracks
                print("\nTesting audio features for your playlist tracks...")
                playlist_features = sp.audio_features(track_ids)
                
                if playlist_features and any(playlist_features):
                    print("✅ Successfully retrieved audio features for your playlist tracks!")
                    
                    # Print sample features
                    for i, feature in enumerate(playlist_features):
                        if feature:
                            track_name = results['items'][i]['track']['name']
                            print(f"\nAudio features for '{track_name}':")
                            print(f"- Danceability: {feature['danceability']}")
                            print(f"- Energy: {feature['energy']}")
                            print(f"- Tempo: {feature['tempo']} BPM")
                            print(f"- Valence: {feature['valence']}")
                    
                    return True
                else:
                    print("❌ No audio features returned for your playlist tracks")
                    print("This could be due to region restrictions or track availability.")
                    return False
            else:
                print("❌ No audio features returned for popular track")
                return False
                
        except Exception as e:
            print(f"❌ Error fetching audio features: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== SPOTIFY AUTHENTICATION TEST WITH AUDIO FEATURES ===")
    print("\nIMPORTANT: Make sure your Spotify Developer App settings match these values:")
    print("- Redirect URI: http://127.0.0.1:8888/callback")
    print("- Check that you have added this exact URI to your app in the Spotify Developer Dashboard")
    
    result = test_audio_features_auth()
    
    if result:
        print("\n✅ TEST PASSED: Authentication and audio features access successful!")
        print("You can now run your main script: python src/data_collection/spotify_ML.py")
    else:
        print("\n❌ TEST FAILED: Could not authenticate or access audio features") 