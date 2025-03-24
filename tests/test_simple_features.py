import os
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Change to the Spotify directory
spotify_dir = Path(__file__).parent.parent
os.chdir(str(spotify_dir))
print(f"Changed working directory to: {os.getcwd()}")

# Load environment variables from .env file
print("Loading environment variables...")
load_dotenv()

# Get Spotify credentials
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

print(f"Client ID present: {'Yes' if CLIENT_ID else 'No'}")
print(f"Client Secret present: {'Yes' if CLIENT_SECRET else 'No'}")

# Set up the client credentials flow - the simplest authentication method
# This only requires your client ID and client secret (no user authorization)
client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

# Create Spotify client
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
print("Created Spotify client with Client Credentials flow")

def test_top_track_audio_features():
    """Test getting audio features for a global top track (which should be accessible)"""
    print("\n=== Testing Audio Features for Global Top Track ===")
    
    # Get a popular track that should definitely have audio features
    popular_track_id = "5HNCy40Ni5BZJFw1TKzRsC"  # Shape of You by Ed Sheeran
    
    try:
        # First get the track info
        track = sp.track(popular_track_id)
        print(f"Track: {track['name']} by {track['artists'][0]['name']}")
        
        # Try to get audio features
        print("\nGetting audio features...")
        features = sp.audio_features(popular_track_id)[0]
        
        if features:
            print("✅ Successfully retrieved audio features for popular track!")
            print(f"- Danceability: {features['danceability']}")
            print(f"- Energy: {features['energy']}")
            print(f"- Tempo: {features['tempo']} BPM")
            print(f"- Valence: {features['valence']}")
            return True
        else:
            print("❌ No audio features returned (features object was None)")
            return False
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_playlist_track_audio_features():
    """Test getting audio features for tracks in your playlist"""
    print("\n=== Testing Audio Features for Your Playlist Tracks ===")
    
    # Your playlist ID
    playlist_id = "6jgCEkpKSc7LQI8ZWdAVr6"  # Your car classics playlist
    
    try:
        # First get basic playlist info
        playlist = sp.playlist(playlist_id, fields="name,owner.display_name")
        print(f"Playlist: {playlist['name']}")
        
        # Get a couple tracks
        results = sp.playlist_tracks(playlist_id, limit=2)
        
        if not results or 'items' not in results or not results['items']:
            print("❌ Failed to get tracks from playlist")
            return False
            
        # Extract track IDs and info
        track_ids = []
        track_names = []
        
        for item in results['items']:
            if item['track']:
                track = item['track']
                name = f"{track['name']} by {track['artists'][0]['name']}"
                print(f"Found track: {name} (ID: {track['id']})")
                track_ids.append(track['id'])
                track_names.append(name)
        
        # Get audio features for these tracks
        print("\nGetting audio features for playlist tracks...")
        features = sp.audio_features(track_ids)
        
        if features and any(features):
            print("✅ Successfully retrieved audio features for playlist tracks!")
            
            # Print sample features
            for i, feature in enumerate(features):
                if feature:
                    print(f"\nAudio features for '{track_names[i]}':")
                    print(f"- Danceability: {feature['danceability']}")
                    print(f"- Energy: {feature['energy']}")
                    print(f"- Tempo: {feature['tempo']} BPM")
                    print(f"- Valence: {feature['valence']}")
            
            return True
        else:
            print("❌ No audio features returned for playlist tracks")
            return False
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== SIMPLE SPOTIFY AUDIO FEATURES TEST ===")
    print("Testing with public Client Credentials flow (no user auth required)")
    
    # Test with popular track first
    popular_result = test_top_track_audio_features()
    
    # Then test with playlist tracks
    playlist_result = test_playlist_track_audio_features()
    
    # Overall result
    if popular_result and playlist_result:
        print("\n✅ BOTH TESTS PASSED: Successfully accessed audio features!")
        print("This means your Spotify credentials are working correctly.")
        print("You should be able to run your main script now.")
    elif popular_result:
        print("\n⚠️ PARTIAL SUCCESS: Could access audio features for popular track but not for your playlist.")
        print("This may indicate an issue with playlist permissions or region restrictions.")
    else:
        print("\n❌ ALL TESTS FAILED: Could not access audio features.")
        print("Please check your Spotify Developer Dashboard settings and credentials.") 