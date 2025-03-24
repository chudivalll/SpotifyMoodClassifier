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

def test_playlist_audio_features():
    """Test fetching tracks from a playlist and retrieving their audio features"""
    print("\n=== Testing Playlist Audio Features Flow ===")
    
    # Define required scopes for accessing private playlists and audio features
    scope = [
        'playlist-read-private',
        'playlist-read-collaborative',
        'user-library-read'
    ]
    
    cache_path = '.spotify_cache'
    
    # Initialize the Spotify client with OAuth
    print("Setting up OAuth authentication...")
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri='http://127.0.0.1:8888/callback',  # Must match exactly in Spotify Dashboard
        scope=' '.join(scope),
        cache_path=cache_path,
        open_browser=False
    )
    
    # Get access token
    print("Getting access token...")
    token_info = auth_manager.get_cached_token()
    
    if not token_info:
        print("\n⚠️ No cached token found. You need to authenticate.")
        auth_url = auth_manager.get_authorize_url()
        print(f"\nPlease navigate to this URL in your browser:\n{auth_url}\n")
        print("After authorizing, copy the FULL URL you are redirected to (including 'http://127.0.0.1...')")
        redirect_url = input("\nEnter the URL you were redirected to: ")
        
        code = auth_manager.parse_response_code(redirect_url)
        token_info = auth_manager.get_access_token(code)
    
    print("Creating Spotify client with OAuth token...")
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    # Use the car classics playlist ID
    playlist_id = "6jgCEkpKSc7LQI8ZWdAVr6"
    
    try:
        # Step 1: Get playlist info to debug
        print(f"Fetching playlist info for ID: {playlist_id}")
        playlist_info = sp.playlist(playlist_id, fields='name,tracks.total')
        print(f"Playlist name: {playlist_info['name']}")
        print(f"Total tracks in playlist: {playlist_info['tracks']['total']}")
        
        # Step 2: Get a few tracks from the playlist (just 3 for testing)
        print("\nFetching first 3 tracks from playlist...")
        results = sp.playlist_tracks(
            playlist_id,
            limit=3,
            fields='items.track.name,items.track.artists,items.track.id'
        )
        
        if not results or not results['items']:
            print("No tracks found in the playlist")
            return False
            
        # Step 3: Extract track IDs and track info
        tracks = []
        track_ids = []
        
        for item in results['items']:
            track = item['track']
            if track:
                artist_names = ", ".join([artist['name'] for artist in track['artists']])
                print(f"Found track: {track['name']} by {artist_names} (ID: {track['id']})")
                
                tracks.append({
                    'name': track['name'],
                    'artists': artist_names,
                    'id': track['id']
                })
                track_ids.append(track['id'])
        
        if not track_ids:
            print("Failed to extract track IDs")
            return False
            
        print(f"\nFound {len(track_ids)} track IDs")
        
        # Step 4: Get audio features for these track IDs
        print("Fetching audio features for these tracks...")
        time.sleep(1)  # Short delay before API call
        
        audio_features = sp.audio_features(track_ids)
        
        if not audio_features or not any(audio_features):
            print("Failed to get audio features")
            return False
            
        # Step 5: Display audio features for first track
        print("\nAudio features for first track:")
        for i, features in enumerate(audio_features):
            if features:
                print(f"\nTrack: {tracks[i]['name']} by {tracks[i]['artists']}")
                print(f"- Danceability: {features['danceability']}")
                print(f"- Energy: {features['energy']}")
                print(f"- Tempo: {features['tempo']}")
                print(f"- Valence (positivity): {features['valence']}")
                print(f"- Acousticness: {features['acousticness']}")
        
        print("\nTest completed successfully! The audio features endpoint is working correctly.")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_playlist_audio_features()
    
    if success:
        print("\n✅ TEST PASSED: Successfully retrieved playlist tracks and their audio features")
        print("You can now run your main script: python src/data_collection/spotify_ML.py")
    else:
        print("\n❌ TEST FAILED: Could not complete the playlist audio features test") 