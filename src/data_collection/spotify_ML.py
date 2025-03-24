# Import required libraries
# spotipy: Python library for Spotify Web API
# SpotifyClientCredentials: Class for client credentials flow authentication
# pandas: Data manipulation library for creating and handling dataframes
import sys
import os
from pathlib import Path
import time
from spotipy.exceptions import SpotifyException

# Change to the Spotify directory
spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))
print(f"Changed working directory to: {os.getcwd()}")

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv

# Add these debug lines right after load_dotenv
print("Current working directory:", os.getcwd())
print("Looking for .env file at:", project_root / '.env')
load_dotenv(project_root / '.env')

# Get credentials and print actual values (be careful not to share these)
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

print("\nCredential Debug Info:")
print(f"Client ID length: {len(CLIENT_ID) if CLIENT_ID else 'Not found'}")
print(f"Client Secret length: {len(CLIENT_SECRET) if CLIENT_SECRET else 'Not found'}")

def create_spotify_client(force_new_token=False):
    """Create Spotify client with OAuth for accessing private playlists and audio features"""
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("Missing Spotify credentials in .env file")
        
    print(f"Initializing Spotify client with OAuth...")
    
    # Define required scopes - this is crucial for accessing audio features
    scope = "user-read-private user-read-email playlist-read-private playlist-read-collaborative user-top-read user-library-read"
    
    cache_path = '.spotify_cache'
    
    # Clear cache if forced new token is requested
    if force_new_token and os.path.exists(cache_path):
        os.remove(cache_path)
        print("Removed existing cache file to force new authentication")
    
    # Create OAuth manager
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri='http://127.0.0.1:8888/callback',  # Must match exactly in Spotify Dashboard
        scope=scope,
        cache_path=cache_path,
        show_dialog=force_new_token  # Force showing dialog if new token needed
    )
    
    # Get access token
    token_info = auth_manager.get_cached_token()
    
    if not token_info:
        print("\n⚠️ No cached token found. You need to authenticate.")
        auth_url = auth_manager.get_authorize_url()
        print(f"\nPlease navigate to this URL in your browser:\n{auth_url}\n")
        print("After authorizing, copy the FULL URL you are redirected to (including 'http://127.0.0.1:8888/callback?code=...')")
        redirect_url = input("\nEnter the URL you were redirected to: ")
        
        code = auth_manager.parse_response_code(redirect_url)
        token_info = auth_manager.get_access_token(code)
        
        if not token_info or 'access_token' not in token_info:
            raise ValueError("Failed to get access token after authorization")
    
    # Create and verify Spotify client
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    # Quick verification that authentication worked
    try:
        user = sp.current_user()
        print(f"Successfully authenticated as: {user['display_name']}")
        return sp
    except Exception as e:
        print(f"Error verifying authentication: {str(e)}")
        # Try one more time with a fresh token
        if not force_new_token:
            print("Attempting to refresh authentication...")
            return create_spotify_client(force_new_token=True)
        else:
            raise ValueError("Authentication failed even after token refresh")

# Function to extract playlist ID from full URL if needed
def get_playlist_id(playlist_input):
    """Extract playlist ID from full Spotify URL or return the ID if already in correct format"""
    if 'spotify.com' in str(playlist_input):
        # Split by '/' and take the last part, then remove any query parameters
        return playlist_input.split('/')[-1].split('?')[0]
    return playlist_input

def get_playlist_tracks(playlist_id, batch_size=50, delay_between_batches=2, max_tracks=1200):
    """Retrieves tracks from a Spotify playlist with rate limit handling"""
    # Create a fresh client for this operation
    sp = create_spotify_client()
    
    tracks = []
    offset = 0
    
    try:
        # First get playlist info
        playlist_info = sp.playlist(playlist_id, fields='name,tracks.total')
        total = min(playlist_info['tracks']['total'], max_tracks)  # Limit total tracks
        print(f"Found {playlist_info['tracks']['total']} tracks in playlist '{playlist_info['name']}'")
        print(f"Will process first {total} tracks")
        
        # Get tracks in batches
        while offset < total:
            current_batch = min(batch_size, total - offset)
            print(f"Fetching tracks {offset} to {offset + current_batch}...")
            
            # Add delay between batches
            if offset > 0:
                time.sleep(delay_between_batches)
            
            try:
                results = sp.playlist_tracks(
                    playlist_id,
                    limit=current_batch,
                    offset=offset,
                    fields='items.track.name,items.track.artists,items.track.album.name,'
                           'items.track.album.release_date,items.track.id,items.track.preview_url'
                )
                
                for item in results['items']:
                    track = item['track']
                    if track:
                        track_info = {
                            'Name': track['name'],
                            'Artist': ", ".join([artist['name'] for artist in track['artists']]),
                            'Album': track['album']['name'],
                            'Release Date': track['album']['release_date'],
                            'Track ID': track['id'],
                            'Preview URL': track.get('preview_url', '')
                        }
                        tracks.append(track_info)
                
                offset += current_batch
                print(f"Retrieved {len(tracks)} tracks so far...")
                
            except SpotifyException as e:
                print(f"Error during track retrieval: {str(e)}")
                if "rate limit" in str(e).lower():
                    wait_time = 30
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # Create a fresh client to handle potential token issues
                    sp = create_spotify_client()
                    time.sleep(5)  # Wait before retry
                continue
            
        print(f"\nSuccessfully retrieved {len(tracks)} tracks in total")
        return tracks
        
    except Exception as e:
        print(f"Error fetching tracks: {str(e)}")
        if tracks:  # If we have some tracks, return them instead of raising error
            print(f"Returning {len(tracks)} tracks that were successfully retrieved")
            return tracks
        raise

def get_audio_features(track_ids, batch_size=20, delay_between_batches=3):
    """Get audio features with smaller batches and fresh client for each batch"""
    # Create a fresh client for this operation
    sp = create_spotify_client()
    
    all_features = []
    total_tracks = len(track_ids)
    
    print(f"\nFetching audio features for {total_tracks} tracks...")
    
    # Process in smaller batches
    for i in range(0, total_tracks, batch_size):
        batch = track_ids[i:i + batch_size]
        print(f"Processing batch {(i//batch_size) + 1} of {(total_tracks + batch_size - 1)//batch_size}")
        
        # Add delay between batches
        if i > 0:
            print(f"Waiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)
        
        try:
            features = sp.audio_features(batch)
            if features and any(features):
                valid_features = [f for f in features if f is not None]
                all_features.extend(valid_features)
                print(f"Successfully processed {len(valid_features)} tracks in this batch")
                print(f"Total tracks processed: {len(all_features)}/{total_tracks}")
            else:
                print(f"No features returned for batch {(i//batch_size) + 1}")
                
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            if "rate limit" in str(e).lower():
                wait_time = 30
                print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                # Create a fresh client
                sp = create_spotify_client()
                # Try again with the same batch
                try:
                    features = sp.audio_features(batch)
                    if features and any(features):
                        valid_features = [f for f in features if f is not None]
                        all_features.extend(valid_features)
                        print(f"Successfully processed {len(valid_features)} tracks after retry")
                    continue
                except:
                    pass
            print("Continuing with next batch after delay...")
            time.sleep(5)  # Wait before next batch
            continue
            
    return all_features

def main():
    """Main execution flow with error handling"""
    try:
        # Define your playlist ID
        PLAYLIST_ID = "6jgCEkpKSc7LQI8ZWdAVr6"
        print(f"Processing playlist with ID: {PLAYLIST_ID}")
        
        # 1. Get tracks from playlist
        tracks_data = get_playlist_tracks(PLAYLIST_ID, batch_size=50, max_tracks=1200)
        
        if not tracks_data:
            print("No tracks were retrieved. Exiting.")
            return
            
        # 2. Extract track IDs from the playlist tracks
        track_ids = [track['Track ID'] for track in tracks_data]
        print(f"\nExtracting {len(track_ids)} track IDs from playlist tracks")
        
        # 3. Get audio features for these track IDs
        audio_features = get_audio_features(track_ids, batch_size=20, delay_between_batches=3)
        
        if not audio_features:
            print("No audio features were retrieved. Exiting.")
            return
            
        # 4. Convert to DataFrames and merge
        df_tracks = pd.DataFrame(tracks_data)
        
        # Select features for ML
        feature_cols = [
            'id', 'danceability', 'energy', 'tempo', 'valence', 
            'acousticness', 'speechiness', 'instrumentalness', 'liveness'
        ]
        
        # Make sure all columns exist in audio_features before selecting
        available_cols = []
        for col in feature_cols:
            if col in audio_features[0]:
                available_cols.append(col)
            elif col == 'id' and 'id' not in audio_features[0]:
                # Special case for id which we need for merging
                print("Warning: 'id' not found in audio features, will use track_id if available")
                if 'track_id' in audio_features[0]:
                    available_cols.append('track_id')
                    # Will rename later
        
        df_features = pd.DataFrame(audio_features)
        df_features = df_features[available_cols]
        
        # Rename id or track_id to Track ID for merging
        if 'id' in df_features.columns:
            df_features = df_features.rename(columns={'id': 'Track ID'})
        elif 'track_id' in df_features.columns:
            df_features = df_features.rename(columns={'track_id': 'Track ID'})
            
        # Merge and save
        df_final = pd.merge(df_tracks, df_features, on='Track ID', how='inner')
        
        # Save full dataset including preview URLs
        output_file = "spotify_playlist_analysis.xlsx"
        df_final.to_excel(output_file, index=False)
        
        # Create a version for ML that excludes preview URLs
        model_columns = [col for col in df_final.columns if col != 'Preview URL']
        df_model = df_final[model_columns]
        
        # Save ML ready dataset
        model_output_file = "spotify_playlist_ml_data.xlsx"
        df_model.to_excel(model_output_file, index=False)
        
        # Make sure data directory exists
        data_dir = project_root / 'data' / 'processed'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to data directory as well
        df_final.to_excel(data_dir / output_file, index=False)
        df_model.to_excel(data_dir / model_output_file, index=False)
        
        print(f"\nAnalysis complete!")
        print(f"Successfully processed {len(df_final)} tracks")
        print(f"Results saved to {output_file}")
        print(f"ML-ready data saved to {model_output_file}")
        
        # Print some statistics for mood classification features
        print("\nFeature Statistics for Mood Classification:")
        for col in ['danceability', 'energy', 'valence', 'acousticness']:
            if col in df_final.columns:
                print(f"Average {col.capitalize()}: {df_final[col].mean():.2f}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
