import os
from pathlib import Path
import time
import sys
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv
from spotipy.exceptions import SpotifyException
import json

# Change to the Spotify directory
spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))
print(f"Changed working directory to: {os.getcwd()}")

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
print("Loading environment variables...")
load_dotenv(project_root / '.env')

# Get credentials and print debug info
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

print("\nCredential Debug Info:")
print(f"Client ID length: {len(CLIENT_ID) if CLIENT_ID else 'Not found'}")
print(f"Client Secret length: {len(CLIENT_SECRET) if CLIENT_SECRET else 'Not found'}")

def create_spotify_client(force_new_token=False):
    """Create Spotify client with OAuth for accessing playlists"""
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("Missing Spotify credentials in .env file")
        
    print(f"Initializing Spotify client with OAuth...")
    
    # Define required scopes - extended with all possible scopes to maximize access
    scope = "user-read-private user-read-email playlist-read-private playlist-read-collaborative user-library-read"
    
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
                           'items.track.album.release_date,items.track.popularity,items.track.id,'
                           'items.track.preview_url,items.track.explicit,items.track.duration_ms'
                )
                
                for item in results['items']:
                    track = item['track']
                    if track:
                        artists = [artist['name'] for artist in track['artists']]
                        
                        track_info = {
                            'Name': track['name'],
                            'Artist': ", ".join(artists),
                            'Album': track['album']['name'],
                            'Release Date': track['album']['release_date'],
                            'Track ID': track['id'],
                            'Preview URL': track.get('preview_url', ''),
                            'Popularity': track.get('popularity', 0),
                            'Explicit': track.get('explicit', False),
                            'Duration (ms)': track.get('duration_ms', 0),
                            'Primary Artist': artists[0] if artists else '',
                            'Number of Artists': len(artists)
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

def get_artist_genres(artist_ids, batch_size=20, delay_between_batches=2):
    """Get genre information for artists"""
    sp = create_spotify_client()
    
    all_artist_info = []
    total_artists = len(artist_ids)
    
    print(f"\nFetching genre information for {total_artists} artists...")
    
    # Process in smaller batches
    for i in range(0, total_artists, batch_size):
        batch = artist_ids[i:i + batch_size]
        print(f"Processing batch {(i//batch_size) + 1} of {(total_artists + batch_size - 1)//batch_size}")
        
        # Add delay between batches
        if i > 0:
            print(f"Waiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)
        
        try:
            artists = sp.artists(batch)
            if artists and 'artists' in artists:
                all_artist_info.extend(artists['artists'])
                print(f"Successfully processed {len(artists['artists'])} artists in this batch")
                
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            if "rate limit" in str(e).lower():
                wait_time = 30
                print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                # Create a fresh client
                sp = create_spotify_client()
                continue
            else:
                print("Continuing with next batch after delay...")
                time.sleep(5)  # Wait before next batch
                continue
            
    return all_artist_info

def get_track_audio_analysis(track_id):
    """Get audio analysis for a track - try with direct API access"""
    sp = create_spotify_client()
    
    try:
        analysis = sp.audio_analysis(track_id)
        if analysis:
            # Extract key features from track analysis
            track_analysis = {}
            
            # Get tempo (BPM)
            if 'track' in analysis:
                track_analysis['Tempo'] = analysis['track'].get('tempo', 0)
                track_analysis['Key'] = analysis['track'].get('key', -1)
                track_analysis['Mode'] = analysis['track'].get('mode', -1)
                track_analysis['Time_Signature'] = analysis['track'].get('time_signature', 4)
                track_analysis['Loudness'] = analysis['track'].get('loudness', 0)
                
            # Get section data (may contain energy, valence info)
            if 'sections' in analysis and analysis['sections']:
                first_section = analysis['sections'][0]
                track_analysis['Section_Loudness'] = first_section.get('loudness', 0)
                track_analysis['Section_Tempo'] = first_section.get('tempo', 0)
                track_analysis['Section_Key'] = first_section.get('key', -1)
                track_analysis['Section_Mode'] = first_section.get('mode', -1)
                
            return track_analysis
    except Exception as e:
        print(f"Error getting audio analysis for track {track_id}: {str(e)}")
    
    return {}

def try_get_audio_features(track_ids, batch_size=1, max_tracks=20):
    """Attempt to get audio features for tracks, one at a time to avoid rate limiting"""
    sp = create_spotify_client()
    
    features_list = []
    total_tracks = min(len(track_ids), max_tracks)  # Limit to max_tracks
    
    print(f"\nAttempting to get audio features for {total_tracks} tracks...")
    print("Note: This may fail due to API restrictions, but we'll try anyway")
    
    for i in range(total_tracks):
        track_id = track_ids[i]
        track_data = {'Track ID': track_id}
        
        # Add delay to avoid rate limits
        if i > 0:
            time.sleep(2)
            
        try:
            # Try the audio_features endpoint
            features = sp.audio_features(track_id)
            if features and features[0]:
                feature = features[0]
                # Add each feature to our track data
                track_data['Danceability'] = feature.get('danceability', 0)
                track_data['Energy'] = feature.get('energy', 0)
                track_data['Valence'] = feature.get('valence', 0)
                track_data['Acousticness'] = feature.get('acousticness', 0)
                track_data['Speechiness'] = feature.get('speechiness', 0)
                track_data['Instrumentalness'] = feature.get('instrumentalness', 0)
                track_data['Liveness'] = feature.get('liveness', 0)
                track_data['Tempo'] = feature.get('tempo', 0)
                
                print(f"Successfully got audio features for track {i+1}/{total_tracks}")
        except Exception as e:
            print(f"Failed to get audio features for track {i+1}: {str(e)}")
            
        # Try the audio_analysis endpoint instead
        try:
            analysis_data = get_track_audio_analysis(track_id)
            # Add analysis data to track data
            track_data.update(analysis_data)
            print(f"Got audio analysis for track {i+1}/{total_tracks}")
        except Exception as e:
            print(f"Failed to get audio analysis for track {i+1}: {str(e)}")
            
        features_list.append(track_data)
            
    return features_list

def main():
    """Main execution flow with error handling"""
    try:
        # Define your playlist ID
        PLAYLIST_ID = "1j7MQNNTtHauvdwj109V9k"
        print(f"Processing playlist with ID: {PLAYLIST_ID}")
        
        # 1. Get tracks from playlist
        tracks_data = get_playlist_tracks(PLAYLIST_ID, batch_size=50, max_tracks=1200)
        
        if not tracks_data:
            print("No tracks were retrieved. Exiting.")
            return
        
        # 2. Create DataFrame from track data
        df_tracks = pd.DataFrame(tracks_data)
        
        # 3. Add calculated fields
        df_tracks['Duration (min)'] = df_tracks['Duration (ms)'] / 60000
        df_tracks['Release Year'] = df_tracks['Release Date'].str[:4]
        
        # Make sure the Preview URL is visible
        print(f"\nNumber of tracks with preview URLs: {df_tracks['Preview URL'].notna().sum()}")
        
        # 4. Extract primary artists for genre lookup
        primary_artists = df_tracks['Primary Artist'].drop_duplicates().tolist()
        
        # 5. Search for each primary artist to get artist IDs
        print(f"\nSearching for {len(primary_artists)} primary artists...")
        sp = create_spotify_client()
        
        artist_lookup = {}
        for i, artist in enumerate(primary_artists):
            if i > 0 and i % 20 == 0:
                print(f"Processed {i}/{len(primary_artists)} artists")
                time.sleep(2)  # Avoid rate limits
                
            try:
                results = sp.search(q=f'artist:"{artist}"', type='artist', limit=1)
                if results and 'artists' in results and results['artists']['items']:
                    artist_id = results['artists']['items'][0]['id']
                    artist_lookup[artist] = artist_id
            except Exception as e:
                print(f"Error searching for artist '{artist}': {str(e)}")
                time.sleep(3)  # Wait longer after error
        
        print(f"Found IDs for {len(artist_lookup)}/{len(primary_artists)} artists")
        
        # 6. Get artist genres
        artist_ids = list(artist_lookup.values())
        artist_info = get_artist_genres(artist_ids[:100])  # Limit to first 100 to avoid rate limits
        
        # 7. Create artist genres dictionary
        artist_genres = {}
        for artist in artist_info:
            artist_id = artist['id']
            genres = artist.get('genres', [])
            popularity = artist.get('popularity', 0)
            
            # Find artist name from lookup
            artist_name = next((name for name, id in artist_lookup.items() if id == artist_id), "Unknown")
            
            artist_genres[artist_name] = {
                'genres': genres,
                'primary_genre': genres[0] if genres else "Unknown",
                'popularity': popularity
            }
        
        # 8. Add genre information to tracks
        df_tracks['Artist Genres'] = df_tracks['Primary Artist'].map(
            lambda x: artist_genres.get(x, {}).get('genres', [])
        )
        df_tracks['Primary Genre'] = df_tracks['Primary Artist'].map(
            lambda x: artist_genres.get(x, {}).get('primary_genre', "Unknown")
        )
        df_tracks['Artist Popularity'] = df_tracks['Primary Artist'].map(
            lambda x: artist_genres.get(x, {}).get('popularity', 0)
        )
        
        # 9. Try to get audio features for a sample of tracks
        track_ids = df_tracks['Track ID'].tolist()
        audio_features = try_get_audio_features(track_ids, max_tracks=20)
        
        # 10. Create audio features DataFrame
        df_features = pd.DataFrame(audio_features)
        
        # 11. Merge track data with audio features if we have any
        if not df_features.empty:
            df_merged = pd.merge(df_tracks, df_features, on='Track ID', how='left')
        else:
            df_merged = df_tracks
            
        # 12. Save dataset to Excel
        output_file = "spotify_playlist_enhanced.xlsx"
        df_merged.to_excel(output_file, index=False)
        
        # 13. Create version for ML without preview URLs
        model_columns = [col for col in df_merged.columns if col != 'Preview URL']
        df_model = df_merged[model_columns]
        
        model_output_file = "spotify_playlist_ml_enhanced.xlsx"
        df_model.to_excel(model_output_file, index=False)
        
        # 14. Make sure data directory exists
        data_dir = project_root / 'data' / 'processed'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to data directory as well
        df_merged.to_excel(data_dir / output_file, index=False)
        df_model.to_excel(data_dir / model_output_file, index=False)
        
        # 15. Print summary stats
        print(f"\nAnalysis complete!")
        print(f"Successfully processed {len(df_merged)} tracks")
        print(f"Results saved to {output_file}")
        print(f"ML-ready data saved to {model_output_file}")
        
        print("\nPlaylist Statistics:")
        print(f"Most common artist: {df_merged['Primary Artist'].value_counts().idxmax()}")
        print(f"Most common genre: {df_merged['Primary Genre'].value_counts().idxmax()}")
        print(f"Average track popularity: {df_merged['Popularity'].mean():.1f}/100")
        print(f"Average track duration: {df_merged['Duration (min)'].mean():.2f} minutes")
        print(f"Explicit tracks: {df_merged['Explicit'].sum()} ({df_merged['Explicit'].mean()*100:.1f}%)")
        
        # Print which audio features were successfully retrieved
        audio_feature_cols = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                              'Speechiness', 'Instrumentalness', 'Liveness', 'Tempo']
        
        print("\nAudio Features Retrieved:")
        for col in audio_feature_cols:
            if col in df_merged.columns:
                non_null_count = df_merged[col].notna().sum()
                print(f"- {col}: {non_null_count} tracks")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 