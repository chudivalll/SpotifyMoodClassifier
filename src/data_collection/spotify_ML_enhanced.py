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

spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

load_dotenv(project_root / '.env')

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')


def create_spotify_client(force_new_token=False):
    """Create and authenticate a Spotify client using OAuth."""
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

    if not client_id or not client_secret:
        raise ValueError("Missing Spotify credentials in .env file")

    scope = "user-read-private user-read-email playlist-read-private playlist-read-collaborative user-library-read"
    cache_path = '.spotify_cache'

    if force_new_token and os.path.exists(cache_path):
        os.remove(cache_path)

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri='http://127.0.0.1:8888/callback',
        scope=scope,
        cache_path=cache_path,
        show_dialog=force_new_token
    )

    token_info = auth_manager.get_cached_token()

    if not token_info:
        print("\nNo cached token — authenticate via browser.")
        auth_url = auth_manager.get_authorize_url()
        print(f"\nOpen: {auth_url}\n")
        redirect_url = input("Paste redirect URL: ")

        code = auth_manager.parse_response_code(redirect_url)
        token_info = auth_manager.get_access_token(code)

        if not token_info or 'access_token' not in token_info:
            raise ValueError("Failed to get access token")

    sp = spotipy.Spotify(auth_manager=auth_manager)

    try:
        user = sp.current_user()
        print(f"Authenticated as: {user['display_name']}")
        return sp
    except Exception as e:
        if not force_new_token:
            return create_spotify_client(force_new_token=True)
        raise ValueError(f"Auth failed: {e}")


def get_playlist_id(playlist_input):
    if 'spotify.com' in str(playlist_input):
        return playlist_input.split('/')[-1].split('?')[0]
    return playlist_input


def get_playlist_tracks(playlist_id, batch_size=50, delay_between_batches=2, max_tracks=1200):
    """Fetch tracks with metadata (popularity, duration, explicit flag)."""
    sp = create_spotify_client()
    tracks = []
    offset = 0

    try:
        playlist_info = sp.playlist(playlist_id, fields='name,tracks.total')
        total = min(playlist_info['tracks']['total'], max_tracks)
        print(f"Playlist '{playlist_info['name']}': {total} tracks to process")

        while offset < total:
            current_batch = min(batch_size, total - offset)

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
                        artists = [a['name'] for a in track['artists']]
                        tracks.append({
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
                        })

                offset += current_batch
                print(f"  {len(tracks)}/{total}")

            except SpotifyException as e:
                if "rate limit" in str(e).lower():
                    time.sleep(30)
                else:
                    sp = create_spotify_client()
                    time.sleep(5)
                continue

        return tracks

    except Exception as e:
        print(f"Error: {e}")
        if tracks:
            return tracks
        raise


def get_artist_genres(artist_ids, batch_size=20, delay_between_batches=2):
    """Fetch genre info for a list of artist IDs."""
    sp = create_spotify_client()
    all_info = []

    for i in range(0, len(artist_ids), batch_size):
        batch = artist_ids[i:i + batch_size]

        if i > 0:
            time.sleep(delay_between_batches)

        try:
            result = sp.artists(batch)
            if result and 'artists' in result:
                all_info.extend(result['artists'])
        except Exception as e:
            if "rate limit" in str(e).lower():
                time.sleep(30)
                sp = create_spotify_client()
            else:
                time.sleep(5)
            continue

    return all_info


def get_track_audio_analysis(track_id):
    """Try to pull detailed audio analysis for a single track."""
    sp = create_spotify_client()

    try:
        analysis = sp.audio_analysis(track_id)
        if not analysis:
            return {}

        result = {}
        if 'track' in analysis:
            t = analysis['track']
            result['Tempo'] = t.get('tempo', 0)
            result['Key'] = t.get('key', -1)
            result['Mode'] = t.get('mode', -1)
            result['Time_Signature'] = t.get('time_signature', 4)
            result['Loudness'] = t.get('loudness', 0)

        if 'sections' in analysis and analysis['sections']:
            s = analysis['sections'][0]
            result['Section_Loudness'] = s.get('loudness', 0)
            result['Section_Tempo'] = s.get('tempo', 0)
            result['Section_Key'] = s.get('key', -1)
            result['Section_Mode'] = s.get('mode', -1)

        return result
    except Exception as e:
        print(f"Audio analysis failed for {track_id}: {e}")
        return {}


def try_get_audio_features(track_ids, batch_size=1, max_tracks=20):
    """Try getting audio features one at a time, falling back to audio analysis."""
    sp = create_spotify_client()
    features_list = []
    total = min(len(track_ids), max_tracks)

    for i in range(total):
        track_id = track_ids[i]
        track_data = {'Track ID': track_id}

        if i > 0:
            time.sleep(2)

        # try audio_features first
        try:
            features = sp.audio_features(track_id)
            if features and features[0]:
                f = features[0]
                for key in ['danceability', 'energy', 'valence', 'acousticness',
                            'speechiness', 'instrumentalness', 'liveness', 'tempo']:
                    track_data[key.capitalize()] = f.get(key, 0)
        except Exception:
            pass

        # fallback to audio_analysis
        try:
            analysis = get_track_audio_analysis(track_id)
            track_data.update(analysis)
        except Exception:
            pass

        features_list.append(track_data)

    return features_list


def main():
    PLAYLIST_ID = "1j7MQNNTtHauvdwj109V9k"

    tracks_data = get_playlist_tracks(PLAYLIST_ID, batch_size=50, max_tracks=1200)
    if not tracks_data:
        print("No tracks retrieved.")
        return

    df_tracks = pd.DataFrame(tracks_data)
    df_tracks['Duration (min)'] = df_tracks['Duration (ms)'] / 60000
    df_tracks['Release Year'] = df_tracks['Release Date'].str[:4]

    # look up artist genres
    primary_artists = df_tracks['Primary Artist'].drop_duplicates().tolist()
    sp = create_spotify_client()

    artist_lookup = {}
    for i, artist in enumerate(primary_artists):
        if i > 0 and i % 20 == 0:
            time.sleep(2)
        try:
            results = sp.search(q=f'artist:"{artist}"', type='artist', limit=1)
            if results and results['artists']['items']:
                artist_lookup[artist] = results['artists']['items'][0]['id']
        except Exception:
            time.sleep(3)

    print(f"Found IDs for {len(artist_lookup)}/{len(primary_artists)} artists")

    artist_ids = list(artist_lookup.values())
    artist_info = get_artist_genres(artist_ids[:100])

    # map genres back to artist names
    artist_genres = {}
    for info in artist_info:
        genres = info.get('genres', [])
        popularity = info.get('popularity', 0)
        name = next((n for n, aid in artist_lookup.items() if aid == info['id']), None)
        if name:
            artist_genres[name] = {
                'genres': genres,
                'primary_genre': genres[0] if genres else "Unknown",
                'popularity': popularity
            }

    df_tracks['Artist Genres'] = df_tracks['Primary Artist'].map(
        lambda x: artist_genres.get(x, {}).get('genres', []))
    df_tracks['Primary Genre'] = df_tracks['Primary Artist'].map(
        lambda x: artist_genres.get(x, {}).get('primary_genre', "Unknown"))
    df_tracks['Artist Popularity'] = df_tracks['Primary Artist'].map(
        lambda x: artist_genres.get(x, {}).get('popularity', 0))

    # try to grab audio features for a sample
    track_ids = df_tracks['Track ID'].tolist()
    audio_features = try_get_audio_features(track_ids, max_tracks=20)
    df_features = pd.DataFrame(audio_features)

    if not df_features.empty:
        df_merged = pd.merge(df_tracks, df_features, on='Track ID', how='left')
    else:
        df_merged = df_tracks

    df_merged.to_excel("spotify_playlist_enhanced.xlsx", index=False)

    df_model = df_merged[[c for c in df_merged.columns if c != 'Preview URL']]
    df_model.to_excel("spotify_playlist_ml_enhanced.xlsx", index=False)

    data_dir = project_root / 'data' / 'processed'
    data_dir.mkdir(parents=True, exist_ok=True)
    df_merged.to_excel(data_dir / "spotify_playlist_enhanced.xlsx", index=False)
    df_model.to_excel(data_dir / "spotify_playlist_ml_enhanced.xlsx", index=False)

    print(f"\nDone — {len(df_merged)} tracks")
    print(f"Top artist: {df_merged['Primary Artist'].value_counts().idxmax()}")
    print(f"Top genre: {df_merged['Primary Genre'].value_counts().idxmax()}")
    print(f"Avg popularity: {df_merged['Popularity'].mean():.1f}/100")
    print(f"Avg duration: {df_merged['Duration (min)'].mean():.1f} min")
    print(f"Explicit: {df_merged['Explicit'].sum()} ({df_merged['Explicit'].mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
