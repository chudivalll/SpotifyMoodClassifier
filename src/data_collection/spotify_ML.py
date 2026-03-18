import sys
import os
from pathlib import Path
import time
from spotipy.exceptions import SpotifyException

spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv

load_dotenv(project_root / '.env')

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')


def create_spotify_client(force_new_token=False):
    """Create and authenticate a Spotify client using OAuth."""
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

    if not client_id or not client_secret:
        raise ValueError("Missing Spotify credentials in .env file")

    scope = "user-read-private user-read-email playlist-read-private playlist-read-collaborative user-top-read user-library-read"
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
        print("\nNo cached token found — you need to authenticate.")
        auth_url = auth_manager.get_authorize_url()
        print(f"\nOpen this URL:\n{auth_url}\n")
        print("After authorizing, paste the full redirect URL below.")
        redirect_url = input("\nRedirect URL: ")

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
    """Extract playlist ID from a Spotify URL, or return as-is."""
    if 'spotify.com' in str(playlist_input):
        return playlist_input.split('/')[-1].split('?')[0]
    return playlist_input


def get_playlist_tracks(playlist_id, batch_size=50, delay_between_batches=2, max_tracks=1200):
    """Fetch tracks from a playlist in batches."""
    sp = create_spotify_client()
    tracks = []
    offset = 0

    try:
        playlist_info = sp.playlist(playlist_id, fields='name,tracks.total')
        total = min(playlist_info['tracks']['total'], max_tracks)
        print(f"Playlist '{playlist_info['name']}': {playlist_info['tracks']['total']} tracks (processing {total})")

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
                           'items.track.album.release_date,items.track.id,items.track.preview_url'
                )

                for item in results['items']:
                    track = item['track']
                    if track:
                        tracks.append({
                            'Name': track['name'],
                            'Artist': ", ".join([a['name'] for a in track['artists']]),
                            'Album': track['album']['name'],
                            'Release Date': track['album']['release_date'],
                            'Track ID': track['id'],
                            'Preview URL': track.get('preview_url', '')
                        })

                offset += current_batch
                print(f"  {len(tracks)}/{total} tracks fetched")

            except SpotifyException as e:
                if "rate limit" in str(e).lower():
                    print(f"Rate limited, waiting 30s...")
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


def get_audio_features(track_ids, batch_size=20, delay_between_batches=3):
    """Fetch audio features in batches.

    Returns empty list if the endpoint is restricted (403) — this happens
    for apps registered after Nov 2024 due to Spotify API changes.
    """
    sp = create_spotify_client()
    all_features = []
    total = len(track_ids)

    print(f"Fetching audio features for {total} tracks...")

    for i in range(0, total, batch_size):
        batch = track_ids[i:i + batch_size]

        if i > 0:
            time.sleep(delay_between_batches)

        try:
            features = sp.audio_features(batch)
            if features:
                valid = [f for f in features if f is not None]
                all_features.extend(valid)
                print(f"  {len(all_features)}/{total} processed")
        except SpotifyException as e:
            if e.http_status == 403:
                print("\nAudio features endpoint returned 403 — your app doesn't have access.")
                print("This is expected for apps registered after Nov 27, 2024.")
                print("The pipeline will fall back to estimated features.\n")
                return []
            if "rate limit" in str(e).lower():
                time.sleep(30)
                sp = create_spotify_client()
                try:
                    features = sp.audio_features(batch)
                    if features:
                        all_features.extend([f for f in features if f])
                    continue
                except:
                    pass
            time.sleep(5)
            continue
        except Exception:
            time.sleep(5)
            continue

    return all_features


def main():
    PLAYLIST_ID = "6jgCEkpKSc7LQI8ZWdAVr6"

    tracks_data = get_playlist_tracks(PLAYLIST_ID, batch_size=50, max_tracks=1200)
    if not tracks_data:
        print("No tracks retrieved.")
        return

    track_ids = [t['Track ID'] for t in tracks_data]
    audio_features = get_audio_features(track_ids, batch_size=20, delay_between_batches=3)
    if not audio_features:
        print("No audio features retrieved.")
        return

    df_tracks = pd.DataFrame(tracks_data)

    feature_cols = ['id', 'danceability', 'energy', 'tempo', 'valence',
                    'acousticness', 'speechiness', 'instrumentalness', 'liveness']
    available_cols = [c for c in feature_cols if c in audio_features[0]]

    df_features = pd.DataFrame(audio_features)[available_cols]

    if 'id' in df_features.columns:
        df_features = df_features.rename(columns={'id': 'Track ID'})
    elif 'track_id' in df_features.columns:
        df_features = df_features.rename(columns={'track_id': 'Track ID'})

    df_final = pd.merge(df_tracks, df_features, on='Track ID', how='inner')

    df_final.to_excel("spotify_playlist_analysis.xlsx", index=False)

    # ML version without preview URLs
    df_model = df_final[[c for c in df_final.columns if c != 'Preview URL']]
    df_model.to_excel("spotify_playlist_ml_data.xlsx", index=False)

    data_dir = project_root / 'data' / 'processed'
    data_dir.mkdir(parents=True, exist_ok=True)
    df_final.to_excel(data_dir / "spotify_playlist_analysis.xlsx", index=False)
    df_model.to_excel(data_dir / "spotify_playlist_ml_data.xlsx", index=False)

    print(f"\nDone — {len(df_final)} tracks processed")
    for col in ['danceability', 'energy', 'valence', 'acousticness']:
        if col in df_final.columns:
            print(f"  avg {col}: {df_final[col].mean():.2f}")


if __name__ == "__main__":
    main()
