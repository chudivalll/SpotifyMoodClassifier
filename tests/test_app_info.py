import os
from pathlib import Path
import requests
import base64
from dotenv import load_dotenv

os.chdir(str(Path(__file__).parent.parent))
load_dotenv()

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Missing Spotify credentials in .env")


def get_client_token():
    """Get a client credentials token directly from the API."""
    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

    resp = requests.post('https://accounts.spotify.com/api/token',
                         headers={'Authorization': f'Basic {auth_header}',
                                  'Content-Type': 'application/x-www-form-urlencoded'},
                         data={'grant_type': 'client_credentials'})

    if resp.status_code == 200:
        return resp.json()
    print(f"Token request failed: {resp.status_code} {resp.text}")
    return None


def check_audio_features():
    """Check if our app can access audio features."""
    token_data = get_client_token()
    if not token_data:
        return

    token = token_data['access_token']
    print(f"Got token (expires in {token_data.get('expires_in')}s)")

    # test with Shape of You
    track_id = "5HNCy40Ni5BZJFw1TKzRsC"
    resp = requests.get(f'https://api.spotify.com/v1/audio-features/{track_id}',
                        headers={'Authorization': f'Bearer {token}'})

    if resp.status_code == 200:
        f = resp.json()
        print(f"Audio features OK: dance={f['danceability']}, energy={f['energy']}, tempo={f['tempo']}")
    else:
        print(f"Audio features failed: {resp.status_code}")
        print(f"  {resp.text}")
        if resp.status_code == 403:
            print("  -> Your app may not have access (post-Nov 2024 restriction)")


if __name__ == "__main__":
    check_audio_features()
