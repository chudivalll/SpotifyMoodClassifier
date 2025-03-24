import os
from pathlib import Path
import requests
import base64
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

print(f"Client ID: {CLIENT_ID}")
# Don't print actual secret for security reasons
print(f"Client Secret length: {len(CLIENT_SECRET) if CLIENT_SECRET else 'Not found'}")

# Function to get an access token with client credentials flow
def get_client_token():
    """Get a client credentials access token directly from Spotify API"""
    auth_url = 'https://accounts.spotify.com/api/token'
    
    # Create basic auth header
    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    
    headers = {
        'Authorization': f'Basic {auth_header}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    payload = {'grant_type': 'client_credentials'}
    
    try:
        response = requests.post(auth_url, headers=headers, data=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        token_data = response.json()
        return token_data
    except requests.exceptions.RequestException as e:
        print(f"Error getting token: {str(e)}")
        if response and response.text:
            print(f"Response: {response.text}")
        return None

# Function to check if the audio-features endpoint requires additional permissions
def check_audio_features_permissions():
    """Check the audio-features endpoint to see what permissions might be needed"""
    token_data = get_client_token()
    
    if not token_data or 'access_token' not in token_data:
        print("❌ Failed to get access token")
        return
    
    access_token = token_data['access_token']
    token_type = token_data.get('token_type', 'Bearer')
    expires_in = token_data.get('expires_in', 'unknown')
    
    print(f"✅ Successfully got access token")
    print(f"Token type: {token_type}")
    print(f"Expires in: {expires_in} seconds")
    
    # Popular track ID for testing
    track_id = "5HNCy40Ni5BZJFw1TKzRsC"  # Shape of You by Ed Sheeran
    
    # Try to get audio features with the token
    features_url = f'https://api.spotify.com/v1/audio-features/{track_id}'
    headers = {'Authorization': f'{token_type} {access_token}'}
    
    try:
        print(f"\nChecking audio features for track: {track_id}")
        response = requests.get(features_url, headers=headers)
        
        if response.status_code == 200:
            print("✅ Successfully accessed audio features!")
            features = response.json()
            print(f"- Danceability: {features.get('danceability')}")
            print(f"- Energy: {features.get('energy')}")
            print(f"- Tempo: {features.get('tempo')} BPM")
        else:
            print(f"❌ Failed to access audio features. Status: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {str(e)}")

# Check app restrictions
def check_app_restrictions():
    """Check if there are any app restrictions that might be causing the 403 error"""
    print("\n=== CHECKING IF YOUR APP HAS RESTRICTIONS ===")
    print("Some causes of 403 errors with Spotify API:")
    print("1. Your app might be in development mode (limited to 25 users)")
    print("2. Your app might not have the correct scopes set up")
    print("3. Your account might not have the correct subscription level")
    print("4. There might be regional restrictions for the content")
    print("\nPlease check your Spotify Developer Dashboard for app settings.")
    print("URL: https://developer.spotify.com/dashboard/applications")
    
    # Prompt for app status
    print("\nIs your app in development mode? (Yes/No)")
    dev_mode = input("> ").strip().lower()
    
    if dev_mode == 'yes':
        print("\nDevelopment mode restricts your app to 25 users.")
        print("If you're the only user, this should not be a problem.")
    
    # Prompt for redirect URI
    print("\nWhat is the exact redirect URI configured in your app?")
    redirect_uri = input("> ").strip()
    
    print(f"\nYour redirect URI: {redirect_uri}")
    print("This MUST match exactly what you use in your code.")
    print("Current code is using: http://127.0.0.1:8888/callback")
    
    if redirect_uri != "http://127.0.0.1:8888/callback":
        print("\n⚠️ Your redirect URI doesn't match what's in the code!")
        print("You should update your code or your Spotify Dashboard settings.")

if __name__ == "__main__":
    print("=== SPOTIFY APP DIAGNOSTICS ===")
    print("This script will help diagnose issues with your Spotify API access.")
    
    # Check if we can get a token directly
    print("\n=== CHECKING CLIENT CREDENTIALS ===")
    check_audio_features_permissions()
    
    # Check app restrictions
    check_app_restrictions() 