import os
import sys
import time
import json
import random
import hashlib
import base64
import urllib.parse
from pathlib import Path
from flask import Flask, render_template, request, redirect, session, jsonify
import requests
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mood_classifier import classify_mood, assign_moods

app = Flask(__name__)
app.secret_key = os.urandom(32)

# Spotify OAuth config — set these in .env or environment
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', '')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', '')
REDIRECT_URI = 'http://127.0.0.1:5000/callback'
SCOPE = 'playlist-read-private playlist-read-collaborative user-library-read'

SPOTIFY_AUTH_URL = 'https://accounts.spotify.com/authorize'
SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token'
SPOTIFY_API_BASE = 'https://api.spotify.com/v1'

# pre-written cultural context for songs (keyed by lowercase track name)
SONG_STORIES = {
    "blinding lights": {
        "story": "The Weeknd wrote this after a late night in Las Vegas, channeling the loneliness behind the neon. The synth line was inspired by a-ha's 'Take On Me' and the production draws from the same Yamaha DX7 that defined 80s pop. The song broke the record for longest-charting Hot 100 hit ever.",
        "era": "2020 — Released during the early days of the pandemic, it became the unofficial soundtrack to empty city streets lit by nothing but traffic lights.",
        "image_mood": "neon_night"
    },
    "bohemian rhapsody": {
        "story": "Freddie Mercury spent three weeks layering 180 vocal overdubs in a tiny London studio. The song was rejected by every radio programmer until DJ Kenny Everett played it 14 times in two days. It fuses opera, ballad, and hard rock in a way nobody had attempted before — or has fully replicated since.",
        "era": "1975 — Recorded at Rockfield Studios in rural Wales, where Queen escaped the London music scene to create something nobody expected.",
        "image_mood": "dramatic"
    },
    "billie jean": {
        "story": "Michael Jackson wrote the bassline walking down a street in LA — the melody was so consuming he didn't notice his car had caught fire behind him. The song is about an obsessive fan who claimed he was the father of her child. The drum pattern, played by Ndugu Chancler, became one of the most sampled beats in history.",
        "era": "1982 — The Thriller era. Jackson performed the moonwalk for the first time during this song on Motown 25, and pop music was never the same.",
        "image_mood": "iconic"
    },
    "smells like teen spirit": {
        "story": "Kurt Cobain wanted to write 'the ultimate pop song' as a joke, ripping off the Pixies' quiet-loud-quiet formula. The title came from a friend writing 'Kurt smells like Teen Spirit' on his wall — it was a deodorant brand. Cobain didn't know that. He thought it was a revolutionary slogan.",
        "era": "1991 — Grunge killed hair metal overnight. This song was the gunshot that ended the 80s.",
        "image_mood": "raw"
    },
    "hotel california": {
        "story": "Don Henley described it as a journey from innocence to experience, a metaphor for the dark underbelly of the American Dream. The dual guitar harmony in the outro, played by Don Felder and Joe Walsh, was composed in a single take and became the most recognizable guitar moment in rock history.",
        "era": "1977 — Post-Vietnam, post-Watergate. America was questioning everything it believed in, and this song put that disillusionment to music.",
        "image_mood": "sunset"
    },
    "humble.": {
        "story": "Kendrick recorded this in one take after hearing the Mike WiLL Made-It beat. The music video, shot by Dave Meyers, recreates Renaissance paintings with Kendrick as the subject — a deliberate commentary on who gets to be immortalized in art. The song won a Grammy and Billboard Music Award.",
        "era": "2017 — DAMN. came out during heightened conversations about race in America. Kendrick became the first rapper to win the Pulitzer Prize for Music the following year.",
        "image_mood": "powerful"
    },
    "drivers license": {
        "story": "Olivia Rodrigo wrote this at 17, processing a real breakup. She recorded the bridge vocals crying in the studio. The song broke Spotify's record for most streams in a single day within a week of release — all without a major label push, just raw teenage heartbreak that millions recognized as their own.",
        "era": "2021 — Gen Z's first great heartbreak anthem. It proved that a teenager with a piano could still break through in the age of algorithms.",
        "image_mood": "melancholy"
    },
    "levels": {
        "story": "Avicii built this track around a vocal sample from Etta James' 'Something's Got a Hold on Me,' bridging 1960s soul with 2011 festival EDM. Tim Bergling was 21 when this changed dance music forever. His struggle with the touring lifestyle that followed would become one of music's most tragic stories.",
        "era": "2011 — The EDM boom. Suddenly DJs were headlining stadiums and electronic music crossed from underground to mainstream pop.",
        "image_mood": "euphoric"
    },
    "one more time": {
        "story": "Daft Punk ran Romanthony's vocals through a vocoder until they became something between human and machine — the thesis statement of their entire career. The song opened the album Discovery, which was inspired by the French duo's childhood memories of Saturday morning cartoons and disco records.",
        "era": "2001 — The dawn of the digital age. Daft Punk made robots feel more human than most pop stars.",
        "image_mood": "futuristic"
    },
    "god's plan": {
        "story": "Drake was given a $996,631 budget for the music video and gave every dollar away to people in Miami — students, shoppers, families. The video is just footage of him handing out money and watching people react. It's one of the most expensive music videos ever made, and not a cent went to production.",
        "era": "2018 — Peak Drake. The song debuted at #1 and stayed there for 11 weeks, defining the streaming era's relationship with rap.",
        "image_mood": "generous"
    },
    "stairway to heaven": {
        "story": "Jimmy Page wrote the guitar parts in a remote cottage in Wales called Bron-Yr-Aur, which had no running water or electricity. The song builds from acoustic folk to explosive rock over 8 minutes — a structure that broke every rule radio had about song length. It remains the most requested song on FM radio despite never being released as a single.",
        "era": "1971 — Led Zeppelin IV didn't even have the band's name on the cover. They let the music speak for itself.",
        "image_mood": "mystical"
    },
    "bad guy": {
        "story": "Billie Eilish and her brother Finneas recorded this in their childhood bedroom in Highland Park, LA. The bass that shakes your chest was made on a MacBook. Billie was 17 when this hit #1, making her the first artist born in the 2000s to top the chart. The whispered vocals were a deliberate rejection of pop's belting tradition.",
        "era": "2019 — Bedroom pop went from a niche genre tag to the sound that dethroned 'Old Town Road.' The music industry realized it didn't need million-dollar studios anymore.",
        "image_mood": "dark_playful"
    },
    "sicko mode": {
        "story": "Travis Scott stitched three completely different beats into one song — a structure borrowed from prog rock and applied to trap. The beat switches became a meme, a TikTok trend, and a blueprint for a generation of producers who stopped thinking in verses and choruses.",
        "era": "2018 — ASTROWORLD turned Travis Scott from a rapper into a cultural architect. The album was named after a demolished Houston amusement park from his childhood.",
        "image_mood": "chaotic"
    },
    "shape of you": {
        "story": "Ed Sheeran originally wrote this for Rihanna. The marimba riff was a placeholder that was supposed to be replaced — but it worked so well they kept it. The song spent 33 weeks in the top 10, longer than any song in Hot 100 history at the time.",
        "era": "2017 — Streaming had fully taken over, and Sheeran proved that a guy with an acoustic guitar could compete with the biggest pop productions.",
        "image_mood": "warm"
    },
    "imagine": {
        "story": "John Lennon wrote this on a white grand piano in his Tittenhurst Park mansion — the irony of a millionaire singing 'imagine no possessions' wasn't lost on critics. But the simplicity of the melody, just a few chords and a plea, made it the closest thing the world has to a secular hymn.",
        "era": "1971 — The Vietnam War was still raging. Lennon had just broken up the Beatles. This was his answer to everything.",
        "image_mood": "peaceful"
    },
    "happy": {
        "story": "Pharrell wrote this for the Despicable Me 2 soundtrack and originally envisioned it for CeeLo Green. After nine other attempts at the song failed, he wrote the final version in about 10 minutes. The 24-hour music video — the world's first — featured 400 people dancing across Los Angeles.",
        "era": "2013 — The song became an anthem for movements worldwide, from flash mobs to political protests. UN designated March 20 as International Day of Happiness, and this became its unofficial theme.",
        "image_mood": "joyful"
    },
    "creep": {
        "story": "Radiohead nearly didn't release this because they thought it was too simple. Thom Yorke wrote it at Exeter University about a woman he was obsessed with. Jonny Greenwood hated the song so much he tried to sabotage it during recording by playing the jarring guitar crunch before each chorus — but that accidental aggression became the song's most iconic moment.",
        "era": "1993 — Britpop was about to explode, but Radiohead were already writing about alienation and self-loathing. They were too weird for the mainstream and too catchy for the underground.",
        "image_mood": "alienated"
    },
    "lovely": {
        "story": "Billie Eilish recorded this with Khalid for the 13 Reasons Why soundtrack. The song captures the feeling of depression as a physical space you can't leave — the room gets smaller, the walls close in. It became an anthem for Gen Z mental health conversations.",
        "era": "2018 — The mental health conversation in music shifted from metaphor to direct honesty. Artists stopped pretending everything was fine.",
        "image_mood": "heavy"
    },
    "heat waves": {
        "story": "Dave Bayley wrote this about missing a close friend during the pandemic lockdowns. The song initially underperformed, then slowly climbed the charts over 59 weeks — the longest journey to #1 in Hot 100 history. It was carried entirely by TikTok and fan-made Minecraft videos.",
        "era": "2020 — The pandemic era, when missing someone wasn't a choice but a mandate. The song's slow burn mirrored how time felt during lockdown — endless and aching.",
        "image_mood": "hazy"
    },
    "uptown funk": {
        "story": "Mark Ronson and Bruno Mars fought over this song for months — Mars wanted it funkier, Ronson wanted it tighter. They rewrote it dozens of times, channeling Morris Day, James Brown, and Prince. The result was the most successful funk record since the genre's 1970s peak.",
        "era": "2014 — Retro-funk hadn't been commercially viable in decades. This song single-handedly brought it back and proved that groove never goes out of style.",
        "image_mood": "flashy"
    },
}

# generic stories for songs not in the dict
GENERIC_STORIES = [
    "Every song carries the fingerprint of its era — the production techniques, the cultural anxieties, the technology available. This track is a time capsule of the moment it was made.",
    "Music doesn't exist in a vacuum. Every chord progression, every drum pattern, every vocal inflection was shaped by the artists who came before and the world happening outside the studio.",
    "Behind every three-minute song is a story — months of writing, rewriting, late nights, creative arguments, and the specific alchemy of the right people in the right room at the right time.",
]


def get_song_story(track_name):
    """Get cultural context for a song."""
    key = track_name.lower().strip()
    # exact match
    if key in SONG_STORIES:
        return SONG_STORIES[key]
    # partial match
    for k, v in SONG_STORIES.items():
        if k in key or key in k:
            return v
    # generic fallback
    return {
        "story": random.choice(GENERIC_STORIES),
        "era": "",
        "image_mood": "default"
    }


def analyze_playlist_data(tracks):
    """Run mood classification and build the wrapped-style analysis."""
    df = pd.DataFrame(tracks)

    # try to assign moods if we have the features
    if 'energy' in df.columns and 'valence' in df.columns:
        df = assign_moods(df)
    else:
        df['mood'] = 'unknown'

    mood_counts = df['mood'].value_counts().to_dict() if 'mood' in df.columns else {}

    # genre breakdown
    genre_counts = {}
    if 'primary_genre' in df.columns:
        genre_counts = df['primary_genre'].value_counts().head(8).to_dict()
    elif 'genre' in df.columns:
        genre_counts = df['genre'].value_counts().head(8).to_dict()

    # top artists
    artist_col = 'artist' if 'artist' in df.columns else 'Artist' if 'Artist' in df.columns else None
    top_artists = []
    if artist_col:
        top_artists = df[artist_col].value_counts().head(5).index.tolist()

    # decade breakdown
    decade_counts = {}
    date_col = 'release_date' if 'release_date' in df.columns else 'Release Date' if 'Release Date' in df.columns else None
    if date_col:
        years = pd.to_numeric(df[date_col].astype(str).str[:4], errors='coerce').dropna()
        decades = (years // 10 * 10).astype(int)
        decade_counts = decades.value_counts().sort_index().to_dict()
        decade_counts = {f"{k}s": v for k, v in decade_counts.items()}

    # build track cards with stories
    name_col = 'name' if 'name' in df.columns else 'Name'
    artist_display = 'artist' if 'artist' in df.columns else 'Artist'
    album_col = 'album' if 'album' in df.columns else 'Album'

    cards = []
    for _, row in df.iterrows():
        track_name = row.get(name_col, 'Unknown')
        story = get_song_story(track_name)
        cards.append({
            'name': track_name,
            'artist': row.get(artist_display, 'Unknown'),
            'album': row.get(album_col, ''),
            'mood': row.get('mood', 'unknown'),
            'energy': round(row.get('energy', 0.5), 2),
            'valence': round(row.get('valence', 0.5), 2),
            'danceability': round(row.get('danceability', 0.5), 2),
            'tempo': round(row.get('tempo', 120), 0),
            'story': story['story'],
            'era': story.get('era', ''),
            'image_mood': story.get('image_mood', 'default'),
            'album_art': row.get('album_art', ''),
            'release_date': str(row.get(date_col, '') if date_col else ''),
        })

    return {
        'mood_counts': mood_counts,
        'genre_counts': genre_counts,
        'top_artists': top_artists,
        'decade_counts': decade_counts,
        'total_tracks': len(df),
        'cards': cards,
    }


@app.route('/')
def index():
    logged_in = 'access_token' in session
    return render_template('index.html', logged_in=logged_in, client_id=CLIENT_ID)


@app.route('/login')
def login():
    params = {
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': REDIRECT_URI,
        'scope': SCOPE,
        'show_dialog': 'true',
    }
    url = f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return redirect(url)


@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return redirect('/')

    resp = requests.post(SPOTIFY_TOKEN_URL, data={
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    })

    if resp.status_code != 200:
        return redirect('/')

    tokens = resp.json()
    session['access_token'] = tokens['access_token']
    session['refresh_token'] = tokens.get('refresh_token', '')

    return redirect('/playlists')


@app.route('/playlists')
def playlists():
    token = session.get('access_token')
    if not token:
        return redirect('/login')

    headers = {'Authorization': f'Bearer {token}'}

    # get user profile
    user_resp = requests.get(f'{SPOTIFY_API_BASE}/me', headers=headers)
    user = user_resp.json() if user_resp.status_code == 200 else {}

    # get playlists
    pl_resp = requests.get(f'{SPOTIFY_API_BASE}/me/playlists?limit=50', headers=headers)
    playlists_data = []
    if pl_resp.status_code == 200:
        for pl in pl_resp.json().get('items', []):
            img = pl.get('images', [{}])
            playlists_data.append({
                'id': pl['id'],
                'name': pl['name'],
                'tracks': pl['tracks']['total'],
                'image': img[0]['url'] if img else '',
                'owner': pl.get('owner', {}).get('display_name', ''),
            })

    return render_template('index.html',
                           logged_in=True,
                           show_playlists=True,
                           playlists=playlists_data,
                           user=user,
                           client_id=CLIENT_ID)


@app.route('/analyze/<playlist_id>')
def analyze(playlist_id):
    token = session.get('access_token')
    if not token:
        return redirect('/login')

    headers = {'Authorization': f'Bearer {token}'}

    # get playlist info
    pl_resp = requests.get(f'{SPOTIFY_API_BASE}/playlists/{playlist_id}', headers=headers)
    if pl_resp.status_code != 200:
        return redirect('/playlists')
    playlist_info = pl_resp.json()

    # get all tracks
    tracks = []
    offset = 0
    total = playlist_info['tracks']['total']

    while offset < min(total, 200):  # cap at 200 for performance
        tr_resp = requests.get(
            f'{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks?offset={offset}&limit=50',
            headers=headers
        )
        if tr_resp.status_code != 200:
            break

        for item in tr_resp.json().get('items', []):
            t = item.get('track')
            if not t:
                continue
            artists = [a['name'] for a in t.get('artists', [])]
            album_images = t.get('album', {}).get('images', [])

            track_data = {
                'name': t['name'],
                'artist': ', '.join(artists),
                'album': t.get('album', {}).get('name', ''),
                'release_date': t.get('album', {}).get('release_date', ''),
                'popularity': t.get('popularity', 50),
                'explicit': t.get('explicit', False),
                'duration_ms': t.get('duration_ms', 0),
                'track_id': t['id'],
                'album_art': album_images[0]['url'] if album_images else '',
            }

            # try to get audio features (will 403 for new apps)
            # we'll batch these after
            tracks.append(track_data)

        offset += 50

    # try to get audio features in batch
    track_ids = [t['track_id'] for t in tracks if t.get('track_id')]
    audio_features = {}

    for i in range(0, len(track_ids), 50):
        batch = track_ids[i:i+50]
        ids_str = ','.join(batch)
        af_resp = requests.get(
            f'{SPOTIFY_API_BASE}/audio-features?ids={ids_str}',
            headers=headers
        )
        if af_resp.status_code == 200:
            for feat in af_resp.json().get('audio_features', []):
                if feat:
                    audio_features[feat['id']] = feat
        elif af_resp.status_code == 403:
            break  # API restricted, fall back to estimation
        time.sleep(0.5)

    # merge audio features into tracks
    for track in tracks:
        tid = track.get('track_id', '')
        if tid in audio_features:
            af = audio_features[tid]
            track['energy'] = af.get('energy', 0.5)
            track['valence'] = af.get('valence', 0.5)
            track['danceability'] = af.get('danceability', 0.5)
            track['tempo'] = af.get('tempo', 120)
            track['acousticness'] = af.get('acousticness', 0.5)
            track['speechiness'] = af.get('speechiness', 0.1)
            track['instrumentalness'] = af.get('instrumentalness', 0.0)
            track['loudness'] = af.get('loudness', -6.0)

    # if no audio features, estimate them
    if not audio_features:
        from src.data_processing.add_missing_columns import estimate_audio_features
        df = pd.DataFrame(tracks)
        df = estimate_audio_features(df)
        tracks = df.to_dict('records')

    # get artist genres for tracks
    artist_names = list(set(t['artist'].split(',')[0].strip() for t in tracks))
    for i in range(0, min(len(artist_names), 50), 50):
        # search each artist for genre
        for artist_name in artist_names[i:i+50]:
            try:
                search_resp = requests.get(
                    f'{SPOTIFY_API_BASE}/search?q=artist:"{artist_name}"&type=artist&limit=1',
                    headers=headers
                )
                if search_resp.status_code == 200:
                    items = search_resp.json().get('artists', {}).get('items', [])
                    if items:
                        genres = items[0].get('genres', [])
                        primary_genre = genres[0] if genres else 'pop'
                        for t in tracks:
                            if t['artist'].split(',')[0].strip() == artist_name:
                                t['primary_genre'] = primary_genre
            except Exception:
                pass
            time.sleep(0.1)

    analysis = analyze_playlist_data(tracks)

    playlist_img = ''
    if playlist_info.get('images'):
        playlist_img = playlist_info['images'][0]['url']

    return render_template('index.html',
                           logged_in=True,
                           show_analysis=True,
                           analysis=analysis,
                           playlist_name=playlist_info['name'],
                           playlist_image=playlist_img,
                           client_id=CLIENT_ID)


@app.route('/demo')
def demo():
    """Run analysis on sample data without Spotify login."""
    sample_path = project_root / 'data' / 'sample' / 'sample_playlist.csv'
    df = pd.read_csv(sample_path)

    tracks = df.to_dict('records')
    # add placeholder album art
    for t in tracks:
        if not t.get('album_art'):
            t['album_art'] = ''

    analysis = analyze_playlist_data(tracks)

    return render_template('index.html',
                           logged_in=False,
                           show_analysis=True,
                           analysis=analysis,
                           playlist_name='Sample Playlist',
                           playlist_image='',
                           client_id=CLIENT_ID)


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', '')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', '')
    app.run(debug=True, port=5000)
