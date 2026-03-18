import os
import pandas as pd
from pathlib import Path

spotify_dir = Path(__file__).parent.parent.parent
os.chdir(str(spotify_dir))

input_file = "spotify_playlist_complete.xlsx"
output_file = "spotify_playlist_bpm_enhanced.xlsx"

df = pd.read_excel(input_file)
print(f"Loaded {len(df)} tracks")

if 'Actual_BPM' not in df.columns:
    df['Actual_BPM'] = None
if 'Music_Key' not in df.columns:
    df['Music_Key'] = None

# manually verified BPM/key values — (track, artist) -> (bpm, key)
# all lowercase for matching
KNOWN_BPM = {
    ("i'll erase away your pain", "the whatnauts"): (121, "Ab Major"),

    # pop hits
    ("blinding lights", "the weeknd"): (171, "C# Minor"),
    ("don't start now", "dua lipa"): (124, "Bb Minor"),
    ("watermelon sugar", "harry styles"): (95, "C Major"),
    ("bad guy", "billie eilish"): (135, "G Minor"),
    ("stay", "the kid laroi"): (170, "Bb Minor"),
    ("levitating", "dua lipa"): (103, "B Minor"),
    ("save your tears", "the weeknd"): (118, "C Major"),
    ("montero (call me by your name)", "lil nas x"): (179, "G Major"),
    ("peaches", "justin bieber"): (90, "F# Major"),
    ("dynamite", "bts"): (114, "B Major"),
    ("heat waves", "glass animals"): (81, "E Minor"),
    ("good 4 u", "olivia rodrigo"): (166, "A Minor"),
    ("drivers license", "olivia rodrigo"): (144, "F# Major"),
    ("industry baby", "lil nas x"): (150, "F Minor"),
    ("positions", "ariana grande"): (144, "D Minor"),
    ("mood", "24kgoldn"): (91, "C# Minor"),
    ("without you", "the kid laroi"): (92, "F# Minor"),
    ("leave the door open", "silk sonic"): (148, "Bb Major"),
    ("we're good", "dua lipa"): (124, "Bb Major"),

    # classics
    ("bohemian rhapsody", "queen"): (72, "Bb Major"),
    ("hotel california", "eagles"): (75, "B Minor"),
    ("billie jean", "michael jackson"): (117, "F# Minor"),
    ("sweet child o' mine", "guns n' roses"): (126, "D Major"),
    ("stairway to heaven", "led zeppelin"): (82, "A Minor"),
    ("imagine", "john lennon"): (75, "C Major"),
    ("purple haze", "jimi hendrix"): (108, "E Major"),
    ("smells like teen spirit", "nirvana"): (117, "F Minor"),
    ("hey jude", "the beatles"): (74, "F Major"),
    ("like a rolling stone", "bob dylan"): (96, "C Major"),

    # hip hop
    ("in da club", "50 cent"): (90, "F# Minor"),
    ("hotline bling", "drake"): (135, "D Minor"),
    ("love the way you lie", "eminem"): (87, "G Minor"),
    ("empire state of mind", "jay-z"): (173, "F# Minor"),
    ("alright", "kendrick lamar"): (110, "F Minor"),
    ("god's plan", "drake"): (77, "A Minor"),
    ("humble", "kendrick lamar"): (150, "F# Minor"),
    ("sicko mode", "travis scott"): (155, "F Minor"),
    ("no role modelz", "j. cole"): (100, "F# Minor"),
    ("goosebumps", "travis scott"): (130, "C# Minor"),

    # electronic
    ("levels", "avicii"): (126, "C# Minor"),
    ("one more time", "daft punk"): (123, "D Major"),
    ("strobe", "deadmau5"): (128, "Bb Minor"),
    ("animals", "martin garrix"): (128, "F# Minor"),
    ("titanium", "david guetta"): (126, "Eb Major"),
    ("dont you worry child", "swedish house mafia"): (129, "G Major"),
    ("lean on", "major lazer"): (98, "G Major"),
    ("closer", "the chainsmokers"): (95, "Ab Major"),
    ("scary monsters and nice sprites", "skrillex"): (140, "A Minor"),
    ("wake me up", "avicii"): (124, "B Minor"),

    # alt/rock
    ("creep", "radiohead"): (92, "G Major"),
    ("seven nation army", "the white stripes"): (124, "E Minor"),
    ("somebody that i used to know", "gotye"): (129, "D Major"),
    ("take me to church", "hozier"): (129, "E Minor"),
    ("viva la vida", "coldplay"): (138, "Ab Major"),
    ("do i wanna know?", "arctic monkeys"): (85, "G Minor"),
    ("radioactive", "imagine dragons"): (136, "B Minor"),
    ("sex on fire", "kings of leon"): (153, "A Major"),
    ("the less i know the better", "tame impala"): (117, "Eb Major"),
    ("high and dry", "radiohead"): (92, "F# Major"),
}

# figure out column names
TRACK_COL_NAMES = ['Track Name', 'track_name', 'Name', 'title', 'track']
ARTIST_COL_NAMES = ['Artist', 'artist_name', 'artist', 'artists']

track_col = next((v for v in TRACK_COL_NAMES if v in df.columns), None)
artist_col = next((v for v in ARTIST_COL_NAMES if v in df.columns), None)

if not track_col or not artist_col:
    print(f"ERROR: can't find track/artist columns in {list(df.columns)}")
    exit(1)

matched = 0

for idx, row in df.iterrows():
    track_lower = str(row[track_col]).lower()
    artist_lower = str(row[artist_col]).lower()

    # exact match
    key = (track_lower, artist_lower)
    if key in KNOWN_BPM:
        bpm, music_key = KNOWN_BPM[key]
        df.at[idx, 'Actual_BPM'] = bpm
        df.at[idx, 'Music_Key'] = music_key
        matched += 1
        continue

    # fuzzy: check if known track/artist is a substring
    for (known_track, known_artist), (bpm, music_key) in KNOWN_BPM.items():
        if ((known_track in track_lower or track_lower in known_track) and
                (known_artist in artist_lower or artist_lower in known_artist)):
            df.at[idx, 'Actual_BPM'] = bpm
            df.at[idx, 'Music_Key'] = music_key
            matched += 1
            break

# update tempo column if it exists
tempo_col = next((c for c in df.columns if c in ['Estimated_Tempo', 'Tempo', 'tempo', 'bpm', 'BPM']), None)
if tempo_col:
    mask = df['Actual_BPM'].notna()
    df.loc[mask, tempo_col] = df.loc[mask, 'Actual_BPM']
    print(f"Updated {mask.sum()} tempo values")

# save
df.to_excel(output_file, index=False)

ml_file = "spotify_playlist_ml_bpm_enhanced.xlsx"
ml_cols = [c for c in df.columns if 'preview' not in c.lower()]
df[ml_cols].to_excel(ml_file, index=False)

data_dir = spotify_dir / 'data' / 'processed'
data_dir.mkdir(parents=True, exist_ok=True)
df.to_excel(data_dir / output_file, index=False)
df[ml_cols].to_excel(data_dir / ml_file, index=False)

bpm_count = df['Actual_BPM'].notna().sum()
print(f"\nMatched {matched}/{len(df)} tracks from known BPM data")
print(f"Total with BPM: {bpm_count} ({bpm_count/len(df)*100:.0f}%)")
if bpm_count > 0:
    print(f"BPM range: {df['Actual_BPM'].min():.0f} - {df['Actual_BPM'].max():.0f}")
