"""Generate sample_playlist.csv from known song data with realistic audio features."""
import csv
import random
import os

random.seed(42)

# known songs with verified BPM + realistic audio features
# format: (name, artist, genre, bpm, key, energy, valence, danceability, acousticness, speechiness, instrumentalness, loudness, popularity, explicit, duration_ms, album, release_date)
SONGS = [
    # pop — generally happy/energetic
    ("Blinding Lights", "The Weeknd", "pop", 171, "C# Minor", 0.73, 0.81, 0.51, 0.00, 0.06, 0.00, -4.1, 87, False, 200040, "After Hours", "2020-03-20"),
    ("Don't Start Now", "Dua Lipa", "pop", 124, "Bb Minor", 0.79, 0.67, 0.79, 0.01, 0.08, 0.00, -4.6, 84, False, 183290, "Future Nostalgia", "2020-03-27"),
    ("Watermelon Sugar", "Harry Styles", "pop", 95, "C Major", 0.82, 0.56, 0.55, 0.12, 0.05, 0.00, -4.9, 83, False, 174000, "Fine Line", "2019-12-13"),
    ("Levitating", "Dua Lipa", "pop", 103, "B Minor", 0.73, 0.91, 0.70, 0.01, 0.07, 0.00, -3.8, 82, False, 203064, "Future Nostalgia", "2020-03-27"),
    ("Save Your Tears", "The Weeknd", "pop", 118, "C Major", 0.63, 0.64, 0.68, 0.03, 0.03, 0.00, -5.5, 80, False, 215627, "After Hours", "2020-03-20"),
    ("Dynamite", "BTS", "pop", 114, "B Major", 0.76, 0.94, 0.75, 0.01, 0.10, 0.00, -4.4, 78, False, 199054, "Dynamite", "2020-08-21"),
    ("Peaches", "Justin Bieber", "pop", 90, "F# Major", 0.68, 0.68, 0.68, 0.32, 0.12, 0.00, -6.2, 76, True, 198082, "Justice", "2021-03-19"),
    ("Positions", "Ariana Grande", "pop", 144, "D Minor", 0.68, 0.65, 0.76, 0.44, 0.06, 0.00, -5.9, 75, True, 172325, "Positions", "2020-10-30"),
    ("Stay", "The Kid LAROI", "pop", 170, "Bb Minor", 0.76, 0.50, 0.59, 0.04, 0.05, 0.00, -5.3, 85, True, 141806, "F*ck Love 3", "2021-07-09"),
    ("We're Good", "Dua Lipa", "pop", 124, "Bb Major", 0.59, 0.72, 0.62, 0.05, 0.04, 0.00, -6.3, 68, False, 165507, "Future Nostalgia", "2021-02-11"),

    # sad/melancholy — low energy, low valence
    ("Drivers License", "Olivia Rodrigo", "pop", 144, "F# Major", 0.43, 0.13, 0.59, 0.72, 0.06, 0.00, -8.8, 81, True, 242014, "SOUR", "2021-05-21"),
    ("Without You", "The Kid LAROI", "pop", 92, "F# Minor", 0.47, 0.26, 0.56, 0.55, 0.04, 0.00, -7.5, 74, True, 160493, "F*ck Love", "2020-07-24"),
    ("Creep", "Radiohead", "alternative", 92, "G Major", 0.40, 0.22, 0.52, 0.16, 0.04, 0.02, -6.1, 71, False, 238640, "Pablo Honey", "1993-02-22"),
    ("High and Dry", "Radiohead", "alternative", 92, "F# Major", 0.36, 0.35, 0.45, 0.05, 0.03, 0.01, -8.2, 62, False, 257200, "The Bends", "1995-03-13"),
    ("Hotel California", "Eagles", "rock", 75, "B Minor", 0.38, 0.29, 0.42, 0.08, 0.03, 0.01, -9.5, 78, False, 391000, "Hotel California", "1977-02-22"),
    ("Stairway to Heaven", "Led Zeppelin", "rock", 82, "A Minor", 0.34, 0.21, 0.33, 0.23, 0.07, 0.04, -10.3, 72, False, 482000, "Led Zeppelin IV", "1971-11-08"),
    ("Imagine", "John Lennon", "rock", 75, "C Major", 0.22, 0.49, 0.38, 0.87, 0.04, 0.00, -11.2, 70, False, 187733, "Imagine", "1971-10-11"),

    # energetic/aggressive — high energy, low valence
    ("Bad Guy", "Billie Eilish", "pop", 135, "G Minor", 0.43, 0.56, 0.70, 0.33, 0.38, 0.13, -11.0, 86, False, 194088, "WHEN WE ALL FALL ASLEEP", "2019-03-29"),
    ("Good 4 U", "Olivia Rodrigo", "pop", 166, "A Minor", 0.66, 0.69, 0.56, 0.02, 0.15, 0.00, -5.0, 82, True, 178147, "SOUR", "2021-05-21"),
    ("MONTERO", "Lil Nas X", "pop", 179, "G Major", 0.71, 0.61, 0.61, 0.08, 0.12, 0.00, -4.7, 79, True, 137876, "MONTERO", "2021-09-17"),
    ("Industry Baby", "Lil Nas X", "hip hop", 150, "F Minor", 0.71, 0.54, 0.74, 0.12, 0.22, 0.00, -5.7, 77, True, 212000, "MONTERO", "2021-09-17"),
    ("Smells Like Teen Spirit", "Nirvana", "rock", 117, "F Minor", 0.91, 0.26, 0.49, 0.00, 0.07, 0.01, -6.4, 75, False, 301920, "Nevermind", "1991-09-24"),
    ("HUMBLE.", "Kendrick Lamar", "hip hop", 150, "F# Minor", 0.62, 0.42, 0.91, 0.00, 0.10, 0.00, -6.7, 78, True, 177000, "DAMN.", "2017-04-14"),
    ("SICKO MODE", "Travis Scott", "hip hop", 155, "F Minor", 0.73, 0.45, 0.83, 0.01, 0.22, 0.00, -3.7, 76, True, 312820, "ASTROWORLD", "2018-08-03"),
    ("Goosebumps", "Travis Scott", "hip hop", 130, "C# Minor", 0.59, 0.40, 0.84, 0.06, 0.05, 0.00, -5.3, 74, True, 243827, "Birds in the Trap", "2016-09-02"),
    ("No Role Modelz", "J. Cole", "hip hop", 100, "F# Minor", 0.63, 0.51, 0.78, 0.25, 0.30, 0.00, -6.1, 73, True, 293067, "2014 Forest Hills Drive", "2014-12-09"),
    ("Scary Monsters and Nice Sprites", "Skrillex", "electronic", 140, "A Minor", 0.93, 0.20, 0.50, 0.02, 0.13, 0.65, -4.1, 65, False, 243345, "Scary Monsters", "2010-06-07"),
    ("Purple Haze", "Jimi Hendrix", "rock", 108, "E Major", 0.86, 0.39, 0.35, 0.03, 0.10, 0.09, -9.7, 60, False, 170893, "Are You Experienced", "1967-05-12"),

    # chill — low energy, high valence
    ("Heat Waves", "Glass Animals", "pop", 81, "E Minor", 0.53, 0.34, 0.76, 0.29, 0.06, 0.00, -7.6, 80, False, 238805, "Dreamland", "2020-08-07"),
    ("Mood", "24kGoldn", "hip hop", 91, "C# Minor", 0.64, 0.73, 0.70, 0.18, 0.05, 0.00, -5.4, 72, True, 140533, "El Dorado", "2021-03-26"),
    ("Lean On", "Major Lazer", "electronic", 98, "G Major", 0.66, 0.47, 0.71, 0.01, 0.07, 0.00, -4.1, 73, False, 176561, "Peace is the Mission", "2015-06-01"),
    ("Closer", "The Chainsmokers", "electronic", 95, "Ab Major", 0.52, 0.66, 0.75, 0.41, 0.03, 0.00, -5.9, 77, False, 244960, "Collage", "2016-07-29"),
    ("Leave The Door Open", "Silk Sonic", "r&b", 148, "Bb Major", 0.46, 0.74, 0.59, 0.62, 0.04, 0.00, -7.4, 75, False, 242096, "An Evening with Silk Sonic", "2021-11-12"),
    ("Do I Wanna Know?", "Arctic Monkeys", "rock", 85, "G Minor", 0.54, 0.32, 0.55, 0.16, 0.03, 0.00, -6.6, 76, False, 272394, "AM", "2013-09-09"),
    ("The Less I Know the Better", "Tame Impala", "alternative", 117, "Eb Major", 0.74, 0.78, 0.64, 0.02, 0.03, 0.01, -5.2, 78, False, 216320, "Currents", "2015-07-17"),

    # classic rock — mixed moods
    ("Bohemian Rhapsody", "Queen", "rock", 72, "Bb Major", 0.40, 0.23, 0.39, 0.28, 0.05, 0.00, -9.9, 80, False, 354320, "A Night at the Opera", "1975-10-31"),
    ("Billie Jean", "Michael Jackson", "pop", 117, "F# Minor", 0.82, 0.88, 0.85, 0.06, 0.06, 0.01, -3.2, 82, False, 294227, "Thriller", "1982-11-30"),
    ("Sweet Child O' Mine", "Guns N' Roses", "rock", 126, "D Major", 0.82, 0.48, 0.42, 0.02, 0.04, 0.01, -5.4, 76, False, 356000, "Appetite for Destruction", "1987-07-21"),
    ("Hey Jude", "The Beatles", "rock", 74, "F Major", 0.36, 0.54, 0.38, 0.41, 0.03, 0.00, -7.9, 74, False, 431000, "Past Masters", "1968-08-26"),
    ("Like a Rolling Stone", "Bob Dylan", "rock", 96, "C Major", 0.55, 0.52, 0.46, 0.52, 0.06, 0.00, -8.5, 65, False, 369600, "Highway 61 Revisited", "1965-08-30"),
    ("Viva la Vida", "Coldplay", "rock", 138, "Ab Major", 0.62, 0.49, 0.53, 0.11, 0.03, 0.00, -6.1, 78, False, 242282, "Viva la Vida", "2008-06-12"),
    ("Seven Nation Army", "The White Stripes", "rock", 124, "E Minor", 0.84, 0.38, 0.73, 0.01, 0.05, 0.00, -4.3, 71, False, 231733, "Elephant", "2003-04-01"),
    ("Radioactive", "Imagine Dragons", "rock", 136, "B Minor", 0.77, 0.24, 0.52, 0.11, 0.06, 0.00, -3.9, 76, False, 186813, "Night Visions", "2012-09-04"),
    ("Sex on Fire", "Kings of Leon", "rock", 153, "A Major", 0.85, 0.58, 0.49, 0.01, 0.04, 0.00, -4.2, 73, False, 203573, "Only by the Night", "2008-09-19"),
    ("Take Me to Church", "Hozier", "alternative", 129, "E Minor", 0.66, 0.43, 0.57, 0.19, 0.05, 0.00, -5.4, 77, False, 241693, "Hozier", "2014-09-19"),
    ("Somebody That I Used to Know", "Gotye", "alternative", 129, "D Major", 0.53, 0.61, 0.64, 0.27, 0.04, 0.00, -7.3, 74, False, 244907, "Making Mirrors", "2011-08-26"),

    # electronic/dance — mostly energetic
    ("Levels", "Avicii", "electronic", 126, "C# Minor", 0.89, 0.65, 0.53, 0.06, 0.05, 0.70, -3.2, 70, False, 203493, "True", "2013-09-13"),
    ("One More Time", "Daft Punk", "electronic", 123, "D Major", 0.85, 0.86, 0.61, 0.01, 0.04, 0.01, -5.8, 72, False, 320357, "Discovery", "2001-03-13"),
    ("Strobe", "deadmau5", "electronic", 128, "Bb Minor", 0.58, 0.30, 0.36, 0.32, 0.03, 0.89, -8.1, 62, False, 635533, "For Lack of a Better Name", "2009-09-22"),
    ("Animals", "Martin Garrix", "electronic", 128, "F# Minor", 0.93, 0.30, 0.55, 0.00, 0.06, 0.83, -3.4, 68, False, 304893, "Animals", "2013-06-17"),
    ("Titanium", "David Guetta", "electronic", 126, "Eb Major", 0.88, 0.53, 0.55, 0.01, 0.08, 0.00, -4.9, 71, False, 245040, "Nothing but the Beat", "2011-08-26"),
    ("Don't You Worry Child", "Swedish House Mafia", "electronic", 129, "G Major", 0.82, 0.72, 0.48, 0.01, 0.05, 0.05, -4.5, 70, False, 213000, "Until Now", "2012-10-22"),
    ("Wake Me Up", "Avicii", "electronic", 124, "B Minor", 0.78, 0.72, 0.53, 0.02, 0.04, 0.00, -5.6, 73, False, 247427, "True", "2013-09-13"),

    # hip hop — various
    ("In Da Club", "50 Cent", "hip hop", 90, "F# Minor", 0.78, 0.71, 0.86, 0.04, 0.23, 0.00, -4.8, 74, True, 193893, "Get Rich or Die Tryin'", "2003-02-06"),
    ("Hotline Bling", "Drake", "hip hop", 135, "D Minor", 0.63, 0.49, 0.87, 0.18, 0.13, 0.00, -5.8, 76, True, 267067, "Views", "2016-04-29"),
    ("Love The Way You Lie", "Eminem", "hip hop", 87, "G Minor", 0.93, 0.64, 0.75, 0.02, 0.23, 0.00, -3.6, 73, True, 263427, "Recovery", "2010-06-18"),
    ("Empire State of Mind", "JAY-Z", "hip hop", 173, "F# Minor", 0.67, 0.55, 0.72, 0.06, 0.28, 0.00, -5.1, 71, True, 276400, "The Blueprint 3", "2009-09-08"),
    ("Alright", "Kendrick Lamar", "hip hop", 110, "F Minor", 0.62, 0.54, 0.66, 0.06, 0.35, 0.00, -7.6, 72, True, 219973, "To Pimp a Butterfly", "2015-03-15"),
    ("God's Plan", "Drake", "hip hop", 77, "A Minor", 0.45, 0.36, 0.75, 0.33, 0.11, 0.00, -9.2, 79, True, 198973, "Scorpion", "2018-06-29"),

    # r&b
    ("I'll Erase Away Your Pain", "The Whatnauts", "r&b", 121, "Ab Major", 0.48, 0.62, 0.66, 0.55, 0.04, 0.00, -10.1, 35, False, 265000, "Reaching for the Stars", "1971-01-01"),

    # extra tracks to round out the dataset
    ("Shape of You", "Ed Sheeran", "pop", 96, "C# Minor", 0.65, 0.93, 0.83, 0.58, 0.08, 0.00, -2.8, 88, False, 233713, "÷", "2017-03-03"),
    ("Uptown Funk", "Mark Ronson", "pop", 115, "D Minor", 0.90, 0.96, 0.86, 0.03, 0.08, 0.00, -4.4, 82, False, 269667, "Uptown Special", "2015-01-13"),
    ("Happy", "Pharrell Williams", "pop", 160, "F Minor", 0.81, 0.96, 0.65, 0.22, 0.07, 0.00, -4.7, 80, False, 232720, "GIRL", "2014-03-03"),
    ("Sunflower", "Post Malone", "hip hop", 90, "D Major", 0.48, 0.91, 0.76, 0.55, 0.08, 0.00, -6.6, 81, False, 158040, "Hollywood's Bleeding", "2019-09-06"),
    ("Starboy", "The Weeknd", "pop", 186, "G Minor", 0.59, 0.49, 0.68, 0.14, 0.28, 0.00, -7.0, 79, True, 230453, "Starboy", "2016-11-25"),
    ("Old Town Road", "Lil Nas X", "hip hop", 136, "G# Minor", 0.62, 0.64, 0.88, 0.13, 0.33, 0.00, -6.0, 80, True, 113000, "7 EP", "2019-06-21"),
    ("Circles", "Post Malone", "pop", 120, "C Major", 0.36, 0.55, 0.70, 0.19, 0.04, 0.00, -6.2, 78, False, 215280, "Hollywood's Bleeding", "2019-09-06"),
    ("Rockstar", "Post Malone", "hip hop", 80, "Bb Minor", 0.52, 0.13, 0.59, 0.13, 0.07, 0.00, -6.0, 77, True, 218147, "Beerbongs & Bentleys", "2018-04-27"),
    ("Believer", "Imagine Dragons", "rock", 125, "Ab Minor", 0.78, 0.42, 0.78, 0.06, 0.13, 0.00, -3.7, 77, False, 204347, "Evolve", "2017-06-23"),
    ("Thunder", "Imagine Dragons", "rock", 168, "C Major", 0.81, 0.65, 0.60, 0.02, 0.07, 0.04, -5.6, 76, False, 187147, "Evolve", "2017-06-23"),
    ("Lovely", "Billie Eilish", "pop", 115, "E Minor", 0.30, 0.12, 0.35, 0.93, 0.03, 0.02, -13.2, 75, False, 200187, "13 Reasons Why S2", "2018-04-19"),
    ("Stressed Out", "Twenty One Pilots", "alternative", 170, "D Minor", 0.65, 0.60, 0.73, 0.14, 0.14, 0.00, -5.6, 74, False, 202333, "Blurryface", "2015-05-19"),
    ("Heathens", "Twenty One Pilots", "alternative", 90, "Bb Minor", 0.60, 0.32, 0.73, 0.06, 0.10, 0.00, -6.1, 73, False, 195920, "Blurryface", "2015-05-19"),
    ("Havana", "Camila Cabello", "pop", 105, "Bb Minor", 0.63, 0.39, 0.77, 0.18, 0.03, 0.00, -4.3, 79, False, 217307, "Camila", "2018-01-12"),
    ("Lucid Dreams", "Juice WRLD", "hip hop", 84, "F Minor", 0.42, 0.24, 0.51, 0.35, 0.20, 0.00, -7.2, 77, True, 239947, "Goodbye & Good Riddance", "2018-05-23"),
    ("Someone You Loved", "Lewis Capaldi", "pop", 110, "C Major", 0.41, 0.45, 0.50, 0.75, 0.03, 0.00, -5.7, 78, False, 182000, "Divinely Uninspired", "2019-05-17"),
]


def classify_mood(energy, valence, danceability=None):
    if 0.35 < energy < 0.65 and 0.35 < valence < 0.65:
        if danceability and danceability > 0.7:
            return "happy"
        return "chill"
    if energy >= 0.5 and valence >= 0.5:
        return "happy"
    elif energy >= 0.5 and valence < 0.5:
        return "aggressive"
    elif energy < 0.5 and valence >= 0.5:
        return "chill"
    else:
        return "sad"


header = [
    "track_id", "name", "artist", "album", "release_date",
    "popularity", "explicit", "duration_ms",
    "primary_genre",
    "tempo", "danceability", "energy", "valence",
    "acousticness", "speechiness", "instrumentalness", "loudness",
    "mood"
]

rows = []
for i, s in enumerate(SONGS):
    name, artist, genre, bpm, key, energy, valence, dance, acoust, speech, instr, loud, pop, explicit, dur, album, date = s
    mood = classify_mood(energy, valence, dance)
    rows.append([
        f"sample_{i:03d}", name, artist, album, date,
        pop, explicit, dur,
        genre,
        bpm, round(dance, 3), round(energy, 3), round(valence, 3),
        round(acoust, 3), round(speech, 3), round(instr, 3), round(loud, 1),
        mood
    ])

out_path = os.path.join(os.path.dirname(__file__), "sample_playlist.csv")
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print(f"Generated {len(rows)} tracks -> {out_path}")

# mood distribution
from collections import Counter
moods = Counter(r[-1] for r in rows)
for mood, count in sorted(moods.items()):
    print(f"  {mood}: {count}")
