"""
Mood classification based on audio features using the energy/valence quadrant model.

         High Valence
              |
   chill      |     happy
  (relaxed)   |   (energetic)
              |
Low Energy ---+--- High Energy
              |
    sad       |    aggressive
 (melancholy) |    (intense)
              |
         Low Valence

Tracks near the center get labeled "focused" as a 5th category.
"""

MOOD_LABELS = ["happy", "sad", "energetic", "chill", "aggressive"]


def classify_mood(energy, valence, danceability=None, tempo=None):
    """Assign a mood label based on audio features.

    Primary axes: energy (arousal) and valence (positivity).
    Danceability and tempo are used as tiebreakers.
    """
    # center zone — not strongly in any quadrant
    if 0.35 < energy < 0.65 and 0.35 < valence < 0.65:
        if danceability is not None and danceability > 0.7:
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


def assign_moods(df):
    """Add a 'mood' column to a DataFrame with audio features.

    Expects columns: energy, valence. Optionally: danceability, tempo.
    """
    def _row_mood(row):
        e = row.get('energy', 0.5)
        v = row.get('valence', 0.5)
        d = row.get('danceability', None)
        t = row.get('tempo', None)
        return classify_mood(e, v, d, t)

    df = df.copy()
    df['mood'] = df.apply(_row_mood, axis=1)
    return df
