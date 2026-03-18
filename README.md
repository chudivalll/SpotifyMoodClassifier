# Spotify Mood Classifier

Classifies songs by mood (happy, sad, aggressive, chill) using audio features and ML clustering. Includes a web interface that gives you a mini "Spotify Wrapped" for any playlist — mood breakdown, genre DNA, decade timeline, and cultural context for each song.

## Quick start

```bash
python -m venv spotify_env
source spotify_env/bin/activate
pip install -r requirements.txt
```

### Run the website

```bash
python web/app.py
```

Open `http://127.0.0.1:5000` — click "Try Demo" to see the full experience without any API keys.

To connect your own Spotify account, add your credentials to `.env` (copy from `.env.example`) and click "Connect Spotify" on the landing page. The app will show your playlists and let you analyze any of them.

### Run the CLI pipeline

```bash
python run_pipeline.py                                    # sample data
python run_pipeline.py --mode live --playlist <ID>        # your playlist
```

## How mood classification works

Songs are classified into 4 moods based on energy (arousal) and valence (positivity):

```
         High Valence
              |
   chill      |     happy
              |
Low Energy ---+--- High Energy
              |
    sad       |   aggressive
              |
         Low Valence
```

## What the website shows

- **Mood breakdown** — visual split of happy/sad/aggressive/chill tracks
- **Genre DNA** — genre fingerprint of your taste
- **Decade timeline** — when your music was made
- **Album cover grid** — click any cover to reveal the cultural story behind the song (where samples came from, what the artist was going through, why the song mattered historically)

## Project structure

- `web/` — Flask app + templates for the website
- `run_pipeline.py` — CLI entry point
- `src/mood_classifier.py` — mood labeling logic
- `src/data_collection/` — Spotify API scripts
- `src/data_processing/` — feature estimation, BPM scraping
- `src/pipelines/` — ML training pipeline + Airflow DAG
- `src/aws_deployment/` — SageMaker training/inference
- `data/sample/` — bundled 76-track sample dataset

## API restrictions

As of Nov 2024, Spotify locked down audio features for new apps. The project handles this by catching the 403, falling back to heuristic estimation, and shipping a sample dataset so everything works out of the box.
