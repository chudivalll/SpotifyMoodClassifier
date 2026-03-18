# Spotify Mood Classifier

Analyzes Spotify playlists and classifies songs by mood using audio features + ML clustering.

Built this to break down my playlists into mood categories (chill, hype, study, etc.) based on tempo, energy, danceability, valence, and other audio features from the Spotify API.

## How it works

1. Pull tracks + audio features from a Spotify playlist via the API
2. Scrape BPM data from external sources where Spotify's data is missing
3. Estimate missing audio features using genre/popularity heuristics
4. Train clustering models (KMeans) to group songs by mood
5. Optionally deploy to AWS SageMaker for inference

## Setup

```bash
python -m venv spotify_env
source spotify_env/bin/activate  # or spotify_env\Scripts\activate on Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Spotify API credentials from the [developer dashboard](https://developer.spotify.com/dashboard).

## Running

```bash
# collect playlist data
python src/data_collection/spotify_ML.py

# add estimated features for tracks missing audio data
python src/data_processing/add_missing_columns.py

# scrape actual BPM values
python src/data_processing/fetch_bpm_data.py

# run the ML pipeline
python src/pipelines/ml_pipeline.py
```

Update `PLAYLIST_ID` in the data collection scripts to point at your playlist.

## Project structure

- `src/data_collection/` - Spotify API scripts (basic, enhanced, simple variants)
- `src/data_processing/` - BPM scraping, feature estimation, data cleanup
- `src/pipelines/` - ML training pipeline + Airflow DAG
- `src/aws_deployment/` - SageMaker training/inference scripts
- `notebooks/` - Jupyter notebooks for exploration
- `tests/` - API connection and auth validation scripts

## API restrictions

As of Nov 2024, Spotify locked down audio features, audio analysis, and recommendations for new apps. This project was built before those restrictions. If you're starting fresh, you'll need to either use pre-collected data or apply for extended API access.
