# Spotify Mood Classifier

Classifies songs by mood (happy, sad, aggressive, chill) using audio features and ML clustering.

Uses the energy/valence quadrant model from music information retrieval — high energy + high valence = happy, low energy + low valence = sad, etc. Trains multiple classifiers (Naive Bayes, Random Forest, Gradient Boosting, Decision Tree) and compares accuracy.

## Quick start

```bash
python -m venv spotify_env
source spotify_env/bin/activate
pip install -r requirements.txt

python run_pipeline.py
```

That's it — runs on the bundled sample dataset, no API keys needed.

## Using your own playlist

If you have Spotify API credentials from before Nov 2024 (with audio features access):

```bash
cp .env.example .env
# fill in your credentials

python run_pipeline.py --mode live --playlist 6jgCEkpKSc7LQI8ZWdAVr6
```

If your app was registered after Nov 2024, audio features will return 403. The pipeline automatically falls back to heuristic estimation based on genre, popularity, and release date.

You can also run on any CSV with the right columns:

```bash
python run_pipeline.py --data my_tracks.csv
```

## How the mood classification works

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

The pipeline trains 4 classifiers on these labels and picks the best one. It also runs KMeans clustering as an unsupervised alternative.

## Project structure

- `run_pipeline.py` — main entry point
- `src/mood_classifier.py` — mood labeling logic (energy/valence quadrants)
- `src/data_collection/` — Spotify API scripts (basic, enhanced, simple)
- `src/data_processing/` — feature estimation, BPM scraping, data cleanup
- `src/pipelines/` — ML training pipeline + Airflow DAG
- `src/aws_deployment/` — SageMaker training/inference scripts
- `data/sample/` — bundled sample dataset (76 tracks with audio features)
- `notebooks/` — Jupyter notebooks for exploration
- `tests/` — API connection and auth validation scripts

## API restrictions

As of Nov 2024, Spotify locked down audio features, audio analysis, and recommendations for new apps. This project handles that by:

1. Gracefully catching the 403 and logging a clear message
2. Falling back to heuristic feature estimation from available metadata
3. Shipping a sample dataset so the ML pipeline works out of the box
