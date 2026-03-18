import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import json
from datetime import datetime

from src.data_collection.spotify_ML import get_playlist_tracks, get_audio_features


class SpotifyMLPipeline:
    def __init__(self, playlist_id):
        self.playlist_id = playlist_id
        self.data_dir = project_root / 'data'
        self.processed_dir = self.data_dir / 'processed'
        self.results_dir = self.data_dir / 'results'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def collect_data(self):
        tracks_data = get_playlist_tracks(self.playlist_id)
        track_ids = [t['Track ID'] for t in tracks_data]
        audio_features = get_audio_features(track_ids)

        df_tracks = pd.DataFrame(tracks_data)
        df_features = pd.DataFrame(audio_features)
        self.df = pd.merge(df_tracks, df_features, on="Track ID")
        self.df.to_csv(self.processed_dir / 'spotify_features.csv', index=False)
        print(f"Collected {len(self.df)} tracks")

    def preprocess_data(self):
        feature_cols = ['BPM', 'Danceability', 'Energy', 'Loudness', 'Valence']
        self.X = self.df[feature_cols]
        self.y = self.df['Energy'] > 0.5

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

    def train_models(self):
        models = {
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

        results = {}
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, preds)
            results[name] = acc
            print(f"  {name}: {acc:.4f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.results_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run(self):
        print("Starting pipeline...")
        self.collect_data()
        self.preprocess_data()
        results = self.train_models()
        print("Pipeline complete.")
        return results


def main():
    PLAYLIST_ID = "your_playlist_id_here"
    pipeline = SpotifyMLPipeline(PLAYLIST_ID)
    results = pipeline.run()

    print("\nResults:")
    for model, acc in results.items():
        print(f"  {model}: {acc:.4f}")


if __name__ == "__main__":
    main()
