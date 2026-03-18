import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

from src.mood_classifier import assign_moods, MOOD_LABELS


FEATURE_COLS = ['tempo', 'danceability', 'energy', 'valence',
                'acousticness', 'speechiness', 'instrumentalness', 'loudness']


class SpotifyMLPipeline:
    def __init__(self, playlist_id=None, data_path=None):
        self.playlist_id = playlist_id
        self.data_path = data_path
        self.data_dir = project_root / 'data'
        self.processed_dir = self.data_dir / 'processed'
        self.results_dir = self.data_dir / 'results'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_sample_data(self):
        """Load the bundled sample dataset."""
        sample_path = self.data_path or (self.data_dir / 'sample' / 'sample_playlist.csv')
        self.df = pd.read_csv(sample_path)
        print(f"Loaded {len(self.df)} tracks from {sample_path}")

    def collect_from_api(self):
        """Collect from Spotify API with fallback to estimation."""
        from src.data_collection.spotify_ML import get_playlist_tracks, get_audio_features
        from src.data_processing.add_missing_columns import estimate_audio_features

        tracks_data = get_playlist_tracks(self.playlist_id)
        if not tracks_data:
            raise ValueError("No tracks retrieved from playlist")

        track_ids = [t['Track ID'] for t in tracks_data]
        audio_features = get_audio_features(track_ids)

        df_tracks = pd.DataFrame(tracks_data)

        if audio_features:
            df_features = pd.DataFrame(audio_features)
            # keep only the feature columns + id for merging
            keep = ['id'] + [c for c in FEATURE_COLS if c in df_features.columns]
            df_features = df_features[keep].rename(columns={'id': 'Track ID'})
            self.df = pd.merge(df_tracks, df_features, on='Track ID', how='left')
        else:
            self.df = df_tracks

        # fill gaps with estimates
        self.df = estimate_audio_features(self.df)

        self.df.to_csv(self.processed_dir / 'spotify_features.csv', index=False)
        print(f"Collected {len(self.df)} tracks")

    def preprocess(self):
        """Assign mood labels and prepare features for training."""
        # make sure we have mood labels
        if 'mood' not in self.df.columns:
            self.df = assign_moods(self.df)

        # check which feature columns are available
        available = [c for c in FEATURE_COLS if c in self.df.columns]
        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        if missing:
            print(f"Warning: missing feature columns {missing}, using {available}")

        self.X = self.df[available].fillna(0)
        self.y = self.df['mood']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42, stratify=self.y
        )
        print(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        print(f"Mood distribution:\n{self.y.value_counts().to_string()}\n")

    def train_classifiers(self):
        """Train multiple classifiers and compare."""
        models = {
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42)
        }

        results = {}
        best_acc = 0
        best_name = None

        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, preds)
            results[name] = acc

            if acc > best_acc:
                best_acc = acc
                best_name = name
                self.best_model = model

            print(f"  {name}: {acc:.3f}")

        print(f"\nBest: {best_name} ({best_acc:.3f})")
        print(f"\nClassification report ({best_name}):")
        preds = self.best_model.predict(self.X_test)
        print(classification_report(self.y_test, preds, zero_division=0))

        # save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            'timestamp': timestamp,
            'accuracies': results,
            'best_model': best_name,
            'best_accuracy': best_acc,
            'n_train': len(self.X_train),
            'n_test': len(self.X_test),
            'features_used': list(self.X.columns),
        }
        with open(self.results_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump(output, f, indent=2)

        return results

    def train_clusters(self, n_clusters=5):
        """KMeans clustering as an alternative to supervised classification."""
        X = self.df[FEATURE_COLS].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(X_scaled)

        print(f"\nKMeans clustering ({n_clusters} clusters):")
        for c in range(n_clusters):
            cluster_tracks = self.df[self.df['cluster'] == c]
            n = len(cluster_tracks)
            name_col = 'name' if 'name' in self.df.columns else 'Name'
            sample = cluster_tracks[name_col].head(3).tolist() if name_col in self.df.columns else []
            avg_energy = cluster_tracks['energy'].mean()
            avg_valence = cluster_tracks['valence'].mean()
            print(f"  Cluster {c} ({n} tracks): energy={avg_energy:.2f}, valence={avg_valence:.2f}")
            if sample:
                print(f"    e.g. {', '.join(sample)}")

        return kmeans

    def run(self, mode='sample'):
        """Run the full pipeline."""
        print("=== Spotify Mood Classifier ===\n")

        if mode == 'sample':
            self.load_sample_data()
        else:
            self.collect_from_api()

        self.preprocess()
        results = self.train_classifiers()
        self.train_clusters()

        print("\nPipeline complete.")
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Spotify Mood Classification Pipeline')
    parser.add_argument('--mode', choices=['sample', 'live'], default='sample',
                        help='sample: use bundled data, live: fetch from Spotify API')
    parser.add_argument('--playlist', type=str, default=None,
                        help='Spotify playlist ID (required for live mode)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CSV dataset (overrides sample/live)')
    args = parser.parse_args()

    if args.data:
        pipeline = SpotifyMLPipeline(data_path=args.data)
        pipeline.run(mode='sample')
    elif args.mode == 'live':
        if not args.playlist:
            print("Error: --playlist required for live mode")
            sys.exit(1)
        pipeline = SpotifyMLPipeline(playlist_id=args.playlist)
        pipeline.run(mode='live')
    else:
        pipeline = SpotifyMLPipeline()
        pipeline.run(mode='sample')


if __name__ == "__main__":
    main()
