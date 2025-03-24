import sys
import os
from pathlib import Path

# Add project root to Python path
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

# Import our Spotify data collection functions
from src.data_collection.spotify_ML import get_playlist_tracks, get_audio_features

class SpotifyMLPipeline:
    def __init__(self, playlist_id):
        self.playlist_id = playlist_id
        self.data_dir = project_root / 'data'
        self.processed_dir = self.data_dir / 'processed'
        self.results_dir = self.data_dir / 'results'
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def collect_data(self):
        """Collect data from Spotify API"""
        print("Collecting playlist data...")
        tracks_data = get_playlist_tracks(self.playlist_id)
        track_ids = [track['Track ID'] for track in tracks_data]
        audio_features = get_audio_features(track_ids)
        
        # Create DataFrames
        df_tracks = pd.DataFrame(tracks_data)
        df_features = pd.DataFrame(audio_features)
        self.df_final = pd.merge(df_tracks, df_features, on="Track ID")
        
        # Save raw data
        self.df_final.to_csv(self.processed_dir / 'spotify_features.csv', index=False)
        print(f"Data collected and saved: {len(self.df_final)} tracks")

    def preprocess_data(self):
        """Preprocess the data for ML"""
        print("Preprocessing data...")
        # Select features for ML
        feature_columns = ['BPM', 'Danceability', 'Energy', 'Loudness', 'Valence']
        self.X = self.df_final[feature_columns]
        
        # Create target variable (you can modify this based on what you want to predict)
        # For example, predicting if a song has high energy
        self.y = self.df_final['Energy'] > 0.5
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        print("Data preprocessing complete")

    def train_models(self):
        """Train multiple ML models and evaluate them"""
        print("Training models...")
        models = {
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }
        
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            results[name] = accuracy
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.results_dir / f'model_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def run_pipeline(self):
        """Run the complete pipeline"""
        try:
            print("Starting Spotify ML Pipeline...")
            self.collect_data()
            self.preprocess_data()
            results = self.train_models()
            print("\nPipeline completed successfully!")
            return results
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise

def main():
    # Replace with your playlist ID
    PLAYLIST_ID = "your_playlist_id_here"
    
    pipeline = SpotifyMLPipeline(PLAYLIST_ID)
    results = pipeline.run_pipeline()
    
    # Print final results
    print("\nFinal Results:")
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.4f}")

if __name__ == "__main__":
    main() 