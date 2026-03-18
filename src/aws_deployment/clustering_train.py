import pandas as pd
import numpy as np
import os
import json
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-clusters', type=int, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args, _ = parser.parse_known_args()

    df = pd.read_csv(os.path.join(args.train, 'spotify_playlist_ml_ready.csv'))
    print(f"Loaded {len(df)} tracks")

    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

    # prefer actual BPM over spotify's tempo estimate
    if 'Actual_BPM' in df.columns and df['Actual_BPM'].notna().any():
        df.loc[df['Actual_BPM'].notna(), 'tempo'] = df.loc[df['Actual_BPM'].notna(), 'Actual_BPM']

    X = df[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Cluster sizes: {dict(zip(unique, counts))}")

    # show cluster centers in original scale
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    print(f"\nCluster centers:\n{centers}")

    # save artifacts
    with open(os.path.join(args.model_dir, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(args.model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.model_dir, 'features.json'), 'w') as f:
        json.dump(features, f)

    print(f"Model saved to {args.model_dir}")
