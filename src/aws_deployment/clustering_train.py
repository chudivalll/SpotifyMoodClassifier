# clustering_train.py
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
    
    # Hyperparameters
    parser.add_argument('--n-clusters', type=int, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    
    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))
    
    # Load training data
    train_data_file = os.path.join(args.train, 'spotify_playlist_ml_ready.csv')
    print(f"Loading training data from: {train_data_file}")
    df = pd.read_csv(train_data_file)
    print(f"Loaded {len(df)} tracks")
    
    # Select features for clustering
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    
    print(f"Selected features for clustering: {features}")
    
    # Use Actual_BPM if available, otherwise use tempo
    if 'Actual_BPM' in df.columns and df['Actual_BPM'].notna().any():
        # Replace tempo with Actual_BPM where available
        df.loc[df['Actual_BPM'].notna(), 'tempo'] = df.loc[df['Actual_BPM'].notna(), 'Actual_BPM']
        print("Replaced tempo with Actual_BPM where available")
    
    # Prepare features for clustering
    X = df[features].fillna(0)
    print(f"Feature matrix shape: {X.shape}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans clustering model
    print(f"Training KMeans with {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state)
    kmeans.fit(X_scaled)
    
    # Get cluster counts
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    print(f"Cluster distribution: {cluster_counts}")
    
    # Calculate cluster centers in original feature space
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=features)
    print("\nCluster centers (original feature space):")
    print(centers_df)
    
    # Save model artifacts
    model_path = os.path.join(args.model_dir, 'kmeans_model.pkl')
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    
    print(f"Saving model to: {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    
    print(f"Saving scaler to: {scaler_path}")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature list for inference
    features_path = os.path.join(args.model_dir, 'features.json')
    print(f"Saving feature list to: {features_path}")
    with open(features_path, 'w') as f:
        json.dump(features, f)
    
    print(f"Training complete. Model saved at {args.model_dir}") 