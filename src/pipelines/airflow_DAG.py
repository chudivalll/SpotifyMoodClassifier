import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Import functions from our spotify_ML module
from src.data_collection.spotify_ML import get_playlist_tracks, get_audio_features

# Load Data
def load_data():
    PLAYLIST_ID = "your_playlist_id"  # Replace with your playlist ID
    tracks_data = get_playlist_tracks(PLAYLIST_ID)
    track_ids = [track['Track ID'] for track in tracks_data]
    audio_features = get_audio_features(track_ids)
    
    # Create and merge DataFrames
    df_tracks = pd.DataFrame(tracks_data)
    df_features = pd.DataFrame(audio_features)
    df_final = pd.merge(df_tracks, df_features, on="Track ID")
    
    # Save to data/processed directory
    df_final.to_csv('data/processed/spotify_features.csv', index=False)

# Preprocess Data
def preprocess_data():
    df = pd.read_csv('data/processed/spotify_features.csv')
    # Add your preprocessing steps here
    # For example, selecting features for ML:
    features_df = df[['BPM', 'Danceability', 'Energy', 'Loudness', 'Valence']]
    features_df.to_csv('data/processed/preprocessed_features.csv', index=False)

# Split Data
def split_data():
    df = pd.read_csv('data/processed/preprocessed_features.csv')
    # You'll need to define your target variable
    # For example, if classifying by energy level:
    X = df.drop(columns=['Energy'])
    y = df['Energy'] > 0.5  # Binary classification example
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Save split data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/y_test.csv', index=False)

# Train and Predict Functions
def train_and_evaluate(model, model_name):
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save results
    with open(f'data/processed/{model_name}_results.txt', 'w') as f:
        f.write(f'{model_name} Accuracy: {accuracy}\n')

def train_naive_bayes():
    train_and_evaluate(GaussianNB(), 'naive_bayes')

def train_gradient_boosting():
    train_and_evaluate(GradientBoostingClassifier(), 'gradient_boosting')

def train_decision_tree():
    train_and_evaluate(DecisionTreeClassifier(), 'decision_tree')

def train_random_forest():
    train_and_evaluate(RandomForestClassifier(), 'random_forest')

# Define DAG
with DAG('spotify_ml_pipeline', 
         start_date=datetime(2024, 3, 16),
         schedule_interval=None,
         catchup=False) as dag:

    task_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )
    
    task_preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )
    
    task_split = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )
    
    task_naive_bayes = PythonOperator(
        task_id='train_naive_bayes',
        python_callable=train_naive_bayes
    )
    
    task_gradient_boosting = PythonOperator(
        task_id='train_gradient_boosting',
        python_callable=train_gradient_boosting
    )
    
    task_decision_tree = PythonOperator(
        task_id='train_decision_tree',
        python_callable=train_decision_tree
    )
    
    task_random_forest = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_random_forest
    )

    # Set Task Dependencies
    task_load >> task_preprocess >> task_split
    task_split >> [task_naive_bayes, task_gradient_boosting, task_decision_tree, task_random_forest]
