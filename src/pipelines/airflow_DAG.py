import sys
from pathlib import Path

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

from src.data_collection.spotify_ML import get_playlist_tracks, get_audio_features
from src.data_processing.add_missing_columns import estimate_audio_features
from src.mood_classifier import assign_moods

FEATURE_COLS = ['tempo', 'danceability', 'energy', 'loudness', 'valence',
                'acousticness', 'speechiness', 'instrumentalness']


def load_data():
    PLAYLIST_ID = "your_playlist_id"
    tracks_data = get_playlist_tracks(PLAYLIST_ID)
    track_ids = [t['Track ID'] for t in tracks_data]
    audio_features = get_audio_features(track_ids)

    df_tracks = pd.DataFrame(tracks_data)

    if audio_features:
        df_features = pd.DataFrame(audio_features)
        df = pd.merge(df_tracks, df_features, left_on='Track ID', right_on='id', how='left')
    else:
        df = df_tracks

    df = estimate_audio_features(df)
    df.to_csv('data/processed/spotify_features.csv', index=False)


def preprocess_data():
    df = pd.read_csv('data/processed/spotify_features.csv')
    df = assign_moods(df)
    available = [c for c in FEATURE_COLS if c in df.columns]
    df[available + ['mood']].to_csv('data/processed/preprocessed_features.csv', index=False)


def split_data():
    df = pd.read_csv('data/processed/preprocessed_features.csv')
    X = df.drop(columns=['mood'])
    y = df['mood']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/y_test.csv', index=False)


def train_and_evaluate(model, model_name):
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    with open(f'data/processed/{model_name}_results.txt', 'w') as f:
        f.write(f'{model_name} Accuracy: {acc}\n')


def train_naive_bayes():
    train_and_evaluate(GaussianNB(), 'naive_bayes')

def train_gradient_boosting():
    train_and_evaluate(GradientBoostingClassifier(), 'gradient_boosting')

def train_decision_tree():
    train_and_evaluate(DecisionTreeClassifier(), 'decision_tree')

def train_random_forest():
    train_and_evaluate(RandomForestClassifier(), 'random_forest')


with DAG('spotify_ml_pipeline',
         start_date=datetime(2024, 3, 16),
         schedule_interval=None,
         catchup=False) as dag:

    t_load = PythonOperator(task_id='load_data', python_callable=load_data)
    t_preprocess = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data)
    t_split = PythonOperator(task_id='split_data', python_callable=split_data)

    t_nb = PythonOperator(task_id='train_naive_bayes', python_callable=train_naive_bayes)
    t_gb = PythonOperator(task_id='train_gradient_boosting', python_callable=train_gradient_boosting)
    t_dt = PythonOperator(task_id='train_decision_tree', python_callable=train_decision_tree)
    t_rf = PythonOperator(task_id='train_random_forest', python_callable=train_random_forest)

    t_load >> t_preprocess >> t_split
    t_split >> [t_nb, t_gb, t_dt, t_rf]
