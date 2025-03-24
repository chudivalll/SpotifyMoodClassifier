# Spotify Playlist Analysis and Mood Classification Project

This project analyzes Spotify playlists to extract audio features and uses machine learning to classify songs into different mood categories (chill, hype, study, etc.).

## Project Purpose

The primary goal of this project is to:
1. Collect audio features from Spotify playlists
2. Train a machine learning model to categorize songs into mood-based categories
3. Create a system that can automatically classify songs based on their audio characteristics
4. Break playlists down into mood-based sections for better music organization

## Project Structure
```python
Spotify/
├── src/                    # Source code
│   ├── data_collection/    # Spotify API interaction
│   ├── models/             # ML model training and evaluation
│   └── pipelines/          # Data processing pipelines
├── data/                  # Data storage
│   ├── raw/               # Raw data from Spotify
│   └── processed/         # Processed data
├── notebooks/            # Jupyter notebooks
└── tests/                # Unit tests
```

## Setup Instructions

1. **Create a Spotify Developer Account**
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new application
   - Get your Client ID and Client Secret

2. **Environment Setup**
   - Create a virtual environment:
     ```bash
     python -m venv spotify_env
     ```
   - Activate the virtual environment:
     - Windows: `spotify_env\Scripts\activate`
     - Unix/MacOS: `source spotify_env/bin/activate`

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   - Copy `.env.example` to `.env`
   - Add your Spotify credentials to `.env`:
     ```
     SPOTIFY_CLIENT_ID=your_client_id_here
     SPOTIFY_CLIENT_SECRET=your_client_secret_here
     ```

## Usage

1. **Data Collection**
   - Update the `PLAYLIST_ID` in `src/data_collection/spotify_ML.py` with your desired playlist
   - You can use either the full Spotify URL or just the playlist ID
   - Example:
     ```python
     PLAYLIST_ID = "6jgCEkpKSc7LQI8ZWdAVr6"
     # or
     PLAYLIST_ID = "https://open.spotify.com/playlist/6jgCEkpKSc7LQI8ZWdAVr6"
     ```

2. **Run the Analysis**
   ```bash
   python src/data_collection/spotify_ML.py
   ```

3. **Model Training** (Future Implementation)
   ```bash
   python src/models/train_model.py
   ```

## Features

- **Playlist Track Retrieval**
  - Fetches track information including name, artist, album, and preview URLs
  - Handles large playlists with batch processing
  - Configurable maximum track limit (default: 1200)

- **Audio Feature Analysis**
  - Retrieves audio features for each track:
    - BPM (Tempo)
    - Danceability
    - Energy
    - Loudness
    - Valence
    - Acousticness
    - Speechiness
    - Instrumentalness

- **Machine Learning Classification**
  - Uses audio features to train models for mood classification
  - Categories include: chill, hype, study, etc.
  - Provides classification metrics and model evaluation

- **Data Export**
  - Saves results to Excel file (`spotify_playlist_analysis.xlsx`)
  - Includes preview URLs for song sampling
  - Excludes preview URLs from model training data
  - Includes basic statistical analysis of audio features

## ML Model Purpose

The machine learning model aims to:
1. Learn patterns in audio features that correspond to different moods
2. Create accurate classifications that match human perception of song moods
3. Help users organize their music libraries by mood
4. Enable mood-based recommendations

## Configuration Options

- `batch_size`: Number of tracks to process per request (default: 100 for tracks, 20 for features)
- `delay_between_batches`: Time to wait between API calls (default: 2-3 seconds)
- `max_tracks`: Maximum number of tracks to process from a playlist (default: 1200)

## Troubleshooting

1. **Authentication Issues**
   - Verify your credentials in `.env` file
   - Check that the `.env` file is in the correct location
   - Ensure the Spotify application is properly configured

2. **Rate Limiting**
   - The script includes automatic handling for rate limits
   - Implements delays between requests
   - Retries failed requests with exponential backoff

3. **Missing Data**
   - The script will continue processing even if some tracks fail
   - Check the console output for progress and error messages
   - Final results will include all successfully processed tracks

## Dependencies

- spotipy
- pandas
- python-dotenv
- openpyxl (for Excel export)
- scikit-learn (for ML model training)
- matplotlib/seaborn (for visualizations)

## Notes

- The script uses Spotify's Client Credentials flow for authentication
- No user authorization is required for public playlists
- Preview URLs are included in the Excel output but excluded from ML training
- Rate limits are handled automatically with retry logic
- Progress updates are provided during execution

## Output

The script generates an Excel file containing:
- Track information (name, artist, album, release date)
- Preview URLs for song sampling
- Audio features (BPM, danceability, energy, loudness, valence, etc.)
- Basic statistical analysis of the features

## Development
- Use `notebooks/` for data exploration and model experimentation
- Add tests in `tests/` directory
- Keep raw data in `data/raw/`
- Store processed data in `data/processed/`

## Important Note on Spotify API Restrictions

As of November 27, 2024, Spotify has implemented significant restrictions on their Web API access. New Web API applications can no longer access several crucial endpoints that this project previously utilized:

1. Audio Features - The core data source for our ML models
2. Audio Analysis - Detailed track analytics we relied upon
3. Recommendations - Previously used for suggesting similar tracks
4. 30-second preview URLs in multi-get responses

These changes only affect new applications and existing applications still in development mode. If your application already had extended mode Web API access before this date, it should remain unaffected.

### Impact on This Project

This project was developed prior to these restrictions and demonstrates techniques that new developers may not be able to fully replicate with freshly registered API credentials. Options to work around these limitations include:

1. Using pre-collected datasets (like the ones included in this repository)
2. Applying for extended API access through Spotify's partner program
3. Exploring alternative audio analysis libraries and APIs

See the [official Spotify announcement](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api) for more details.

## AWS Deployment Guide

This section provides detailed instructions on how to deploy this project to AWS using Amazon S3 and SageMaker for machine learning model training and deployment.

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Python 3.6+ with pip
- Git for version control

### Step 1: Set Up AWS Environment

#### 1.1 Install and Configure AWS CLI

```bash
# Install AWS CLI
pip install awscli

# Configure AWS CLI with your credentials
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, default region (e.g., us-east-1), and output format (json recommended).

#### 1.2 Create S3 Bucket for Project Data

```bash
# Create a bucket for the project
aws s3 mb s3://spotify-playlist-analysis --region us-east-1
```

### Step 2: Upload Data to S3

```bash
# Upload the Excel files to S3
aws s3 cp spotify_playlist_complete.xlsx s3://spotify-playlist-analysis/data/
aws s3 cp spotify_playlist_ml_complete.xlsx s3://spotify-playlist-analysis/data/

# Upload source code to S3
aws s3 cp --recursive src/ s3://spotify-playlist-analysis/code/
```

### Step 3: Adapt Data Processing for SageMaker

Create a SageMaker processing script to handle BPM data and feature engineering:

```python
# spotify_data_processor.py
import pandas as pd
import os
import sys
import numpy as np

# SageMaker paths
input_path = '/opt/ml/processing/input/'
output_path = '/opt/ml/processing/output/'

# Load data
df = pd.read_excel(os.path.join(input_path, 'spotify_playlist_complete.xlsx'))

# Identify track and artist columns
track_column = None
artist_column = None

for col in df.columns:
    if col.lower() in ['track name', 'track_name', 'name', 'title', 'track']:
        track_column = col
    if col.lower() in ['artist', 'artist_name', 'artists']:
        artist_column = col

# Preprocess features
# Create Actual_BPM and Music_Key columns if they don't exist
if 'Actual_BPM' not in df.columns:
    df['Actual_BPM'] = np.nan
if 'Music_Key' not in df.columns:
    df['Music_Key'] = ''

# Fill missing BPM values using Spotify's tempo feature
if 'tempo' in df.columns and 'Actual_BPM' in df.columns:
    df.loc[df['Actual_BPM'].isna(), 'Actual_BPM'] = df.loc[df['Actual_BPM'].isna(), 'tempo']

# Prepare data for ML
ml_df = df.copy()
if 'preview_url' in ml_df.columns:
    ml_df.drop(columns=['preview_url'], inplace=True)

# Save processed data
df.to_csv(os.path.join(output_path, 'spotify_playlist_processed.csv'), index=False)
ml_df.to_csv(os.path.join(output_path, 'spotify_playlist_ml_ready.csv'), index=False)

print(f"Processed {len(df)} tracks. Data saved to output directory.")
```

### Step 4: Create SageMaker Notebook Instance

Use the AWS Console to create a SageMaker notebook:

1. Navigate to SageMaker in the AWS Console
2. Create a new notebook instance
3. Select an instance type (ml.t3.medium is sufficient for starting)
4. Create a new IAM role with S3 access or use an existing one
5. Launch the notebook instance

### Step 5: Create Clustering Model Training Script

```python
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
    
    args = parser.parse_args()
    
    # Load training data
    df = pd.read_csv(os.path.join(args.train, 'spotify_playlist_ml_ready.csv'))
    
    # Select features for clustering
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    
    # Use Actual_BPM if available, otherwise use tempo
    if 'Actual_BPM' in df.columns and df['Actual_BPM'].notna().any():
        # Replace tempo with Actual_BPM where available
        df.loc[df['Actual_BPM'].notna(), 'tempo'] = df.loc[df['Actual_BPM'].notna(), 'Actual_BPM']
    
    # Prepare features for clustering
    X = df[features].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans clustering model
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state)
    kmeans.fit(X_scaled)
    
    # Save model artifacts
    model_path = os.path.join(args.model_dir, 'kmeans_model.pkl')
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature list for inference
    with open(os.path.join(args.model_dir, 'features.json'), 'w') as f:
        json.dump(features, f)
    
    print(f"Training complete. Model saved at {args.model_dir}")
```

### Step 6: Create SageMaker Inference Script

```python
# clustering_inference.py
import os
import json
import pickle
import flask
import pandas as pd
import numpy as np

# Paths for model artifacts
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# Load model and artifacts
def model_fn(model_dir):
    kmeans = pickle.load(open(os.path.join(model_dir, 'kmeans_model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(model_dir, 'scaler.pkl'), 'rb'))
    with open(os.path.join(model_dir, 'features.json'), 'r') as f:
        features = json.load(f)
    
    return {
        'kmeans': kmeans,
        'scaler': scaler,
        'features': features
    }

# Main prediction function
def predict_fn(input_data, model):
    kmeans = model['kmeans']
    scaler = model['scaler']
    features = model['features']
    
    # Handle both single prediction and batch predictions
    if isinstance(input_data, dict):
        # Single prediction - convert to DataFrame
        df = pd.DataFrame([input_data])
    else:
        # Batch prediction
        df = pd.DataFrame(input_data)
    
    # Extract only needed features
    X = df[features].fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get cluster predictions
    clusters = kmeans.predict(X_scaled)
    
    # Get distances to each cluster center
    distances = kmeans.transform(X_scaled)
    
    # Prepare response
    predictions = []
    for i, cluster in enumerate(clusters):
        cluster_distances = {f"cluster_{j}": dist for j, dist in enumerate(distances[i])}
        
        prediction = {
            'cluster': int(cluster),
            'distances': cluster_distances
        }
        predictions.append(prediction)
    
    return predictions

# Input handling
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# Output handling
def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
```

### Step 7: Train and Deploy the Model with SageMaker SDK

Create a SageMaker notebook with the following code:

```python
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
import pandas as pd
import numpy as np
import os
import time

# Initialize SageMaker session
role = get_execution_role()
sagemaker_session = sagemaker.Session()
bucket = 'spotify-playlist-analysis'
prefix = 'spotify-clusters'

# Prepare data for processing
processing_input = f's3://{bucket}/data/'
processing_output = f's3://{bucket}/{prefix}/processed/'

# Define a processing job
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

processor = ScriptProcessor(
    command=['python3'],
    image_uri='737474898029.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large'
)

# Run processing job
processor.run(
    code='spotify_data_processor.py',
    inputs=[
        ProcessingInput(
            source=processing_input,
            destination='/opt/ml/processing/input/'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output/',
            destination=processing_output
        )
    ]
)

# Define estimator for training
sklearn_estimator = SKLearn(
    entry_point='clustering_train.py',
    role=role,
    instance_type='ml.m5.large',
    instance_count=1,
    framework_version='0.23-1',
    hyperparameters={
        'n-clusters': 5,
        'random-state': 42
    },
    sagemaker_session=sagemaker_session
)

# Define data channels
train_data = sagemaker_session.upload_data(
    path=f's3://{bucket}/{prefix}/processed/spotify_playlist_ml_ready.csv',
    bucket=bucket,
    key_prefix=f'{prefix}/train'
)

# Train the model
sklearn_estimator.fit({'train': train_data})

# Deploy the model to an endpoint
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# Test the endpoint with a sample
sample_data = {
    'danceability': 0.8,
    'energy': 0.6,
    'loudness': -5.0,
    'speechiness': 0.05,
    'acousticness': 0.2,
    'instrumentalness': 0.0,
    'liveness': 0.1,
    'valence': 0.7,
    'tempo': 120.0,
    'duration_ms': 210000
}

# Make a prediction
response = predictor.predict(sample_data)
print(f"Prediction: {response}")

# Get endpoint name for later reference
endpoint_name = predictor.endpoint_name
print(f"Endpoint name: {endpoint_name}")
```

### Step 8: Monitor and Evaluate the Model

```python
import boto3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set up CloudWatch client
cloudwatch = boto3.client('cloudwatch')

# Get metrics for the endpoint
def get_endpoint_metrics(endpoint_name, start_time=None, end_time=None, period=300):
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(hours=1)
    if end_time is None:
        end_time = datetime.utcnow()
    
    metrics = [
        'Invocations',
        'InvocationsPerInstance',
        'ModelLatency',
        'OverheadLatency',
        'Invocation4XXErrors',
        'Invocation5XXErrors'
    ]
    
    results = {}
    
    for metric in metrics:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName=metric,
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                },
                {
                    'Name': 'VariantName',
                    'Value': 'AllTraffic'
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=['Average', 'Sum', 'Maximum', 'Minimum']
        )
        
        results[metric] = response['Datapoints']
    
    return results

# Example: Get metrics and plot them
metrics = get_endpoint_metrics(endpoint_name)

for metric_name, datapoints in metrics.items():
    if not datapoints:
        continue
    
    timestamps = [d['Timestamp'] for d in datapoints]
    values = [d['Average'] for d in datapoints]
    
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, values)
    plt.title(f'{metric_name} for {endpoint_name}')
    plt.xlabel('Time')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

### Step 9: Clean Up Resources

When you're done with the project, clean up the resources to avoid additional charges:

```python
# Delete the endpoint
predictor.delete_endpoint()

# Optionally delete the model
sagemaker_client = boto3.client('sagemaker')
sagemaker_client.delete_model(ModelName=predictor.model_name)
```

### AWS Cost Considerations

- **S3 Storage**: Costs based on the amount of data stored (~$0.023 per GB)
- **SageMaker Notebook**: ml.t3.medium instance (~$0.05 per hour)
- **SageMaker Processing**: ml.m5.large instance (~$0.115 per hour)
- **SageMaker Training**: ml.m5.large instance (~$0.115 per hour)
- **SageMaker Deployment**: ml.t2.medium instance (~$0.065 per hour)

It's recommended to shut down endpoints and notebook instances when not in use to minimize costs.

## Project Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│             │     │             │     │                 │
│  Spotify    │────▶│  Amazon S3  │────▶│  SageMaker      │
│  Excel Data │     │  Storage    │     │  Processing Job │
│             │     │             │     │                 │
└─────────────┘     └─────────────┘     └─────────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│             │     │             │     │                 │
│  SageMaker  │◀────│  SageMaker  │◀────│  SageMaker      │
│  Endpoint   │     │  Model      │     │  Training Job   │
│             │     │             │     │                 │
└─────────────┘     └─────────────┘     └─────────────────┘
        │
        ▼
┌─────────────┐     ┌─────────────┐
│             │     │             │
│  CloudWatch │     │  Client     │
│  Monitoring │◀────│  Application│
│             │     │             │
└─────────────┘     └─────────────┘
```

## Conclusion

This project demonstrates advanced skills in:

1. **Data Processing**: Working with Spotify playlist data and audio features
2. **Machine Learning**: Implementing clustering algorithms to group similar songs
3. **Cloud Deployment**: Using AWS services (S3, SageMaker) for scalable ML workloads
4. **Monitoring and Maintenance**: Setting up proper monitoring with CloudWatch

By following this guide, you can fully replicate the end-to-end workflow from local development to AWS cloud deployment.