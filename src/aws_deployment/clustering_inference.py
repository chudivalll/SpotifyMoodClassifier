# clustering_inference.py
import os
import json
import pickle
import pandas as pd
import numpy as np

# Paths for model artifacts
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# Load model and artifacts
def model_fn(model_dir):
    """
    Load the model and artifacts from the model directory
    """
    print(f"Loading model from: {model_dir}")
    kmeans = pickle.load(open(os.path.join(model_dir, 'kmeans_model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(model_dir, 'scaler.pkl'), 'rb'))
    with open(os.path.join(model_dir, 'features.json'), 'r') as f:
        features = json.load(f)
    
    print(f"Model loaded successfully, features: {features}")
    return {
        'kmeans': kmeans,
        'scaler': scaler,
        'features': features
    }

# Main prediction function
def predict_fn(input_data, model):
    """
    Generate predictions based on the input data
    """
    kmeans = model['kmeans']
    scaler = model['scaler']
    features = model['features']
    
    print(f"Received input: {type(input_data)}")
    
    # Handle both single prediction and batch predictions
    if isinstance(input_data, dict):
        # Single prediction - convert to DataFrame
        df = pd.DataFrame([input_data])
        print("Single prediction mode")
    else:
        # Batch prediction
        df = pd.DataFrame(input_data)
        print(f"Batch prediction mode, batch size: {len(df)}")
    
    # Extract only needed features and ensure all required features are present
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feature in missing_features:
            df[feature] = 0  # Default value for missing features
    
    # Select only required features
    X = df[features].fillna(0)
    print(f"Feature matrix shape: {X.shape}")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get cluster predictions
    clusters = kmeans.predict(X_scaled)
    print(f"Generated predictions for {len(clusters)} instances")
    
    # Get distances to each cluster center
    distances = kmeans.transform(X_scaled)
    
    # Prepare response
    predictions = []
    for i, cluster in enumerate(clusters):
        cluster_distances = {f"cluster_{j}": float(dist) for j, dist in enumerate(distances[i])}
        
        prediction = {
            'cluster': int(cluster),
            'distances': cluster_distances
        }
        predictions.append(prediction)
    
    print(f"Returning {len(predictions)} predictions")
    return predictions

# Input handling
def input_fn(request_body, request_content_type):
    """
    Parse input data from the request
    """
    print(f"Received request with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

# Output handling
def output_fn(prediction, response_content_type):
    """
    Format the prediction result to return to the client
    """
    print(f"Formatting output with content type: {response_content_type}")
    
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}") 