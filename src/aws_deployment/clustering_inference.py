import os
import json
import pickle
import pandas as pd
import numpy as np

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


def model_fn(model_dir):
    kmeans = pickle.load(open(os.path.join(model_dir, 'kmeans_model.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(model_dir, 'scaler.pkl'), 'rb'))
    with open(os.path.join(model_dir, 'features.json'), 'r') as f:
        features = json.load(f)

    return {'kmeans': kmeans, 'scaler': scaler, 'features': features}


def predict_fn(input_data, model):
    kmeans = model['kmeans']
    scaler = model['scaler']
    features = model['features']

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = pd.DataFrame(input_data)

    # fill in any missing features with 0
    for f in features:
        if f not in df.columns:
            df[f] = 0

    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)

    clusters = kmeans.predict(X_scaled)
    distances = kmeans.transform(X_scaled)

    predictions = []
    for i, cluster in enumerate(clusters):
        predictions.append({
            'cluster': int(cluster),
            'distances': {f"cluster_{j}": float(d) for j, d in enumerate(distances[i])}
        })

    return predictions


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")
