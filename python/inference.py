import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import librosa
import os
from config import (
    SAMPLE_RATE, CHUNK_DURATION, N_MFCC, FRAME_LENGTH,
    MODEL_PATH, SCALER_PATH, CLASS_LABELS,
    FEATURE_CLIP_MIN, FEATURE_CLIP_MAX, SAMPLE_AUDIO_PATH
)

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Scaler loaded from {SCALER_PATH}")
else:
    print(f"ERROR: Model files not found at {MODEL_PATH} or {SCALER_PATH}")
    print("Please train the model first using the notebook.")
    model = None
    scaler = None

def extract_features_from_chunks(audio_file, sr=SAMPLE_RATE, chunk_duration=CHUNK_DURATION):
    """
    Extract MFCC mean and std features from chunks of the audio file.
    Returns a list of feature vectors.
    """
    try:
        audio, _ = librosa.load(audio_file, sr=sr, mono=True)
        chunk_size = int(chunk_duration * sr)
        hop_length = int(FRAME_LENGTH * sr)
        features_list = []

        for i in range(0, len(audio) - chunk_size + 1, chunk_size):
            chunk = audio[i:i + chunk_size]

            # Compute MFCCs
            mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length)

            # Combine mean and std for MFCCs only
            features = np.concatenate([
                np.mean(mfccs, axis=1), np.std(mfccs, axis=1)
            ])
            # Clip outliers using config values
            features = np.clip(features, FEATURE_CLIP_MIN, FEATURE_CLIP_MAX)
            features_list.append(features)

        return np.array(features_list)
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return np.array([])

def preprocess_input(features):
    """
    Preprocess input features to match training data shape and scaling.
    Assumes features are extracted as a 2D array (n_chunks, 26).
    """
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    return features_scaled

def aggregate_predictions(predictions):
    """
    Aggregate chunk predictions using majority voting with tie-breaking.
    Returns the most frequent class, or High Noise if tied.
    """
    if len(predictions) == 0:
        return None
    unique, counts = np.unique(predictions, return_counts=True)
    max_count = np.max(counts)
    most_common = unique[counts == max_count]
    if len(most_common) > 1:  # Tie case
        return 2  # Default to High Noise (2) for safety
    return most_common[0]

def predict_class(audio_file):
    """
    Predict and aggregate class (0: Low Noise, 1: Mid Noise, 2: High Noise) for a given audio file.
    """
    if model is None or scaler is None:
        print("ERROR: Model or scaler not loaded. Cannot perform prediction.")
        return None
        
    features = extract_features_from_chunks(audio_file)
    if len(features) == 0:
        return None
    features_scaled = preprocess_input(features)
    predictions = model.predict(features_scaled)
    aggregated_class = aggregate_predictions(predictions)
    return aggregated_class

if __name__ == "__main__":
    sample_audio = SAMPLE_AUDIO_PATH
    
    if not os.path.exists(sample_audio):
        print(f"ERROR: Sample audio file not found at {sample_audio}")
        print("Please update SAMPLE_AUDIO_PATH in config.py or provide a valid audio file path.")
        exit(1)
    
    print(f"Processing audio file: {sample_audio}")
    predicted_class = predict_class(sample_audio)
    
    if predicted_class is not None:
        print(f"Aggregated predicted class: {CLASS_LABELS[predicted_class]}")
        
        features = extract_features_from_chunks(sample_audio)
        if len(features) > 0:
            features_scaled = preprocess_input(features)
            chunk_predictions = model.predict(features_scaled)
            print(f"\nChunk-wise predictions ({len(chunk_predictions)} chunks):")
            for i, pred in enumerate(chunk_predictions):
                print(f"Chunk {i+1}: {CLASS_LABELS[pred]}")
    else:
        print("No valid predictions due to processing error.")