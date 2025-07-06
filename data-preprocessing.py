import os
import librosa
import gc
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    SAMPLE_RATE, FRAME_LENGTH, CHUNK_DURATION, N_MFCC,
    CLASS1_DIR, CLASS2_DIR, CLASS3_DIR,
    FEATURES_PATH, LABELS_PATH, PLOTS_DIR,
    FEATURE_CLIP_MIN, FEATURE_CLIP_MAX, CLASS_LABELS,
    create_directories
)

def augment_audio(audio, sr):
    """
    Apply lightweight time-shifting augmentation.
    """
    shift = np.random.randint(-sr//10, sr//10)
    audio_shifted = np.roll(audio, shift)
    return audio_shifted

def extract_features_from_chunks(audio_file, sr=SAMPLE_RATE, chunk_duration=CHUNK_DURATION):
    """
    Extract MFCC mean and std features from short chunks of the audio file.
    Returns a list of feature vectors.
    """
    try:
        audio, _ = librosa.load(audio_file, sr=sr, mono=True)
        chunk_size = int(chunk_duration * sr)
        hop_length = int(FRAME_LENGTH * sr)
        features_list = []

        audios = [audio, augment_audio(audio, sr)]
        for audio_data in audios:
            for i in range(0, len(audio_data) - chunk_size + 1, chunk_size):
                chunk = audio_data[i:i + chunk_size]

                mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length)


                features = np.concatenate([
                    np.mean(mfccs, axis=1), np.std(mfccs, axis=1)
                ])
                features = np.clip(features, FEATURE_CLIP_MIN, FEATURE_CLIP_MAX)
                features_list.append(features)

        return features_list
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return []

def load_chunked_dataset(class1_dir, class2_dir, class3_dir, chunk_duration=CHUNK_DURATION):
    """
    Load audio files, extract features, and assign labels (0: low noise, 1: mid noise, 2: high noise).
    Returns feature matrix X and label vector y.
    """
    if os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH):
        print(f"Loading saved dataset from {FEATURES_PATH} and {LABELS_PATH}...")
        X = np.load(FEATURES_PATH)
        y = np.load(LABELS_PATH)
        print("Class distribution:", np.bincount(y))
        return X, y

    X, y = [], []

    # Process Class 1: Low noise (label = 0)
    for file in os.listdir(class1_dir):
        if file.endswith('.wav'):
            features_list = extract_features_from_chunks(
                os.path.join(class1_dir, file), chunk_duration=chunk_duration
            )
            X.extend(features_list)
            y.extend([0] * len(features_list))
            gc.collect()
            print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    # Process Class 2: Mid noise (label = 1)
    for file in os.listdir(class2_dir):
        if file.endswith('.wav'):
            features_list = extract_features_from_chunks(
                os.path.join(class2_dir, file), chunk_duration=chunk_duration
            )
            X.extend(features_list)
            y.extend([1] * len(features_list))
            gc.collect()
            print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    # Process Class 3: High noise (label = 2)
    for file in os.listdir(class3_dir):
        if file.endswith('.wav'):
            features_list = extract_features_from_chunks(
                os.path.join(class3_dir, file), chunk_duration=chunk_duration
            )
            X.extend(features_list)
            y.extend([2] * len(features_list))
            gc.collect()
            print(f"Memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    X = np.array(X)
    y = np.array(y)
    print("Class distribution:", np.bincount(y))

    np.save(FEATURES_PATH, X)
    np.save(LABELS_PATH, y)
    print(f"Dataset saved as {FEATURES_PATH} and {LABELS_PATH}")

    return X, y

def plot_feature_distributions(features, labels):
    """
    Plot distribution of the first MFCC mean to check class separability.
    """
    create_directories()
    plt.figure(figsize=(10, 6))
    for label in np.unique(labels):
        plt.hist(features[labels == label, 0], bins=30, alpha=0.5, label=f'{CLASS_LABELS[label]}')
    plt.title('Distribution of First MFCC Mean')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_distribution.png'))
    plt.show()

def plot_confusion_matrix(cm, classes=CLASS_LABELS):
    """
    Plot the confusion matrix using seaborn with Blues color map.
    """
    create_directories()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix (Blues Color Map)')
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    plt.show()

def main():
    """
    Main function to demonstrate dataset loading and feature extraction.
    Update the dataset paths in config.py before running.
    """
    # Create necessary directories
    create_directories()
    
    # Load dataset with configured paths
    print("Loading dataset and extracting MFCC features...")
    print(f"Class 1 (Low Noise): {CLASS1_DIR}")
    print(f"Class 2 (Mid Noise): {CLASS2_DIR}")
    print(f"Class 3 (High Noise): {CLASS3_DIR}")
    
    # Check if directories exist
    if not all(os.path.exists(path) for path in [CLASS1_DIR, CLASS2_DIR, CLASS3_DIR]):
        print("ERROR: One or more dataset directories do not exist.")
        print("Please update the paths in config.py or create the directories.")
        return
    
    X, y = load_chunked_dataset(CLASS1_DIR, CLASS2_DIR, CLASS3_DIR, chunk_duration=CHUNK_DURATION)
    print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features each")
    
    # Plot feature distributions
    plot_feature_distributions(X, y)

if __name__ == "__main__":
    main()