"""
Configuration file for Acoustic Noise Classifier
Modify these parameters according to your dataset and requirements
"""

import os

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Dataset paths - Update these paths to point to your audio data directories
CLASS1_DIR = "data/low_noise"     # Low noise audio files (Class 0)
CLASS2_DIR = "data/mid_noise"     # Mid noise audio files (Class 1)  
CLASS3_DIR = "data/high_noise"    # High noise audio files (Class 2)

# =============================================================================
# AUDIO PROCESSING PARAMETERS
# =============================================================================

# Audio sample rate (Hz) - MS-SNSD standard, suitable for hearing aids
SAMPLE_RATE = 16000

# Frame length for MFCC computation (seconds)
FRAME_LENGTH = 0.02  # 20ms frames

# Duration of audio chunks for processing (seconds)
CHUNK_DURATION = 2   # 2-second chunks for real-time processing

# Number of MFCC coefficients to extract
N_MFCC = 13

# =============================================================================
# MODEL CONFIGURATION  
# =============================================================================

# Model save paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "mlp_scaler.pkl")

# Data save paths
DATA_DIR = "data"
FEATURES_PATH = os.path.join(DATA_DIR, "X.npy")
LABELS_PATH = os.path.join(DATA_DIR, "y.npy")

# =============================================================================
# SPEECH METRICS CONFIGURATION
# =============================================================================

# Model identifier for speech metrics calculation
MODEL_NAME = "DS64"

# Paths for speech quality metrics calculation (update as needed)
NOISY_AUDIO_PATH = "./audio_samples/noisy"
CLEAN_AUDIO_PATH = "./audio_samples/clean"
METRICS_OUTPUT_PATH = "./metrics/results.csv"

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Output directory for plots and visualizations
PLOTS_DIR = "plots"

# Class labels for visualization
CLASS_LABELS = ["Low Noise", "Mid Noise", "High Noise"]

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

# Default audio file for testing inference (update path as needed)
SAMPLE_AUDIO_PATH = "./sample_audio/test_file.wav"

# Feature clipping bounds (to handle outliers)
FEATURE_CLIP_MIN = -100
FEATURE_CLIP_MAX = 100

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_directories():
    """Create necessary directories if they don't exist"""
    dirs_to_create = [MODEL_DIR, DATA_DIR, PLOTS_DIR]
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)

def validate_paths():
    """Validate that required dataset paths exist"""
    paths_to_check = [CLASS1_DIR, CLASS2_DIR, CLASS3_DIR]
    missing_paths = []
    
    for path in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("WARNING: The following dataset paths do not exist:")
        for path in missing_paths:
            print(f"  - {path}")
        print("Please update the paths in config.py or create the directories.")
        return False
    return True

if __name__ == "__main__":
    create_directories()
    print("Directories created successfully!")
    
    if validate_paths():
        print("All dataset paths exist!")
    else:
        print("Please update dataset paths in config.py")
