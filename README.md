# Acoustic Noise Classifier

A machine learning project for classifying acoustic noise levels in audio signals using MFCC features and Random Forest/MLP classifiers.

## Overview

This project implements an acoustic noise classifier that can categorize audio signals into three noise levels:

- **Low Noise** (Class 0)
- **Mid Noise** (Class 1)
- **High Noise** (Class 2)

The classifier uses Mel-Frequency Cepstral Coefficients (MFCC) features extracted from short audio chunks and employs both Random Forest and Multi-Layer Perceptron (MLP) models for classification.

## Features

- **Audio Feature Extraction**: MFCC-based feature extraction with configurable parameters
- **Data Augmentation**: Time-shifting augmentation for improved model robustness
- **Multiple Models**: Support for Random Forest and MLP classifiers
- **Real-time Processing**: Chunk-based processing suitable for real-time applications
- **Cross-Platform Inference**: Python and Rust implementations for maximum deployment flexibility
- **Edge Device Support**: Optimized Rust implementation for Raspberry Pi and embedded systems
- **ONNX Model Export**: Cross-platform model format for deployment
- **Model Evaluation**: Comprehensive evaluation with confusion matrices and metrics
- **Speech Quality Metrics**: Calculate PESQ, STOI, MOS, and other speech quality metrics
- **Stream Processing**: Real-time audio stream classification

## Project Structure

```
├── README.md                                   # Project documentation
├── requirements.txt                            # Python dependencies
├── config.py                                   # Configuration parameters
├── data-preprocessing.py                       # Data loading and preprocessing
├── acoustic-noise-classifier.ipynb           # Model training (Random Forest + MLP)
├── evaluation.ipynb                           # Model evaluation and visualization
├── calculate-speech-metrics.py               # Speech quality metrics calculation
├── inference-rfe.py                          # Legacy Python inference script
├── python/                                   # Python inference implementation
│   └── inference.py                         # Optimized Python inference
├── rust/                                     # Rust inference implementation
│   ├── Cargo.toml                           # Rust dependencies and config
│   └── src/                                 # Rust source code
│       ├── main.rs                         # CLI interface (file & stream modes)
│       ├── lib.rs                          # Library exports
│       ├── audio_processor.rs              # Audio I/O and processing
│       ├── feature_extractor.rs            # MFCC feature extraction
│       ├── model_inference.rs              # ONNX model inference
│       ├── scaler.rs                       # Feature scaling
│       ├── stream_processor.rs             # Real-time stream processing
│       └── model.onnx                      # Exported model for inference
├── models/                                   # Directory for saved models
├── data/                                     # Directory for datasets
│   └── MS-SNSD-classes-synthesizer.py      # Dataset synthesis script
└── plots/                                    # Generated visualization outputs
```

## Installation

### Python Environment

1. Clone the repository:

```bash
git clone https://github.com/yourusername/acoustic-noise-classifier.git
cd acoustic-noise-classifier
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: For dataset synthesis, you'll also need the MS-SNSD audiolib module which will be available when you clone the MS-SNSD repository.

### Rust Environment (for high-performance inference)

1. Install Rust (if not already installed):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

2. Build the Rust inference engine:

```bash
cd rust
cargo build --release
```

3. The compiled binary will be available at `rust/target/release/audio_noise_classifier`

## Configuration

Modify `config.py` to set your dataset paths and parameters:

```python
# Dataset paths
CLASS1_DIR = "path/to/low_noise_audio_files"    # Low noise samples
CLASS2_DIR = "path/to/mid_noise_audio_files"    # Mid noise samples
CLASS3_DIR = "path/to/high_noise_audio_files"   # High noise samples

# Audio processing parameters
SAMPLE_RATE = 16000      # Audio sample rate
CHUNK_DURATION = 2       # Duration of audio chunks in seconds
N_MFCC = 13             # Number of MFCC coefficients
```

## Usage

### 1. Dataset Preparation

#### Option A: Generate Synthetic Dataset (Recommended)

```bash
# Clone MS-SNSD dataset
cd data
git clone https://github.com/microsoft/MS-SNSD.git

# Generate balanced dataset for three noise classes
python MS-SNSD-classes-synthesizer.py
```

#### Option B: Use Custom Dataset

Organize your audio files in the required directory structure (see Dataset Format section below).

### 2. Data Preprocessing

Extract MFCC features from your dataset:

```bash
python data-preprocessing.py
```

This will:

- Load audio files from the configured directories
- Extract MFCC features from audio chunks
- Apply data augmentation
- Save processed features as `X.npy` and `y.npy`

### 3. Model Training

Use the Jupyter notebook for interactive model training and evaluation:

```bash
jupyter notebook acoustic-noise-classifier.ipynb
```

This notebook includes:

- Random Forest classifier with RFE feature selection
- Multi-Layer Perceptron (MLP) with hyperparameter optimization
- Model evaluation and performance comparison
- ONNX model export for cross-platform deployment

### 4. Model Inference

#### Python Inference (Recommended for Development)

```bash
# Using the optimized inference script
python python/inference.py

# Or using the legacy script
python inference-rfe.py
```

#### Rust Inference (Recommended for Production/Edge Devices)

```bash
# Build the Rust inference engine (one time)
cd rust
cargo build --release

# File-based inference
./target/release/audio_noise_classifier file path/to/audio.wav

# Real-time stream processing
./target/release/audio_noise_classifier stream

# Using custom model path
./target/release/audio_noise_classifier file audio.wav path/to/custom/model.onnx
```

#### Deployment Options:

- **Raspberry Pi**: Use the Rust implementation for optimal performance
- **Edge Devices**: Rust binary with minimal dependencies
- **Web Services**: Python implementation with REST API wrapper
- **Mobile Apps**: ONNX model can be integrated with mobile ML frameworks

### 5. Model Evaluation

Visualize and compare model performance:

```bash
jupyter notebook evaluation.ipynb
```

This provides:

- Speech quality metrics visualization
- Multi-model performance comparison
- Correlation analysis between metrics
- Comprehensive radar charts

### 6. Calculate Speech Metrics

Evaluate speech quality metrics:

```bash
python calculate-speech-metrics.py
```

Configure the input paths in the script for your noisy and clean audio files.

## Dataset Generation

### Automatic Dataset Synthesis (MS-SNSD)

The project includes a synthesis script to automatically generate training data with three noise classes using the MS-SNSD dataset.

#### Prerequisites

1. Clone the MS-SNSD dataset in the `data/` folder:

```bash
cd data
git clone https://github.com/microsoft/MS-SNSD.git
```

2. Update paths in the synthesis script if needed:

```bash
# Edit data/MS-SNSD-classes-synthesizer.py
# Update CLEAN_DIR and NOISE_DIR paths to point to your MS-SNSD location:
CLEAN_DIR = './MS-SNSD/clean_train'  # Clean speech files
NOISE_DIR = './MS-SNSD/noise_train'  # Noise files
```

#### Generate Synthetic Dataset

```bash
cd data
python MS-SNSD-classes-synthesizer.py
```

This script will generate:

- **~20 hours** of balanced training data across three classes
- **Automatic SNR distribution** based on hearing aid requirements:
  - **Class 1 (Low Noise)**: 30-40 dB SNR (quiet environments)
  - **Class 2 (Mid Noise)**: 15-25 dB SNR (conversational settings)
  - **Class 3 (High Noise)**: 0-10 dB SNR (noisy environments)

#### Output Structure

```
data/
├── NoisySpeech_training_class1/    # Class 1: Low noise audio
├── NoisySpeech_training_class2/    # Class 2: Mid noise audio
├── NoisySpeech_training_class3/    # Class 3: High noise audio
├── CleanSpeech_training_class1/    # Corresponding clean speech
├── CleanSpeech_training_class2/    # Corresponding clean speech
├── CleanSpeech_training_class3/    # Corresponding clean speech
├── Noise_training_class1/          # Corresponding noise signals
├── Noise_training_class2/          # Corresponding noise signals
└── Noise_training_class3/          # Corresponding noise signals
```

#### Configuration

Update `config.py` to point to the generated dataset:

```python
CLASS1_DIR = "data/NoisySpeech_training_class1"  # Low noise
CLASS2_DIR = "data/NoisySpeech_training_class2"  # Mid noise
CLASS3_DIR = "data/NoisySpeech_training_class3"  # High noise
```

## Dataset Format

The classifier expects audio files organized in three directories:

```
data/
├── low_noise/     # Class 0: Low noise audio files (.wav)
├── mid_noise/     # Class 1: Mid noise audio files (.wav)
└── high_noise/    # Class 2: High noise audio files (.wav)
```

**Audio Requirements:**

- Format: WAV files
- Sample rate: 16 kHz (configurable)
- Mono channel
- Minimum duration: 2 seconds per file

## Model Performance

The project includes comprehensive evaluation metrics:

- **Classification Accuracy**: Overall model performance across noise levels
- **Confusion Matrices**: Detailed breakdown of classification performance
- **Precision, Recall, F1-score**: Per-class performance metrics
- **Feature Importance Analysis**: Understanding which features drive predictions
- **Speech Quality Metrics**: PESQ, STOI, MOS, SDR, SAR, SI-SDR
- **Learning Curves**: Training and validation performance over time
- **Radar Charts**: Multi-dimensional performance visualization

## Deployment Guide

### Edge Devices (Raspberry Pi, etc.)

The Rust implementation is optimized for resource-constrained environments:

```bash
# Cross-compile for Raspberry Pi (from x86_64)
rustup target add armv7-unknown-linux-gnueabihf
cargo build --release --target armv7-unknown-linux-gnueabihf

# Transfer binary to Raspberry Pi
scp target/armv7-unknown-linux-gnueabihf/release/audio_noise_classifier pi@your-pi:/home/pi/

# Run on Raspberry Pi
./audio_noise_classifier stream
```
## Technical Architecture

### Python Stack

- **Preprocessing**: librosa for audio processing and MFCC extraction
- **ML Models**: scikit-learn for Random Forest and MLP classifiers
- **Evaluation**: matplotlib/seaborn for visualization
- **Metrics**: speechmetrics, PESQ for quality assessment

### Rust Stack

- **Audio I/O**: hound for WAV file handling, cpal for real-time audio
- **Signal Processing**: rustfft for FFT operations, custom MFCC implementation
- **ML Inference**: ort (ONNX Runtime) for cross-platform model execution
- **Async Processing**: tokio for concurrent stream handling

### Model Pipeline

1. **Audio Input** → 2-second chunks at 16kHz
2. **Feature Extraction** → 13 MFCC coefficients (mean + std = 26 features)
3. **Preprocessing** → StandardScaler normalization
4. **Classification** → MLP/RF model inference
5. **Aggregation** → Majority voting across chunks
6. **Output** → Low/Mid/High noise classification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Libraries**: Built using scikit-learn, librosa, rustfft, and ONNX Runtime
- **Audio Processing**: MFCC feature extraction based on librosa implementation
- **Datasets**: Compatible with MS-SNSD and VoiceBank+DEMAND datasets
- **Cross-Platform**: ONNX format enables deployment across multiple platforms
- **Edge Computing**: Rust implementation optimized for resource-constrained devices
