from pesq import pesq
import speechmetrics
import pandas as pd
import librosa
import os
from config import MODEL_NAME, NOISY_AUDIO_PATH, CLEAN_AUDIO_PATH, METRICS_OUTPUT_PATH, create_directories

MODEL = MODEL_NAME

NOISY_PATH = NOISY_AUDIO_PATH
CLEAN_PATH = CLEAN_AUDIO_PATH  
CSV_PATH = METRICS_OUTPUT_PATH

NOISY_PATH = NOISY_PATH.replace("Model", MODEL)
CSV_PATH = CSV_PATH.replace("Model", MODEL)

def calculate_metrics(noisy_path, clean_path):
    clean_audio, sr = librosa.load(clean_path, sr=None)
    noisy_audio, _ = librosa.load(noisy_path, sr=sr)  

    min_len = min(len(noisy_audio), len(clean_audio))
    noisy_audio = noisy_audio[:min_len]
    clean_audio = clean_audio[:min_len]

    metrics = speechmetrics.load(['stoi', 'mosnet', 'sisdr', 'bsseval'], window=None)

    results = metrics(noisy_audio, clean_audio, rate=sr)

    try:
        if sr in [8000, 16000]:
            pesq_score = pesq(sr, noisy_audio, clean_audio, 'wb' if sr == 16000 else 'nb')
            results['pesq'] = pesq_score
        else:
            print(f"PESQ cannot be computed for sampling rate {sr}. Resample to 16kHz or 8kHz if required.")
    
    except Exception as e:
        print(f"Error computing PESQ: {e}")
        results['pesq'] = None
        
    return {
        "MOS": results['mosnet'][0][0],
        "SDR": results['sdr'][0][0],
        "SAR": results['sar'][0][0],
        "SI-SDR": results['sisdr'],
        "STOI": results['stoi'],
        "PESQ": results['pesq']
    }

def main():
    """
    Main function to calculate speech quality metrics for all audio files.
    """
    create_directories()
    
    if not os.path.exists(NOISY_PATH):
        print(f"ERROR: Noisy audio path does not exist: {NOISY_PATH}")
        print("Please update NOISY_AUDIO_PATH in config.py")
        return
        
    if not os.path.exists(CLEAN_PATH):
        print(f"ERROR: Clean audio path does not exist: {CLEAN_PATH}")
        print("Please update CLEAN_AUDIO_PATH in config.py")
        return
    
    print("Processing audio files...")
    print(f"Noisy audio path: {NOISY_PATH}")
    print(f"Clean audio path: {CLEAN_PATH}")
    print(f"Output CSV path: {CSV_PATH}")
    
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    
    metrics_list = []
    audio_files = [f for f in os.listdir(NOISY_PATH) if f.endswith(".wav")]
    
    if not audio_files:
        print(f"No WAV files found in {NOISY_PATH}")
        return
    
    for i, file_name in enumerate(audio_files):
        noisy_path = os.path.join(NOISY_PATH, file_name)
        clean_path = os.path.join(CLEAN_PATH, file_name)
        
        if not os.path.exists(clean_path):
            print(f"WARNING: Clean file not found for {file_name}, skipping...")
            continue
        
        try:
            metrics = calculate_metrics(noisy_path=noisy_path, clean_path=clean_path)
            metrics['File Name'] = file_name
            metrics_list.append(metrics)
            
            print(f"Processed {i+1}/{len(audio_files)} files: {file_name}")
            
        except Exception as e:
            print(f"ERROR processing {file_name}: {e}")
            continue
    
    if metrics_list:
        df = pd.DataFrame(metrics_list)
        df.to_csv(CSV_PATH, index=False)
        print(f"\nMetrics saved to: {CSV_PATH}")
        print(f"Processed {len(metrics_list)} files successfully")
        
        print("\nSummary Statistics:")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        print(df[numeric_columns].describe())
    else:
        print("No files were processed successfully.")

if __name__ == "__main__":
    main()