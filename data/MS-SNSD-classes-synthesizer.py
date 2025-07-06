import glob
import numpy as np
import soundfile as sf
import os
from audiolib import audioread, audiowrite, snr_mixer

SAMPLING_RATE = 16000
AUDIO_LENGTH_SEC = 30
SILENCE_LENGTH_SEC = 0
AUDIO_LENGTH = int(SAMPLING_RATE * AUDIO_LENGTH_SEC)
SILENCE_SAMPLES = int(SAMPLING_RATE * SILENCE_LENGTH_SEC)

CLASS_1_SNR = (30, 40)
CLASS_2_SNR = (15, 25)
CLASS_3_SNR = (0, 10) 

CLEAN_DIR = './clean_train'
NOISE_DIR = './noise_train'
OUTPUT_NOISY_DIRS = {
    1: './NoisySpeech_training_class1',
    2: './NoisySpeech_training_class2',
    3: './NoisySpeech_training_class3',
}
OUTPUT_CLEAN_DIRS = {
    1: './CleanSpeech_training_class1',
    2: './CleanSpeech_training_class2',
    3: './CleanSpeech_training_class3',
}
OUTPUT_NOISE_DIRS = {
    1: './Noise_training_class1',
    2: './Noise_training_class2',
    3: './Noise_training_class3',
}

for d in OUTPUT_NOISY_DIRS.values():
    os.makedirs(d, exist_ok=True)
for d in OUTPUT_CLEAN_DIRS.values():
    os.makedirs(d, exist_ok=True)
for d in OUTPUT_NOISE_DIRS.values():
    os.makedirs(d, exist_ok=True)

clean_files = glob.glob(os.path.join(CLEAN_DIR, '*.wav'))
noise_files = glob.glob(os.path.join(NOISE_DIR, '*.wav'))

rng = np.random.default_rng(42) 

def get_random_snr_and_class(filecounter):
    """Cycle through classes for balanced distribution, with clean speech for Class 1"""
    class_choice = (filecounter % 3) + 1
    if class_choice == 1:
        if rng.random() < 0.5:
            snr = rng.uniform(35, 40)  
        else:
            snr = rng.uniform(30, 35) 
    elif class_choice == 2:
        snr = rng.uniform(*CLASS_2_SNR)
    else:  # class 3
        snr = rng.uniform(*CLASS_3_SNR)
    return snr, class_choice

filecounter = 0
total_minutes = 60*20  
target_samples = int(total_minutes * 60 * SAMPLING_RATE)
num_samples = 0

class_counts = {1: 0, 2: 0, 3: 0}

while num_samples < target_samples:
    idx_s = rng.integers(len(clean_files))
    clean, fs = audioread(clean_files[idx_s])

    while len(clean) < AUDIO_LENGTH:
        idx_s = (idx_s + 1) % len(clean_files)
        newclean, fs = audioread(clean_files[idx_s])
        clean = np.append(clean, np.zeros(SILENCE_SAMPLES))
        clean = np.append(clean, newclean)
    clean = clean[:AUDIO_LENGTH]

    idx_n = rng.integers(len(noise_files))
    noise, fs = audioread(noise_files[idx_n])

    while len(noise) < len(clean):
        idx_n = (idx_n + 1) % len(noise_files)
        newnoise, fs = audioread(noise_files[idx_n])
        noise = np.append(noise, np.zeros(SILENCE_SAMPLES))
        noise = np.append(noise, newnoise)
    noise = noise[:len(clean)]

    snr_value, noise_class = get_random_snr_and_class(filecounter)
    if noise_class == 1 and snr_value >= 35:
        clean_snr = clean
        noise_snr = np.zeros_like(clean)
        noisy_snr = clean_snr
    else:
        clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=snr_value)

    filecounter += 1
    class_counts[noise_class] += 1
    
    noisy_path = os.path.join(OUTPUT_NOISY_DIRS[noise_class], f'noisy{filecounter}_SNRdb_{snr_value:.2f}_class{noise_class}_clnsp{filecounter}.wav')
    clean_path = os.path.join(OUTPUT_CLEAN_DIRS[noise_class], f'clnsp{filecounter}.wav')
    noise_path = os.path.join(OUTPUT_NOISE_DIRS[noise_class], f'noisy{filecounter}_SNRdb_{snr_value:.2f}_class{noise_class}.wav')

    audiowrite(noisy_snr, fs, noisy_path, norm=False)
    audiowrite(clean_snr, fs, clean_path, norm=False)
    audiowrite(noise_snr, fs, noise_path, norm=False)

    num_samples += len(noisy_snr)
    
    if filecounter % 50 == 0:
        print(f"Generated {filecounter} files, ~{num_samples / SAMPLING_RATE / 60:.1f} minutes")
        print(f"Class distribution: {class_counts}")

print(f"Done. Generated ~{num_samples / SAMPLING_RATE / 60:.2f} minutes of data.")
print(f"Final class distribution: {class_counts}")
print(f"Class 1 (DSP): {class_counts[1]} files")
print(f"Class 2 (Simple ML): {class_counts[2]} files")
print(f"Class 3 (Deep Learning): {class_counts[3]} files")