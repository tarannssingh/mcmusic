import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import random

INPUT_DIR = "GTZAN/genres_processed"
OUTPUT_DIR = "GTZAN/genres_augmented"
AUG_PER_FILE = 3
SAMPLE_RATE = 22050
DURATION = 5  # seconds
TARGET_SAMPLES = SAMPLE_RATE * DURATION

def time_shift(y, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def apply_pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=np.random.uniform(-n_steps, n_steps))

def apply_time_stretch(y, rate_range=(0.8, 1.2)):
    rate = np.random.uniform(*rate_range)
    try:
        return librosa.effects.time_stretch(y, rate)
    except:
        return y

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def volume_scale(y, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    return y * scale

def augment_audio(y, sr):
    if random.random() < 0.5:
        y = time_shift(y)
    if random.random() < 0.5:
        y = apply_pitch_shift(y, sr)
    if random.random() < 0.5:
        y = apply_time_stretch(y)
    if random.random() < 0.5:
        y = add_noise(y)
    if random.random() < 0.5:
        y = volume_scale(y)
    return y


def augment_dataset():
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in tqdm(files, desc="Augmenting"):
            if not filename.lower().endswith(".wav"):
                continue

            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(root, INPUT_DIR)
            output_subdir = os.path.join(OUTPUT_DIR, rel_path)
            os.makedirs(output_subdir, exist_ok=True)

            try:
                y, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"Skipping {input_path}: {e}")
                continue

            for i in range(AUG_PER_FILE):
                y_aug = augment_audio(np.copy(y), sr)
                y_aug = librosa.util.fix_length(y_aug, size=22050)
                output_filename = filename.replace(".wav", f"_aug{i+1}.wav")
                output_path = os.path.join(output_subdir, output_filename)
                sf.write(output_path, y_aug, SAMPLE_RATE)

if __name__ == "__main__":
    augment_dataset()
