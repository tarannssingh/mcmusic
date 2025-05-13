import os
import librosa
import soundfile as sf
import random

RAW_DATA_DIR = "GTZAN/genres_original/"
PROCESSED_DATA_DIR = "GTZAN/genres_processed/"
SAMPLE_RATE = 22050
DURATION = 5

def preprocess_audio(input_path, output_path, sample_rate=SAMPLE_RATE, duration=DURATION):
    try:
        y, sr = librosa.load(input_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"Failed to load {input_path}: {e}")
        return

    target_samples = duration * sample_rate

    if len(y) < target_samples:
        y = librosa.util.fix_length(y, target_samples)
    else:
        max_start = len(y) - target_samples
        start_sample = random.randint(0, max_start)
        y = y[start_sample:start_sample + target_samples]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y, sample_rate)


def process_dataset():
    for genre in os.listdir(RAW_DATA_DIR):
        genre_dir = os.path.join(RAW_DATA_DIR, genre)
        if not os.path.isdir(genre_dir):
            continue
        for filename in os.listdir(genre_dir):
            if not filename.endswith(".wav"):
                continue
            input_path = os.path.join(genre_dir, filename)
            output_path = os.path.join(PROCESSED_DATA_DIR, genre, filename)
            preprocess_audio(input_path, output_path)
            print(f"Processed: {input_path} -> {output_path}")

if __name__ == "__main__":
    process_dataset()
