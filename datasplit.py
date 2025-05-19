import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Constants
INPUT_DIR = "GTZAN/genres_processed"
OUTPUT_DIR = "GTZAN/genres_split"
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED = 42

random.seed(SEED)

def split_files(file_list):
    random.shuffle(file_list)
    total = len(file_list)
    n_train = int(SPLIT_RATIOS["train"] * total)
    n_val = int(SPLIT_RATIOS["val"] * total)

    return {
        "train": file_list[:n_train],
        "val": file_list[n_train:n_train + n_val],
        "test": file_list[n_train + n_val:]
    }

def split_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    genres = [d.name for d in Path(INPUT_DIR).iterdir() if d.is_dir()]

    for genre in genres:
        genre_path = Path(INPUT_DIR) / genre
        files = [f for f in genre_path.glob("*.wav")]

        split_data = split_files(files)

        for split_type, file_list in split_data.items():
            out_dir = Path(OUTPUT_DIR) / split_type / genre
            os.makedirs(out_dir, exist_ok=True)
            for file_path in file_list:
                shutil.copy(file_path, out_dir / file_path.name)
    print("✅ Dataset successfully split into train/val/test sets.")

if __name__ == "__main__":
    split_dataset()
