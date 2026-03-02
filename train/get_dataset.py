import json
import os
import glob
import random
from dataset.verse import VerseDataset



def create_and_save_split(data_folder, split_path, train_ratio=0.8, val_ratio=0.1):
    print("No split found. Creating a new split 80/10/10...")

    all_files = sorted(
        glob.glob(os.path.join(data_folder, "**", "*.pickle"), recursive=True)
    )

    all_files_rel = [os.path.relpath(p, data_folder) for p in all_files]

    if len(all_files_rel) == 0:
        raise ValueError(f"No .pickle files found in {data_folder}")

    random.shuffle(all_files_rel)

    n_total = len(all_files_rel)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": all_files_rel[:n_train],
        "val":   all_files_rel[n_train:n_train + n_val],
        "test":  all_files_rel[n_train + n_val:]
    }

    with open(split_path, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Split saved in: {split_path}")
    print(
        f"Train={len(splits['train'])}, "
        f"Val={len(splits['val'])}, "
        f"Test={len(splits['test'])}"
    )

    return splits


def get_dataset(cfg, overfit_n=None):
    data_folder = cfg.dataset.folder

    # 🔹 split TOUJOURS stocké dans le dossier des données
    split_path = os.path.join(data_folder, "splits.json")

    if os.path.exists(split_path):
        print(f"Loading existing split: {split_path}")
        with open(split_path, "r") as f:
            splits = json.load(f)
    else:
        splits = create_and_save_split(
            data_folder=data_folder,
            split_path=split_path
        )

    train_filenames = splits["train"]

    if overfit_n is not None and overfit_n > 0:
        print(f"Overfit mode with {overfit_n} CTs")
        train_filenames = train_filenames[:overfit_n]

    train_dataset = VerseDataset(
        folder=data_folder,
        filenames=train_filenames,
    )

    val_dataset = VerseDataset(
        folder=data_folder,
        filenames=splits["val"],
    )

    test_dataset = VerseDataset(
        folder=data_folder,
        filenames=splits.get("test", []),
    )

    return train_dataset, val_dataset, test_dataset