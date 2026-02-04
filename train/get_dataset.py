import json
import os


from dataset.verse import VerseDataset

def get_dataset(cfg):

    data_folder = cfg.dataset.folder 
    
    split_path = getattr(cfg.dataset, 'split_path', './splits.json')

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split not found : {split_path}")

    with open(split_path, 'r') as f:
        splits = json.load(f)

    # Train
    train_dataset = VerseDataset(
        folder=data_folder,
        filenames=splits['train'],
        num_train_angles=None
    )

    # Val
    val_dataset = VerseDataset(
        folder=data_folder,
        filenames=splits['val'],
        num_train_angles=None
    )

    # Test
    test_dataset = VerseDataset(
        folder=data_folder,
        filenames=splits.get('test', []),
        num_train_angles=None
    )

    print(f"--- Datasets Setup ---")
    print(f"Split File : {split_path}")
    print(f"Train size : {len(train_dataset)}")
    print(f"Val size   : {len(val_dataset)}")
    
    return train_dataset, val_dataset, test_dataset