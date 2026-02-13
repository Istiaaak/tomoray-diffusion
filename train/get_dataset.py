import json
import os


from dataset.verse import VerseDataset

def get_dataset(cfg, overfit_n=None):

    data_folder = cfg.dataset.folder 
    
    split_path = getattr(cfg.dataset, 'split_path', './splits.json')

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"split not found : {split_path}")

    with open(split_path, 'r') as f:
        splits = json.load(f)

    train_filenames = splits["train"]

    if overfit_n is not None and overfit_n > 0:
        print(f"Overfit mode with {overfit_n} CTs")
        train_filenames = train_filenames[:overfit_n]



    # Train
    train_dataset = VerseDataset(
        folder=data_folder,
        filenames=train_filenames,
    )

    # Val
    val_dataset = VerseDataset(
        folder=data_folder,
        filenames=splits['val'],
    )

    # Test
    test_dataset = VerseDataset(
        folder=data_folder,
        filenames=splits.get('test', []),
    )

    return train_dataset, val_dataset, test_dataset