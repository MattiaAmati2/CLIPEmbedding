import random
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets, DatasetDict

def create_custom_splits(hf_dataset_id, train_pct=0.70, val_pct=0.15, seed=42):
    print(f"Downloading {hf_dataset_id} from Hugging Face...")
    dataset = load_dataset(hf_dataset_id)

    splits_to_concat = [dataset[split] for split in dataset.keys()]
    all_ds = concatenate_datasets(splits_to_concat)

    label_key_map = {
        "food101": "label",
        "Alanox/stanford-dogs": "target",
        "Donghyun99/FGVC-Aircraft": "variation"
    }
    label_key = label_key_map.get(hf_dataset_id, "label")

    labels = all_ds[label_key]
    grouped_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped_indices[label].append(idx)

    random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []

    for label, idxs in grouped_indices.items():
        random.shuffle(idxs)

        total_items = len(idxs)
        train_end = int(total_items * train_pct)
        val_end = train_end + int(total_items * val_pct)

        train_idx.extend(idxs[:train_end])
        val_idx.extend(idxs[train_end:val_end])
        test_idx.extend(idxs[val_end:])

    custom_splits = DatasetDict({
        "train": all_ds.select(train_idx),
        "val": all_ds.select(val_idx),
        "test": all_ds.select(test_idx)
    })

    print(f"Split Complete for {hf_dataset_id}:")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    return custom_splits, label_key