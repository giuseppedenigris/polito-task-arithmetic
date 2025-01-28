## Global imports ##
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
import random
from collections import defaultdict

class BalancedDataset(Dataset):
    def __init__(self, dataset: Dataset, cache_path: str=None):
        self.dataset = dataset

        if cache_path is not None and os.path.exists(cache_path):
            # Use already computed and cached balanced_indices
            with open(cache_path) as fp:
                self.balanced_indices = json.load(fp)
        else:
            # Group indices according to the class to which they belong                
            class_indices = defaultdict(list)
            for idx, (_, label) in tqdm(enumerate(self.dataset), "Balancing.."):
                # Unwrap the integer if it's in a tensor
                if isinstance(label, torch.Tensor):
                    label = label.item()
                class_indices[label].append(idx)

            # Find class with min samples
            self.min_class_count = min(len(indices) for indices in class_indices.values())
                
            # Build the array of down-sampled indices
            self.balanced_indices = []
            for indices in class_indices.values():
                self.balanced_indices.extend(indices[:self.min_class_count])    # Truncate to min_class_count
            
            # Shuffle the indices (reproducible thanks to seed)
            random.Random(0).shuffle(self.balanced_indices)

            if cache_path is not None:
                # Save balanced_indices to file, to retrieve it easily without recomputing it
                print(f"Balanced Dataset | Saving down-sampled indices to '{cache_path}'")
                with open(cache_path, "w") as fp:
                    json.dump(self.balanced_indices, fp)

    def __getitem__(self, idx):
        balanced_idx = self.balanced_indices[idx]
        return self.dataset[balanced_idx]

    def __len__(self):
        return len(self.balanced_indices)