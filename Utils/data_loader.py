import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# Comment and uncomment depending on which dataset you need to use.

dataset = torch.load("../Preprocessed Datasets/qm9_3d_fgn_dataset.pt", weights_only=False)
#dataset = torch.load("../Preprocessed Datasets/tox21_3d_fgn_dataset.pt", weights_only=False)

total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - (train_size + val_size)

print(f"Splitting: Train: {train_size} |Val: {val_size} | Test: {test_size}")

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

#PyG DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)