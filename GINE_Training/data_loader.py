import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# Comment and uncomment depending on which dataset you need to use.

tox21_dataset = torch.load(r"D:\MoleColyte\Preprocessed Datasets\tox21_3d_fgn_dataset.pt", weights_only=False)

tox21_total_size = len(tox21_dataset)
tox21_train_size = int(0.8 * tox21_total_size)
tox21_val_size = int(0.1 * tox21_total_size)
tox21_test_size = tox21_total_size - (tox21_train_size + tox21_val_size)

print(f"Splitting: Train: {tox21_train_size} |Val: {tox21_val_size} | Test: {tox21_test_size}")

tox21_train_dataset, tox21_val_dataset, tox21_test_dataset = random_split(tox21_dataset, [tox21_train_size, tox21_val_size, tox21_test_size])

#PyG DataLoader
tox21_train_loader = DataLoader(tox21_train_dataset, batch_size=32, shuffle=True)
tox21_val_loader = DataLoader(tox21_val_dataset, batch_size=32, shuffle=False)
tox21_test_loader = DataLoader(tox21_test_dataset, batch_size=32, shuffle=False)


# ********************************************************************************************************************************************************************

qm9_dataset = torch.load(r"D:\MoleColyte\Preprocessed Datasets\qm9_3d_fgn_dataset.pt", weights_only=False)

qm9_total_size = len(qm9_dataset)
qm9_train_size = int(0.8 * qm9_total_size)
qm9_val_size = int(0.1 * qm9_total_size)
qm9_test_size = qm9_total_size - (qm9_train_size + qm9_val_size)

print(f"Splitting: Train: {qm9_train_size} |Val: {qm9_val_size} | Test: {qm9_test_size}")

qm9_train_dataset, qm9_val_dataset, qm9_test_dataset = random_split(qm9_dataset, [qm9_train_size, qm9_val_size, qm9_test_size])

#PyG DataLoader
qm9_train_loader = DataLoader(qm9_train_dataset, batch_size=32, shuffle=True)
qm9_val_loader = DataLoader(qm9_val_dataset, batch_size=32, shuffle=False)
qm9_test_loader = DataLoader(qm9_test_dataset, batch_size=32, shuffle=False)