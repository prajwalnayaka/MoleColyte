import torch
import dgl
from torch.utils.data import Dataset, DataLoader, random_split


# ===========================================================================
#  WRAPPER: Makes the list of (dgl.graph, label) tuples behave like a
#  proper PyTorch Dataset so random_split and DataLoader work on it.
# ===========================================================================
class MoleculeDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of (dgl.DGLGraph, y_tensor) tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    dgl.batch() merges N separate molecule graphs into one large disconnected
    graph, tracking which nodes/edges belong to which molecule internally.
    This is the DGL equivalent of PyG's automatic Batch.from_data_list().
    """
    graphs, labels = zip(*batch)
    batched_graph  = dgl.batch(graphs)
    batched_labels = torch.stack(labels, dim=0).squeeze(1)
    return batched_graph, batched_labels


# ===========================================================================
#  TOX21 LOADERS  (classification — 12 binary labels)
# ===========================================================================
def get_tox21_loaders(path=r"/kaggle/input/datasets/prajwalnayakat/molecolyte-datasets/tox21_3d_egnn_dataset.pt", batch_size=32):
    raw     = torch.load(path, weights_only=False)
    dataset = MoleculeDataset(raw)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size   = int(0.1 * total)
    test_size  = total - train_size - val_size

    print(f"Tox21 Split → Train: {train_size} | Val: {val_size} | Test: {test_size}")

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


# ===========================================================================
#  QM9 LOADERS  (regression — 19 quantum properties, we train on column 7)
# ===========================================================================
def get_qm9_loaders(path=r"/kaggle/input/datasets/prajwalnayakat/molecolyte-datasets/qm9_3d_egnn_dataset.pt", batch_size=32):
    raw     = torch.load(path, weights_only=False)
    dataset = MoleculeDataset(raw)

    total = len(dataset)
    train_size = int(0.8 * total)
    val_size   = int(0.1 * total)
    test_size  = total - train_size - val_size

    print(f"QM9 Split → Train: {train_size} | Val: {val_size} | Test: {test_size}")

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


# ===========================================================================
#  EXPOSE LOADERS FOR IMPORT
#  egnn_layer.py and train_Tox21.py import directly from here.
# ===========================================================================
tox21_train_loader, tox21_val_loader, tox21_test_loader = get_tox21_loaders()
qm9_train_loader,   qm9_val_loader,   qm9_test_loader   = get_qm9_loaders()


# ===========================================================================
#  VERIFICATION — run this file directly to confirm batches look right
# ===========================================================================
if __name__ == "__main__":
    print("\n--- Tox21 Batch Check ---")
    for batch_graph, batch_labels in tox21_train_loader:
        print(f"Graphs in batch      : {batch_graph.batch_size}")
        print(f"Total nodes          : {batch_graph.num_nodes()}")
        print(f"Total edges          : {batch_graph.num_edges()}")
        print(f"Node feature shape   : {batch_graph.ndata['x'].shape}")
        print(f"Position shape       : {batch_graph.ndata['pos'].shape}")
        print(f"Edge attr shape      : {batch_graph.edata['edge_attr'].shape}")
        print(f"Label shape          : {batch_labels.shape}")   # should be (32, 12)
        break

    print("\n--- QM9 Batch Check ---")
    for batch_graph, batch_labels in qm9_train_loader:
        print(f"Graphs in batch      : {batch_graph.batch_size}")
        print(f"Total nodes          : {batch_graph.num_nodes()}")
        print(f"Total edges          : {batch_graph.num_edges()}")
        print(f"Node feature shape   : {batch_graph.ndata['x'].shape}")
        print(f"Position shape       : {batch_graph.ndata['pos'].shape}")
        print(f"Edge attr shape      : {batch_graph.edata['edge_attr'].shape}")
        print(f"Label shape          : {batch_labels.shape}")   # should be (32, 19)
        break