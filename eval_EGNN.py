import os
os.environ["DGL_SKIP_GRAPHBOLT"] = "1"

import torch
import time
import torch.nn as nn
import dgl
from sklearn.metrics import roc_auc_score
from EGNN_Training.data_loader import tox21_test_loader
from EGNN_Training.egnn_layer import EGNNLayer


# ==========================================
# MODEL ARCHITECTURE (must match train_Tox21.py exactly)
# ==========================================
class MoleColyteEGNN(nn.Module):
    def __init__(self, in_node_features=8, hidden_dim=128, edge_attr_dim=5, num_layers=3, out_features=12):
        super().__init__()

        self.input_proj = nn.Linear(in_node_features, hidden_dim)

        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim, edge_attr_dim=edge_attr_dim)
            for _ in range(num_layers)
        ])

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, g, x, pos, edge_attr):
        h = self.input_proj(x)

        for layer in self.egnn_layers:
            h, pos = layer(g, h, pos, edge_attr)

        g.ndata['h'] = h
        mol_embedding = dgl.mean_nodes(g, 'h')

        return self.prediction_head(mol_embedding)


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kiln: {device}")

    model = MoleColyteEGNN(in_node_features=8, hidden_dim=128, edge_attr_dim=5, num_layers=3, out_features=12)
    model.load_state_dict(
        torch.load(
            r"/kaggle/input/models/prajwalnayakat/egnn-tox21/pytorch/default/1/molecolyte_egnn_tox21_best.pt",
            weights_only=False,
            map_location=device
        )
    )
    model = model.to(device)
    model.eval()

    all_preds   = []
    all_targets = []

    print("Running inference on the unseen Test Set...")
    with torch.no_grad():
        for batch_graph, batch_labels in tox21_test_loader:
            batch_graph  = batch_graph.to(device)
            batch_labels = batch_labels.to(device)

            x         = batch_graph.ndata['x'].to(torch.float)
            pos       = batch_graph.ndata['pos'].to(torch.float)
            edge_attr = batch_graph.edata['edge_attr'].to(torch.float)

            logits = model(batch_graph, x, pos, edge_attr)   # (batch, 12)
            probs  = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_targets.append(batch_labels.cpu())

    # Concatenate all batches — stay in pure torch
    all_preds   = torch.cat(all_preds,   dim=0)   # (N, 12)
    all_targets = torch.cat(all_targets, dim=0)   # (N, 12)

    roc_aucs = []

    print("\n=== MoleColyte EGNN — Tox21 Evaluation Results ===")
    for i in range(12):
        # NaN mask in pure torch
        valid_mask   = all_targets[:, i] == all_targets[:, i]
        task_targets = all_targets[valid_mask, i].tolist()   # plain Python list, zero numpy
        task_preds   = all_preds[valid_mask,   i].tolist()   # plain Python list, zero numpy

        if len(set(task_targets)) > 1:
            score = roc_auc_score(task_targets, task_preds)
            roc_aucs.append(score)
            print(f"Assay {i + 1:02d} AUC-ROC: {score:.4f}")
        else:
            print(f"Assay {i + 1:02d} Skipped (Only one class present in test set)")

    print("--------------------------------------------------")
    mean_auc = sum(roc_aucs) / len(roc_aucs)
    print(f"🏆 FINAL MEAN AUC-ROC: {mean_auc:.4f}")


if __name__ == "__main__":
    since = time.time()
    evaluate()
    print(f"Time: {(time.time() - since) / 60:.2f} minutes.")