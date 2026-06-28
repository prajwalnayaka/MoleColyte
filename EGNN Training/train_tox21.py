import time
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy
from egnn_layer import EGNNLayer
from data_loader import tox21_train_loader, tox21_val_loader


# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class MoleColyteEGNN(nn.Module):
    def __init__(self, in_node_features=8, hidden_dim=128, edge_attr_dim=5, num_layers=3, out_features=1):
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


# ==========================================
# 2. DYNAMIC WEIGHT CALCULATION
# ==========================================
def calculate_dynamic_weights(loader):
    print("Scanning training data to calculate exact imbalance penalties...")
    num_pos = torch.zeros(12)
    num_neg = torch.zeros(12)

    for batch_graph, batch_labels in loader:
        targets = batch_labels.to(torch.float)

        for i in range(12):
            col       = targets[:, i]
            valid_col = col[col == col]  # NaN mask

            num_pos[i] += (valid_col == 1).sum()
            num_neg[i] += (valid_col == 0).sum()

    dynamic_weights = num_neg / (num_pos + 1e-5)
    print(f"Calculated Pathway Penalties: {dynamic_weights.cpu().tolist()}\n")
    return dynamic_weights


# ==========================================
# 3. FINE-TUNING LOOP
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kiln: {device}")

    # Load the pre-trained QM9 model (out_features=1, as trained)
    model = MoleColyteEGNN(out_features=1)
    torch.load("../EGNN Weights/molecolyte_egnn_qm9_best.pt", weights_only=True)

    # Swap the prediction head for Tox21's 12 binary targets
    model.prediction_head = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 12)   # 12 toxicity assay outputs
    )
    model = model.to(device)

    optimizer       = optim.Adam(model.parameters(), lr=1e-3)
    dynamic_penalty = calculate_dynamic_weights(tox21_train_loader).to(device)
    criterion       = nn.BCEWithLogitsLoss(pos_weight=dynamic_penalty, reduction='none')

    EPOCHS    = 20
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        # ------------------------------------------
        # TRAINING PHASE
        # ------------------------------------------
        model.train()
        total_train_loss = 0
        print(f"Starting Epoch {epoch + 1}")

        for step, (batch_graph, batch_labels) in enumerate(tox21_train_loader):
            batch_graph  = batch_graph.to(device)
            batch_labels = batch_labels.to(device)

            x         = batch_graph.ndata['x'].to(torch.float)
            pos       = batch_graph.ndata['pos'].to(torch.float)
            edge_attr = batch_graph.edata['edge_attr'].to(torch.float)

            optimizer.zero_grad()

            predictions  = model(batch_graph, x, pos, edge_attr)   # (batch, 12)
            target_flags = batch_labels.to(torch.float)             # (batch, 12)

            is_valid = target_flags == target_flags                  # NaN mask
            raw_loss = criterion(predictions, target_flags)
            loss     = raw_loss[is_valid].mean()

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if step % 500 == 0:
                print(f"\t Epoch {epoch + 1}/{EPOCHS} | Batch {step} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(tox21_train_loader)
        print(f"==> Epoch {epoch + 1} Train Complete | Average BCE Loss: {avg_train_loss:.4f}")

        # ------------------------------------------
        # VALIDATION PHASE
        # ------------------------------------------
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_graph, batch_labels in tox21_val_loader:
                batch_graph  = batch_graph.to(device)
                batch_labels = batch_labels.to(device)

                x         = batch_graph.ndata['x'].to(torch.float)
                pos       = batch_graph.ndata['pos'].to(torch.float)
                edge_attr = batch_graph.edata['edge_attr'].to(torch.float)

                predictions  = model(batch_graph, x, pos, edge_attr)
                target_flags = batch_labels.to(torch.float)

                is_valid = target_flags == target_flags
                raw_loss = criterion(predictions, target_flags)
                loss     = raw_loss[is_valid].mean()

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(tox21_val_loader)
        print(f"\tValidation Phase | Avg Val Loss: {avg_val_loss:.4f}")

        # ------------------------------------------
        # CHECKPOINTING
        # ------------------------------------------
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "../EGNN Weights/molecolyte_egnn_tox21_best.pt")
            print(f"🏆 New best Tox21 model saved! (Lowest Val Loss: {best_loss:.4f})\n")
        else:
            print(f"Model did not improve. Best Val loss remains: {best_loss:.4f}\n")


if __name__ == "__main__":
    since = time.time()
    train()
    print(f"Time: {(time.time() - since) / 60:.2f} minutes.")