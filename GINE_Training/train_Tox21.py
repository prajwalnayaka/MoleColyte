import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GINEConv, global_mean_pool
from .data_loader import tox21_train_loader, tox21_val_loader
from .layer_embedding import BondEmbedding


# ==========================================
# 1. THE MAIN MODEL ARCHITECTURE (Reverted Baseline)
# ==========================================
class MoleColyteModel(nn.Module):
    def __init__(self, in_node_features=8, emb_dim=128, hidden_dim=64, out_features=1):
        super().__init__()
        self.bond_emb = BondEmbedding(emb_dim)

        self.mlp1 = nn.Sequential(nn.Linear(in_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.conv1 = GINEConv(self.mlp1, edge_dim=emb_dim + 1)
        self.conv2 = GINEConv(self.mlp2, edge_dim=emb_dim + 1)

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )

    def forward(self, x, pos, edge_index, edge_attr, batch):
        edge_embedding = self.bond_emb(edge_attr)
        row, col = edge_index
        distances = torch.pairwise_distance(pos[row], pos[col]).unsqueeze(-1)
        physics_edges = torch.cat([edge_embedding, distances], dim=-1)

        # Pass 1
        x = self.conv1(x, edge_index, edge_attr=physics_edges)
        x = torch.relu(x)

        # Pass 2
        x = self.conv2(x, edge_index, edge_attr=physics_edges)

        mol_features = global_mean_pool(x, batch)
        return self.prediction_head(mol_features)


# ==========================================
# 2. DYNAMIC WEIGHT CALCULATION
# ==========================================
def calculate_dynamic_weights(loader):
    print("Scanning training data to calculate exact imbalance penalties...")
    num_pos = torch.zeros(12)
    num_neg = torch.zeros(12)

    for batch in loader:
        targets = batch.y.to(torch.float)

        for i in range(12):
            col = targets[:, i]
            valid_col = col[col == col]  # NaN Mask

            num_pos[i] += (valid_col == 1).sum()
            num_neg[i] += (valid_col == 0).sum()

    dynamic_weights = num_neg / (num_pos + 1e-5)
    print(f"Calculated Pathway Penalties: {dynamic_weights.numpy().round(1)}\n")
    return dynamic_weights


# ==========================================
# 3. THE STAGE 2 FINE-TUNING LOOP
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kiln: {device}")

    # Import the model as it is
    model = MoleColyteModel(out_features=1)

    # Loading pre-trained physics weights from training on QM9 (Standard baseline)
    model.load_state_dict(torch.load(r"../GINE weights/molecolyte_gine_qm9_best.pt"))

    # Modifying the model's prediction head according to the target features of Tox21
    model.prediction_head = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 12)  # 12 targets for Tox21
    )
    model = model.to(device)

    # Note: Using lr=0.001 since the 64-dim architecture without dropout handles it well
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dynamic_penalty = calculate_dynamic_weights(tox21_train_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=dynamic_penalty, reduction='none')

    EPOCHS = 20
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        # TRAINING PHASE
        model.train()
        total_train_loss = 0
        print("Starting Epoch", epoch + 1)

        for step, batch in enumerate(tox21_train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            predictions = model(batch.x.to(torch.float), batch.pos, batch.edge_index, batch.edge_attr, batch.batch)
            target_flags = batch.y.to(torch.float)

            is_valid = target_flags == target_flags

            raw_loss = criterion(predictions, target_flags)
            loss = raw_loss[is_valid].mean()

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
            for batch in tox21_val_loader:
                batch = batch.to(device)

                predictions = model(batch.x.to(torch.float), batch.pos, batch.edge_index, batch.edge_attr, batch.batch)
                target_flags = batch.y.to(torch.float)

                is_valid = target_flags == target_flags

                raw_loss = criterion(predictions, target_flags)
                loss = raw_loss[is_valid].mean()

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(tox21_val_loader)
        print(f"\tValidation Phase | Avg Val Loss: {avg_val_loss:.4f}")

        # ------------------------------------------
        # CHECKPOINTING LOGIC
        # ------------------------------------------
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Restored standard path naming
            torch.save(model.state_dict(), r"../GINE weights/molecolyte_gine_tox21_best.pt")
            print(f"🏆 New best Tox21 model saved! (Lowest Val Loss: {best_loss:.4f})\n")
        else:
            print(f"Model did not improve. Best Val loss remains: {best_loss:.4f}\n")


if __name__ == "__main__":
    since = time.time()
    train()
    print(f"Time: {(time.time()-since)/60:.2f} minutes.")