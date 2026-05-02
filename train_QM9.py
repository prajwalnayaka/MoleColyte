import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GINEConv, global_mean_pool

from layer_embedding import BondEmbedding
from data_loader import train_loader


# ==========================================
# 1. THE MAIN MODEL ARCHITECTURE (MoleColyte)
# ==========================================
class MoleColyteModel(nn.Module):
    def __init__(self, in_node_features=8, emb_dim=128, hidden_dim=64):
        super().__init__()

        self.bond_emb = BondEmbedding(emb_dim)
                                             # 8                 # 64                             # 64         # 64
        self.mlp1 = nn.Sequential(nn.Linear(in_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                                             # 64         # 64                              # 64        # 64
        self.mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        self.conv1 = GINEConv(self.mlp1, edge_dim=emb_dim + 1)
        self.conv2 = GINEConv(self.mlp2, edge_dim=emb_dim + 1)

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # One target output (internal energy) because we just want the model to learn the physics and calculating internal energy requires a lot of physics knowledge
        )

    def forward(self, x, pos, edge_index, edge_attr, batch):
        edge_embedding = self.bond_emb(edge_attr)

        # Calculate the exact 3D distance between connected atoms
        row, col = edge_index
        distances = torch.pairwise_distance(pos[row], pos[col]).unsqueeze(-1)
        physics_edges = torch.cat([edge_embedding, distances], dim=-1)

        x = self.conv1(x, edge_index, edge_attr=physics_edges)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr=physics_edges)

        mol_features = global_mean_pool(x, batch)
        return self.prediction_head(mol_features)


# ==========================================
# 2. THE TRAINING LOOP
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kiln: {device}")
    print("Training model on QM9 dataset")

    model = MoleColyteModel(in_node_features=8, emb_dim=128).to(device) # 6 to match Tox21
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # MSE because the final output will be numerical


    EPOCHS = 10
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print("Starting Epoch", epoch + 1)

        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()

            predictions = model(batch.x, batch.pos, batch.edge_index, batch.edge_attr, batch.batch)

            target_property = batch.y[:, 7].to(torch.float)
            loss = criterion(predictions.view(-1), target_property.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 500 == 0:
                print(f"\t Epoch {epoch + 1}/{EPOCHS} | Batch {step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"==> Epoch {epoch + 1} complete | Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), r"Trained Models/molecolyte_qm9_pretrained_best.pt")
            print(f"🏆 New best model saved! (Lowest Loss: {best_loss:.4f})\n")
        else:
            print(f"Model did not improve. Best loss remains: {best_loss:.4f}\n")


if __name__ == "__main__":
    since = time.time()
    train()
    print(f"Time: {(time.time()-since)/60:.2f} minutes.")