import torch
import time
import torch.nn as nn
import torch.optim as optim
import dgl
from egnn_layer import EGNNLayer
from data_loader import qm9_train_loader, qm9_val_loader


# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class MoleColyteEGNN(nn.Module):
    """
    MoleColyte built on EGNN layers instead of GINEConv.

    Key difference from the GINEConv version:
    - The node embedding is projected to hidden_dim before any message passing
    - Each EGNNLayer updates both node hidden states AND 3D coordinates
    - Final readout uses the updated coordinates implicitly via mean pooling
    """
    def __init__(self, in_node_features=8, hidden_dim=128, edge_attr_dim=5, num_layers=3, out_features=1):
        super().__init__()

        # Project raw atom features into the hidden dimension
        self.input_proj = nn.Linear(in_node_features, hidden_dim)

        # Stack of EGNN layers — each one refines both node states and positions
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim, edge_attr_dim=edge_attr_dim)
            for _ in range(num_layers)
        ])

        # Readout head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )

    def forward(self, g, x, pos, edge_attr):
        # Project input features into hidden space
        h = self.input_proj(x)    # (N, hidden_dim)

        # Run through EGNN layers — pos evolves each layer
        for layer in self.egnn_layers:
            h, pos = layer(g, h, pos, edge_attr)

        # Global mean pool — average node states per molecule in the batch
        g.ndata['h'] = h
        mol_embedding = dgl.mean_nodes(g, 'h')   # (batch_size, hidden_dim)

        return self.prediction_head(mol_embedding)


# ==========================================
# 2. TRAINING LOOP
# ==========================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kiln: {device}")
    print("Pre-training EGNN on QM9 (target: U0, internal energy at 0K)")

    model     = MoleColyteEGNN(out_features=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    EPOCHS    = 10
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}")

        for step, (batch_graph, batch_labels) in enumerate(qm9_train_loader):
            batch_graph  = batch_graph.to(device)
            batch_labels = batch_labels.to(device)

            x         = batch_graph.ndata['x'].to(torch.float)
            pos       = batch_graph.ndata['pos'].to(torch.float)
            edge_attr = batch_graph.edata['edge_attr'].to(torch.float)

            optimizer.zero_grad()

            predictions     = model(batch_graph, x, pos, edge_attr)
            target_property = batch_labels[:, 0].to(torch.float)   # U0: internal energy
            loss            = criterion(predictions.view(-1), target_property.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 500 == 0:
                print(f"\t Epoch {epoch + 1}/{EPOCHS} | Batch {step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(qm9_train_loader)
        print(f"==> Epoch {epoch + 1} complete | Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "../EGNN Weights/molecolyte_egnn_qm9_best.pt")
            print(f"🏆 New best model saved! (Lowest Loss: {best_loss:.4f})\n")
        else:
            print(f"Model did not improve. Best loss remains: {best_loss:.4f}\n")


if __name__ == "__main__":
    since = time.time()
    train()
    print(f"Time: {(time.time() - since) / 60:.2f} minutes.")