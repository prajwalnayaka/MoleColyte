import torch.nn as nn

class BondEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.bond_translator = nn.Linear(5, emb_dim)

    def forward(self, edge_attr):
        translated_bonds = self.bond_translator(edge_attr)
        return translated_bonds