import torch.nn as nn

class BondEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.bond_translator = nn.Linear(5, emb_dim)