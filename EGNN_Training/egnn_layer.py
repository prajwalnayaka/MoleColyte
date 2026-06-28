import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class EGNNLayer(nn.Module):
    """
    One layer of an Equivariant Graph Neural Network (EGNN).
    Based on: "E(n) Equivariant Graph Neural Networks" (Satorras et al., 2021)

    What makes EGNN different from GINEConv:
    - GINEConv: uses pre-computed distances as extra edge features. Coordinates
      are static — they never change during message passing.
    - EGNN: computes distances live from pos during every forward pass, AND
      updates the 3D coordinates of every node as part of the layer itself.
      This means the geometry evolves as information flows through the network,
      making it sensitive to the actual 3D shape of the molecule.

    Per-layer operations:
    1. For every edge: compute distance from current pos, run edge MLP
    2. For every node: aggregate neighbour messages, run node MLP → new hidden state
    3. For every node: compute a weighted sum of relative position vectors → update pos
    """

    def __init__(self, hidden_dim, edge_attr_dim=5):
        super().__init__()

        # Edge MLP: takes [h_i, h_j, distance, edge_attr] → message
        # hidden_dim * 2 for the two node states + 1 for distance + edge_attr_dim for bond type
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_attr_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node MLP: takes [h_i, aggregated messages] → new h_i
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate MLP: takes edge message → scalar weight for pos update
        # Output is a single scalar that scales the relative position vector (pos_i - pos_j)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def edge_message(self, edges):
        """
        Runs on every edge simultaneously.
        Computes the distance between the two endpoint atoms from their
        current positions, then feeds everything into the edge MLP.
        """
        # Relative position vector and its scalar distance
        rel_pos  = edges.src['pos'] - edges.dst['pos']          # (E, 3)
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)    # (E, 1)

        # Concatenate: source hidden state, dest hidden state, distance, bond type
        edge_input = torch.cat([
            edges.src['h'],           # (E, hidden_dim)
            edges.dst['h'],           # (E, hidden_dim)
            distance,                  # (E, 1)
            edges.data['edge_attr'],   # (E, edge_attr_dim)
        ], dim=-1)

        message    = self.edge_mlp(edge_input)     # (E, hidden_dim)
        coord_weight = self.coord_mlp(message)     # (E, 1) — scalar for pos update

        return {
            'message':      message,
            'coord_weight': coord_weight * rel_pos,  # (E, 3) — weighted relative vector
        }

    def node_update(self, nodes):
        """
        Runs on every node simultaneously.
        Aggregates incoming messages and updates the node's hidden state.
        """
        # 'agg_msg' is the sum of all incoming messages (set by dgl after edge_message)
        node_input = torch.cat([nodes.data['h'], nodes.data['agg_msg']], dim=-1)
        new_h      = self.node_mlp(node_input)
        return {'h': new_h}

    def forward(self, g, h, pos, edge_attr):
        with g.local_scope():
            g.ndata['h']         = h
            g.ndata['pos']       = pos
            g.edata['edge_attr'] = edge_attr

            # Step 1: compute messages and coordinate weights along every edge
            g.apply_edges(self.edge_message)

            # Step 2: aggregate messages into each node
            g.update_all(fn.copy_e('message', 'm'), fn.sum('m', 'agg_msg'))

            # Step 3: update node hidden states
            g.apply_nodes(self.node_update)

            # Step 4: update coordinates
            # Sum the weighted relative vectors arriving at each node
            g.update_all(fn.copy_e('coord_weight', 'cw'), fn.sum('cw', 'agg_cw'))
            new_pos = pos + g.ndata['agg_cw']   # (N, 3)

            return g.ndata['h'], new_pos
