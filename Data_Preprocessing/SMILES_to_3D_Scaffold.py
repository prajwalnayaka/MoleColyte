import pandas as pd
import torch
import time
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType

"""
 N => Number of atoms in a molecule
 F => Number of features of an atom (Atomic Number, Hybridization, Electrical Charge etc.)
 E => Number of bonds in a molecule
 K => Number of functional groups in a molecule

 EGNN NOTE:
 Unlike GINEConv which needed pre-computed distances baked into edge_attr,
 EGNN computes pairwise distances dynamically from pos during message passing.
 pos must be stored as raw 3D coordinates — which is exactly what we were
 already doing. The preprocessing pipeline is structurally identical.

 SCAFFOLD SPLIT NOTE:
 The original SMILES string is now saved alongside each graph so that
 Bemis-Murcko scaffolds can be computed later for scaffold-based splitting.
 This does not affect QM9 preprocessing — QM9 pretraining still uses random
 splitting, consistent with the EGNN/SchNet/DimeNet literature.
"""

# ===========================================================================
#  FUNCTIONAL GROUP NODE (FGN) DETECTION
# ===========================================================================
FUNCTIONAL_GROUP_SMARTS = [
    ("benzene",  "c1ccccc1"),
    ("carbonyl", "[CX3]=[OX1]"),
    ("carboxyl",  "[CX3](=O)[OX2H1]"),
    ("hydroxyl",  "[OX2H]"),
    ("amine",     "[NX3;H2,H1;!$(NC=O)]"),
    ("amide",     "[NX3][CX3](=[OX1])"),
    ("nitro",     "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("ether",     "[OD2]([#6])[#6]"),
    ("thiol",     "[SX2H]"),
    ("halogen",   "[F,Cl,Br,I]"),
    ("sulfonyl",  "[$([#16X4](=[OX1])=[OX1])]"),
    ("phosphate", "[PX4](=O)"),
]

COMPILED_FG_PATTERNS = [
    (name, Chem.MolFromSmarts(smarts)) for name, smarts in FUNCTIONAL_GROUP_SMARTS
]


def detect_functional_groups(mol):
    found = []
    seen_sets = set()
    for fg_name, pattern in COMPILED_FG_PATTERNS:
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            key = frozenset(match)
            if key not in seen_sets:
                seen_sets.add(key)
                found.append((fg_name, match))
    return found


def build_fgn_augmented_graph(mol, x_tensor, pos_tensor, edge_index_tensor, edge_attr_tensor):
    """
    Takes the base graph tensors and appends K Functional Group Nodes (FGNs).

    Returns:
        x_aug        : (N+K) x (F+1) — original features + is_fgn flag
        pos_aug      : (N+K) x 3     — original coords + FGN centroids
        edge_aug     : 2 x (E+E_new) — original bonds + bipartite FGN edges
        edge_attr_aug: (E+E_new) x 5 — original bond attrs + virtual bond attrs
    """
    n_atoms = x_tensor.size(0)

    is_fgn_flag = torch.zeros(n_atoms, 1, dtype=torch.float)
    x_aug = torch.cat([x_tensor, is_fgn_flag], dim=1)

    fgn_features_list = []
    fgn_pos_list = []
    new_edges = []
    new_edge_attrs = []

    fg_matches = detect_functional_groups(mol)

    for fg_idx, (fg_name, atom_indices) in enumerate(fg_matches):
        virtual_node_idx = n_atoms + fg_idx

        member_features = x_tensor[list(atom_indices)]
        mean_features   = member_features.mean(dim=0)
        fgn_row         = torch.cat([mean_features, torch.tensor([1.0])])
        fgn_features_list.append(fgn_row)

        member_pos = pos_tensor[list(atom_indices)]
        centroid   = member_pos.mean(dim=0)
        fgn_pos_list.append(centroid)

        for atom_idx in atom_indices:
            new_edges.append([atom_idx,         virtual_node_idx])
            new_edges.append([virtual_node_idx, atom_idx])
            virtual_flag = [0, 0, 0, 0, 1]
            new_edge_attrs.append(virtual_flag)
            new_edge_attrs.append(virtual_flag)

    if fgn_features_list:
        fgn_features = torch.stack(fgn_features_list, dim=0)
        fgn_pos      = torch.stack(fgn_pos_list,      dim=0)
        x_aug   = torch.cat([x_aug,      fgn_features], dim=0)
        pos_aug = torch.cat([pos_tensor, fgn_pos],      dim=0)
    else:
        pos_aug = pos_tensor

    if new_edges:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        edge_aug        = torch.cat([edge_index_tensor, new_edge_tensor], dim=1)
        new_attr_tensor = torch.tensor(new_edge_attrs, dtype=torch.float)
        edge_attr_aug   = torch.cat([edge_attr_tensor, new_attr_tensor], dim=0)
    else:
        edge_aug      = edge_index_tensor
        edge_attr_aug = edge_attr_tensor

    return x_aug, pos_aug, edge_aug, edge_attr_aug


# ===========================================================================
#  MAIN PIPELINE
# ===========================================================================
print("Loading dataset...")
ds = pd.read_csv("tox21.csv").fillna(0)

label_columns = [col for col in ds.columns if col not in ["smiles", "mol_id"]]

successful_graphs = []
failed_count = 0

print(f"Starting 2D-to-3D + FGN Pipeline for {len(ds)} molecules.")
start_time = time.time()

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    "balls. Drake and One Piece goated asl"  # DO NOT DELETE — placeholder for virtual bond column
]

for index, row in ds.iterrows():
    smiles_string = row["smiles"]

    if index % 500 == 0:
        print(f"Processing Molecule {index} / {len(ds)}...")

    # --- PHASE 1: 2D Graph Construction ---
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        failed_count += 1
        continue
    mol = Chem.AddHs(mol)

    # --- PHASE 2: 3D Geometry & Physics ---
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if res != 0:
        failed_count += 1
        continue

    ff_result = AllChem.MMFFOptimizeMolecule(mol)
    if ff_result != 0:
        failed_count += 1
        continue

    # --- PHASE 3: Geometry Tensor ---
    conformer  = mol.GetConformer()
    pos_tensor = torch.tensor(conformer.GetPositions(), dtype=torch.float)

    # --- PHASE 4: Feature Tensor ---
    atom_features = []
    for atom in mol.GetAtoms():
        hybridization = atom.GetHybridization()
        feature_vector = [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            1 if hybridization == HybridizationType.SP   else 0,
            1 if hybridization == HybridizationType.SP2  else 0,
            1 if hybridization == HybridizationType.SP3  else 0,
        ]
        atom_features.append(feature_vector)

    x_tensor = torch.tensor(atom_features, dtype=torch.float)

    # --- PHASE 5: Bonds Tensor ---
    edge_indices = []
    edge_attrs   = []
    for bond in mol.GetBonds():
        start_idx    = bond.GetBeginAtomIdx()
        end_idx      = bond.GetEndAtomIdx()
        b_type       = bond.GetBondType()
        bond_feature = [int(b_type == t) for t in BOND_TYPES]
        edge_indices.append([start_idx, end_idx])
        edge_indices.append([end_idx,   start_idx])
        edge_attrs.append(bond_feature)
        edge_attrs.append(bond_feature)

    if len(edge_indices) > 0:
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr_tensor  = torch.tensor(edge_attrs,   dtype=torch.float)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor  = torch.empty((0, len(BOND_TYPES)), dtype=torch.float)

    # --- PHASE 6: FGN AUGMENTATION ---
    x_aug, pos_aug, edge_index_aug, edge_attr_aug = build_fgn_augmented_graph(
        mol, x_tensor, pos_tensor, edge_index_tensor, edge_attr_tensor
    )

    # --- PHASE 7: Label Tensor ---
    labels   = row[label_columns].values.astype(float)
    y_tensor = torch.tensor(labels, dtype=torch.float)  # shape: (12,)

    # --- PHASE 8: Build DGL Graph ---
    src = edge_index_aug[0]
    dst = edge_index_aug[1]

    g = dgl.graph((src, dst), num_nodes=x_aug.size(0))
    g.ndata['x']         = x_aug           # (N+K, 8)   node features
    g.ndata['pos']       = pos_aug          # (N+K, 3)   3D coordinates — EGNN reads and updates these
    g.edata['edge_attr'] = edge_attr_aug    # (E+E_new, 5) bond type one-hot

    # SCAFFOLD SPLIT: smiles_string saved alongside graph + label
    successful_graphs.append((g, y_tensor, smiles_string))

print(f"\nConversion Completed")
print("-" * 35)
print(f"Successfully generated {len(successful_graphs)} 3D DGL Graphs.")
print(f"Skipped {failed_count} physically impossible molecules.")
print("-" * 35)

torch.save(successful_graphs, "tox21_3d_egnn_dataset_scaffold.pt")
end_time   = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time / 60:.2f} minutes.")
print("Saved successfully as 'tox21_3d_egnn_dataset_scaffold.pt'.")
