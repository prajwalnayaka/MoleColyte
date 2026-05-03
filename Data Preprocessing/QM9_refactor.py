import torch
import time
from rdkit import Chem
from torch_geometric.datasets import QM9
from torch_geometric.data import Data

# ===========================================================================
#  FUNCTIONAL GROUP NODE (FGN) DETECTION
# ===========================================================================
FUNCTIONAL_GROUP_SMARTS = [
    ("benzene", "c1ccccc1"),
    ("carbonyl", "[CX3]=[OX1]"),
    ("carboxyl", "[CX3](=O)[OX2H1]"),
    ("hydroxyl", "[OX2H]"),
    ("amine", "[NX3;H2,H1;!$(NC=O)]"),
    ("amide", "[NX3][CX3](=[OX1])"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("thiol", "[SX2H]"),
    ("halogen", "[F,Cl,Br,I]"),
    ("sulfonyl", "[$([#16X4](=[OX1])=[OX1])]"),
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
    n_atoms = x_tensor.size(0)

    # Expand x with is_fgn_flag=0 column for all real atoms
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
        mean_features = member_features.mean(dim=0)
        fgn_row = torch.cat([mean_features, torch.tensor([1.0])])
        fgn_features_list.append(fgn_row)

        member_pos = pos_tensor[list(atom_indices)]
        centroid = member_pos.mean(dim=0)
        fgn_pos_list.append(centroid)

        # Bipartite edges AND the virtual flag
        for atom_idx in atom_indices:
            new_edges.append([atom_idx, virtual_node_idx])
            new_edges.append([virtual_node_idx, atom_idx])
            virtual_flag = [0, 0, 0, 0, 1]
            new_edge_attrs.append(virtual_flag)
            new_edge_attrs.append(virtual_flag)

    if fgn_features_list:
        fgn_features = torch.stack(fgn_features_list, dim=0)
        fgn_pos = torch.stack(fgn_pos_list, dim=0)
        x_aug = torch.cat([x_aug, fgn_features], dim=0)
        pos_aug = torch.cat([pos_tensor, fgn_pos], dim=0)
    else:
        pos_aug = pos_tensor

    if new_edges:
        new_edge_tensor = (torch.tensor(new_edges, dtype=torch.long).t().contiguous())
        edge_aug = torch.cat([edge_index_tensor, new_edge_tensor], dim=1)
        new_attr_tensor = torch.tensor(new_edge_attrs, dtype=torch.float)
        edge_attr_aug = torch.cat([edge_attr_tensor, new_attr_tensor], dim=0)
    else:
        edge_aug = edge_index_tensor
        edge_attr_aug = edge_attr_tensor

    return x_aug, pos_aug, edge_aug, edge_attr_aug


# ===========================================================================
#  MAIN PIPELINE
# ===========================================================================
print("Loading RDKit molecules from raw SDF to guarantee perfect 3D alignment...")
sdf_path = '../QM9_dataset/raw/gdb9.sdf'
supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
mol_dict = {}
for mol in supplier:
    if mol is not None:
        name = mol.GetProp('_Name')
        mol_dict[name] = mol

print("Loading compiled PyG QM9 Dataset...")
dataset = QM9(root='QM9_dataset')

successful_graphs = []
failed_count = 0

print(f"Starting FGN Pipeline for {len(dataset)} QM9 molecules.")
start_time = time.time()

for i, data in enumerate(dataset):
    if i % 10000 == 0:
        print(f"Processing Molecule {i} / {len(dataset)}...")

    mol_name = data.name
    mol = mol_dict.get(mol_name)

    if mol is None:
        failed_count += 1
        continue

    # --- THE PATH A SLICE: Instant Tensor Translation ---
    N = data.x.size(0)

    # Rebuild Tox21's 7-column feature tensor perfectly from PyG's 11-column matrix
    atomic_num = data.x[:, 5:6]
    formal_charge = torch.zeros((N, 1), dtype=torch.float)  # QM9 chemicals are neutral ground states
    is_aromatic = data.x[:, 6:7]
    num_hs = data.x[:, 10:11]
    sp = data.x[:, 7:8]
    sp2 = data.x[:, 8:9]
    sp3 = data.x[:, 9:10]

    # Shape: [N, 7] - Perfectly matching the Tox21 loop
    x_tensor = torch.cat([atomic_num, formal_charge, is_aromatic, num_hs, sp, sp2, sp3], dim=1)

    # Pad the QM9 edge attributes with the 5th column for virtual bonds
    E = data.edge_attr.size(0)
    virtual_flag_padding = torch.zeros((E, 1), dtype=torch.float)

    # Shape: [E, 5]
    edge_attr_tensor = torch.cat([data.edge_attr, virtual_flag_padding], dim=1)

    # --- PHASE 6: FGN AUGMENTATION ---
    x_aug, pos_aug, edge_index_aug, edge_attr_aug = build_fgn_augmented_graph(
        mol, x_tensor, data.pos, data.edge_index, edge_attr_tensor
    )

    # --- PHASE 8: Build and Store ---
    # We keep QM9's highly accurate quantum target properties (data.y)
    molecule_graph = Data(x=x_aug, edge_index=edge_index_aug, edge_attr=edge_attr_aug, pos=pos_aug, y=data.y)
    successful_graphs.append(molecule_graph)

print(f"\nConversion Completed")
print("-" * 35)
print(f"Successfully generated {len(successful_graphs)} QM9-FGN PyTorch Graphs.")
print(f"Failed/Missing: {failed_count}")
print("-" * 35)

torch.save(successful_graphs, "../Preprocessed Datasets/qm9_3d_fgn_dataset.pt")
total_time = time.time() - start_time
print(f"Total time: {total_time / 60:.2f} minutes.")
print("Saved successfully as 'qm9_3d_fgn_dataset.pt'.")