# MoleColyte: The Pharmaceutical Acolyte 🧪

MoleColyte is an end-to-end in-silico toxicity prediction pipeline built on 3D Equivariant Graph Neural Networks (EGNN), designed to predict molecular toxicity across 12 biological assays from the NIH Tox21 dataset. It is intended as a pre-screening tool for pharmaceutical drug discovery — given a list of candidate molecules, MoleColyte filters out high-toxicity candidates early, reducing the cost and time of downstream R&D and minimising the need for in-vivo testing.

## The Problem

Early-stage drug discovery is expensive and slow. A significant portion of drug candidates fail late in development due to toxicity — a problem that could have been caught earlier with better computational screening. Traditional toxicity testing is lab-intensive, time-consuming, and ethically fraught when it involves animal testing. MoleColyte addresses this by providing a fast, physics-informed computational pre-screen that narrows 1000 candidates down to a handful worth investigating further.

---

## How It Works

Each molecule is given as a SMILES string. MoleColyte converts it into a physics-optimised 3D molecular graph, augments it with virtual Functional Group Nodes (FGNs), and passes it through a pretrained EGNN to produce toxicity probabilities across all 12 Tox21 assays.

```
SMILES string
     ↓
3D graph construction (ETKDGv3 + MMFF optimisation)
     ↓
FGN augmentation (12 SMARTS patterns)
     ↓
EGNN pretrained on QM9 → fine-tuned on Tox21
     ↓
12 toxicity probabilities (one per assay)
```

---

## Architecture

### EGNN - Equivariant Graph Neural Network

The core model is built on E(n) Equivariant Graph Neural Networks (Satorras et al., 2021). Unlike standard GNNs that treat 3D coordinates as static edge features, EGNN dynamically recomputes pairwise distances at every message passing step and updates atomic positions as part of the forward pass. This makes the model natively equivariant to rotations, reflections, and translations — the same molecule in any orientation produces the same prediction.

Each EGNN layer performs three operations in sequence:

- **Edge MLP** — computes a message between 2 atoms, which is amalgamation of  hidden states of the atoms, the distance between the atoms, and bond type features
- **Node MLP** — assimilates the incoming messages from all the immediate neighbors of the atom and uses it to update the hidden state of the atom
- **Coordinate MLP** — produces the new position of an atom based on messages coming in from all the immediate neighboring atoms and the relative distances from them.

Three EGNN layers are stacked, giving each atom a 3-hop receptive field. SiLU activations are used to introduce non-linearity without dying neurons. The final molecule representation is obtained via global mean pooling over all node hidden states.

### GINE - Graph Isomorphism Network

This model served as the initial approach, hence setting the baseline performance to be beaten by the EGNN model. This model doesn’t inherently acknowledge or support the use of the pos tensor which contain the X, Y and Z coordinates of the atoms of the molecule, so the
workaround was to calculate the Cartesian distance between the atoms using their X, Y and Z coordinates from the pos tensor and append it to the edge_attr tensor. This way the model has implicit knowledge of the 3D position of the atoms.


### Functional Group Nodes (FGNs)

Before any neural network processing, each molecular graph is augmented with virtual FGN nodes — a custom augmentation developed for this project. Twelve SMARTS patterns detect functional groups including benzene rings, carbonyls, hydroxyls, amines, halogens, and others. For each match:

- A virtual node is appended with features equal to the mean of its member atoms
- Its 3D position is set to the geometric centroid of its member atoms
- Bidirectional virtual edges connect the FGN to every member atom

FGNs allow information to travel across chemically meaningful substructures in fewer message passing steps and provide the model with a hierarchical representation of molecular topology.

### Transfer Learning Strategy

The model follows a two-stage transfer learning approach:

1. **QM9 Pretraining** — the EGNN is trained to predict internal energy at 0K (U0) across 133,885 small organic molecules with DFT-computed quantum mechanical properties. This forces the model to develop deep understanding of 3D molecular geometry and atomic interactions.
2. **Tox21 Fine-tuning** — the pretrained EGNN layers are retained and only the prediction head is replaced and fine-tuned for 12-label binary toxicity classification.

### Dynamic Penalty Weights

Tox21 is severely class-imbalanced — some assays have up to 20:1 non-toxic to toxic ratios. A dynamic penalty weight is computed per assay:

$$w_k = \frac{N_{\text{non-toxic}, k}}{N_{\text{toxic}, k}}$$

This weight is applied to the positive class in BCEWithLogitsLoss, penalising misses on rare toxic molecules proportionally to how rare they are in each assay. Without this, the model would default to predicting non-toxic for everything and still achieve misleadingly high accuracy.

---

## Datasets

### Tox21

7,831 molecules with 12 binary toxicity labels across nuclear receptor and stress response assays.

| Assay | Type | Biological Target |
|---|---|---|
| NR-AR | Nuclear Receptor | Androgen Receptor |
| NR-AR-LBD | Nuclear Receptor | Androgen Receptor Ligand Binding Domain |
| NR-AhR | Nuclear Receptor | Aryl Hydrocarbon Receptor |
| NR-Aromatase | Nuclear Receptor | Aromatase Enzyme |
| NR-ER | Nuclear Receptor | Estrogen Receptor |
| NR-ER-LBD | Nuclear Receptor | Estrogen Receptor Ligand Binding Domain |
| NR-PPAR-gamma | Nuclear Receptor | Peroxisome Proliferator Activated Receptor |
| SR-ARE | Stress Response | Antioxidant Response Element |
| SR-ATAD5 | Stress Response | DNA Damage Indicator |
| SR-HSE | Stress Response | Heat Shock Element |
| SR-MMP | Stress Response | Mitochondrial Membrane Potential |
| SR-p53 | Stress Response | DNA Damage / Cancer Suppression |

### QM9

133,885 small organic molecules with up to 9 heavy atoms, each with 19 quantum mechanical properties computed using Density Functional Theory (DFT). Pretraining uses U0 — internal energy at 0K — as the regression target.

---

## Results

Performance is measured using AUC-ROC per assay on the held-out test set. Missing labels are excluded from scoring via NaN masking.

| Model | Framework | Mean AUC-ROC |
|---|---|--------------|
| GINE (3D augmented) | PyTorch Geometric | 0.67         |
| **EGNN (equivariant)** | **DGL** | **0.76**     |
### GINE ROC-AUC:
<img width="837" height="594" alt="GINE-AUC" src="https://github.com/user-attachments/assets/ecd11419-4352-45a6-930d-803a2b7ba6ff" />                    
### EGNN ROC-AUC:
<img width="494" height="389" alt="EGNN_AUC" src="https://github.com/user-attachments/assets/42d29205-aeaa-443c-8106-d1cec38b30d9" />


### A Note on Evaluation

Both models were evaluated using random splitting. Scaffold-based splitting — the industry standard for molecular property prediction — would provide a more rigorous evaluation by testing on structurally dissimilar molecules. This is identified as future work.

The 9 percentage point improvement from GINE to EGNN demonstrates that rotational equivariance and dynamic 3D geometry are meaningful signals for toxicity prediction. For context, the NIH Tox21 Challenge (2014) saw winning ensembled models score ~0.84 AUC using 5-12 layer DNNs with extensive feature engineering. Our single-model EGNN at 0.76 is competitive with published single-model baselines on this dataset, which typically range from 0.60 to 0.75.

---

## Pipeline

### File Structure

```
├── dataset/
│   └── tox21.csv
│
├── preprocessing/
│   ├── SMILES_to_3D.py       ← Tox21 3D graph construction + FGN augmentation
│   └── QM9_refactor.py       ← QM9 DGL graph construction + FGN augmentation
│
├── reference-papers/
│   └── (reference papers)
│
└── training/
    ├── egnn_layer.py          ← EGNN layer implementation (edge, node, coord MLPs)
    ├── data_loader.py         ← DGL dataset wrapper + collate function
    ├── train_QM9.py           ← QM9 pretraining loop
    ├── train_Tox21.py         ← Tox21 fine-tuning loop + dynamic penalty weights
    └── eval_molecolyte.py     ← AUC-ROC evaluation on test set
```

### Graph Format

Every molecule is stored as a DGL graph with the following tensors:

| Tensor | Shape | Description |
|---|---|---|
| `g.ndata['x']` | (N+K, 8) | Node features: 7 chemistry features + is_fgn flag |
| `g.ndata['pos']` | (N+K, 3) | 3D coordinates in Angstroms |
| `g.edata['edge_attr']` | (E+E_new, 5) | Bond type one-hot + virtual bond flag |
| `y` | (12,) for Tox21, (1, 19) for QM9 | Labels |

Where N = real atoms, K = FGN nodes, E = directed chemical bonds, E_new = FGN bipartite edges.

### Node Features

| Index | Feature |
|---|---|
| 0 | Atomic number |
| 1 | Formal charge |
| 2 | Is aromatic |
| 3 | Number of hydrogens |
| 4 | SP hybridization |
| 5 | SP2 hybridization |
| 6 | SP3 hybridization |
| 7 | is_fgn flag (0 = real atom, 1 = virtual FGN) |

### Edge Features

| Index | Feature |
|---|---|
| 0 | Single bond |
| 1 | Double bond |
| 2 | Triple bond |
| 3 | Aromatic bond |
| 4 | Virtual bond flag (0 = real, 1 = FGN edge) |

---

## Training of EGNN

### QM9 Pretraining

- **Task** — Regression (internal energy U0)
- **Loss** — Mean Squared Error (MSE)
- **Optimiser** — Adam (lr=1e-3)
- **Epochs** — 10
- **Batch size** — 32
- **Split** — 80 / 10 / 10
- **Best loss** — 0.4298 (normalised units — PyG applies Z-score standardisation to QM9 targets)

### Tox21 Fine-tuning

- **Task** — Multi-label binary classification (12 assays)
- **Loss** — BCEWithLogitsLoss with dynamic per-assay positive weights
- **Optimiser** — Adam (lr=1e-3)
- **Epochs** — 20
- **Batch size** — 32
- **Split** — 80 / 10 / 10
- **Class imbalance** — Dynamic positive weights computed per assay (negatives / positives)
- **Missing labels** — NaN masking applied during loss computation and evaluation

### Model Configuration

| Hyperparameter | Value |
|---|---|
| Input node features | 8 |
| Hidden dimension | 128 |
| Edge attribute dimension | 5 |
| EGNN layers | 3 |
| Activation (EGNN layers) | SiLU |
| Activation (prediction head) | ReLU |
| Output (QM9) | 1 |
| Output (Tox21) | 12 |

---

## Installation

```bash
# Python 3.10 recommended — DGL does not support Python 3.13 on Windows
py -3.10 -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install torch==2.2.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torchdata==0.7.1
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install pandas rdkit torch-geometric scikit-learn numpy==1.26.4
```

**Known issue:** DGL's graphbolt submodule requires `torchdata.datapipes` which was removed in torchdata ≥ 0.8.0. Pin to 0.7.1 as shown above, or set `os.environ["DGL_SKIP_GRAPHBOLT"] = "1"` before importing DGL.

**Known issue:** PyG's QM9 dataset class contains a bug where `mol.GetNumAtoms()` is called on a potentially None molecule object. A one-line patch is required before running `QM9_refactor.py`, so use this specific fork of the PyG library:

https://github.com/prajwalnayaka/pytorch_geometric

---

## Usage

Run the pipeline in order:

```bash
# Step 1 — Preprocess Tox21
python Data Preprocessing/SMILES_to_3D.py

# Step 2 — Preprocess QM9
python Data_Preprocessing/QM9_refactor.py

# Step 3 — Create the data loader objects
python EGNN_Training/data_loader.py

# Step 4 — Pretrain on QM9
python EGNN_Training/train_QM9.py

# Step 5 — Fine-tune on Tox21
python trainiEGNN_Trainingng/train_Tox21.py

# Step 6 — Evaluate
python eval_EGNN.ipynb
```

---

## Dependencies

| Package                  | Purpose |
|--------------------------|---|
| PyTorch                  | Deep learning framework |
| DGL                      | Graph neural network library |
| RDKit                    | SMILES parsing, 3D embedding (ETKDGv3), MMFF optimisation |
| PyTorch Geometric        | QM9 dataset download and preprocessing |
| scikit-learn             | AUC-ROC evaluation |
| NumPy / Pandas           | Data handling |

---

## What I'd Improve Next

- **Scaffold splitting** — replace random splitting with scaffold-based splitting for a more honest evaluation on structurally unseen molecules
- **Learning rate scheduler** — cosine annealing or step decay during both pretraining and fine-tuning
- **Attention pooling** — replace mean pooling with a learned attention mechanism for the graph readout
- **GUI** — a web interface for pharmaceutical teams to submit SMILES lists and receive toxicity reports
- **Threshold calibration** — tune per-assay decision thresholds based on the desired sensitivity/specificity tradeoff for pharmaceutical use

---

## Contributors

- **Prajwal Nayaka T** ([GitHub](https://github.com/prajwalnayakat))
  - Designed and built the GINE pipeline on PyTorch Geometric (data preprocessing, model architecture, training loop, evaluation)
  - Engineered the Functional Group Node (FGN) augmentation system
  - Designed the dynamic per-assay penalty weight system for class imbalance
  - Built the 3D molecular graph construction pipeline (ETKDGv3 + MMFF) for both Tox21 and QM9
  - Resolved PyG QM9 dataset bug (opened PR on PyG repository)
  - Trained and evaluated both GINE and EGNN models

- **Pragya MV** ([GitHub](https://github.com/pragyamv))
  - Designed and implemented the EGNN layer (edge MLP, node MLP, coordinate MLP)
  - Engineered the FGN creation feature for the 3D molecules
  - Designed the training scripts for QM9 pretraining and Tox21 fine-tuning
  - Designed the data loader and DGL graph collation pipeline
  - Literature review and reference paper curation

- **Suhana Bakshi** ([GitHub](https://github.com/suhanabakshi))
  - Dataset acquisition and initial exploratory analysis

---

## References

- Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) Equivariant Graph Neural Networks. *ICML 2021*
- Tox21 Data Challenge — https://tripod.nih.gov/tox21
- QM9 Dataset — Ramakrishnan et al. (2014), *Scientific Data*
- PyTorch Geometric QM9 — https://pytorch-geometric.readthedocs.io
