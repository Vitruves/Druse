#!/usr/bin/env python3
"""
Train a GNN-based pKa predictor for Druse.

Architecture: Message-passing GNN operating on molecular graphs.
- Atom features (25-dim): element one-hot, degree, hybridization, aromaticity,
                           formal charge, in-ring, num-Hs, electronegativity
- Bond features (10-dim): bond type one-hot, conjugated, in-ring, stereo
- 4 rounds of message passing with edge-conditioned convolution
- Per-atom pKa readout (is_ionizable head + pKa regression head)

Export: Binary .weights file for Metal inference (same DRAF-style format).

Usage:
    python train_pka.py train --csv /path/to/pka.csv [--epochs 200] [--device cuda]
    python train_pka.py eval  --csv /path/to/pka.csv --checkpoint best_pka.pt
    python train_pka.py export --checkpoint best_pka.pt --out ../Models/druse-models/pKaGNN.weights
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ImportError:
    print("ERROR: RDKit is required. Install with: pip install rdkit")
    sys.exit(1)

try:
    from torch_geometric.nn import NNConv, Set2Set
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("WARNING: torch-geometric not found. Using built-in message passing.")

# =============================================================================
# Constants — MUST match Metal shader
# =============================================================================

NUM_ATOM_FEATURES = 25
NUM_BOND_FEATURES = 10
HIDDEN_DIM = 128
NUM_MSG_LAYERS = 4
READOUT_DIM = 64

# Element vocabulary (covers >99% of drug-like molecules)
ELEMENT_MAP = {
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,
    'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'Se': 9,
    'B': 10, 'Si': 11, 'I': 12,
}
NUM_ELEMENTS = 14  # 13 known + 1 other

# Pauling electronegativity (scaled to [0,1])
ELECTRONEG = {
    'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'Se': 2.55,
    'B': 2.04, 'Si': 1.90, 'I': 2.66,
}
MAX_EN = 3.98


# =============================================================================
# Featurization
# =============================================================================

def atom_features(atom):
    """25-dimensional atom feature vector."""
    feat = np.zeros(NUM_ATOM_FEATURES, dtype=np.float32)
    sym = atom.GetSymbol()

    # Element one-hot (0-13)
    feat[ELEMENT_MAP.get(sym, NUM_ELEMENTS - 1)] = 1.0

    # Degree (14) — normalized
    feat[14] = min(atom.GetDegree(), 6) / 6.0

    # Hybridization one-hot (15-17): sp, sp2, sp3
    hyb = atom.GetHybridization()
    if hyb == Chem.rdchem.HybridizationType.SP:
        feat[15] = 1.0
    elif hyb == Chem.rdchem.HybridizationType.SP2:
        feat[16] = 1.0
    elif hyb == Chem.rdchem.HybridizationType.SP3:
        feat[17] = 1.0

    # Aromaticity (18)
    feat[18] = float(atom.GetIsAromatic())

    # Formal charge (19) — clipped
    feat[19] = max(-2, min(2, atom.GetFormalCharge())) / 2.0

    # In ring (20)
    feat[20] = float(atom.IsInRing())

    # Number of Hs (21) — normalized
    feat[21] = min(atom.GetTotalNumHs(), 4) / 4.0

    # Electronegativity (22) — normalized
    feat[22] = ELECTRONEG.get(sym, 2.5) / MAX_EN

    # Number of radical electrons (23)
    feat[23] = min(atom.GetNumRadicalElectrons(), 2) / 2.0

    # Is heteroatom (24)
    feat[24] = float(atom.GetAtomicNum() not in (1, 6))

    return feat


def bond_features(bond):
    """10-dimensional bond feature vector."""
    feat = np.zeros(NUM_BOND_FEATURES, dtype=np.float32)

    # Bond type one-hot (0-3)
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        feat[0] = 1.0
    elif bt == Chem.rdchem.BondType.DOUBLE:
        feat[1] = 1.0
    elif bt == Chem.rdchem.BondType.TRIPLE:
        feat[2] = 1.0
    elif bt == Chem.rdchem.BondType.AROMATIC:
        feat[3] = 1.0

    # Conjugated (4)
    feat[4] = float(bond.GetIsConjugated())

    # In ring (5)
    feat[5] = float(bond.IsInRing())

    # Ring size features (6-8): 5-ring, 6-ring, other ring
    if bond.IsInRing():
        ri = bond.GetOwningMol().GetRingInfo()
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for ring in ri.AtomRings():
            if a1 in ring and a2 in ring:
                rsize = len(ring)
                if rsize == 5:
                    feat[6] = 1.0
                elif rsize == 6:
                    feat[7] = 1.0
                else:
                    feat[8] = 1.0
                break

    # Stereo (9)
    feat[9] = float(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)

    return feat


# =============================================================================
# Ionizable site detection (pure RDKit, mirrors C++ SMARTS logic)
# =============================================================================

# Simplified ionizable group patterns for labeling training data.
# We only need to identify WHICH atom is ionizable and whether it's an acid;
# the model will learn the actual pKa value.
IONIZABLE_PATTERNS = [
    # Acids (deprotonate — lose H, pKa is for HA → A- + H+)
    # Oxygen acids
    ("[OX2H1]S(=O)(=O)",    True,  0),   # Sulfonic acid
    ("[OX2H1]P(=O)",        True,  0),   # Phosphoric
    ("[CX3](=O)[OX2H1]",   True,  2),   # Carboxylic acid (OH is atom 2)
    ("[OX2H1]c",            True,  0),   # Phenol
    ("[OX2H1]C=C",          True,  0),   # Enol
    ("[OX2H1]B",            True,  0),   # Boronic acid
    ("[OX2H1]S(=O)",        True,  0),   # Sulfinic acid
    ("[SX2H1]",             True,  0),   # Thiol
    ("[SeX2H1]",            True,  0),   # Selenol
    # NH acids
    ("[NH1](S(=O)(=O))",    True,  0),   # Sulfonamide
    ("[NH1](C(=O))C(=O)",   True,  0),   # Imide
    ("[OX2H1]/N=C",         True,  0),   # Oxime
    ("[nH1]",               True,  0),   # Aromatic NH
    ("[NH1]C(=S)",          True,  0),   # Thioamide
    ("[OX2H1]NC(=O)",       True,  0),   # Hydroxamic acid
    # Carbon acids
    ("[CH2](C(=O))C(=O)",   True,  0),   # 1,3-dicarbonyl
    ("[CH1](C(=O))C(=O)",   True,  0),   # Substituted 1,3-dicarbonyl
    ("[CH2](C#N)C#N",       True,  0),   # Malononitrile
    ("[CH2](C#N)C(=O)",     True,  0),   # Cyanoacetate
    ("[CH2,CH3,CH1]([CX4,c,H])[NX3+](=O)[O-]", True, 0),  # Nitroalkane
    ("[OX2H1][CX4]",        True,  0),   # Alcohol

    # Bases (protonate — gain H, pKa is for BH+ → B + H+)
    ("[NX3]C(=[NX2])[NX3]", False, 0),   # Guanidine
    ("[NX3]C(=[NX2])[!N]",  False, 0),   # Amidine
    ("[nH0;X2]1cc[nH1]c1",  False, 0),   # Imidazole
    ("[nH0;X2;R1]",         False, 0),   # Aromatic N (pyridine etc.)
    ("[NX3H2;!$(NC=O);!$(NS=O);!$(Nc)]", False, 0),  # Primary amine
    ("[NX3H1;!$(NC=O);!$(NS=O);!$(Nc);!R]", False, 0),  # Secondary amine
    ("[NX3H0;!$(NC=O);!$(NS=O);!$(Nc);!R]([CX4])([CX4])[CX4]", False, 0),  # Tertiary amine
    ("[NX3H1;R;!$(NC=O);!$(NS(=O)=O);!a]", False, 0),  # Ring NH saturated
    ("[NX3H0;R;!$(NC=O);!$(NS(=O)=O);!a]", False, 0),  # Ring NR saturated
    ("[NX3H2]c",            False, 0),   # Aniline
    ("[NX3H1;!$(NC=O)](C)c", False, 0),  # N-alkyl aniline
    ("[NX3H2][NX3]",        False, 0),   # Hydrazine
    ("[NX3H2]O",            False, 0),   # Hydroxylamine
]


def detect_ionizable_sites(mol):
    """Detect ionizable sites in a molecule.

    Returns list of (atom_idx, is_acid) for each unique ionizable atom.
    """
    seen = set()
    sites = []
    for smarts, is_acid, target_idx in IONIZABLE_PATTERNS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        for match in mol.GetSubstructMatches(pat):
            if not match:
                continue
            idx = match[min(target_idx, len(match) - 1)]
            if idx not in seen:
                seen.add(idx)
                sites.append((idx, is_acid))
    return sites


# =============================================================================
# Dataset
# =============================================================================

def smiles_to_graph(smiles):
    """Convert SMILES to molecular graph (PyG Data or dict)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Canonicalize for deterministic atom ordering
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)

    # Detect ionizable sites on the heavy-atom molecule
    sites = detect_ionizable_sites(mol)
    site_map = {idx: is_acid for idx, is_acid in sites}

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    # Atom features
    x = np.zeros((num_atoms, NUM_ATOM_FEATURES), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = atom_features(atom)

    # Bond features + edge index
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if len(edge_index) == 0:
        # Single atom molecule — add self-loop
        edge_index = [[0, 0]]
        edge_attr = [np.zeros(NUM_BOND_FEATURES, dtype=np.float32)]

    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_attr = np.array(edge_attr, dtype=np.float32)

    # Per-atom labels: is_ionizable (binary) + pKa (regression)
    is_ionizable = np.zeros(num_atoms, dtype=np.float32)
    is_acid = np.zeros(num_atoms, dtype=np.float32)

    for idx, acid in site_map.items():
        if idx < num_atoms:
            is_ionizable[idx] = 1.0
            is_acid[idx] = 1.0 if acid else 0.0

    return {
        'x': torch.from_numpy(x),
        'edge_index': torch.from_numpy(edge_index),
        'edge_attr': torch.from_numpy(edge_attr),
        'is_ionizable': torch.from_numpy(is_ionizable),
        'is_acid': torch.from_numpy(is_acid),
        'num_atoms': num_atoms,
    }


def _estimate_pka_prior(mol, atom_idx, is_acid):
    """Rough pKa estimate from atom environment for bipartite matching.

    This doesn't need to be accurate — just good enough to distinguish
    e.g. a carboxylic acid (pKa ~4) from a phenol (pKa ~10) from an amine (pKa ~10)
    so the greedy matcher assigns the right experimental pKa to the right atom.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    sym = atom.GetSymbol()
    arom = atom.GetIsAromatic()

    if is_acid:
        if sym == 'O':
            # Check neighbors
            for nbr in atom.GetNeighbors():
                nsym = nbr.GetSymbol()
                if nsym == 'S' and nbr.GetDegree() >= 3:
                    return -2.0   # sulfonic/sulfinic
                if nsym == 'P':
                    return 2.0    # phosphoric
                if nsym == 'C':
                    # Check if C has =O neighbor (carboxylic acid)
                    for nn in nbr.GetNeighbors():
                        if nn.GetIdx() != atom_idx and nn.GetSymbol() == 'O' and \
                           mol.GetBondBetweenAtoms(nbr.GetIdx(), nn.GetIdx()).GetBondTypeAsDouble() == 2.0:
                            return 4.0  # carboxylic acid
                    if nbr.GetIsAromatic():
                        return 10.0  # phenol
                    return 16.0  # alcohol
                if nsym == 'B':
                    return 8.8    # boronic acid
                if nsym == 'N':
                    return 9.0    # hydroxamic/oxime
            return 10.0
        elif sym == 'S':
            if arom or any(n.GetIsAromatic() for n in atom.GetNeighbors()):
                return 6.5   # thiophenol
            return 10.0      # aliphatic thiol
        elif sym == 'N':
            if arom:
                return 14.0  # pyrrole/indole NH
            # Check for sulfonamide, imide
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'S':
                    return 10.0  # sulfonamide
                if nbr.GetSymbol() == 'C':
                    for nn in nbr.GetNeighbors():
                        if nn.GetSymbol() == 'O' and \
                           mol.GetBondBetweenAtoms(nbr.GetIdx(), nn.GetIdx()).GetBondTypeAsDouble() == 2.0:
                            return 9.0  # imide/amide NH
            return 10.0
        elif sym == 'C':
            return 10.0  # carbon acid
        elif sym == 'Se':
            return 5.2
        return 10.0
    else:
        # Base — conjugate acid pKa
        if sym == 'N':
            if arom:
                # Aromatic N
                if any(n.GetSymbol() == 'N' for n in atom.GetNeighbors()):
                    return 3.0  # diazine
                return 5.0      # pyridine
            # Check for guanidine/amidine
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'C':
                    c_n_count = sum(1 for nn in nbr.GetNeighbors() if nn.GetSymbol() == 'N')
                    if c_n_count >= 2:
                        return 12.0  # guanidine/amidine
            # Check attached to aromatic
            if any(n.GetIsAromatic() for n in atom.GetNeighbors()):
                return 4.5  # aniline
            if atom.IsInRing():
                return 10.0  # saturated ring N
            nh = atom.GetTotalNumHs()
            if nh >= 2:
                return 10.5  # primary amine
            elif nh == 1:
                return 10.5  # secondary amine
            else:
                return 9.8   # tertiary amine
        return 5.0


class PKaDataset(Dataset):
    """Dataset of (molecule graph, ionizable_atom_idx, experimental_pKa) triples.

    Uses SMARTS-based pKa priors to do bipartite matching between experimental
    pKa values and detected ionizable atoms, giving explicit atom-level supervision
    even for multi-site molecules.
    """

    def __init__(self, df, smiles_cache=None):
        """
        Args:
            df: DataFrame with columns ['SMILES', 'PKA'].
                Multiple rows per SMILES allowed (multiple ionizable sites).
        """
        self.samples = []
        self.graphs = {}
        self.smiles_cache = smiles_cache or {}

        # Group by SMILES and collect all pKa values
        grouped = df.groupby('SMILES')['PKA'].apply(list).to_dict()

        skipped = 0
        for smi, pka_list in grouped.items():
            if smi in self.smiles_cache:
                graph = self.smiles_cache[smi]
            else:
                graph = smiles_to_graph(smi)
                if graph is None:
                    skipped += 1
                    continue
                self.smiles_cache[smi] = graph

            self.graphs[smi] = graph

            # Find ionizable atoms with SMARTS-based pKa priors
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped += 1
                continue

            ion_atoms = []  # list of (atom_idx, prior_pka, is_acid)
            for i in range(graph['num_atoms']):
                if graph['is_ionizable'][i] > 0.5:
                    prior = _estimate_pka_prior(mol, i, graph['is_acid'][i].item() > 0.5)
                    ion_atoms.append((i, prior, graph['is_acid'][i].item() > 0.5))

            if not ion_atoms:
                skipped += 1
                continue

            # Deduplicate experimental pKa: average values within 0.5 of each other
            sorted_pkas = sorted(pka_list)
            dedup_pkas = []
            cluster = [sorted_pkas[0]]
            for p in sorted_pkas[1:]:
                if p - cluster[-1] < 0.5:
                    cluster.append(p)
                else:
                    dedup_pkas.append(float(np.mean(cluster)))
                    cluster = [p]
            dedup_pkas.append(float(np.mean(cluster)))

            # Bipartite matching: pair each exp pKa with the ionizable atom whose
            # SMARTS prior pKa is closest, using greedy 1:1 assignment.
            # Build all (site_idx, exp_idx, |prior - exp|) candidates.
            candidates = []
            for si, (atom_idx, prior, is_acid) in enumerate(ion_atoms):
                for ei, exp_pka in enumerate(dedup_pkas):
                    candidates.append((si, ei, abs(prior - exp_pka)))
            candidates.sort(key=lambda c: c[2])

            used_sites = set()
            used_exp = set()
            for si, ei, _ in candidates:
                if si in used_sites or ei in used_exp:
                    continue
                used_sites.add(si)
                used_exp.add(ei)
                atom_idx = ion_atoms[si][0]
                self.samples.append((smi, atom_idx, dedup_pkas[ei]))

            # Any unmatched experimental pKa values — assign to the remaining
            # closest ionizable atom (allows multiple pKa per atom in rare cases)
            for ei, exp_pka in enumerate(dedup_pkas):
                if ei in used_exp:
                    continue
                # Find closest by prior
                best_si = min(range(len(ion_atoms)),
                              key=lambda si: abs(ion_atoms[si][1] - exp_pka))
                atom_idx = ion_atoms[best_si][0]
                self.samples.append((smi, atom_idx, exp_pka))

        if skipped > 0:
            print(f"  Skipped {skipped} molecules (parse failure or no ionizable sites)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        smi, atom_idx, pka = self.samples[idx]
        graph = self.graphs[smi]

        if HAS_PYG:
            data = Data(
                x=graph['x'],
                edge_index=graph['edge_index'],
                edge_attr=graph['edge_attr'],
                is_ionizable=graph['is_ionizable'],
                is_acid=graph['is_acid'],
            )
        else:
            data = graph

        return data, atom_idx, pka


def collate_fn(batch):
    """Custom collate: batch graphs + per-sample atom_idx and pKa.

    Pre-computes global atom indices so training loop is fully vectorized.
    """
    graphs, atom_idxs, pkas = zip(*batch)

    if HAS_PYG:
        batched = Batch.from_data_list(graphs)
    else:
        batched = _manual_batch(graphs)

    # Compute per-molecule start offsets from batch vector, then convert
    # local atom_idxs to global indices in the batched graph.
    local_idxs = torch.tensor(atom_idxs, dtype=torch.long)
    pkas = torch.tensor(pkas, dtype=torch.float32)

    # batch.batch is sorted (0,0,..,1,1,..,2,..) so mol boundaries are where
    # the value changes. Compute start offsets via bincount cumsum.
    batch_vec = batched.batch
    counts = torch.bincount(batch_vec)
    mol_starts = torch.zeros_like(counts)
    torch.cumsum(counts[:-1], dim=0, out=mol_starts[1:])

    global_idxs = mol_starts + local_idxs

    return batched, global_idxs, pkas


def _manual_batch(graphs):
    """Batch graphs without PyG (fallback)."""
    xs, edge_indices, edge_attrs = [], [], []
    is_ions, is_acids = [], []
    batch_vec = []
    offset = 0
    for i, g in enumerate(graphs):
        n = g['x'].shape[0]
        xs.append(g['x'])
        edge_indices.append(g['edge_index'] + offset)
        edge_attrs.append(g['edge_attr'])
        is_ions.append(g['is_ionizable'])
        is_acids.append(g['is_acid'])
        batch_vec.extend([i] * n)
        offset += n

    class FakeBatch:
        pass

    b = FakeBatch()
    b.x = torch.cat(xs, dim=0)
    b.edge_index = torch.cat(edge_indices, dim=1)
    b.edge_attr = torch.cat(edge_attrs, dim=0)
    b.is_ionizable = torch.cat(is_ions, dim=0)
    b.is_acid = torch.cat(is_acids, dim=0)
    b.batch = torch.tensor(batch_vec, dtype=torch.long)
    return b


# =============================================================================
# Model
# =============================================================================

class EdgeMLP(nn.Module):
    """Edge-conditioned MLP: transforms bond features into message weight matrix."""
    def __init__(self, edge_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, edge_attr):
        return self.net(edge_attr).view(-1, self.hidden_dim, self.hidden_dim)


class MessagePassingLayer(nn.Module):
    """Edge-conditioned message passing with gated residual."""
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]

        # Edge-conditioned message
        edge_weight = self.edge_mlp(edge_attr)         # [E, H]
        msg = self.msg_mlp(x[src]) * edge_weight       # [E, H]

        # Aggregate (mean)
        agg = torch.zeros_like(x)
        count = torch.zeros(x.shape[0], 1, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.shape[0], 1, device=x.device))
        count = count.clamp(min=1)
        agg = agg / count

        # Gated residual
        gate = self.gate(torch.cat([x, agg], dim=-1))
        x = x + gate * agg
        x = self.norm(x)
        return x


class PKaGNN(nn.Module):
    """Graph Neural Network for per-atom pKa prediction.

    Architecture:
        1. Atom encoder: Linear(25→128) + GELU + Linear(128→128)
        2. 4× message passing layers (edge-conditioned, gated residual, LayerNorm)
        3. Ionizable head: Linear(128→64) + GELU + Linear(64→1) → sigmoid
        4. pKa head: Linear(128→64) + GELU + Linear(64→1)
        5. Acid head: Linear(128→64) + GELU + Linear(64→1) → sigmoid

    Total: ~150K parameters.
    """

    def __init__(self):
        super().__init__()

        # Atom encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(NUM_ATOM_FEATURES, HIDDEN_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

        # Message passing layers
        self.msg_layers = nn.ModuleList([
            MessagePassingLayer(HIDDEN_DIM, NUM_BOND_FEATURES)
            for _ in range(NUM_MSG_LAYERS)
        ])

        # Per-atom ionizable classification head
        self.ion_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, READOUT_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(READOUT_DIM, 1),
        )

        # Per-atom pKa regression head
        self.pka_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, READOUT_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(READOUT_DIM, 1),
        )

        # Per-atom acid/base classification head
        self.acid_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, READOUT_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(READOUT_DIM, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: [N, 25] atom features
            edge_index: [2, E] edge indices
            edge_attr: [E, 10] bond features

        Returns:
            ion_logits: [N, 1] ionizable probability (pre-sigmoid)
            pka_pred: [N, 1] predicted pKa
            acid_logits: [N, 1] acid probability (pre-sigmoid)
        """
        h = self.atom_encoder(x)

        for layer in self.msg_layers:
            h = layer(h, edge_index, edge_attr)

        ion_logits = self.ion_head(h)
        pka_pred = self.pka_head(h)
        acid_logits = self.acid_head(h)

        return ion_logits, pka_pred, acid_logits


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_pka_loss = 0
    total_ion_loss = 0
    total_acid_loss = 0
    n_batches = 0

    for batch, atom_idxs, target_pkas in loader:
        if HAS_PYG:
            batch = batch.to(device)
        else:
            for k in ['x', 'edge_index', 'edge_attr', 'is_ionizable', 'is_acid', 'batch']:
                setattr(batch, k, getattr(batch, k).to(device))

        atom_idxs = atom_idxs.to(device)
        target_pkas = target_pkas.to(device)

        ion_logits, pka_pred, acid_logits = model(
            batch.x, batch.edge_index, batch.edge_attr
        )

        # 1. Ionizable classification loss (all atoms)
        ion_loss = F.binary_cross_entropy_with_logits(
            ion_logits.squeeze(-1),
            batch.is_ionizable,
            pos_weight=torch.tensor(5.0, device=device),  # ionizable atoms are rare
        )

        # 2. Acid/base classification loss (ionizable atoms only)
        ion_mask = batch.is_ionizable > 0.5
        if ion_mask.any():
            acid_loss = F.binary_cross_entropy_with_logits(
                acid_logits.squeeze(-1)[ion_mask],
                batch.is_acid[ion_mask],
            )
        else:
            acid_loss = torch.tensor(0.0, device=device)

        # 3. pKa regression loss — fully vectorized atom-level supervision
        # atom_idxs are pre-computed global indices from collate_fn
        pred_pkas = pka_pred.squeeze(-1)[atom_idxs]   # [batch_size]
        pka_loss = F.huber_loss(pred_pkas, target_pkas, delta=2.0)

        # Combined loss
        loss = pka_loss + 0.3 * ion_loss + 0.1 * acid_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        total_pka_loss += pka_loss.item()
        total_ion_loss += ion_loss.item()
        total_acid_loss += acid_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'pka_loss': total_pka_loss / max(n_batches, 1),
        'ion_loss': total_ion_loss / max(n_batches, 1),
        'acid_loss': total_acid_loss / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred = []
    all_true = []
    all_ion_pred = []
    all_ion_true = []

    for batch, atom_idxs, target_pkas in loader:
        if HAS_PYG:
            batch = batch.to(device)
        else:
            for k in ['x', 'edge_index', 'edge_attr', 'is_ionizable', 'is_acid', 'batch']:
                setattr(batch, k, getattr(batch, k).to(device))

        atom_idxs = atom_idxs.to(device)
        target_pkas = target_pkas.to(device)

        ion_logits, pka_pred, acid_logits = model(
            batch.x, batch.edge_index, batch.edge_attr
        )

        # Ionizable classification
        ion_prob = torch.sigmoid(ion_logits.squeeze(-1))
        all_ion_pred.append(ion_prob.cpu())
        all_ion_true.append(batch.is_ionizable.cpu())

        # pKa regression — vectorized atom-level readout
        pred_pkas = pka_pred.squeeze(-1)[atom_idxs]
        all_pred.append(pred_pkas.cpu())
        all_true.append(target_pkas.cpu())

    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    errors = pred - true

    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    median_ae = np.median(np.abs(errors))

    # R² and Pearson
    ss_res = (errors ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    cov = np.mean((true - true.mean()) * (pred - pred.mean()))
    pearson = cov / (true.std() * pred.std()) if true.std() > 0 and pred.std() > 0 else 0

    within_05 = (np.abs(errors) <= 0.5).mean()
    within_10 = (np.abs(errors) <= 1.0).mean()
    within_20 = (np.abs(errors) <= 2.0).mean()

    # Ionizable classification AUC
    ion_pred = torch.cat(all_ion_pred).numpy()
    ion_true = torch.cat(all_ion_true).numpy()
    try:
        from sklearn.metrics import roc_auc_score
        ion_auc = roc_auc_score(ion_true, ion_pred)
    except Exception:
        ion_auc = 0.0

    return {
        'mae': mae,
        'rmse': rmse,
        'median_ae': median_ae,
        'r2': r2,
        'pearson': pearson,
        'within_05': within_05,
        'within_10': within_10,
        'within_20': within_20,
        'ion_auc': ion_auc,
        'n_samples': len(true),
    }


def run_train(args):
    print("=== pKa GNN Training ===\n")

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load data
    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    # Handle BOM
    df.columns = [c.strip().strip('\ufeff') for c in df.columns]
    # Coerce pKa to numeric, drop non-numeric rows (e.g. "<1", "~15", ranges)
    df['PKA'] = pd.to_numeric(df['PKA'], errors='coerce')
    n_before = len(df)
    df = df.dropna(subset=['PKA'])
    if n_before - len(df) > 0:
        print(f"  Dropped {n_before - len(df)} rows with non-numeric pKa")
    # Canonicalize SMILES to merge duplicates and ensure reproducible splits
    df['SMILES'] = df['SMILES'].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
    )
    print(f"  {len(df)} entries, {df['SMILES'].nunique()} unique SMILES")

    # Split by SMILES (not by row) to avoid data leakage
    unique_smiles = df['SMILES'].unique()
    train_smi, val_smi = train_test_split(
        unique_smiles, test_size=0.15, random_state=42
    )
    train_df = df[df['SMILES'].isin(train_smi)]
    val_df = df[df['SMILES'].isin(val_smi)]

    print(f"  Train: {len(train_df)} entries ({len(train_smi)} molecules)")
    print(f"  Val:   {len(val_df)} entries ({len(val_smi)} molecules)")

    # Build datasets with shared SMILES cache
    print("\nBuilding molecular graphs ...")
    cache = {}
    train_ds = PKaDataset(train_df, smiles_cache=cache)
    val_ds = PKaDataset(val_df, smiles_cache=cache)
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0,
                            pin_memory=(device.type == 'cuda'))

    # Model
    model = PKaGNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: PKaGNN ({n_params:,} parameters, {n_params * 4 / 1024:.1f} KB)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_mae = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']

        improved = ""
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            patience_counter = 0
            improved = " *"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
            }, args.out_checkpoint)
        else:
            patience_counter += 1

        if epoch % 5 == 1 or epoch == args.epochs or improved:
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={train_metrics['loss']:.4f}  "
                  f"pKa={train_metrics['pka_loss']:.4f}  "
                  f"val MAE={val_metrics['mae']:.3f}  "
                  f"RMSE={val_metrics['rmse']:.3f}  "
                  f"R²={val_metrics['r2']:.3f}  "
                  f"r={val_metrics['pearson']:.3f}  "
                  f"≤1.0={val_metrics['within_10']:.1%}  "
                  f"ionAUC={val_metrics['ion_auc']:.3f}  "
                  f"lr={lr:.2e}{improved}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\nBest val MAE: {best_mae:.3f}")
    print(f"Checkpoint: {args.out_checkpoint}")

    # Final evaluation
    ckpt = torch.load(args.out_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    final = evaluate(model, val_loader, device)
    print(f"\n=== Final Validation ===")
    print(f"  MAE:       {final['mae']:.3f}")
    print(f"  RMSE:      {final['rmse']:.3f}")
    print(f"  Median AE: {final['median_ae']:.3f}")
    print(f"  R²:        {final['r2']:.3f}")
    print(f"  Pearson r: {final['pearson']:.3f}")
    print(f"  ≤0.5:      {final['within_05']:.1%}")
    print(f"  ≤1.0:      {final['within_10']:.1%}")
    print(f"  ≤2.0:      {final['within_20']:.1%}")
    print(f"  Ion AUC:   {final['ion_auc']:.3f}")
    print(f"  N samples: {final['n_samples']}")


def run_eval(args):
    print("=== pKa GNN Evaluation ===\n")

    device = torch.device(args.device if args.device != 'auto' else 'cpu')

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().strip('\ufeff') for c in df.columns]
    df['PKA'] = pd.to_numeric(df['PKA'], errors='coerce')
    df = df.dropna(subset=['PKA'])
    df['SMILES'] = df['SMILES'].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
    )

    cache = {}
    ds = PKaDataset(df, smiles_cache=cache)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    model = PKaGNN().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    metrics = evaluate(model, loader, device)
    print(f"MAE:       {metrics['mae']:.3f}")
    print(f"RMSE:      {metrics['rmse']:.3f}")
    print(f"Median AE: {metrics['median_ae']:.3f}")
    print(f"R²:        {metrics['r2']:.3f}")
    print(f"Pearson r: {metrics['pearson']:.3f}")
    print(f"≤0.5:      {metrics['within_05']:.1%}")
    print(f"≤1.0:      {metrics['within_10']:.1%}")
    print(f"≤2.0:      {metrics['within_20']:.1%}")
    print(f"Ion AUC:   {metrics['ion_auc']:.3f}")
    print(f"N samples: {metrics['n_samples']}")


# =============================================================================
# Export to Metal binary format
# =============================================================================

EXPORT_WEIGHT_ORDER = [
    # Atom encoder (0-3)
    "atom_encoder.0.weight",     # [128, 25]
    "atom_encoder.0.bias",       # [128]
    "atom_encoder.2.weight",     # [128, 128]
    "atom_encoder.2.bias",       # [128]
    # Message passing layer 0 (4-12)
    "msg_layers.0.edge_mlp.0.weight",    # [128, 10]
    "msg_layers.0.edge_mlp.0.bias",      # [128]
    "msg_layers.0.edge_mlp.2.weight",    # [128, 128]
    "msg_layers.0.edge_mlp.2.bias",      # [128]
    "msg_layers.0.msg_mlp.0.weight",     # [128, 128]
    "msg_layers.0.msg_mlp.0.bias",       # [128]
    "msg_layers.0.msg_mlp.2.weight",     # [128, 128]
    "msg_layers.0.msg_mlp.2.bias",       # [128]
    "msg_layers.0.gate.0.weight",        # [128, 256]
    "msg_layers.0.gate.0.bias",          # [128]
    "msg_layers.0.norm.weight",          # [128]
    "msg_layers.0.norm.bias",            # [128]
    # Message passing layer 1 (16-27)
    "msg_layers.1.edge_mlp.0.weight",
    "msg_layers.1.edge_mlp.0.bias",
    "msg_layers.1.edge_mlp.2.weight",
    "msg_layers.1.edge_mlp.2.bias",
    "msg_layers.1.msg_mlp.0.weight",
    "msg_layers.1.msg_mlp.0.bias",
    "msg_layers.1.msg_mlp.2.weight",
    "msg_layers.1.msg_mlp.2.bias",
    "msg_layers.1.gate.0.weight",
    "msg_layers.1.gate.0.bias",
    "msg_layers.1.norm.weight",
    "msg_layers.1.norm.bias",
    # Message passing layer 2 (28-39)
    "msg_layers.2.edge_mlp.0.weight",
    "msg_layers.2.edge_mlp.0.bias",
    "msg_layers.2.edge_mlp.2.weight",
    "msg_layers.2.edge_mlp.2.bias",
    "msg_layers.2.msg_mlp.0.weight",
    "msg_layers.2.msg_mlp.0.bias",
    "msg_layers.2.msg_mlp.2.weight",
    "msg_layers.2.msg_mlp.2.bias",
    "msg_layers.2.gate.0.weight",
    "msg_layers.2.gate.0.bias",
    "msg_layers.2.norm.weight",
    "msg_layers.2.norm.bias",
    # Message passing layer 3 (40-51)
    "msg_layers.3.edge_mlp.0.weight",
    "msg_layers.3.edge_mlp.0.bias",
    "msg_layers.3.edge_mlp.2.weight",
    "msg_layers.3.edge_mlp.2.bias",
    "msg_layers.3.msg_mlp.0.weight",
    "msg_layers.3.msg_mlp.0.bias",
    "msg_layers.3.msg_mlp.2.weight",
    "msg_layers.3.msg_mlp.2.bias",
    "msg_layers.3.gate.0.weight",
    "msg_layers.3.gate.0.bias",
    "msg_layers.3.norm.weight",
    "msg_layers.3.norm.bias",
    # Ionizable head (52-55)
    "ion_head.0.weight",     # [64, 128]
    "ion_head.0.bias",       # [64]
    "ion_head.2.weight",     # [1, 64]
    "ion_head.2.bias",       # [1]
    # pKa head (56-59)
    "pka_head.0.weight",     # [64, 128]
    "pka_head.0.bias",       # [64]
    "pka_head.2.weight",     # [1, 64]
    "pka_head.2.bias",       # [1]
    # Acid head (60-63)
    "acid_head.0.weight",    # [64, 128]
    "acid_head.0.bias",      # [64]
    "acid_head.2.weight",    # [1, 64]
    "acid_head.2.bias",      # [1]
]


def run_export(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    print(f"=== pKa GNN Weight Export ===")
    print(f"  Checkpoint: {args.checkpoint}")
    if "val_metrics" in ckpt:
        m = ckpt["val_metrics"]
        print(f"  Val MAE: {m.get('mae', '?'):.3f}")
        print(f"  Val R²:  {m.get('r2', '?'):.3f}")

    # Verify all weights present
    missing = [k for k in EXPORT_WEIGHT_ORDER if k not in state]
    if missing:
        print(f"\nERROR: {len(missing)} weight tensors missing:")
        for k in missing:
            print(f"  - {k}")
        sys.exit(1)

    tensors = []
    for key in EXPORT_WEIGHT_ORDER:
        t = state[key]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        tensors.append(t.float().numpy().flatten())

    num_tensors = len(tensors)
    offsets = []
    curr = 0
    for t in tensors:
        offsets.append((curr, len(t)))
        curr += len(t)
    total_floats = curr

    out_path = Path(args.out)
    with open(out_path, "wb") as f:
        # Magic: "PKA1"
        f.write(b"PKA1")
        f.write(struct.pack("<I", 1))           # version
        f.write(struct.pack("<I", num_tensors))
        f.write(struct.pack("<I", total_floats))
        # Offset table
        for off, cnt in offsets:
            f.write(struct.pack("<II", off, cnt))
        # Weight data
        for t in tensors:
            f.write(t.astype(np.float32).tobytes())

    size_kb = total_floats * 4 / 1024
    print(f"\n  Exported {num_tensors} tensors, {total_floats:,} floats ({size_kb:.1f} KB)")
    print(f"  PKA1 v1 format → {out_path}")
    print(f"\n  Copy to Models/druse-models/pKaGNN.weights")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="pKa GNN — per-atom pKa prediction")
    sub = parser.add_subparsers(dest="command")

    # train
    tp = sub.add_parser("train", help="Train pKa GNN model")
    tp.add_argument("--csv", type=str, required=True, help="pKa CSV file (SMILES,PKA)")
    tp.add_argument("--epochs", type=int, default=200)
    tp.add_argument("--batch_size", type=int, default=256)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--patience", type=int, default=30,
                    help="Early stopping patience")
    tp.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    tp.add_argument("--out_checkpoint", type=str, default="best_pka.pt",
                    help="Output checkpoint path")

    # eval
    ep = sub.add_parser("eval", help="Evaluate trained model")
    ep.add_argument("--csv", type=str, required=True)
    ep.add_argument("--checkpoint", type=str, required=True)
    ep.add_argument("--device", type=str, default="auto")

    # export
    xp = sub.add_parser("export", help="Export weights for Metal inference")
    xp.add_argument("--checkpoint", type=str, required=True)
    xp.add_argument("--out", type=str, default="../Models/druse-models/pKaGNN.weights")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "export":
        run_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
