

##############################
### File From AGILE github ###
##############################

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType as BT


RDLogger.DisableLog("rdApp.*")

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def smiles_to_data(smiles):
    """Convert a SMILES string into a torch_geometric graph Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    type_idx = []
    chirality_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append(
            [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
            ]
        )
        edge_feat.append(
            [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
            ]
        )

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MultiMolTSVDataset(Dataset):
    """Dataset that loads 4 molecules + scalar features + label from a TSV file."""

    def __init__(
        self,
        tsv_path,
        molecule_cols,
        scalar_cols,
        label_col
    ):
        """Load TSV rows into memory for multi-molecule training."""
        usecols = ["lnp_id"] + list(molecule_cols) + list(scalar_cols) + [label_col]
        df = pd.read_csv(tsv_path, sep="\t", usecols=usecols)
        df = df.dropna(subset=usecols)

        self.lnp_ids = df["lnp_id"].tolist()
        self.smiles = df[molecule_cols].values.tolist()
        self.scalars = df[scalar_cols].astype(np.float32).to_numpy()
        self.labels = df[label_col].astype(np.float32).to_numpy()
        self.num_molecules = len(molecule_cols)

    def __len__(self):
        """Return the number of samples in the TSV."""
        return len(self.labels)

    def __getitem__(self, index):
        """Return (molecule graphs, scalar features, label) for one sample."""
        mol_smiles = self.smiles[index]
        mol_graphs = [smiles_to_data(s) for s in mol_smiles]
        scalar = torch.tensor(self.scalars[index], dtype=torch.float)
        label = torch.tensor([self.labels[index]], dtype=torch.float)
        lnp_id = self.lnp_ids[index]
        return mol_graphs, scalar, label, lnp_id


def multimol_collate_fn(samples):
    """Collate multi-molecule samples into a single batched graph set."""
    mol_list = []
    scalar_list = []
    label_list = []
    lnp_ids = []
    for mols, scalars, label, lnp_id in samples:
        mol_list.extend(mols)
        scalar_list.append(scalars)
        label_list.append(label)
        lnp_ids.append(lnp_id)

    mol_batch = Batch.from_data_list(mol_list)
    scalar_batch = torch.stack(scalar_list, dim=0)
    label_batch = torch.stack(label_list, dim=0)
    return mol_batch, scalar_batch, label_batch, lnp_ids
