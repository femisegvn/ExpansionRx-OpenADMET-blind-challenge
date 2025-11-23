import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
import rdkit.Chem as Chem
from gnn.utils import get_encoded_atom_features, get_bond_features, get_bond_index

#Molecular graph object
class Molecular_Graph:
    def __init__(self, smiles, y=None):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.y = torch.Tensor([[float(y)]])
        self.nodes = self.get_nodes()
        self.edge_index, self.edge_features = self.get_edges()
        self.data = Data(x=self.nodes.long(), edge_index = self.edge_index.long(), edge_attr = self.edge_features.long(),
                        smiles = self.smiles, y = self.y)

    def get_nodes(self):

        mol = self.mol

        atoms = [a for a in mol.GetAtoms()]

        #atom_nodes = torch.Tensor([get_atom_features(atom) for atom in atoms])
        atom_nodes = torch.Tensor([get_encoded_atom_features(atom)[0] for atom in atoms])
        

        return atom_nodes

    def get_edges(self):

        mol = self.mol

        bonds = [b for b in mol.GetBonds()]

        bond_edges_index = torch.Tensor([get_bond_index(bond) for bond in bonds]).t().contiguous()
        bond_edges_features = torch.Tensor([get_bond_features(bond) for bond in bonds])
        
        edges = bond_edges_index, bond_edges_features
        if len(bonds) != 0:
            edges = to_undirected(bond_edges_index, bond_edges_features)

        return edges