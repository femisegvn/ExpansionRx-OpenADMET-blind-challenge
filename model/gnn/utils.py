import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, AllChem

def one_hot_encoder(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x==item) for item in allowable_set]

def get_atom_features(atom):
    return [atom.GetAtomicNum(),
            atom.GetChiralTag(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom.IsInRing()]

def get_encoded_atom_features(atom):
    return [one_hot_encoder(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H', 'Unknown'])
            + one_hot_encoder(atom.GetChiralTag(),[0,1,2,3])
            + one_hot_encoder(atom.GetDegree(),[0,1,2,3,4,5])
            + one_hot_encoder(atom.GetFormalCharge(),[-1,-2,1,2,0])
            + one_hot_encoder(str(atom.GetHybridization()),['SP','SP2','SP3','other'])
            + [atom.GetIsAromatic()]
            + [atom.IsInRing()]]

def get_bond_features(bond):
    return [bond.GetBondType(),
            bond.GetIsAromatic(),
            bond.GetIsConjugated()
            ]

def get_bond_index(bond):
    return [int(bond.GetBeginAtom().GetIdx()),
            int(bond.GetEndAtom().GetIdx())]


def dataset_similarity(train_set, test_set, fingerprint = 'Morgan'):


    if fingerprint == 'Morgan':
        fpgen = rdFingerprintGenerator.GetMorganGenerator() #Morgan fingerprint generator
    elif fingerprint == 'RDKit':
        fpgen = AllChem.GetRDKitFPGenerator() #RDKit fingerprint generator
    
    train_mols = [Chem.MolFromSmiles(d.smiles) for d in train_set]
    test_mols = [Chem.MolFromSmiles(d.smiles) for d in test_set]

    train_fingerprints = [fpgen.GetFingerprint(x) for x in train_mols]
    test_fingerprints = [fpgen.GetFingerprint(x) for x in test_mols]

    max_similarity = max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints) for fingerprint in test_fingerprints)

    return max_similarity