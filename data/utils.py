import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.DataStructs import ConvertToNumpyArray



def compute_rdkit_descriptors(smiles, fingerprint=False):
    '''
    takes just one smiles string as input
    '''
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        print(f"Failed to parse SMILES: '{smiles}'")
        return None

    # Get all descriptor names
    descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]

    # Create calculator
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    # Calculate all descriptors
    values = calculator.CalcDescriptors(mol)

    # Combine into dict
    features = {}
    features['SMILES'] = smiles

    for desc, val in zip(descriptor_names, values):
        features[desc] = val
    
    if fingerprint:
        # --- Morgan fingerprint (radius=2, nBits=1024) ---
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((1,))
        ConvertToNumpyArray(fp, arr)
        for i in range(len(arr)):
            features[f'FP_{i}'] = arr[i]
    

    return features