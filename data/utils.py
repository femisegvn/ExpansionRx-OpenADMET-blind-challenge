import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

from data.train_data import conversion_df


def results_to_submission(results_df, csv_path, reverse_log = True):

    reverse_dict = dict([(x[-1], x[0:-1]) for x in conversion_df.values])

    output_df = results_df[["SMILES", "Molecule Name"]].copy()
    for col in results_df.columns[2:]:
        if col == "dataset":
            continue
        orig_name, log_scale, multiplier = reverse_dict[col]
        output_df[orig_name] = results_df[col]
        if log_scale:
            output_df[orig_name] = 10 ** output_df[orig_name] * 1 / multiplier - 1

    output_df.to_csv(csv_path, index=False)
    print(f'Results written to {csv_path}')

    return output_df


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

def remove_correlated_features(df, threshold=0.9):
    """
    Remove one feature from each pair of correlated features above the given threshold.
    Returns a reduced DataFrame and list of dropped columns.
    """
    # Compute correlation matrix (absolute value)
    corr_matrix = df.corr().abs()

    # Upper triangle only (to avoid duplicate pairs)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    #print(f"Dropping {len(to_drop)} highly correlated features (>{threshold})")
    return df.drop(columns=to_drop), to_drop