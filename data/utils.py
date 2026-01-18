import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors, Lipinski, Crippen, DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
import pandas as pd

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


def compute_rdkit_descriptors(smiles, fingerprint=False, descriptor_filter='all'):
    '''
    takes just one smiles string as input
    '''
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        print(f"Failed to parse SMILES: '{smiles}'")
        return None

    # Get all descriptor names
    all_descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]

    if descriptor_filter == 'none':
        descriptor_names = []
    elif descriptor_filter == 'no_frag':
        descriptor_names = [d for d in all_descriptor_names if not d.startswith('fr_')]
    elif descriptor_filter == 'frag':
        descriptor_names = [d for d in all_descriptor_names if d.startswith('fr_')]
    else:  # 'all'
        descriptor_names = all_descriptor_names

    # Combine into dict
    features = {}
    features['SMILES'] = smiles

    if descriptor_names:
        # Create calculator
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

        # Calculate all descriptors
        values = calculator.CalcDescriptors(mol)

        for desc, val in zip(descriptor_names, values):
            features[desc] = val
    
    if fingerprint:
        # --- Morgan fingerprint (radius=2, nBits=1024) ---
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((1024,))
        ConvertToNumpyArray(fp, arr)
        for i in range(1024):
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

def characterise(df, name = None):
    '''
    Definitely should have a better name for this but I just wanted a
    quick function for calculating like 6 descriptors for exploration
    '''

    smiles = df.SMILES

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    features = {'MWt': rdMolDescriptors.CalcExactMolWt,
                'TPSA': rdMolDescriptors.CalcTPSA,
                'HDonors': Lipinski.NumHDonors,
                'HAcceptors': Lipinski.NumHAcceptors,
                'CLogP': Crippen.MolLogP}
    
    feat_dict = {}
        
    for feat_name, feat in features.items():
        feature = [feat(mol) for mol in mols]
        
        feat_dict[feat_name] = feature

    feat_df = pd.DataFrame(feat_dict)

    df = df.join(feat_df)

    if name is not None:
        df['Dataset'] = name

    return df

def get_train_test_similarity(df : pd.DataFrame ,
                              smiles_col : str = "SMILES",
                              set_col : str = "Dataset") -> pd.DataFrame: 
    """
    Stolen entirely from Fischer et al.
    """
    radius = 2
    fpSize = 2048
    includeChirality=True
    
    df = df.copy()
    fg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize,includeChirality=includeChirality)
    train_mol_list = [Chem.MolFromSmiles(smi) for smi in df.loc[df[set_col] == 'Train',:][smiles_col]]
    test_mol_list = [Chem.MolFromSmiles(smi) for smi in df.loc[df[set_col] != 'Train',:][smiles_col]]

    train_fp = [fg.GetFingerprint(x) for x in train_mol_list]
    test_fp = [fg.GetFingerprint(x) for x in test_mol_list]

    df["train_test_sim"] = np.nan
    train_test_sim = [np.max([DataStructs.TanimotoSimilarity(train, test) for test in test_fp]) for train in train_fp]
    sim_train = [np.max([DataStructs.TanimotoSimilarity(train, test) for test in test_fp]) for train in train_fp]
    sim_test = [np.max([DataStructs.TanimotoSimilarity(train, test) for train in train_fp]) for test in test_fp]
    df.loc[df[set_col] == 'Train', "train_test_sim"] = sim_train
    df.loc[df[set_col] == 'Test', "train_test_sim"] = sim_test
    return df

