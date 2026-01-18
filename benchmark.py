#!/usr/bin/env python3
"""
Benchmark script for ADMET prediction models.
Computes features, runs 5x5 CV for each assay, featurizer, and model.
Saves results to CSV.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.preprocessing import MaxAbsScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

from data.train_data import log_train_df, log_col_names
from data.utils import compute_rdkit_descriptors

# Define featurization functions

def get_morgan_features(smiles_list):
    features = [compute_rdkit_descriptors(s, fingerprint=True, descriptor_filter='none') for s in tqdm(smiles_list, desc="Morgan")]
    df = pd.DataFrame(features)
    fp_cols = [f'FP_{i}' for i in range(1024)]
    return df[fp_cols].values

def get_rdkit_descriptors_no_fragments(smiles_list):
    features = [compute_rdkit_descriptors(s, fingerprint=False, descriptor_filter='no_frag') for s in tqdm(smiles_list, desc="RDKit No Frag")]
    df = pd.DataFrame(features)
    desc_cols = [c for c in df.columns if c != 'SMILES']
    return df[desc_cols].values

def get_rdkit_fragments(smiles_list):
    features = [compute_rdkit_descriptors(s, fingerprint=False, descriptor_filter='frag') for s in tqdm(smiles_list, desc="RDKit Frag")]
    df = pd.DataFrame(features)
    frag_cols = [c for c in df.columns if c != 'SMILES']
    return df[frag_cols].values

def get_rdkit_all(smiles_list):
    features = [compute_rdkit_descriptors(s, fingerprint=False, descriptor_filter='all') for s in tqdm(smiles_list, desc="RDKit All")]
    df = pd.DataFrame(features)
    all_cols = [c for c in df.columns if c != 'SMILES']
    return df[all_cols].values

def get_hybrid_features(smiles_list):
    features = [compute_rdkit_descriptors(s, fingerprint=True, descriptor_filter='all') for s in tqdm(smiles_list, desc="Hybrid")]
    df = pd.DataFrame(features)
    all_cols = [c for c in df.columns if c != 'SMILES']
    return df[all_cols].values

featurizers = {
    'Morgan': get_morgan_features,
    'RDKit_No_Frag': get_rdkit_descriptors_no_fragments,
    'RDKit_Frag': get_rdkit_fragments,
    'RDKit_All': get_rdkit_all,
    'Hybrid': get_hybrid_features
}

# Define models
models = {
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'SVR': SVR(kernel='rbf'),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

def visualize_results(df):
    os.makedirs('plots', exist_ok=True)
    assays = df['Assay'].unique()
    for assay in assays:
        assay_df = df[df['Assay'] == assay]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # R² plot
        sns.barplot(data=assay_df, x='Featurizer', y='Mean_R2', hue='Model', ax=ax1, errorbar='sd', edgecolor='black', linewidth=1, err_kws={"linewidth": 0.8}, capsize=0.2)
        ax1.set_title(f'{assay} - R²')
        ax1.set_ylim(-1, 1)
        
        # MAE plot
        sns.barplot(data=assay_df, x='Featurizer', y='Mean_MAE', hue='Model', ax=ax2, errorbar='sd', edgecolor='black', linewidth=1, err_kws={"linewidth": 0.8}, capsize=0.2)
        ax2.set_title(f'{assay} - MAE')
        
        plt.tight_layout()
        plt.savefig(f'plots/{assay}.svg')
        plt.close(fig)

def main():
    subset_size = None  # For quick testing; set to None for full data
    print("Loading data...")
    if subset_size:
        log_train_df_subset = log_train_df.iloc[:subset_size].copy()
        smiles = log_train_df_subset['SMILES'].tolist()
        print(f"Using subset of {len(smiles)} molecules")
    else:
        log_train_df_subset = log_train_df
        smiles = log_train_df_subset['SMILES'].tolist()
        print(f"Total molecules: {len(smiles)}")

    # Cache features
    cache_file = 'features_cache.pkl'
    if os.path.exists(cache_file):
        print("Loading cached features...")
        with open(cache_file, 'rb') as f:
            X_dict = pickle.load(f)
    else:
        print("Computing features...")
        X_dict = {}
        for name, func in featurizers.items():
            print(f"Computing {name}...")
            X = func(smiles)
            X_dict[name] = X
        print("Saving features cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(X_dict, f)

    print("Running benchmarking...")
    results = []
    # Print sample counts for all assays
    for assay in log_col_names:
        y_full = log_train_df_subset[assay].values
        mask = ~np.isnan(y_full)
        y = y_full[mask]
        print(f'Assay {assay}: {len(y)} samples')

    for assay in tqdm(log_col_names, desc="Assays"):
        y_full = log_train_df_subset[assay].values
        mask = ~np.isnan(y_full)
        y = y_full[mask]
        print(f"Assay {assay}: {len(y)} samples")
        if len(y) < 2:
            print(f"Skipping {assay} due to insufficient samples")
            continue
        if len(y) < 5:
            print(f"Skipping {assay} due to insufficient samples for 5-fold CV")
            continue
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        for feat_name, X_full in tqdm(X_dict.items(), desc=f"Featurizers for {assay}", leave=False):
            X = X_full[mask]
            for model_name, model in tqdm(models.items(), desc=f"Models for {feat_name}", leave=False):
                # Create pipeline with scaling for X and y
                pipeline = Pipeline([
                    ('scaler_X', MaxAbsScaler()),
                    ('regressor', TransformedTargetRegressor(transformer=MaxAbsScaler(), regressor=model))
                ])
                scores = cross_validate(pipeline, X, y, cv=cv, scoring=['r2', 'neg_mean_absolute_error'], n_jobs=-1)
                mean_r2 = np.nanmean(scores['test_r2'])
                std_r2 = np.nanstd(scores['test_r2'])
                mean_mae = -np.nanmean(scores['test_neg_mean_absolute_error'])
                std_mae = np.nanstd(scores['test_neg_mean_absolute_error'])
                results.append({
                    'Assay': assay,
                    'Featurizer': feat_name,
                    'Model': model_name,
                    'Mean_R2': mean_r2,
                    'Std_R2': std_r2,
                    'Mean_MAE': mean_mae,
                    'Std_MAE': std_mae
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv('benchmark_results.csv', index=False)
    print("Results saved to benchmark_results.csv")
    print(results_df.head())

    # Visualize results
    visualize_results(results_df)

if __name__ == '__main__':
    main()