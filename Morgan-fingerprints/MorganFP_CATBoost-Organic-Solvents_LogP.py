import sys
import os
sys.path.append('path_of_iSIM_sampling_codes')

# Download iSIM code from https://github.com/mqcomplab/iSIM/tree/main

import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from matplotlib.markers import MarkerStyle
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
from sklearn import preprocessing
from catboost import CatBoostRegressor
import optuna
from div import *
from sampling import *
from comp import *
from real import calculate_isim_real, calculate_medoid_real, calculate_outlier_real, pairwise_average_real
from utils import binary_fps, minmax_norm, real_fps, pairwise_average, rdkit_pairwise_sim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from comp import calculate_isim, calculate_medoid, calculate_outlier
from real import calculate_isim_real, calculate_medoid_real, calculate_outlier_real, pairwise_average_real
from utils import binary_fps, minmax_norm, real_fps, pairwise_average, rdkit_pairwise_sim
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Load the partition coefficient dataset
input_dir = 'path-of-dataset/'
dataset = pd.read_csv(input_dir + 'Organic-Solvents_LogP.csv')

smiles = df['Canonical_SMILES'].tolist()

# Generation of fingerprints
fps = binary_fps(smiles, fp_type='ECFP4', n_bits=512) # --> fp_type: {'RDKIT', 'ECFP4', 'ECFP6', 'MACCS'}, if ECFP indicate n_bits

# Optional: save the fingerprints in a npy file
fp_output_path = os.path.join(input_dir, 'Organic-Solvents_LogP_MFs.npy')
np.save(fp_output_path, fps)

# === Load fingerprints and define output for t-SNE ===
fingerprints = np.load(fp_output_path)
tSNE_output = os.path.join('tSNE_MFs.csv')

# Define the similarity index for sampling
n_ary = 'JT'

# Define the percentage
percentage = 80

# Get the sampled indexes
strat_samp = stratified_sampling(fingerprints, n_ary, percentage)

# Do tSNE to visualize the sampled indexes
tsne = TSNE(n_components=2)
X =tsne.fit_transform(fingerprints)

# Create a DataFrame with tSNE coordinates
tsne_df = pd.DataFrame(X, columns=['tSNE1', 'tSNE2'])

label_1 = 'Testing'
label_2 = 'Training'
label_3 = 'Validation'

# Initialize the label column as 'testing'
tsne_df['Label'] = label_1

# Label Training samples (80% of data)
tsne_df.loc[strat_samp, 'Label'] = label_2

# Extract Training subset (80% of data)
dataset = pd.read_csv(input_dir + 'Organic-Solvents_LogP.csv')
dataset = pd.concat([dataset, tsne_df], axis=1)
dataset_sampling = dataset[dataset['Label'] == label_2].copy()

# Generate fingerprints for Quota subset
smiles_sampling = dataset_sampling['Canonical_SMILES'].tolist()
fps_quota = binary_fps(smiles_sampling, fp_type='ECFP4', n_bits=512)
np.save(input_dir + 'Organic-Solvents_LogP_MFs_Sampling.npy', fps_quota)
fingerprints_sampling = np.load(input_dir + 'Organic-Solvents_LogP_MFs_Sampling.npy')

# Further split Sampling data into training (70%) and validation (10%)
percentage_new = 87.50
samp_new = stratified_sampling(fingerprints_sampling, n_ary, percentage_new)

# Create training and validation indices within Quota subset
train_indices = dataset_sampling.index[samp_new].tolist()
val_indices = dataset_sampling.index.difference(train_indices).tolist()

# Update tsne_df['Label']
tsne_df.loc[train_indices, 'Label'] = label_2  # Training
tsne_df.loc[val_indices, 'Label'] = label_3   # Validation
tsne_df.loc[dataset[dataset['Label'] == label_1].index, 'Label'] = label_1  # Test

# Combine dataset with updated tsne_df
dataset = pd.concat([dataset.drop(['tSNE1', 'tSNE2', 'Label'], axis=1), tsne_df], axis=1)

# Extract features and labels
mol_smiles = dataset['SMILES']
logP_partition = dataset['Exp_logp']

# Calculate Morgan fingerprints using RDKit
def morgan_fingerprints(smiles, radius=2, n_bits=2048):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) for mol in mols]
    fingerprints_array = [list(fp) for fp in fingerprints]
    return pd.DataFrame(fingerprints_array)

# Function call to calculate Morgan fingerprints
fingerprints_df = morgan_fingerprints(dataset['SMILES'])

print("Morgan fingerprints dataset shape:", fingerprints_df.shape)

# Assign training, validation, and test sets
X_train = fingerprints_df.iloc[train_indices]
y_train = logP_partition[train_indices]
smiles_train = mol_smiles[train_indices]

X_val = fingerprints_df.iloc[val_indices]
y_val = logP_partition[val_indices]
smiles_val = mol_smiles[val_indices]

X_test = fingerprints_df.iloc[dataset[dataset['Label'] == label_1].index]
y_test = logP_partition[dataset[dataset['Label'] == label_1].index]
smiles_test = mol_smiles[dataset[dataset['Label'] == label_1].index]


# Train CatBoost with optimized hyperparameters
cat_model = CatBoostRegressor(verbose = 100, n_estimators = 2459, l2_leaf_reg = 10, learning_rate = 0.06507914893805909, boosting_type = 'Plain', depth = 7)
cat_model.fit(X_train, y_train)

# Predict on train
y_pred_train = cat_model.predict(X_train)

# Print training metrics
print("\nTrain MAE:", mean_absolute_error(y_train, y_pred_train))
print("Train RMSE:", math.sqrt(mean_squared_error(y_train, y_pred_train)))
print("Train R2:", r2_score(y_train, y_pred_train))

# Predict on Validation
y_pred_val = cat_model.predict(X_val)

# Print validation metrics
print("\nValidation MAE:", mean_absolute_error(y_val, y_pred_val))
print("Validation RMSE:", math.sqrt(mean_squared_error(y_val, y_pred_val)))
print("Validation R2:", r2_score(y_val, y_pred_val))

# Predict on test
y_pred_test = cat_model.predict(X_test)

# Print test metrics
print("\nTest MAE:", mean_absolute_error(y_test, y_pred_test))
print("Test RMSE:", math.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Test R2:", r2_score(y_test, y_pred_test))

# Create separate DataFrames for each split
train_df = pd.DataFrame({
    'SMILES': smiles_train,
    'logP_exp': y_train,
    'logP_pred': y_pred_train,
    'Set': 'Training'
})

val_df = pd.DataFrame({
    'SMILES': smiles_val,
    'logP_exp': y_val,
    'logP_pred': y_pred_val,
    'Set': 'Validation'
})

test_df = pd.DataFrame({
    'SMILES': smiles_test,
    'logP_exp': y_test,
    'logP_pred': y_pred_test,
    'Set': 'Testing'
})

# Combine them into one
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Save to CSV
combined_df.to_csv('CATBoost_MorganFP-LogP-Startified.csv', index=False)
