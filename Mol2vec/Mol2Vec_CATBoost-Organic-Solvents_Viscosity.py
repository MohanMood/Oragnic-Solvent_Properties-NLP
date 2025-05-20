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
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import seaborn as sns
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from gensim.models import word2vec
from sklearn import preprocessing
from catboost import CatBoostRegressor
import optuna
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from div import *
from sampling import *
from comp import *
from real import calculate_isim_real, calculate_medoid_real, calculate_outlier_real, pairwise_average_real
from utils import binary_fps, minmax_norm, real_fps, pairwise_average, rdkit_pairwise_sim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Load the viscosity dataset
input_dir = 'path-of-dataset/'
df = pd.read_csv(input_dir + 'Oragnic-Solvents_Viscosity_Canonical_SMILES.csv', encoding='unicode_escape')

smiles = df['Canonical_SMILES'].tolist()

# Generation of fingerprints
fps = binary_fps(smiles, fp_type='ECFP4', n_bits=512)

# Optional: save the fingerprints in a npy file
fp_output_path = os.path.join(input_dir, 'Oragnic-Solvents_Viscosity_MFs.npy')
np.save(fp_output_path, fps)

# === Load fingerprints and define output for t-SNE ===
fingerprints = np.load(fp_output_path)
tSNE_output = os.path.join('tSNE_MFs.csv')

# Define the similarity index for sampling
n_ary = 'JT'

# Define the percentage for initial sampling (90%)
percentage = 90

# Get the sampled indexes (90% of data)
quota_samp = quota_sampling(fingerprints, n_ary, percentage)

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

# Label Quota samples (90% of data)
tsne_df.loc[quota_samp, 'Label'] = label_2

# Extract Quota subset (90% of data)
dataset = pd.read_csv(input_dir + 'Oragnic-Solvents_Viscosity_Canonical_SMILES.csv', encoding='unicode_escape')
dataset = pd.concat([dataset, tsne_df], axis=1)
dataset_sampling = dataset[dataset['Label'] == label_2].copy()

# Generate fingerprints for Quota subset
smiles_sampling = dataset_sampling['Canonical_SMILES'].tolist()
fps_quota = binary_fps(smiles_sampling, fp_type='ECFP4', n_bits=512)
np.save(input_dir + 'Oragnic-Solvents_Viscosity_MFs_Sampling.npy', fps_quota)
fingerprints_sampling = np.load(input_dir + 'Oragnic-Solvents_Viscosity_MFs_Sampling.npy')

# Further split data into 85% training and 5% validation
percentage_new = 94.45 
samp_new = quota_sampling(fingerprints_sampling, n_ary, percentage_new)

# Create training and validation indices within Quota subset
train_indices = dataset_sampling.index[samp_new].tolist()
val_indices = dataset_sampling.index.difference(train_indices).tolist()

# Update tsne_df['Label'] to reflect training and validation splits
tsne_df.loc[train_indices, 'Label'] = label_2  # Training
tsne_df.loc[val_indices, 'Label'] = label_3   # Validation
tsne_df.loc[dataset[dataset['Label'] == label_1].index, 'Label'] = label_1  # Test

# Save the updated tSNE DataFrame
tsne_df.to_csv(input_dir + tSNE_output, index=False)
tsne_df = pd.read_csv(input_dir + tSNE_output, encoding='unicode_escape')

# Combine dataset with updated tsne_df
dataset = pd.concat([dataset.drop(['tSNE1', 'tSNE2', 'Label'], axis=1), tsne_df], axis=1)

# Extract features and labels
mol_smiles = dataset['Canonical_SMILES']
log_visc = dataset['log_visc']
temperature = dataset['Temperature (K)']

# Load the pre-trained Mol2vec model
model_path = '{path_of_Mol2vec}/model_300dim.pkl'
model = word2vec.Word2Vec.load(model_path)

# Convert SMILES to Mol2vec features
os_smiles = [Chem.MolFromSmiles(x) for x in mol_smiles]
os_sentences = [mol2alt_sentence(x, 1) for x in os_smiles]

class DfVec:
    def __init__(self, vector):
        self.vector = vector

def sentences2vec(sentences, model, unseen='UNK'):
    vectors = []
    for sentence in sentences:
        vec = []
        for word in sentence:
            if word in model.wv.key_to_index:
                vec.append(model.wv[word])
            else:
                if unseen in model.wv.key_to_index:
                    vec.append(model.wv[unseen])
                else:
                    vec.append(np.zeros(model.vector_size))
        vectors.append(np.sum(vec, axis=0))
    return vectors

os_mol2vec = [DfVec(x) for x in sentences2vec(os_sentences, model, unseen='UNK')]
np_os_mol2vec = np.array([x.vector for x in os_mol2vec])
print("mol2vec shape:", np_os_mol2vec.shape)

# Combine Mol2vec features and temperature
features = np.hstack((np_os_mol2vec, temperature.values.reshape(-1, 1)))
print("features shape:", features.shape)

# Assign training, validation, and test sets
X_train = features[train_indices]
y_train = log_visc[train_indices]
smiles_train = mol_smiles[train_indices]
temp_train = temperature[train_indices]

X_val = features[val_indices]
y_val = log_visc[val_indices]
smiles_val = mol_smiles[val_indices]
temp_val = temperature[val_indices]

X_test = features[dataset[dataset['Label'] == label_1].index]
y_test = log_visc[dataset[dataset['Label'] == label_1].index]
smiles_test = mol_smiles[dataset[dataset['Label'] == label_1].index]
temp_test = temperature[dataset[dataset['Label'] == label_1].index]

# Train CatBoost with optimized hyperparameters
cat_model = CatBoostRegressor(verbose = 100, n_estimators = 2000, l2_leaf_reg = 5, learning_rate = 0.03, boosting_type = 'Plain', depth = None, 
                              random_strength = 5, rsm = 0.8)
cat_model.fit(X_train, y_train)

# Predict on train
y_pred_train = cat_model.predict(X_train)

# Print metrics
print("\nTrain MAE:", mean_absolute_error(y_train, y_pred_train))
print("Train RMSE:", math.sqrt(mean_squared_error(y_train, y_pred_train)))
print("Train R2:", r2_score(y_train, y_pred_train))

# Predict on Validation
y_pred_val = cat_model.predict(X_val)

# Print metrics
print("\nValidation MAE:", mean_absolute_error(y_val, y_pred_val))
print("Validation RMSE:", math.sqrt(mean_squared_error(y_val, y_pred_val)))
print("Validation R2:", r2_score(y_val, y_pred_val))

# Predict on test
y_pred_test = cat_model.predict(X_test)

# Print metrics
print("\nTest MAE:", mean_absolute_error(y_test, y_pred_test))
print("Test RMSE:", math.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Test R2:", r2_score(y_test, y_pred_test))

# Create separate DataFrames for each split
train_df = pd.DataFrame({
    'SMILES': smiles_train,
    'Temp (K)': temp_train,
    'viscosity_exp': y_train,
    'viscosity_pred': y_pred_train,
    'Set': 'Training'
})

val_df = pd.DataFrame({
    'SMILES': smiles_val,
    'Temp (K)': temp_val,
    'viscosity_exp': y_val,
    'viscosity_pred': y_pred_val,
    'Set': 'Validation'
})

test_df = pd.DataFrame({
    'SMILES': smiles_test,
    'Temp (K)': temp_test,
    'viscosity_exp': y_test,
    'viscosity_pred': y_pred_test,
    'Set': 'Testing'
})

# Combine them into one
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Save to CSV
combined_df.to_csv('CATBoost_Mol2vec-Viscosity-Quota.csv', index=False)
