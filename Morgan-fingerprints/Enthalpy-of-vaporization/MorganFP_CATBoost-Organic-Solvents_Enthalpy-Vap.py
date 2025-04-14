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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from catboost import CatBoostRegressor

# Load dataset
input_dir = 'path-of-the-dataset/'
dataset = pd.read_csv(input_dir + 'Organic-solvents_Enthalpy-of-vaporization.csv', encoding='unicode_escape')

# Sorting dataset with unique SMILES
dataset = dataset.drop_duplicates(subset=['SMILES']).sort_values(by='SMILES')
print("Initial dataset shape:", dataset.shape)

# Select relevant columns
dataset = dataset[['SMILES', 'enthalpy_vap']]

X = dataset['SMILES']
y = dataset['enthalpy_vap']

# Calculate Morgan fingerprints using RDKit
def morgan_fingerprints(smiles, radius=2, n_bits=2048):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) for mol in mols]
    fingerprints_array = [list(fp) for fp in fingerprints]
    return pd.DataFrame(fingerprints_array)

# Function call to calculate Morgan fingerprints
fingerprints_df = morgan_fingerprints(dataset['SMILES'])
print("Morgan fingerprints dataset shape:", fingerprints_df.shape)

# Seed for reproducibility
seed = 536

# Simple train-test split
X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
    np_os_mol2vec, log_visc, mol_smiles, test_size=0.2, random_state=seed)

# Train CatBoost
cat_model = CatBoostRegressor(verbose=100)
cat_model.fit(X_train, y_train)

# Predict on train
y_pred_train = cat_model.predict(X_train)
train_df = pd.DataFrame({
    'SMILES': smiles_train.reset_index(drop=True),
    'log_visc_actual': y_train.reset_index(drop=True),
    'log_visc_pred': y_pred_train
})
train_df.to_csv(f'CATBoost_MorganFP-Training_Enthalpy_vap-{seed}seed.csv', index=False)

# Predict on test
y_pred_test = cat_model.predict(X_test)
test_df = pd.DataFrame({
    'SMILES': smiles_test.reset_index(drop=True),
    'log_visc_actual': y_test.reset_index(drop=True),
    'log_visc_pred': y_pred_test
})
test_df.to_csv(f'CATBoost_MorganFP-Testing_Enthalpy_vap-{seed}seed.csv', index=False)

# Print metrics
print("Test MAE:", mean_absolute_error(y_test, y_pred_test))
print("Test RMSE:", math.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Test R2:", r2_score(y_test, y_pred_test))

# Scatterplot
plt.figure(figsize=(4, 5))
ln = np.arange(min(min(y_train), min(y_test)), max(max(y_train), max(y_test)), 0.2)
plt.plot(ln, ln, 'gray', linestyle='--')

plt.scatter(y_train, y_pred_train, color='red', label='Training', alpha=0.8,
            marker=MarkerStyle("D", fillstyle="right"), s=60)
plt.scatter(y_test, y_pred_test, color='blue', label='Testing', alpha=0.8,
            marker=MarkerStyle("D", fillstyle="right"), s=60)

plt.xlabel('Experimental logP', fontsize=16)
plt.ylabel('Predicted logP', fontsize=16)
plt.xticks(np.arange(12, -9, step=-2), fontsize=14)
plt.yticks(np.arange(12, -9, step=-2), fontsize=14)
plt.grid(color='#D3D3D3', linestyle='--', which='both', axis='both')
plt.legend(loc='upper left', fontsize=16)
plt.show()
