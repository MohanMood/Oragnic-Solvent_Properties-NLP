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
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
from sklearn import preprocessing
from catboost import CatBoostRegressor

# loading organic solvents enthalpy of vaporization data
input_dir = 'path-of-the-dataset/'
dataset = pd.read_csv(input_dir + 'Organic-solvents_Enthalpy-of-vaporization.csv', encoding='unicode_escape')

# Sorting dataset with unique SMILES
dataset = dataset.drop_duplicates(subset=['SMILES']).sort_values(by='SMILES')

print(dataset.shape)

mol_smiles = dataset['SMILES']
enthalpy_vap = dataset['enthalpy_vap']

# Load the pre-trained Word2Vec model
model_path = '/mol2vec-master/models/model_300dim.pkl'
model = word2vec.Word2Vec.load(model_path)

# Extraction of identifiers from molecules
os_smiles = [Chem.MolFromSmiles(x) for x in mol_smiles]
os_sentences = [mol2alt_sentence(x, 1) for x in os_smiles]

# Define the DfVec class
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

# Convert sentences to vectors
os_mol2vec = [DfVec(x) for x in sentences2vec(os_sentences, model, unseen='UNK')]
np_os_mol2vec = np.array([x.vector for x in os_mol2vec])
print("mol2vec shape:", np_os_mol2vec.shape)

# Standardize the mol2vec features
scaler = preprocessing.StandardScaler()
np_os_mol2vec = scaler.fit_transform(np_os_mol2vec)

print("features shape:", np_os_mol2vec.shape)

# Seed for reproducibility
seed = 536

# Simple train-test split
X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
    np_os_mol2vec, enthalpy_vap, mol_smiles, test_size=0.2, random_state=seed)

# Train CatBoost
cat_model = CatBoostRegressor(verbose=100)
cat_model.fit(X_train, y_train)

# Predict on train
y_pred_train = cat_model.predict(X_train)
train_df = pd.DataFrame({
    'SMILES': smiles_train.reset_index(drop=True),
    'enthalpy_vap_actual': y_train.reset_index(drop=True),
    'enthalpy_vap_pred': y_pred_train
})
train_df.to_csv(f'CATBoost_Mol2vec-Training_Enthalpy-Vap-{seed}seed.csv', index=False)

# Predict on test
y_pred_test = cat_model.predict(X_test)
test_df = pd.DataFrame({
    'SMILES': smiles_test.reset_index(drop=True),
    'enthalpy_vap_actual': y_test.reset_index(drop=True),
    'enthalpy_vap_pred': y_pred_test
})
test_df.to_csv(f'CATBoost_Mol2vec-Testing_Enthalpy-Vap-{seed}seed.csv', index=False)

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
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(color='#D3D3D3', linestyle='--', which='both', axis='both')
plt.legend(loc='upper left', fontsize=16)
plt.show()
