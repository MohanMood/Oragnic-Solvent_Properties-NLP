import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from div import *
from sampling import *
from comp import *
from real import calculate_isim_real, calculate_medoid_real, calculate_outlier_real, pairwise_average_real
from utils import binary_fps, minmax_norm, real_fps, pairwise_average, rdkit_pairwise_sim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import datasets
from datasets import load_dataset
import torch
import evaluate
import random
import os
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set seed for reproducibility
seed = 536
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

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
mol_smiles = dataset['Canonical_SMILES']
Exp_logp = dataset['Exp_logp']

# Select relevant columns
dataset = dataset[['Canonical_SMILES', 'Exp_logp', 'Label']]

# Split dataset based on Label
train_data = dataset[dataset['Label'] == 'Training']
val_data = dataset[dataset['Label'] == 'Validation']
test_data = dataset[dataset['Label'] == 'Testing']

# Extract inputs and outputs
X_train = train_data['Canonical_SMILES']
X_val = val_data['Canonical_SMILES']
X_test = test_data['Canonical_SMILES']
y_train = train_data['Exp_logp']
y_val = val_data['Exp_logp']
y_test = test_data['Exp_logp']
smiles_train = train_data['Canonical_SMILES']
smiles_val = val_data['Canonical_SMILES']
smiles_test = test_data['Canonical_SMILES']

# Create DataFrames for Hugging Face datasets
train_dataset = pd.DataFrame({
	'Canonical_SMILES': smiles_train,    
	'Exp_logp': y_train
})
val_dataset = pd.DataFrame({
	'Canonical_SMILES': smiles_val,
	'Exp_logp': y_val
})
test_dataset = pd.DataFrame({
	'Canonical_SMILES': smiles_test,
	'Exp_logp': y_test
})

# Save to CSV
train_dataset.to_csv('LogP_train_dataset_chemBERTa.csv', index=False)
val_dataset.to_csv('LogP_val_dataset_chemBERTa.csv', index=False)
test_dataset.to_csv('LogP_test_dataset_chemBERTa.csv', index=False)

# Load datasets into Hugging Face format
dataset = load_dataset('csv', data_files={
    'train': 'LogP_train_dataset_chemBERTa.csv',
    'validation': 'LogP_val_dataset_chemBERTa.csv',
    'test': 'LogP_test_dataset_chemBERTa.csv'
}, delimiter=',')

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

def tokenize_function(examples):
    return tokenizer(examples["Canonical_SMILES"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Add labels to the tokenized datasets
tokenized_datasets = tokenized_datasets.map(lambda x: {'labels': x['Exp_logp']})

# Prepare datasets for training
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

# Save tokenized datasets to csv
train_dataset.to_csv('LogP_train_ChemBERTa_dataset.csv', index=False)
val_dataset.to_csv('LogP_val_ChemBERTa_dataset.csv', index=False)
test_dataset.to_csv('LogP_test_ChemBERTa_dataset.csv', index=False)

# Define model
model = AutoModelForSequenceClassification.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", num_labels=1)

# Define metrics
mae_metric = evaluate.load("mae")
mse_metric = evaluate.load("mse")
pearsonr_metric = evaluate.load("pearsonr")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    metrics = {
        'mae': mae_metric.compute(predictions=predictions, references=labels)['mae'],
        'rmse': math.sqrt(mse_metric.compute(predictions=predictions, references=labels)['mse']),
        'pearsonr': pearsonr_metric.compute(predictions=predictions, references=labels)['pearsonr'],
        'r2': r2_score(labels, predictions),
    }
    return metrics

# Training arguments
para_output_dir = 'Path_of_model/'
model_output_path = f'{para_output_dir}/model'

training_args = TrainingArguments(output_dir=para_output_dir, 
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  num_train_epochs=100, 
                                  report_to="none",
                                  load_best_model_at_end=True, 
                                  metric_for_best_model="mae", 
                                  greater_is_better=False,
                                  seed = seed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
)

# Fine-tuning
trainer.train()

# Print the number of completed epochs
print(f"Training completed at epoch: {int(trainer.state.epoch)}")

# Save model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model 
model_to_save.save_pretrained(model_output_path)

# Save training predictions
train_results = trainer.predict(small_train_dataset)
pd_pred_train = pd.DataFrame(train_results.predictions, columns=["predict"])
pd_exp_train = pd.DataFrame(train_results.label_ids, columns=["exp"])
pd_smiles_train = pd.DataFrame(X_train.reset_index(drop=True), columns=["SMILES"])
pd_train = pd.concat((pd_smiles_train, pd_exp_train, pd_pred_train), axis=1)

pd_train.to_csv(f'Organic_Solvents_LogP_Train-{seed}Seed.csv', index=False)

# Making prediction
model = AutoModelForSequenceClassification.from_pretrained(model_output_path)

# Arguments for Trainer
test_args = TrainingArguments(
     output_dir=model_output_path,
     do_train=False,
     do_predict=True,
     dataloader_drop_last=False,
     report_to="none"
)

# Init Trainer
trainer = Trainer(
          model=model,
          args=test_args,
          compute_metrics=compute_metrics)

test_results = trainer.predict(test_dataset)

# Print out predictions and metrics in test set
print("Predictions:", test_results.predictions)
print("Metrics:", test_results.metrics)

# MAE, RMSE and R^2 Coefficient of Determination
print("Mean Absolute Error:", mean_absolute_error(test_results.label_ids, test_results.predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(test_results.label_ids, test_results.predictions)))
print("R^2 Score:", r2_score(test_results.label_ids, test_results.predictions))

# Save prediction to csv
pd_pred_test = pd.DataFrame(test_results.predictions, columns=["predict"])
pd_exp_test = pd.DataFrame(test_results.label_ids, columns=["exp"])
pd_smiles = pd.DataFrame(X_test.reset_index(drop=True), columns=["SMILES"])
pd_test = pd.concat((pd_smiles, pd_exp_test, pd_pred_test), axis=1)

pd_test.to_csv(f'Organic_Solvents_LogP_Test-{seed}Seed.csv', index=False)
