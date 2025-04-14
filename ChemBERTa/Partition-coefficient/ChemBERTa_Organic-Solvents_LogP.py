import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import evaluate
import math
from transformers import EarlyStoppingCallback
import random


# Random seed for reproducibility
seed = 302
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Load the partition coefficient dataset
input_dir = 'path-of-dataset/'
dataset = pd.read_csv(input_dir + 'Organic-Solvents_LogP.csv')

# Sorting the dataset with unique SMILES and remove duplicate SMILES
dataset = dataset.drop_duplicates(subset=['SMILES']).sort_values(by='SMILES')
print("Initial dataset shape:", dataset.shape)

dataset = dataset[['SMILES', 'Exp_logp']]

X = dataset['SMILES']
y = dataset['Exp_logp']

# Split the data into training and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=seed)

# Create training, validation, and testing datasets
train_dataset = pd.concat([X_train, y_train], axis=1)
val_dataset = pd.concat([X_val, y_val], axis=1)
test_dataset = pd.concat([X_test, y_test], axis=1)

# Save them to CSV
train_dataset.to_csv('LogP_train_dataset.csv', index=False)
val_dataset.to_csv('LogP_val_dataset.csv', index=False)
test_dataset.to_csv('LogP_test_dataset.csv', index=False)

# Load datasets into HuggingFace format
dataset = load_dataset('csv', data_files={
    'train': 'LogP_train_dataset.csv',
    'validation': 'LogP_val_dataset.csv',
    'test': 'LogP_test_dataset.csv'
}, delimiter=',')

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

def tokenize_function(examples):
    return tokenizer(examples["SMILES"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Add labels to the tokenized datasets
tokenized_datasets = tokenized_datasets.map(lambda x: {'labels': x['Exp_logp']})

# Prepare datasets for training
small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["validation"]
small_test_dataset = tokenized_datasets["test"]


# Save tokenized datasets to csv
small_train_dataset.to_csv('LogP_train_ChemBERTa_dataset.csv', index=False)
small_eval_dataset.to_csv('LogP_val_ChemBERTa_dataset.csv', index=False)
small_test_dataset.to_csv('LogP_test_ChemBERTa_dataset.csv', index=False)

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
para_output_dir = '/home/mmj/Machine-Learning/LLM/LogP/'
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
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
)

# Fine-tuning
trainer.train()
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

pd_train.to_csv(f'Organic_Solvents_LogP_Train_Predictions_{seed}seed.csv', index=False)

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

test_results = trainer.predict(small_test_dataset)

# Print out predictions and metrics in test set
print("Predictions:", test_results.predictions)
print("Metrics:", test_results.metrics)

# MAE, RMSE and R^2 Coefficient of Determination
print("Mean Absolute Error:", mean_absolute_error(test_results.label_ids, test_results.predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(test_results.label_ids, test_results.predictions)))
print("R^2 Score:", r2_score(test_results.label_ids, test_results.predictions))

# Plot exp vs pred in test set
plt.figure()
ln = np.arange(min(test_results.label_ids), max(test_results.label_ids), 0.2)
plt.plot(ln, ln, 'r--')
plt.scatter(test_results.label_ids, test_results.predictions)
plt.xlabel('exp. LogP')
plt.ylabel('pred. LogP')
plt.show()

# Save prediction to csv
pd_pred_test = pd.DataFrame(test_results.predictions, columns=["predict"])
pd_exp_test = pd.DataFrame(test_results.label_ids, columns=["exp"])
pd_smiles = pd.DataFrame(X_test.reset_index(drop=True), columns=["SMILES"])
pd_test = pd.concat((pd_smiles, pd_exp_test, pd_pred_test), axis=1)

pd_test.to_csv(f'Organic_Solvents_LogP_Test_Predictions_{seed}seed.csv', index=False)