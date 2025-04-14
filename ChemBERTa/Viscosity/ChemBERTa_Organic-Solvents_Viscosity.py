import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import datasets
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import evaluate
import math
import random

# Random seed for reproducibility
seed = 637
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Load dataset
input_dir = 'path-of-the-dataset/'
dataset = pd.read_csv(input_dir + 'Oragnic-Solvents_Viscosity.csv', encoding='unicode_escape')

# Select relevant columns and first 100 data points
dataset = dataset[['CANON_SMILES', 'Temperature (K)', 'log_visc']]

# Splitting features and target
X = dataset[['CANON_SMILES', 'Temperature (K)']]
y = dataset['log_visc']

# Split the data into training and testing sets
X_trva, X_test, y_trva, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_trva, y_trva, test_size=0.125, random_state=seed)

# Load dataset into HuggingFace format
train_dataset = pd.concat([X_train, y_train], axis=1)
val_dataset = pd.concat([X_val, y_val], axis=1)
test_dataset = pd.concat([X_test, y_test], axis=1)

train_dataset.to_csv('viscosity_train_dataset.csv', index=False)
val_dataset.to_csv('viscosity_val_dataset.csv', index=False)
test_dataset.to_csv('viscosity_test_dataset.csv', index=False)

# Load datasets
dataset = load_dataset('csv', data_files={'train': 'viscosity_train_dataset.csv',
                                          'val': 'viscosity_val_dataset.csv', 
                                          'test': 'viscosity_test_dataset.csv'}, delimiter=',')

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

def tokenize_function(examples):
    tokens_smiles = tokenizer(examples["CANON_SMILES"], padding="max_length", truncation=True, max_length=504)
    tokens_temp = tokenizer(str(examples["Temperature (K)"]), padding="max_length", truncation=True, max_length=8)
    # Concatenate the input_ids and attention_mask properly
    input_ids =  tokens_smiles['input_ids'] + tokens_temp['input_ids']
    attention_mask = tokens_smiles['attention_mask'] + tokens_temp['attention_mask']
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

tokenized_datasets = dataset.map(tokenize_function, batched=False)

# Add labels to the tokenized datasets
tokenized_datasets = tokenized_datasets.map(lambda x: {'labels': x['log_visc']})

# Prepare datasets for training
small_train_dataset = tokenized_datasets["train"]
small_val_dataset = tokenized_datasets["val"]
small_eval_dataset = tokenized_datasets["test"]

small_train_dataset.to_csv('viscosity_train_ChemBERTa_dataset.csv', index=False)
small_val_dataset.to_csv('viscosity_val_ChemBERTa_dataset.csv', index=False)
small_eval_dataset.to_csv('viscosity_test_ChemBERTa_dataset.csv', index=False)

# Print the shape of input features in the small_train_dataset
print("Shape of input_ids:", len(small_train_dataset['input_ids'][0]), "tokens per input")
print("Shape of attention_mask:", len(small_train_dataset['attention_mask'][0]), "tokens per input")
print("Shape of labels:", len(small_train_dataset['labels']), "data points")

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
        'rmse': math.sqrt(mse_metric.compute(predictions=predictions, references=labels, squared=False)['mse']),
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
    eval_dataset=small_val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]

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
pd_smiles_train = pd.DataFrame(X_train['CANON_SMILES'].reset_index(drop=True), columns=["CANON_SMILES"])
pd_temp_train = pd.DataFrame(X_train['Temperature (K)'].reset_index(drop=True), columns=["Temperature (K)"])
pd_train = pd.concat((pd_smiles_train, pd_temp_train, pd_exp_train, pd_pred_train), axis=1)

pd_train.to_csv(f'organic_solvents_log_visc_train_predictions-{seed}seed.csv', index=False)

# Making prediction
model = AutoModelForSequenceClassification.from_pretrained(model_output_path)

# Arguments for Trainer
test_args = TrainingArguments(
     output_dir=model_output_path,
     do_train=False,
     do_predict=True,
     dataloader_drop_last=False,
     report_to="none",
    seed=seed
)

# Init Trainer
trainer = Trainer(
          model=model,
          args=test_args,
          compute_metrics=compute_metrics)

test_results = trainer.predict(small_eval_dataset)

# Print out predictions and metrics in test set
print("Predictions:", test_results.predictions)
print("Metrics:", test_results.metrics)

# MAE, RMSE and R^2 Coefficient of Determination
print("Mean Absolute Error:", mean_absolute_error(test_results.predictions, test_results.label_ids))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(test_results.predictions, test_results.label_ids)))
print("R^2 Score:", r2_score(test_results.predictions, test_results.label_ids))

# Plot exp vs pred in test set
plt.figure()
ln = np.arange(min(test_results.label_ids), max(test_results.label_ids), 0.2)
plt.plot(ln, ln, 'r--')
plt.scatter(test_results.label_ids, test_results.predictions)
plt.xlabel('exp. log_visc')
plt.ylabel('pred. log_visc')

# Save prediction to csv
pd_pred_test = pd.DataFrame(test_results.predictions, columns=["predict"])
pd_exp_test = pd.DataFrame(test_results.label_ids, columns=["exp"])
pd_smiles = pd.DataFrame(X_test['CANON_SMILES'].reset_index(drop=True), columns=["CANON_SMILES"])
pd_temp = pd.DataFrame(X_test['Temperature (K)'].reset_index(drop=True), columns=["Temperature (K)"])
pd_test = pd.concat((pd_smiles, pd_temp, pd_exp_test, pd_pred_test), axis=1)

pd_test.to_csv(f'organic_solvents_log_visc_test_predictions-{seed}seed.csv', index=False)
