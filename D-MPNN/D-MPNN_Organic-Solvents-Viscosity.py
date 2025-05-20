import pandas as pd
import numpy as np
import torch
import random
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from chemprop import data, models, nn, uncertainty
from chemprop.models import save_model, load_model
from chemprop.cli.predict import find_models
from chemprop.cli.conf import NOW
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 536
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
pl.seed_everything(SEED, workers=True)

def load_csv_data(file_path, smiles_column='SMILES', target_column='viscosity_exp', temp_column='Temp (K)', set_column='Set'):
    """Load a CSV file for viscosity data, preserving all SMILES."""
    try:
        dataset = pd.read_csv(file_path, encoding='unicode_escape')
        print(f"Dataset shape for {file_path}: {dataset.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_path}")

    # Split data based on 'Set' column from Mol2vec predictions
    train_data = dataset[dataset[set_column] == 'Training']
    val_data = dataset[dataset[set_column] == 'Validation']
    test_data = dataset[dataset[set_column] == 'Testing']

    # Store original SMILES to preserve input order
    train_smiles = train_data[smiles_column].tolist()
    val_smiles = val_data[smiles_column].tolist()
    test_smiles = test_data[smiles_column].tolist()

    # Extract SMILES, targets, and temperatures
    train_mol_smiles = train_data[smiles_column]
    train_targets = train_data[target_column].values.reshape(-1, 1)
    train_temps = train_data[temp_column].values.reshape(-1, 1)

    val_mol_smiles = val_data[smiles_column]
    val_targets = val_data[target_column].values.reshape(-1, 1)
    val_temps = val_data[temp_column].values.reshape(-1, 1)

    test_mol_smiles = test_data[smiles_column]
    test_targets = test_data[target_column].values.reshape(-1, 1)
    test_temps = test_data[temp_column].values.reshape(-1, 1)

    return (train_mol_smiles, train_targets, train_temps, train_data, train_smiles,
            val_mol_smiles, val_targets, val_temps, val_data, val_smiles,
            test_mol_smiles, test_targets, test_temps, test_data, test_smiles)

def create_dataloaders(file_path, input_dir, smiles_column='SMILES', target_column='viscosity_exp', temp_column='Temp (K)', set_column='Set'):
    """Create train, validation, and test dataloaders from a single CSV file."""
    # Load dataset
    (train_smiles, train_targets, train_temps, train_dataset, train_original_smiles,
     val_smiles, val_targets, val_temps, val_dataset, val_original_smiles,
     test_smiles, test_targets, test_temps, test_dataset, test_original_smiles) = load_csv_data(
        input_dir + file_path, smiles_column, target_column, temp_column, set_column
    )

    # Create MoleculeDatapoints with temperature as X_d
    def create_datapoints(smiles, targets, temps):
        datapoints = []
        for smi, y, X_d in zip(smiles, targets, temps):
            try:
                datapoint = data.MoleculeDatapoint.from_smi(smi, y, x_d=X_d)
                datapoints.append(datapoint)
            except ValueError as e:
                print(f"Invalid SMILES '{smi}': {e}")
        return datapoints

    train_data = create_datapoints(train_smiles, train_targets, train_temps)
    val_data = create_datapoints(val_smiles, val_targets, val_temps)
    test_data = create_datapoints(test_smiles, test_targets, test_temps)

    if not train_data or not val_data or not test_data:
        raise ValueError("One or more datasets are empty after SMILES validation")

    # Create datasets
    train_dset = data.MoleculeDataset(train_data)
    scaler = train_dset.normalize_targets()
    val_dset = data.MoleculeDataset(val_data)
    val_dset.normalize_targets(scaler)
    test_dset = data.MoleculeDataset(test_data)
    test_dset.normalize_targets(scaler)

    # Create dataloaders
    train_loader = data.build_dataloader(train_dset)
    val_loader = data.build_dataloader(val_dset, shuffle=False)
    test_loader = data.build_dataloader(test_dset, shuffle=False)
    return train_loader, val_loader, test_loader, scaler, test_dset, test_original_smiles, test_dataset

# Load Dataset
input_dir = 'path_of_viscosity_dataset/'
file_path = 'CATBoost_Mol2vec-Viscosity-Quota.csv'

train_loader, val_loader, test_loader, scaler, test_dset, test_original_smiles, test_dataset = create_dataloaders(
    file_path, input_dir, smiles_column='SMILES', target_column='viscosity_exp', temp_column='Temp (K)', set_column='Set'
)

# Define MPNN Model
mp = nn.BondMessagePassing()
agg = nn.MeanAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn_input_dim = mp.output_dim + 1 
ffn = nn.MveFFN(output_transform=output_transform, input_dim=ffn_input_dim)
mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)

# Monitoring Setup
monitor_metric = "val_loss"
monitor_mode = "min"
print(f"Monitoring metric: {monitor_metric} with mode: {monitor_mode}")

# Checkpoint and EarlyStopping
model_output_dir = Path(f"chemprop_training/{NOW}")
model_output_dir.mkdir(parents=True, exist_ok=True)
checkpointing = ModelCheckpoint(
    dirpath=model_output_dir / "checkpoints",
    filename="best-{epoch}-{val_loss:.2f}",
    monitor=monitor_metric,
    mode=monitor_mode,
    save_last=True,
)

early_stop_callback = EarlyStopping(
    monitor=monitor_metric,
    patience=250,
    verbose=True,
    mode=monitor_mode
)

# Trainer
trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="cpu",
    callbacks=[checkpointing, early_stop_callback],
    devices=1,
    max_epochs=3000,
    deterministic=True,
)

# Train Model
trainer.fit(mpnn, train_loader, val_loader)
print(f"Training stopped after {trainer.current_epoch} epochs.")

# Save Best Model
best_model_path = checkpointing.best_model_path
model = mpnn.__class__.load_from_checkpoint(best_model_path)
p_model = model_output_dir / f"Organic-Solvents_Viscosity_{SEED}seed_Mean_NoScale.pt"
save_model(p_model, model)

# Uncertainty Estimation
unc_estimator = uncertainty.MVEEstimator()
trainer = pl.Trainer(logger=False, enable_progress_bar=True, accelerator="cpu", devices=1, deterministic=True)

# Load trained model
model_paths = find_models([model_output_dir])
models = [load_model(model_path, multicomponent=False) for model_path in model_paths]

# Compute Predictions and Uncertainty
test_predss, test_uncss = unc_estimator(test_loader, models, trainer)
test_preds = test_predss.mean(0)
test_uncs = test_uncss.mean(0)

# Use original temperatures from input CSV
test_temps = test_dataset['Temp (K)'].values

# Create Output DataFrame
df_test = pd.DataFrame(
    {
        "smiles": test_original_smiles,
        "Temp": test_temps,
        "target": test_dataset['viscosity_exp'].values,
        "pred": test_preds.reshape(-1),
        "unc": test_uncs.reshape(-1),
    }
)

# Save Output
df_test.to_csv(model_output_dir / f'Log-Viscosity_MPFF_{SEED}seed_Mean-Final_noscale.csv', index=False)
