import pandas as pd
import numpy as np
import torch
import random
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from chemprop import data, models, nn, uncertainty, featurizers, utils
from chemprop.models import save_model, load_model
from chemprop.cli.predict import find_models
from chemprop.cli.conf import NOW

# Set random seed for reproducibility
SEED = 365
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
pl.seed_everything(SEED, workers=True)

# Reading organic solvents viscosity data
input_dir = 'path-of-the-dataset/'
dataset = pd.read_csv(input_dir + 'Oragnic-Solvents_Viscosity.csv', encoding='unicode_escape')

mol_smiles = dataset['CANON_SMILES']
logP_values = dataset['log_visc'].values.reshape(-1, 1)
temperature_values = dataset['Temperature (K)'].values.reshape(-1, 1)
temperature_values = np.array(temperature_values).reshape(len(mol_smiles), 1)

all_data = [
    data.MoleculeDatapoint.from_smi(smi, y, x_d=X_d)
    for smi, y, X_d in zip(
        mol_smiles,
        logP_values,
        temperature_values,
    )
]

# Split Data with fixed seed
mols = [d.mol for d in all_data]
train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.7, 0.1, 0.2), SEED)
train_data, val_data, test_data = data.split_data_by_indices(all_data, train_indices, val_indices, test_indices)

train_dset = data.MoleculeDataset(train_data[0])
scaler = train_dset.normalize_targets()
temperature_values_scaler = train_dset.normalize_inputs("X_d")
val_dset = data.MoleculeDataset(val_data[0])
val_dset.normalize_targets(scaler)
val_dset.normalize_inputs("X_d", temperature_values_scaler)
test_dset = data.MoleculeDataset(test_data[0])
test_dset.normalize_inputs("X_d", temperature_values_scaler)

train_loader = data.build_dataloader(train_dset)
val_loader = data.build_dataloader(val_dset, shuffle=False)
test_loader = data.build_dataloader(test_dset, shuffle=False)

# Define MPNN Model
mp = nn.BondMessagePassing()
agg = nn.MeanAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn_input_dim = mp.output_dim + temperature_values.shape[1]
ffn = nn.MveFFN(output_transform=output_transform, input_dim=ffn_input_dim)
X_d_transform = nn.ScaleTransform.from_standard_scaler(temperature_values_scaler)
mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, X_d_transform=X_d_transform)


# Monitoring Setup
monitor_metric = "val_loss"
monitor_mode = "min"
print(f"Monitoring metric: {monitor_metric} with mode: {monitor_mode}")

# Checkpoint and EarlyStopping
model_output_dir = Path(f"chemprop_training/{NOW}")
checkpointing = ModelCheckpoint(
    dirpath=model_output_dir / "checkpoints",
    filename="best-{epoch}-{val_loss:.2f}",
    monitor=monitor_metric,
    mode=monitor_mode,
    save_last=True,
)

early_stop_callback = EarlyStopping(
    monitor=monitor_metric,
    patience=150,
    verbose=True,
    mode=monitor_mode
)

# Trainer
trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True,
    enable_progress_bar=False,
    accelerator="cpu",
    callbacks=[checkpointing, early_stop_callback],
    devices=1,
    max_epochs=2000,
    deterministic=True,
)

trainer.fit(mpnn, train_loader, val_loader)
print(f"Training stopped after {trainer.current_epoch + 1} epochs.")


best_model_path = checkpointing.best_model_path
model = mpnn.__class__.load_from_checkpoint(best_model_path)
p_model = model_output_dir / "Organic-Solvents_Viscosity_Best.pt"
save_model(p_model, model)

# Load and Process Test Data
test_dset = data.MoleculeDataset(test_data[0])
test_loader = data.build_dataloader(test_dset, shuffle=False)
unc_estimator = uncertainty.MVEEstimator()

# Load trained model
model_paths = find_models([model_output_dir])
models = [load_model(model_path, multicomponent=False) for model_path in model_paths]
trainer = pl.Trainer(logger=False, enable_progress_bar=True, accelerator="cpu", devices=1, deterministic=True)

test_predss, test_uncss = unc_estimator(test_loader, models, trainer)
test_preds = test_predss.mean(0)

# Retrieve the scaled temperature values
scaled_temperature_values = test_dset.X_d

df_test = pd.DataFrame(
    {
        "smiles": test_dset.smiles,
        "Temp": scaled_temperature_values.reshape(-1),
        "target": test_dset.Y.reshape(-1),
        "pred": test_preds.reshape(-1),
    }
)

df_test.to_csv(f'Organic-Solvents-Viscosity_MPNN-model-{SEED}seed.csv', index=False)