import pandas as pd
import numpy as np
import torch
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from chemprop import data, models, nn, uncertainty
from chemprop.models import save_model, load_model
from chemprop.cli.predict import find_models
from chemprop.cli.conf import NOW
import random

# Set random seed for reproducibility
SEED = 214
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
pl.seed_everything(SEED, workers=True)

# Load Dataset
input_dir = 'path-of-the-dataset/'
dataset = pd.read_csv(input_dir + 'Organic-solvents_Enthalpy-of-vaporization.csv')
print(dataset.shape)

# Remove duplicates and sort by SMILES
dataset = dataset.drop_duplicates(subset=['SMILES']).sort_values(by='SMILES')
print(dataset.shape)

mol_smiles = dataset['SMILES']
logP_values = dataset['enthalpy_vap'].values.reshape(-1, 1)
all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(mol_smiles, logP_values)]

# Split Data
mols = [d.mol for d in all_data]
train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.7, 0.1, 0.2), SEED)
train_data, val_data, test_data = data.split_data_by_indices(all_data, train_indices, val_indices, test_indices)

train_dset = data.MoleculeDataset(train_data[0])
scaler = train_dset.normalize_targets()
val_dset = data.MoleculeDataset(val_data[0])
val_dset.normalize_targets(scaler)
test_dset = data.MoleculeDataset(test_data[0])

train_loader = data.build_dataloader(train_dset)
val_loader = data.build_dataloader(val_dset, shuffle=False)
test_loader = data.build_dataloader(test_dset, shuffle=False)

# Define MPNN Model
mp = nn.BondMessagePassing()
agg = nn.MeanAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.MveFFN(output_transform=output_transform)
mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)

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
    patience=50,
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
p_model = model_output_dir / "best.pt"
save_model(p_model, model)

# Load and Process Test Data
test_dset = data.MoleculeDataset(test_data[0])
test_loader = data.build_dataloader(test_dset, shuffle=False)
unc_estimator = uncertainty.MVEEstimator())

# Load trained model
model_paths = find_models([model_output_dir])
models = [load_model(model_path, multicomponent=False) for model_path in model_paths]
trainer = pl.Trainer(logger=False, enable_progress_bar=True, accelerator="cpu", devices=1)

test_predss, test_uncss = unc_estimator(test_loader, models, trainer)
test_preds = test_predss.mean(0)

df_test = pd.DataFrame(
    {
        "smiles": test_dset.smiles,
        "target": test_dset.Y.reshape(-1),
        "pred": test_preds.reshape(-1),
    }
)

df_test.to_csv(f'MPFF_model-Enthalpy-Vap_{SEED}seed-SMILES_CommChem_70-10-20_Early.csv', index=False)
