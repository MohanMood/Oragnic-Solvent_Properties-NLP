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
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
SEED = 536
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
pl.seed_everything(SEED, workers=True)

def load_csv_data(file_path, target_column='dvap', sort_smiles=True, remove_duplicates=True):
    """Load and preprocess a single CSV file."""
    dataset = pd.read_csv(file_path)
    print(f"Dataset shape for {file_path}: {dataset.shape}")
    original_smiles = dataset['SMILES'].tolist()

    # Remove duplicates and sort by SMILES (optional for train/val, disabled for test)
    if remove_duplicates:
        dataset = dataset.drop_duplicates(subset=['SMILES'])
        print(f"Shape after deduplication: {dataset.shape}")
    if sort_smiles:
        dataset = dataset.sort_values(by='SMILES')
        print(f"Shape after sorting: {dataset.shape}")

    # Validate SMILES and target values
    dataset = dataset.dropna(subset=['SMILES', target_column])
    print(f"Shape after removing NA: {dataset.shape}")

    mol_smiles = dataset['SMILES']
    targets = dataset[target_column].values.reshape(-1, 1)
    return mol_smiles, targets, dataset, original_smiles

def create_dataloaders(train_file, val_file, test_file, input_dir, target_column='dvap'):
    """Create train, validation, and test dataloaders from separate CSV files."""
    train_smiles, train_targets, _, _ = load_csv_data(
        input_dir + train_file, target_column, sort_smiles=True, remove_duplicates=True
    )
    val_smiles, val_targets, val_dataset, _ = load_csv_data(
        input_dir + val_file, target_column, sort_smiles=True, remove_duplicates=True
    )
    test_smiles, test_targets, test_dataset, test_original_smiles = load_csv_data(
        input_dir + test_file, target_column, sort_smiles=False, remove_duplicates=False
    )

    # Create MoleculeDatapoints
    train_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(train_smiles, train_targets)]
    val_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(val_smiles, val_targets)]
    test_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(test_smiles, test_targets)]

    # Create datasets
    train_dset = data.MoleculeDataset(train_data)
    scaler = train_dset.normalize_targets()
    val_dset = data.MoleculeDataset(val_data)
    val_dset.normalize_targets(scaler)
    test_dset = data.MoleculeDataset(test_data)

    # Create dataloaders
    train_loader = data.build_dataloader(train_dset)
    val_loader = data.build_dataloader(val_dset, shuffle=False)
    test_loader = data.build_dataloader(test_dset, shuffle=False)
    return train_loader, val_loader, test_loader, scaler, test_dset, test_original_smiles, test_dataset, val_dset

# Load Datasets
input_dir = 'Path_of_datasets/'
train_file = 'training_data.csv'
val_file = 'validation_data.csv'
test_file = 'testing_data.csv'

train_loader, val_loader, test_loader, scaler, test_dset, test_original_smiles, test_dataset, val_dset = create_dataloaders(
    train_file, val_file, test_file, input_dir, target_column='logP_exp'
)

# Define MPNN Model
mp = nn.BondMessagePassing()
agg = nn.SumAggregation()
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

# Train Model
trainer.fit(mpnn, train_loader, val_loader)
print(f"Training stopped after {trainer.current_epoch + 1} epochs.")

# Save Best Model
best_model_path = checkpointing.best_model_path
model = mpnn.__class__.load_from_checkpoint(best_model_path)
p_model = model_output_dir / f"MPFF_model-LogP_{SEED}SEED_Stratified-Sum.pt"
save_model(p_model, model)

# Uncertainty Estimation
unc_estimator = uncertainty.MVEEstimator()
unc_calibrator = uncertainty.ZScalingCalibrator()

# Load trained model
model_paths = find_models([model_output_dir])
models = [load_model(model_path, multicomponent=False) for model_path in model_paths]
trainer = pl.Trainer(logger=False, enable_progress_bar=True, accelerator="cpu", devices=1)

# Compute Predictions and Uncertainty
test_predss, test_uncss = unc_estimator(test_loader, models, trainer)
test_preds = test_predss.mean(0)
test_uncs = test_uncss.mean(0)

# Create Output DataFrame
df_test = pd.DataFrame(
    {
        "smiles": test_original_smiles,
        "target": test_dataset['logP_exp'].values,
        "pred": test_preds.reshape(-1),
        "unc": test_uncs.reshape(-1),
    }
)

# Calibration
cal_loader = data.build_dataloader(val_dset, shuffle=False)
cal_predss, cal_uncss = unc_estimator(cal_loader, models, trainer)
average_cal_preds = cal_predss.mean(0)
average_cal_uncs = cal_uncss.mean(0)
cal_targets = val_dset.Y
cal_mask = torch.from_numpy(np.isfinite(cal_targets))
cal_targets = np.nan_to_num(cal_targets, nan=0.0)
cal_targets = torch.from_numpy(cal_targets)
unc_calibrator.fit(average_cal_preds, average_cal_uncs, cal_targets, cal_mask)

cal_test_uncs = unc_calibrator.apply(test_uncs)
df_test["cal_unc"] = cal_test_uncs

# Evaluate Uncertainty
test_targets = test_dset.Y
test_mask = torch.from_numpy(np.isfinite(test_targets))
test_targets = np.nan_to_num(test_targets, nan=0.0)
test_targets = torch.from_numpy(test_targets)

unc_evaluators = [
    uncertainty.NLLRegressionEvaluator(),
    uncertainty.CalibrationAreaEvaluator(),
    uncertainty.ExpectedNormalizedErrorEvaluator(),
    uncertainty.SpearmanEvaluator(),
]

for evaluator in unc_evaluators:
    evaluation = evaluator.evaluate(test_preds, cal_test_uncs, test_targets, test_mask)
    print(f"{evaluator.alias}: {evaluation.tolist()}")

# Save Output
df_test.to_csv(f'MPFF_model-LogP_{SEED}SEED_Stratified-Sum-Final.csv', index=False)
