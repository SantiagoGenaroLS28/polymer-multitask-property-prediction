import json
import os
import random
import numpy as np
import pandas as pd
import torch

from torch import nn
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import polygnn_trainer as pt


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "polygnn_rho_tg_tm.csv"
SPLIT_FILE = r"C:\Users\santi\PolyGraphMT\splits\rho_tg_tm.json"

RANDOM_SEED = 42
N_FEATURES = 512

# Keep this modest so we don't spend forever on this model
EPOCHS = 100
N_FOLDS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

master_data = pd.read_csv(DATA_FILE, index_col=0)

print()
print("Property counts:")
print(master_data["prop"].value_counts())

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

train_smiles = set(split["train_smiles"])
val_smiles = set(split["val_smiles"])
test_smiles = set(split["test_smiles"])

train_data = master_data[
    master_data["smiles_string"].isin(train_smiles)
].copy()

val_data = master_data[
    master_data["smiles_string"].isin(val_smiles)
].copy()

test_data = master_data[
    master_data["smiles_string"].isin(test_smiles)
].copy()

print()
print("Measurements:")
print("Train:", len(train_data))
print("Val:", len(val_data))
print("Test:", len(test_data))

print()
print("Unique polymers:")
print("Train:", train_data["smiles_string"].nunique())
print("Val:", val_data["smiles_string"].nunique())
print("Test:", test_data["smiles_string"].nunique())


# polyGNN expects this column
train_data["graph_feats"] = [{} for _ in range(len(train_data))]
val_data["graph_feats"] = [{} for _ in range(len(val_data))]
test_data["graph_feats"] = [{} for _ in range(len(test_data))]


# ============================================================
# POLYGNN FEATURIZER FROM THEIR EXAMPLE
# ============================================================

def morgan_featurizer(smile):

    # This is what their example.py does

    mol = Chem.MolFromSmiles(smile)

    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smile}")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=N_FEATURES,
        useChirality=True
    )

    fp = np.asarray(fp, dtype=np.float32)
    fp = np.expand_dims(fp, 0)

    return Data(
        x=torch.tensor(fp, dtype=torch.float)
    )


# ============================================================
# MODEL SETTINGS
# ============================================================

PROPERTY_GROUPS = {
    "thermal": ["rho", "tg", "tm"]
}

# Instead of running expensive hyperparameter optimization,
# use a reasonable fixed configuration.
hps = pt.hyperparameters.HpConfig()

hps.set_values({
    "r_learn": 0.001,
    "batch_size": 50,
    "dropout_pct": 0.1,
    "capacity": 2,
    "activation": nn.functional.leaky_relu,
})


# ============================================================
# TRAIN MULTITASK MODEL
# ============================================================

os.makedirs("polygnn_results", exist_ok=True)

for group, prop_cols in PROPERTY_GROUPS.items():

    print()
    print("=" * 70)
    print("Training properties:", prop_cols)
    print("=" * 70)

    prop_cols = sorted(prop_cols)
    nprops = len(prop_cols)

    selector_dim = nprops if nprops > 1 else 0

    root_dir = os.path.join(
        "polygnn_results",
        group
    )


    group_train = train_data[
        train_data["prop"].isin(prop_cols)
    ].copy()

    group_val = val_data[
        val_data["prop"].isin(prop_cols)
    ].copy()

    group_test = test_data[
        test_data["prop"].isin(prop_cols)
    ].copy()

    # Keep track of rows before preparation
    train_indices = group_train.index.tolist()
    val_indices = group_val.index.tolist()

    # Prepare TRAIN + VAL together because polyGNN needs
    # the features/scalers available for training.
    train_val = pd.concat(
        [group_train, group_val],
        axis=0
    )

    train_val, scaler_dict = pt.prepare.prepare_train(
        train_val,
        smiles_featurizer=morgan_featurizer,
        root_dir=root_dir
    )

    group_train = train_val.loc[train_indices]
    group_val = train_val.loc[val_indices]

    fit_pts = group_train.data.values.tolist()
    val_pts = group_val.data.values.tolist()

    print()
    print("Training measurements:", len(group_train))
    print("Validation measurements:", len(group_val))
    print("Test measurements:", len(group_test))

    print()
    print("Scaler information:")
    print([(k, str(v)) for k, v in scaler_dict.items()])


    # ========================================================
    # TRAIN CONFIG
    # ========================================================

    train_config = pt.train.trainConfig(
        amp=False,
        loss_obj=pt.loss.sh_mse_loss(),
        hps=hps,
        device=DEVICE,
        multi_head=False,
    )

    train_config.epochs = EPOCHS


    # ========================================================
    # TRAIN 3-FOLD ENSEMBLE
    # ========================================================

    pt.train.train_kfold_ensemble(
        dataframe=group_train,

        model_constructor=lambda: pt.models.MlpOut(
            input_dim=N_FEATURES + selector_dim,
            output_dim=1,
            hps=hps,
        ),

        train_config=train_config,
        submodel_trainer=pt.train.train_submodel,
        augmented_featurizer=None,
        scaler_dict=scaler_dict,
        root_dir=root_dir,
        n_fold=N_FOLDS,
        random_seed=RANDOM_SEED,
    )


    # ========================================================
    # LOAD ENSEMBLE
    # ========================================================

    ensemble = pt.load.load_ensemble(
        root_dir,
        pt.models.MlpOut,
        DEVICE,
        {
            "input_dim": N_FEATURES + selector_dim,
            "output_dim": 1,
        },
    )


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    y, y_pred, y_std, selectors = pt.infer.eval_ensemble(
        model=ensemble,
        root_dir=root_dir,
        dataframe=group_test,
        smiles_featurizer=morgan_featurizer,
        device=DEVICE,
        ensemble_kwargs_dict={
            "monte_carlo": False
        },
    )


    # ========================================================
    # CALCULATE METRICS PER PROPERTY
    # ========================================================

    results = {}

    # selectors identifies which property each row belongs to.
    # We use the raw dataframe ordering to keep things simple.

    group_test_reset = group_test.reset_index(drop=True)

    y = np.asarray(y).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_std = np.asarray(y_std).reshape(-1)

    for prop in prop_cols:

        mask = (
            group_test_reset["prop"]
            .values == prop
        )

        y_prop = y[mask]
        pred_prop = y_pred[mask]
        std_prop = y_std[mask]

        r2 = r2_score(
            y_prop,
            pred_prop
        )

        mae = mean_absolute_error(
            y_prop,
            pred_prop
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_prop,
                pred_prop
            )
        )

        results[prop] = {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "n_test": int(mask.sum()),
            "mean_uncertainty": float(
                np.mean(std_prop)
            ),
        }


    print()
    print("=" * 70)
    print("POLYGNN TEST RESULTS")
    print("=" * 70)

    for prop, metrics in results.items():

        print(
            f"{prop}: "
            f"R2={metrics['r2']:.4f}, "
            f"MAE={metrics['mae']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"N={metrics['n_test']}"
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {
        "model": "polyGNN",
        "type": "multitask",
        "properties": prop_cols,
        "representation": "Morgan bit vector",
        "radius": 2,
        "n_bits": N_FEATURES,
        "seed": RANDOM_SEED,
        "epochs": EPOCHS,
        "folds": N_FOLDS,
        "results": results,
    }

    with open(
        "polygnn_results.json",
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )


print()
print("Finished.")
print("Saved: polygnn_results.json")