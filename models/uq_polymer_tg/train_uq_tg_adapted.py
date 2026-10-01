import json
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data/tg_exp.csv"
SPLIT_FILE = "splits/rho_tg_tm.json"

N_ENSEMBLES = 5
EPOCHS = 100
BATCH_SIZE = 256
RADIUS = 3
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD Tg DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

smiles_col = [c for c in df.columns if c.lower() == "smiles"][0]
tg_col = [c for c in df.columns if c.lower() == "tg"][0]

df = df[[smiles_col, tg_col]].copy()
df.columns = ["smiles", "tg"]

df = df.dropna(subset=["smiles", "tg"])

# Average duplicate SMILES, matching PolyGraphMT behavior
df = df.groupby("smiles", as_index=False)["tg"].mean()

print("Unique Tg polymers:", len(df))


# ============================================================
# LOAD COMMON SMILES SPLIT
# ============================================================

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

train_smiles = set(split["train_smiles"])
val_smiles = set(split["val_smiles"])
test_smiles = set(split["test_smiles"])

train_df = df[df["smiles"].isin(train_smiles)].copy()
val_df = df[df["smiles"].isin(val_smiles)].copy()
test_df = df[df["smiles"].isin(test_smiles)].copy()

print("Train Tg polymers:", len(train_df))
print("Validation Tg polymers:", len(val_df))
print("Test Tg polymers:", len(test_df))


# ============================================================
# MORGAN FREQUENCY FINGERPRINTS
# ============================================================

def get_morgan_counts(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprint(mol, radius=RADIUS)
    return fp.GetNonzeroElements()


# Build fingerprint vocabulary ONLY from training data
train_fps = []

for smi in train_df["smiles"]:
    train_fps.append(get_morgan_counts(smi))

valid_train = [fp for fp in train_fps if fp is not None]

# Count how many training polymers contain each Morgan feature
feature_frequency = Counter()

for fp in valid_train:
    for key in fp.keys():
        feature_frequency[key] += 1

# Keep features appearing in at least 2 training polymers
feature_list = sorted([
    key for key, count in feature_frequency.items()
    if count >= 2
])

feature_index = {
    feature: i for i, feature in enumerate(feature_list)
}

print("Morgan frequency features:", len(feature_list))


def make_features(dataframe):
    X = []
    y = []
    smiles_used = []

    for _, row in dataframe.iterrows():

        fp = get_morgan_counts(row["smiles"])

        if fp is None:
            continue

        vector = np.zeros(len(feature_list), dtype=np.float32)

        for key, count in fp.items():
            if key in feature_index:
                vector[feature_index[key]] = count

        X.append(vector)
        y.append(row["tg"])
        smiles_used.append(row["smiles"])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        smiles_used,
    )


X_train, y_train, smiles_train = make_features(train_df)
X_val, y_val, smiles_val = make_features(val_df)
X_test, y_test, smiles_test = make_features(test_df)

print("Feature matrix:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


# ============================================================
# PYTORCH DATA LOADERS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train)
)

val_dataset = TensorDataset(
    torch.tensor(X_val),
    torch.tensor(y_val)
)

test_dataset = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(y_test)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# UQ NEURAL NETWORK
# Original architecture:
# 512 -> 1024 -> 512
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self, n_input):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(model, model_number):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(inputs).squeeze(1)

            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                predictions = model(inputs).squeeze(1)

                loss = criterion(predictions, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch == EPOCHS - 1
        ):

            print(
                f"Model {model_number} | "
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

    model.load_state_dict(best_state)

    return model


# ============================================================
# TRAIN ENSEMBLE
# ============================================================

models = []

for i in range(N_ENSEMBLES):

    print()
    print("=" * 60)
    print(f"Training ensemble model {i+1}/{N_ENSEMBLES}")
    print("=" * 60)

    torch.manual_seed(SEED + i)

    model = NeuralNetwork(
        X_train.shape[1]
    ).to(DEVICE)

    model = train_model(model, i + 1)

    models.append(model)


# ============================================================
# ENSEMBLE PREDICTIONS
# ============================================================

def ensemble_predict(models, X):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    predictions = []

    with torch.no_grad():

        for model in models:

            model.eval()

            pred = (
                model(X_tensor)
                .squeeze(1)
                .cpu()
                .numpy()
            )

            predictions.append(pred)

    predictions = np.asarray(predictions)

    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(axis=0)

    return mean_prediction, std_prediction


pred_train, std_train = ensemble_predict(models, X_train)
pred_val, std_val = ensemble_predict(models, X_val)
pred_test, std_test = ensemble_predict(models, X_test)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        )
    }


train_metrics = calculate_metrics(
    y_train,
    pred_train
)

val_metrics = calculate_metrics(
    y_val,
    pred_val
)

test_metrics = calculate_metrics(
    y_test,
    pred_test
)


print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print("Train:", train_metrics)
print("Validation:", val_metrics)
print("Test:", test_metrics)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "UQ-polymer-Tg neural network ensemble",
    "representation": "Morgan frequency fingerprint",
    "radius": RADIUS,
    "ensemble_size": N_ENSEMBLES,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "seed": SEED,

    "n_train": len(y_train),
    "n_val": len(y_val),
    "n_test": len(y_test),

    "n_features": len(feature_list),

    "train": train_metrics,
    "validation": val_metrics,
    "test": test_metrics,
}

with open(
    "uq_tg_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


prediction_df = pd.DataFrame({
    "smiles": smiles_test,
    "actual_tg": y_test,
    "predicted_tg": pred_test,
    "uncertainty_std": std_test,
})

prediction_df.to_csv(
    "uq_tg_test_predictions.csv",
    index=False
)


print()
print("Saved:")
print("uq_tg_results.json")
print("uq_tg_test_predictions.csv")