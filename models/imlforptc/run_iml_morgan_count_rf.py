import os
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_FILE = r"C:\Users\santi\OneDrive\Desktop\polymers.csv"
ROWS_USED_CSV = r"C:\Users\santi\OneDrive\Desktop\Santiago Research Semester 2\Reseach Semester 2\features_out\rows_used.csv"

SMILES_COL = "SMILES"
TARGET_COL = "TC"

RADIUS = 2
N_BITS = 2048
SEED = 42
TEST_SIZE = 0.2


# --------------------------------------------------
# Load targets and preserve OLD row order
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)
rows_used = pd.read_csv(ROWS_USED_CSV)

df.columns = df.columns.str.strip()
rows_used.columns = rows_used.columns.str.strip()

df_target = (
    df.groupby(SMILES_COL, as_index=False)[TARGET_COL]
    .mean()
)

# Critical: preserve rows_used ordering
aligned = rows_used[[SMILES_COL]].merge(
    df_target,
    on=SMILES_COL,
    how="left",
    sort=False
)

if aligned[TARGET_COL].isna().any():
    print("Warning: unmatched SMILES found")
    print(aligned.loc[aligned[TARGET_COL].isna(), [SMILES_COL]].head())

aligned = aligned.dropna(subset=[TARGET_COL]).reset_index(drop=True)

print("Aligned polymers:", len(aligned))


# --------------------------------------------------
# Morgan COUNT fingerprints
# --------------------------------------------------

X = []
y = []
smiles_used = []

for _, row in aligned.iterrows():

    smi = row[SMILES_COL]
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        print("Invalid SMILES:", smi)
        continue

    fp = AllChem.GetHashedMorganFingerprint(
        mol,
        radius=RADIUS,
        nBits=N_BITS
    )

    arr = np.zeros(N_BITS, dtype=np.float32)

    for idx, count in fp.GetNonzeroElements().items():
        arr[idx] = count

    X.append(arr)
    y.append(row[TARGET_COL])
    smiles_used.append(smi)


X = np.asarray(X)
y = np.asarray(y)

print("Valid polymers:", len(y))
print("Fingerprint shape:", X.shape)


# --------------------------------------------------
# Exact same split logic as old model
# --------------------------------------------------

X_train, X_test, y_train, y_test, smi_train, smi_test = train_test_split(
    X,
    y,
    smiles_used,
    test_size=TEST_SIZE,
    random_state=SEED
)


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=500,
    random_state=SEED,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


def metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred))
    }


print()
print("TRAIN RESULTS")
print(metrics(y_train, pred_train))

print()
print("TEST RESULTS")
print(metrics(y_test, pred_test))


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

out = pd.DataFrame({
    "SMILES": smi_test,
    "actual_TC": y_test,
    "predicted_TC": pred_test
})

out.to_csv(
    "iml_morgan_count_same_split_predictions.csv",
    index=False
)

print()
print("Saved: iml_morgan_count_same_split_predictions.csv")