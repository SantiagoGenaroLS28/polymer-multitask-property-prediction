import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\santi\PolyGraphMT"

DATA_DIR = os.path.join(
    BASE,
    "data",
    "raw"
)

FEATURE_DIR = os.path.join(
    BASE,
    "classical_models",
    "features_rho_tg_tm"
)

SPLIT_FILE = os.path.join(
    BASE,
    "splits",
    "rho_tg_tm.json"
)

OUTDIR = os.path.join(
    BASE,
    "classical_models",
    "results"
)

os.makedirs(
    OUTDIR,
    exist_ok=True
)


PROPERTY_FILES = {
    "rho": os.path.join(DATA_DIR, "rho_exp.csv"),
    "tg": os.path.join(DATA_DIR, "tg_exp.csv"),
    "tm": os.path.join(DATA_DIR, "tm_exp.csv"),
}


FEATURE_FILES = {
    "Morgan_2048": os.path.join(
        FEATURE_DIR,
        "X_morgan_2048.npy"
    ),

    "RDKit_2048": os.path.join(
        FEATURE_DIR,
        "X_rdkit_2048.npy"
    ),

    "MACCS_167": os.path.join(
        FEATURE_DIR,
        "X_maccs_167.npy"
    ),

    "PolyEmb_300": os.path.join(
        FEATURE_DIR,
        "X_polyemb_300.npy"
    ),
}


ROWS_FILE = os.path.join(
    FEATURE_DIR,
    "rows_used.csv"
)

SEED = 42


# ============================================================
# HELPERS
# ============================================================

def find_smiles_column(df):

    for col in df.columns:

        if col.strip().lower() in [
            "smiles",
            "smile"
        ]:

            return col

    raise ValueError(
        "Could not find SMILES column."
    )


def find_target_column(df, prop):

    possibilities = [
        prop,
        prop.upper(),
        prop.lower(),
        "value",
        "Value",
        "y",
        "target"
    ]

    for col in possibilities:

        if col in df.columns:
            return col

    # Density sometimes has descriptive name
    if prop == "rho":

        for col in df.columns:

            if "density" in col.lower():
                return col

    raise ValueError(
        f"Could not identify target for {prop}. "
        f"Columns: {df.columns.tolist()}"
    )


def metrics(y_true, y_pred):

    return {
        "R2": r2_score(
            y_true,
            y_pred
        ),

        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    }


# ============================================================
# LOAD MASTER FEATURE SMILES
# ============================================================

rows = pd.read_csv(
    ROWS_FILE
)

master_smiles = rows[
    "SMILES"
].astype(str).tolist()

smiles_to_index = {
    smi: i
    for i, smi in enumerate(master_smiles)
}

print(
    "Master feature polymers:",
    len(master_smiles)
)


# ============================================================
# LOAD SPLIT
# ============================================================

with open(
    SPLIT_FILE,
    "r"
) as f:

    split = json.load(f)


train_set = set(
    split["train_smiles"]
)

val_set = set(
    split["val_smiles"]
)

test_set = set(
    split["test_smiles"]
)


print(
    "Split:",
    len(train_set),
    len(val_set),
    len(test_set)
)


# ============================================================
# LOAD FEATURES
# ============================================================

feature_arrays = {}

for name, path in FEATURE_FILES.items():

    X = np.load(path)

    if len(X) != len(master_smiles):

        raise ValueError(
            f"{name}: {len(X)} feature rows "
            f"but {len(master_smiles)} SMILES"
        )

    feature_arrays[name] = X


# ============================================================
# RUN EACH PROPERTY
# ============================================================

all_results = []


for prop, path in PROPERTY_FILES.items():

    print()
    print("=" * 70)
    print("PROPERTY:", prop)
    print("=" * 70)

    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    smi_col = find_smiles_column(df)
    target_col = find_target_column(
        df,
        prop
    )

    # Average duplicate SMILES
    temp = (
        df[[smi_col, target_col]]
        .dropna()
        .groupby(
            smi_col,
            as_index=False
        )[target_col]
        .mean()
    )

    temp.columns = [
        "SMILES",
        "target"
    ]

    # Only SMILES with generated features
    temp = temp[
        temp["SMILES"].isin(
            smiles_to_index
        )
    ].copy()

    temp["feature_index"] = (
        temp["SMILES"]
        .map(smiles_to_index)
    )

    train_df = temp[
        temp["SMILES"].isin(
            train_set
        )
    ].copy()

    val_df = temp[
        temp["SMILES"].isin(
            val_set
        )
    ].copy()

    test_df = temp[
        temp["SMILES"].isin(
            test_set
        )
    ].copy()


    print(
        "Train:",
        len(train_df),
        "Val:",
        len(val_df),
        "Test:",
        len(test_df)
    )


    y_train = (
        train_df["target"]
        .values
        .astype(float)
    )

    y_val = (
        val_df["target"]
        .values
        .astype(float)
    )

    y_test = (
        test_df["target"]
        .values
        .astype(float)
    )


    train_idx = (
        train_df["feature_index"]
        .values
        .astype(int)
    )

    val_idx = (
        val_df["feature_index"]
        .values
        .astype(int)
    )

    test_idx = (
        test_df["feature_index"]
        .values
        .astype(int)
    )


    # ========================================================
    # EACH REPRESENTATION
    # ========================================================

    for representation, X_all in feature_arrays.items():

        print()
        print("Representation:", representation)

        X_train = X_all[
            train_idx
        ]

        X_val = X_all[
            val_idx
        ]

        X_test = X_all[
            test_idx
        ]


        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        rf = RandomForestRegressor(
            n_estimators=500,
            random_state=SEED,
            n_jobs=-1
        )

        rf.fit(
            X_train,
            y_train
        )

        pred_val = rf.predict(
            X_val
        )

        pred_test = rf.predict(
            X_test
        )

        val_metrics = metrics(
            y_val,
            pred_val
        )

        test_metrics = metrics(
            y_test,
            pred_test
        )


        print(
            "RF Test:",
            test_metrics
        )


        all_results.append({
            "Property": prop,
            "Model": "RF",
            "Representation": representation,

            "N_train": len(y_train),
            "N_val": len(y_val),
            "N_test": len(y_test),

            "Val_R2": val_metrics["R2"],
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],

            "Test_R2": test_metrics["R2"],
            "Test_MAE": test_metrics["MAE"],
            "Test_RMSE": test_metrics["RMSE"]
        })


        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        xgb = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1
        )

        xgb.fit(
            X_train,
            y_train
        )

        pred_val = xgb.predict(
            X_val
        )

        pred_test = xgb.predict(
            X_test
        )

        val_metrics = metrics(
            y_val,
            pred_val
        )

        test_metrics = metrics(
            y_test,
            pred_test
        )


        print(
            "XGB Test:",
            test_metrics
        )


        all_results.append({
            "Property": prop,
            "Model": "XGB",
            "Representation": representation,

            "N_train": len(y_train),
            "N_val": len(y_val),
            "N_test": len(y_test),

            "Val_R2": val_metrics["R2"],
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],

            "Test_R2": test_metrics["R2"],
            "Test_MAE": test_metrics["MAE"],
            "Test_RMSE": test_metrics["RMSE"]
        })


# ============================================================
# SAVE SUMMARY
# ============================================================

results = pd.DataFrame(
    all_results
)

results = results.sort_values(
    [
        "Property",
        "Test_RMSE"
    ]
)

output_file = os.path.join(
    OUTDIR,
    "rf_xgb_rho_tg_tm_results.csv"
)

results.to_csv(
    output_file,
    index=False
)

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)

print()
print(
    "Saved:",
    output_file
)