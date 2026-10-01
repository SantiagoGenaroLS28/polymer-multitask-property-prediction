import os
import warnings
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, MACCSkeys
from rdkit.Chem.rdmolops import RDKFingerprint

from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence


# ============================================================
# PATHS
# ============================================================

DATA_DIR = r"C:\Users\santi\PolyGraphMT\data\raw"

FILES = {
    "rho": os.path.join(DATA_DIR, "rho_exp.csv"),
    "tg":  os.path.join(DATA_DIR, "tg_exp.csv"),
    "tm":  os.path.join(DATA_DIR, "tm_exp.csv"),
}

# CHANGE THIS to where POLYINFO_PI1M.pkl actually is
EMBED_MODEL_PATH = r"C:\Users\santi\PolyGraphMT\classical_models\POLYINFO_PI1M.pkl"

OUTDIR = "features_rho_tg_tm"
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RDK_FP_SIZE = 2048
RDK_MAX_PATH = 7

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# ============================================================
# HELPERS
# ============================================================

def find_smiles_column(df):
    for col in df.columns:
        if col.strip().lower() in ["smiles", "smile"]:
            return col
    raise ValueError(f"No SMILES column found. Columns: {df.columns.tolist()}")


def smiles_to_embeddings(smiles_list, model, unseen="UNK"):

    kv = model.wv
    vec_size = kv.vector_size
    zero_vec = np.zeros(vec_size, dtype=np.float32)

    embeddings = []

    for smi in smiles_list:

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            warnings.warn(f"Invalid SMILES: {smi}")
            embeddings.append(zero_vec.copy())
            continue

        tokens = mol2alt_sentence(mol, 1)

        vecs = []

        for word in tokens:

            if word in kv.key_to_index:
                vecs.append(kv[word])

            elif unseen in kv.key_to_index:
                vecs.append(kv[unseen])

            else:
                vecs.append(zero_vec)

        if len(vecs) > 0:
            emb = np.mean(vecs, axis=0)
        else:
            emb = zero_vec

        embeddings.append(
            emb.astype(np.float32)
        )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


def smiles_to_rdkfp(smiles_list):

    fps = []

    for smi in smiles_list:

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            fps.append(
                np.zeros(
                    RDK_FP_SIZE,
                    dtype=np.int8
                )
            )
            continue

        bv = RDKFingerprint(
            mol,
            fpSize=RDK_FP_SIZE,
            maxPath=RDK_MAX_PATH
        )

        arr = np.zeros(
            RDK_FP_SIZE,
            dtype=np.int8
        )

        DataStructs.ConvertToNumpyArray(
            bv,
            arr
        )

        fps.append(arr)

    return np.asarray(
        fps,
        dtype=np.int8
    )


def smiles_to_maccs(smiles_list):

    fps = []

    for smi in smiles_list:

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            fps.append(
                np.zeros(167, dtype=np.int8)
            )
            continue

        bv = MACCSkeys.GenMACCSKeys(mol)

        arr = np.zeros(
            bv.GetNumBits(),
            dtype=np.int8
        )

        DataStructs.ConvertToNumpyArray(
            bv,
            arr
        )

        fps.append(arr)

    return np.asarray(
        fps,
        dtype=np.int8
    )


def smiles_to_morgan(smiles_list):

    fps = []

    for smi in smiles_list:

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            fps.append(
                np.zeros(
                    MORGAN_BITS,
                    dtype=np.int8
                )
            )
            continue

        bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            MORGAN_RADIUS,
            nBits=MORGAN_BITS
        )

        arr = np.zeros(
            MORGAN_BITS,
            dtype=np.int8
        )

        DataStructs.ConvertToNumpyArray(
            bv,
            arr
        )

        fps.append(arr)

    return np.asarray(
        fps,
        dtype=np.int8
    )


# ============================================================
# COLLECT ALL UNIQUE SMILES
# ============================================================

all_smiles = []

for prop, path in FILES.items():

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    smi_col = find_smiles_column(df)

    smiles = (
        df[smi_col]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(
        prop,
        "rows:",
        len(df),
        "SMILES:",
        len(smiles)
    )

    all_smiles.extend(smiles)


# Preserve first appearance order
unique_smiles = list(dict.fromkeys(all_smiles))

print()
print("Total unique SMILES:", len(unique_smiles))


# ============================================================
# LOAD POLYMER EMBEDDING
# ============================================================

print()
print("Loading PolyEmb model...")

polyemb_model = word2vec.Word2Vec.load(
    EMBED_MODEL_PATH
)

print(
    "Embedding dimension:",
    polyemb_model.wv.vector_size
)


# ============================================================
# GENERATE FEATURES
# ============================================================

print("\nGenerating Morgan...")
X_morgan = smiles_to_morgan(
    unique_smiles
)

print("Generating RDKit...")
X_rdkit = smiles_to_rdkfp(
    unique_smiles
)

print("Generating MACCS...")
X_maccs = smiles_to_maccs(
    unique_smiles
)

print("Generating PolyEmb...")
X_polyemb = smiles_to_embeddings(
    unique_smiles,
    polyemb_model
)


# ============================================================
# SAVE
# ============================================================

np.save(
    os.path.join(
        OUTDIR,
        "X_morgan_2048.npy"
    ),
    X_morgan
)

np.save(
    os.path.join(
        OUTDIR,
        "X_rdkit_2048.npy"
    ),
    X_rdkit
)

np.save(
    os.path.join(
        OUTDIR,
        "X_maccs_167.npy"
    ),
    X_maccs
)

np.save(
    os.path.join(
        OUTDIR,
        "X_polyemb_300.npy"
    ),
    X_polyemb
)

rows = pd.DataFrame({
    "SMILES": unique_smiles
})

rows.to_csv(
    os.path.join(
        OUTDIR,
        "rows_used.csv"
    ),
    index=False
)

print()
print("Saved features.")
print("Morgan:", X_morgan.shape)
print("RDKit:", X_rdkit.shape)
print("MACCS:", X_maccs.shape)
print("PolyEmb:", X_polyemb.shape)