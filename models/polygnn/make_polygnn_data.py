import pandas as pd

files = {
    "tg": r"C:\Users\santi\PolyGraphMT\data\raw\tg_exp.csv",
    "rho": r"C:\Users\santi\PolyGraphMT\data\raw\rho_exp.csv",
    "tm": r"C:\Users\santi\PolyGraphMT\data\raw\tm_exp.csv",
}

rows = []

for prop, path in files.items():
    df = pd.read_csv(path)

    smiles_col = [c for c in df.columns if c.lower() == "smiles"][0]
    value_col = [c for c in df.columns if c.lower() == prop][0]

    temp = df[[smiles_col, value_col]].copy()
    temp.columns = ["smiles_string", "value"]
    temp["prop"] = prop

    # remove missing values
    temp = temp.dropna(subset=["smiles_string", "value"])

    # average duplicate SMILES within each property
    temp = (
        temp.groupby("smiles_string", as_index=False)["value"]
        .mean()
    )

    temp["prop"] = prop

    rows.append(temp)

combined = pd.concat(rows, ignore_index=True)

combined = combined[
    ["smiles_string", "prop", "value"]
]

combined.to_csv(
    "polygnn_rho_tg_tm.csv",
    index=True
)

print(combined["prop"].value_counts())
print()
print("Total property measurements:", len(combined))
print("Unique SMILES:", combined["smiles_string"].nunique())
print()
print("Saved: polygnn_rho_tg_tm.csv")