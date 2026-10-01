# Polymer Property Benchmarks

This project compares machine learning approaches for polymer property prediction.

Properties of interest:

* Glass transition temperature (Tg)
* Melting temperature (Tm)
* Density
* Thermal conductivity
* Gas permeability

Models explored include:

* Random Forest
* XGBoost
* Polymer embeddings
* Graph neural networks
* Multi-task learning models





\# Polymer Multitask Property Prediction



This repository compares machine learning and deep learning approaches for predicting polymer properties from molecular structure.
\## Properties
The current study includes:
\- Density (`rho`)
\- Glass transition temperature (`Tg`)
\- Melting temperature (`Tm`)
\- Thermal conductivity (`TC`)
\## Models
Models evaluated include:
\- PolyGraphMT
\- polyGNN
\- UQ-polymer-Tg
\- Random Forest
\- XGBoost
\- IMLforPTC-derived models
\## Molecular Representations
Representations tested include:
\- Morgan fingerprints
\- RDKit fingerprints
\- MACCS keys
\- PolyEmb polymer embeddings
\- Graph-based representations
\## Current Best Results
| Property | Model | Representation | Test R² |

|---|---|---|---:|

| Density | PolyGraphMT Multitask | Graph | 0.798 |

| Tg | XGBoost | RDKit-2048 | 0.886 |

| Tm | polyGNN | Morgan-512 / MLP | 0.769 |

| Thermal Conductivity | Random Forest | PolyEmb-300 | 0.748 |



\## Reproducibility



Density, Tg, and Tm models use a shared SMILES-based train/validation/test split:



\- Train: 6,710 unique SMILES

\- Validation: 1,438 unique SMILES

\- Test: 1,438 unique SMILES



The split is stored in:



`splits/rho\_tg\_tm.json`



Full model results are stored in:



`results/tables/combined\_polymer\_model\_results.json`



and



`results/tables/rf\_xgb\_rho\_tg\_tm\_results.csv`



\## Repository Structure



```text

data/

models/

&#x20;   classical\_ml/

&#x20;   imlforptc/

results/

&#x20;   tables/

&#x20;   figures/

splits/

docs/

README.md

requirements.txt

.gitignore

