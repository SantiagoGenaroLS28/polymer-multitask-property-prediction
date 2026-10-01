\# External Model Repositories



This project compares and adapts several open-source polymer property prediction workflows.



\## PolyGraphMT



Repository: https://github.com/sobinalosious/PolyGraphMT



Used for single-task and multitask prediction of density, glass transition temperature, and melting temperature.



The repository was adapted to use a common train/validation/test SMILES split so results could be compared directly with other models.



\## polyGNN



Repository: https://github.com/rishigurnani/polygnn\_trainer



Used for multitask prediction of density, glass transition temperature, and melting temperature.



The tested workflow uses Morgan fingerprints as input to the repository's neural-network training framework.



\## UQ-polymer-Tg



Repository: https://github.com/huangxiang701/UQ-polymer-Tg



Adapted for glass transition temperature prediction using an ensemble neural-network approach and Morgan-count fingerprints.



\## IMLforPTC



Repository: https://github.com/SJTU-MI/IMLforPTC



Used to investigate Morgan and Morgan-count fingerprint representations for thermal conductivity prediction.



\## Classical Machine Learning



Random Forest and XGBoost models were implemented for density, Tg, and Tm using:



\- Morgan-2048 fingerprints

\- RDKit-2048 fingerprints

\- MACCS-167 keys

\- PolyEmb-300 polymer embeddings



These models use the same master SMILES split as the PolyGraphMT experiments.

