# Polymer Property Prediction Model Landscape

## Comparison Table

| Model | Property focus | Representation | Model type | Main advantage | GitHub |
|---|---|---|---|---|---|
| UQ-polymer-Tg | Tg | Morgan fingerprint with frequency counts | Neural network ensemble, GPR, MCD, MVE, BNN, EDL | Predicts Tg with uncertainty | https://github.com/huangxiang701/UQ-polymer-Tg |
| IMLforPTC | Thermal conductivity | Physical descriptors, Morgan, MACCS, Mol2vec | Traditional ML / Random Forest | Interpretable and closely related to prior fingerprint-based work | https://github.com/SJTU-MI/IMLforPTC |
| PolymerGasMembraneML | Gas permeability | Morgan fingerprints / chemical descriptors | DNN ensemble / Random Forest | Pretrained models; predicts He, H2, O2, N2, CO2, and CH4 permeability | https://github.com/jsunn-y/PolymerGasMembraneML |
| polyBERT | Multiple polymer properties | Learned PSMILES embedding | Pretrained Transformer + multitask property prediction | Large pretrained polymer representation | https://github.com/Ramprasad-Group/polyBERT |
| polyGNN | Multiple polymer properties | Molecular graph | Multitask GNN | Learns several polymer properties together | https://github.com/rishigurnani/polygnn_trainer |
| PolyGraphMT | Multiple polymer properties | Molecular graph | Multitask, multi-fidelity GNN | Learns multiple properties and multiple data fidelities | https://github.com/sobinalosious/PolyGraphMT |

## Model Categories

### Property-Specific Models

**UQ-polymer-Tg**
- Target: Tg
- Input: Morgan fingerprints with frequency information
- Models: neural-network ensemble, Gaussian process regression, Monte Carlo dropout, mean-variance estimation, Bayesian neural network, and evidential deep learning
- Main contribution: uncertainty quantification for Tg predictions

**IMLforPTC**
- Target: thermal conductivity
- Inputs: engineered physical descriptors, Morgan fingerprints, MACCS keys, Mol2vec, and other reduced representations
- Includes trained models, feature-engineering notebooks, SHAP analysis, and screening workflows
- Especially useful as a baseline because it is similar to a fingerprint/descriptor + traditional ML pipeline

**PolymerGasMembraneML**
- Target: gas permeability
- Gases: He, H2, O2, N2, CO2, CH4
- Inputs: Morgan fingerprints or chemical descriptors
- Models: DNN ensemble or Random Forest
- Includes pretrained models and screening datasets

### General Polymer Models

**polyBERT**
- Treats polymer PSMILES as a chemical language
- Uses a pretrained Transformer to generate polymer fingerprints
- Designed for fast property prediction across many polymer properties
- Best comparison to learned embedding approaches

**polyGNN**
- Uses polymer molecular graphs
- Supports multitask learning
- Shared GNN representation can be used for several properties at once
- Useful comparison to single-task GNN models

**PolyGraphMT**
- Uses RDKit molecular graphs with GINE, GIN, or GCN backbones
- Supports multitask and multi-fidelity learning
- Can combine experimental, MD, DFT, and other fidelity levels
- Includes released checkpoints, training scripts, inference tools, metrics, and plotting workflows


The newer model families add:
1. **Pretraining** — learn a general polymer representation before fitting a property model.
2. **Multitask learning** — train one model across several properties.
3. **Multi-fidelity learning** — combine data from different sources such as experiment, MD, and DFT.

## Initial Benchmarking Plan

A useful first comparison would be:

1. Previous PolyEmb + Random Forest baseline
2. Property-specific traditional ML baseline such as IMLforPTC
3. Pretrained representation model such as polyBERT
4. Multitask graph model such as polyGNN
5. Multitask, multi-fidelity graph model such as PolyGraphMT
