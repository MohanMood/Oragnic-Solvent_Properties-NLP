# Oragnic-Solvent_Properties-NLP

This repository contains datasets and machine learning workflows for predicting key physicochemical properties of organic solvents—viscosity, partition coefficient (logP), and enthalpy of vaporization—using natural language processing (NLP)-derived molecular embeddings, such as Mol2Vec, ChemBERTa and compared with traditional featurization techniques Morgan fingerprints, DFT, and COSMO-RS-derived sigma profiles.

## Dataset
All datasets are provided in the "Dataset" folder for viscosity, octanol–water partition coefficient (logP), and enthalpy of vaporization

## How to Run
The prediction scripts can be executed using the following commands:

*python Mol2Vec_CATBoost-Organic-Solvents_Viscosity.py*                  # for viscosity predictions

*python Mol2Vec_CATBoost-Organic-Solvents_LogP.py*                       # for partition coefficient predictions

*python Mol2Vec_CATBoost-Organic-Solvents_Vaporization-Enthalpy.py*      # for enthalpy of vaporization predictions

Same procedure for other molecular featurization technuques (e.g., Morgan fingerprints, DFT, Sigma profiles, D-MPNN, and ChemBERTa) by modifying the scripts.

## LM-GAN Model
Please download the LM-GAN model from https://zenodo.org/records/8387351
The generated new molecular sequences (SMILES) are provided in LM-GAN folder

## instant Similarity (iSIM)
Please download it from https://github.com/mqcomplab/iSIM

## Please Cite
Mohan M, Guggilam S, Bhowmik D, Kidder MK, Smith JC. Natural Language Processing in Molecular Chemistry: Property Predictions for Organic Compounds. Green Chemistry, 2025, DOI:
