# Imputation Matters: A Deeper Look into an Overlooked Step in Longitudinal Health and Behavior Sensing Research

This repository contains code for imputation algorithms and evaluation related to the paper "Imputation Matters: A Deeper Look into an Overlooked Step in Longitudinal Health and Behavior Sensing Research."

## Repository Contents

- **amputation.py**: Contains different strategies for introducing missing values (i.e., 'amputation') into user-level datasets
- **imputation.py**: Contains different data imputation strategies used in the paper
- **parameters.py**: Contains some meta-information to run imputation algorithms
- **utils.py**: Contains some utility functions for loading and preprocessing the data
- **example_notebook_prediction.ipynb**: Contains an example of running imputation algorithms for the prediction task
- **example_notebook_reconstruction.ipynb**: Contains an example of running imputation algorithms for the reconstruction task

## GLOBEM Datasets

For additional information on GLOBEM datasets and their pipeline, as well as instructions for downloading the datasets, please refer to the GLOBEM study and procedures at:

**https://the-globem.github.io/**

## Usage

For the prediction task, the paper uses the GLOBEM pipeline "evaluation_single_dataset_within_user" task. The example notebook has instructions on adding the imputed file to the GLOBEM pipeline and running it.

## Publication

This paper is accepted for publication in Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT) and will be presented in **UbiComp 2026**.

## Citation

To cite this paper use:

(BibTeX to be added once available)
