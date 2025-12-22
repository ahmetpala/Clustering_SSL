# Clustering-based analysis framework for underwater acoustic data

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-2.3.5-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.7-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange)
![SciPy](https://img.shields.io/badge/SciPy-1.16.3-orange)
![TQDM](https://img.shields.io/badge/TQDM-4.67.1-orange)
![Xarray](https://img.shields.io/badge/Xarray-2025.11.0-orange)
![Zarr](https://img.shields.io/badge/Zarr-3.1.5-orange)

This repository contains code implementing a clustering-based framework for analyzing echograms, designed to identify sandeel regions using both **self-supervised learning (SSL)** features and raw data. Below is an overview of the key steps and functionality:

## Features and functionality

1. **Data preprocessing**:
   - Loads echogram data, labels, and bottom detection information from pre-defined directories.
   - Extracts multi-frequency data and applies transformations for standardization and filtering.

2. **Patch-based feature extraction**:
   - Divides echograms into patches (e.g., 8x8 pixels) for feature representation.
   - Extracts SSL-based features for each patch using the `Extraction_for_Clustering` module.

3. **Clustering**:
   - Applies k-means clustering to the extracted features and raw data to identify clusters.
   - Generates cluster maps and channels for further analysis.

4. **Cluster selection**:
   - Implements an iterative cluster selection process to optimize metrics like the F1 score for identifying sandeel regions.
   - Evaluates individual clusters and their combinations for maximizing classification performance.

5. **Evaluation**:
   - Calculates F1 score, precision, and recall for each cluster combination.
   - Compares the SSL-based clustering results with raw data clustering and supervised U-Net predictions.

6. **Visualization**:
   - Provides visualization tools for comparing raw data, SSL-based clusters, U-Net predictions, and ground-truth labels.

## Advantages
- Handles **class imbalance** through over-clustering and iterative cluster selection without altering the original data distribution.
- Offers flexibility by supporting multiple clustering methods, such as **k-means**, **DBSCAN**, and **GMM**.
- Enables detailed comparative analysis with supervised models like U-Net, demonstrating the potential of clustering-based approaches for fisheries acoustics.

## Applications
This framework can be applied to a range of fields, including:
- **Fisheries Acoustics**: Identifying sandeel and other marine species.

For additional details and replication, refer to the corresponding author.
