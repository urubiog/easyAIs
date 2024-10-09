"""
Preprocessing Package

This package provides various tools and functions for preprocessing data in machine learning and data analysis tasks. The preprocessing steps are crucial for preparing raw data for analysis, model training, and evaluation. This package includes modules for different preprocessing operations, including dataset management and data transformation.

Subpackages:
- `datasets`: Contains predefined datasets that can be used for testing, experimentation, and benchmarks. Each dataset is provided in a format suitable for use in preprocessing and modeling scenarios.

Modules:
- `cleaning`: Includes functions and classes for cleaning and sanitizing data, such as handling missing values, removing duplicates, and standardizing data formats.
- `transformation`: Provides tools for transforming data, including normalization, scaling, encoding categorical variables, and feature extraction.
- `feature_selection`: Contains methods for selecting relevant features from datasets, including techniques for dimensionality reduction and feature importance evaluation.
- `augmentation`: Includes methods for augmenting datasets, particularly useful in machine learning for increasing the diversity of training data through techniques such as oversampling and synthetic data generation.
- `loaders`: Provides functions and classes for loading data from various sources, including file systems, databases, and online repositories. This module supports different data formats and provides utilities for easy integration into preprocessing workflows.

Usage:
- Import specific preprocessing functions or classes from the relevant module as needed.
- For dataset-related tasks, use the `datasets` subpackage to access and load predefined datasets.
- Utilize other modules for data cleaning, transformation, feature selection, augmentation, and data loading.

Example:
```python
from preprocessing.cleaning import remove_duplicates
from preprocessing.transformation import normalize_data
from preprocessing.loaders import load_csv
from preprocessing.datasets import dataset1

# Load dataset
data = dataset1.load()

# Clean and transform data
clean_data = remove_duplicates(data)
normalized_data = normalize_data(clean_data)

# Load additional data from a CSV file
additional_data = load_csv('path/to/file.csv')
"""
