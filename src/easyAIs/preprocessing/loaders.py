"""
Loaders Module

This module provides functions and a class for loading data from various file formats, including CSV, JSON, and Excel. It is designed to streamline the process of integrating external data sources into preprocessing workflows.

Functions:
- `load_csv(filepath: str) -> pd.DataFrame`: Loads data from a CSV file into a Pandas DataFrame.
- `load_json(filepath: str) -> Dict[str, Any]`: Loads data from a JSON file into a dictionary.
- `load_excel(filepath: str, sheet_name: str = 0) -> pd.DataFrame`: Loads data from an Excel file into a Pandas DataFrame. Allows specifying the sheet to load.

Classes:
- `DataLoader`: A flexible class for loading data from various formats. Supports CSV, JSON, and Excel files. Provides customization and additional handling options for different data sources.

Usage:
- Utilize `load_csv`, `load_json`, and `load_excel` functions for straightforward data loading from specific formats.
- Use the `DataLoader` class for more advanced scenarios and when handling multiple data formats with a unified interface.

Example:
```python
from preprocessing.loaders import load_csv, DataLoader

# Load data from a CSV file
data_csv = load_csv('path/to/file.csv')

# Load data from a JSON file
data_json = load_json('path/to/file.json')

# Load data from an Excel file
data_excel = load_excel('path/to/file.xlsx', sheet_name='Sheet1')

# Using DataLoader for more advanced data loading
data_loader = DataLoader('path/to/file.xlsx', format='excel')
data_loaded = data_loader.load()
"""

def load_data(path: str):
    """
    Docstring.
    """
    pass

