"""
Utils Package

This package provides utility functions and classes that support various operations within the data preprocessing and model training workflows. The utilities are designed to assist with tasks such as data verification, type checking, and performance metrics.

Modules:
- `verifiers`: Contains functions for verifying the types and components of data structures to ensure consistency and correctness.
- `metrics`: Includes classes and functions for tracking and recording the performance metrics of models during training and evaluation.

Usage:
- Import utilities from the `verifiers` module to perform type checks and component verifications.
- Utilize classes and functions from the `metrics` module to track model performance and generate historical records.

Example:
```python
from utils.verifiers import verify_type, verify_components_type
from utils.metrics import History

# Example usage of verify_type
value = verify_type(10, int)  # Ensures value is of type int

# Example usage of History class
history = History()
history.add_metric('accuracy', 0.95)
"""
