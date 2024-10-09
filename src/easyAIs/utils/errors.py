"""
Module: easyAIs.utils.errors

This module defines custom error classes used throughout the application to handle and report various types of exceptions and issues. Custom errors provide clearer context and more specific information about the nature of the problem, which can improve debugging and error handling processes.

Custom Errors:
- `InvalidTypeError`: Raised when a value does not match the expected type.
- `InvalidComponentError`: Raised when a component of a data structure does not meet the expected criteria.
- `OptimizationError`: Raised for issues related to optimization processes and parameters.

Usage:
- Import custom error classes to handle specific exceptions in your code.
- Use these errors to provide detailed and informative error messages, enhancing the overall robustness and clarity of your application's error handling.

Example:
```python
from errors import InvalidTypeError, InvalidComponentError

def process_data(data):
    if not isinstance(data, list):
        raise InvalidTypeError("Expected data to be a list.")
    # Further processing...
"""
