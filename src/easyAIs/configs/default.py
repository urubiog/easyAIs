"""
Module: easyAIs.configs.default

This module defines the default configuration settings for neural network models in the easyAIs framework. It provides a set of predefined parameters and hyperparameters that can be used as a baseline for model training and evaluation.

Functions:
    - `get_default_settings()`: Retrieves the default configuration settings for the models.

Usage:
    Users can use this module to access and utilize the default configuration settings provided by the easyAIs framework. These default settings are intended to offer a starting point for model configuration and can be adjusted according to specific needs.

    Example:
    ```python
    from easyAIs.configs import default

    # Retrieve default settings
    settings = default.get_default_settings()
    ```

Notes:
    - The `get_default_settings()` function should be implemented to return a dictionary or another data structure that contains the default configuration settings.
    - Default settings include parameters such as learning rate, batch size, number of epochs, and other common hyperparameters.
    - Users can override these default settings by defining custom configurations in the `custom.py` module if necessary.

"""

from typing import Any

# Default config for a personalized nn
config: dict[str, Any] = {
    'epochs': 100,
    'learning_rate': 0.01,
    'batch_size': 32,
    'optimizer': 'sgd',
    'seed': 123,
    'activation': 'sigmoid',
    'loss': 'mse',
}

...
