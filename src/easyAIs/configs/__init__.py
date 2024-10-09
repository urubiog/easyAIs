"""
Configs package

This package contains configuration modules for setting up and customizing neural network models within the easyAIs framework.

Modules:
    - `default`: Contains default configuration settings for neural network models, including typical hyperparameters and architecture settings.
    - `custom`: Provides mechanisms for users to define and apply custom configuration settings, allowing for flexibility and experimentation with different model setups.

Usage:
    The configurations in this package are used to define the initial setup and parameters for neural network models. The `default` module provides a set of standard configurations, while the `custom` module allows for user-defined settings.

    Example:
    ```python
    from easyAIs.configs import default, custom

    # Access default settings
    default_settings = default.get_default_settings()

    # Define custom settings
    custom_settings = custom.get_custom_settings()
    ```

Notes:
    - `default.py` typically includes predefined values for learning rates, batch sizes, number of epochs, and other essential model parameters.
    - `custom.py` is intended for users who need to specify their own configurations, potentially overriding the defaults or adding new parameters.

"""

