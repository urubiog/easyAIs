"""
Module: easyAIs.core

The `easyAIs.core` module provides the foundational components and abstractions for building and training neural network models within the easyAIs framework. It encompasses core functionalities, including model structures, layers, loss functions, metrics, activation functions, and optimizers.

Submodules:
    - `activations`: Contains definitions for various activation functions used in neural network layers.
    - `layer`: Defines the base `Layer` class and its derived classes representing different types of layers in neural networks.
    - `loss`: Provides implementations of common loss functions used to evaluate and train models.
    - `metrics`: Includes various performance metrics for assessing the accuracy and effectiveness of models.
    - `model`: Defines the base `Model` class and its derived classes for different types of neural network architectures.
    - `neural_components`: Contains fundamental components such as nodes and neurons that are used in constructing neural networks.
    - `optimizers`: Implements optimization algorithms to update model parameters during training.

Usage:
    Import the required components from the submodules to build and configure neural network models, define custom layers, loss functions, metrics, and optimizers, and perform training and evaluation.

    Example:
    ```python
    from easyAIs.core.model import Model
    from easyAIs.core.layers import Dense, Input
    from easyAIs.loss import MeanSquaredError
    from easyAIs.core.metrics import Accuracy
    from easyAIs.core.optimizers import SGD

    # Define a model
    model = Model(layers=[Input(10), Dense(5, activation="relu")])
    
    # Specify a loss function
    loss_function = MeanSquaredError()
    
    # Define an optimizer
    optimizer = SGD(model, learning_rate=0.01, epochs=10)
    ```

Notes:
    - Each submodule in the `easyAIs.core` package is designed to be modular and reusable, allowing for flexibility in building and training various neural network architectures.
    - Refer to the individual submodules for more detailed documentation on the specific classes and functions they provide.

"""
