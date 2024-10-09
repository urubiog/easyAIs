"""
Module: easyAIs.arquitectures

This module provides implementations of various neural network models and their training procedures.

Classes:
    - Perceptron: A single-layer feedforward neural network, also known as a unitary layer model.
    - MLP: A Multi-Layer Perceptron model that supports a flexible and customizable network architecture.
    - RNN: Represents a Recurrent Neural Network, designed for sequence prediction tasks. (Not yet implemented)
    - CNN: Represents a Convolutional Neural Network, suited for image and spatial data processing. (Not yet implemented)
    - AutoEncoder: Represents an AutoEncoder model for unsupervised learning and data compression. (Not yet implemented)
    - NN: A flexible neural network model that serves as a base class for building various custom neural network architectures. (Not yet implemented)

Dependencies:
    - `easyAIs.utils.verifiers`: Utility functions for type checking and component verification.
    - `easyAIs.core.model`: The base `Model` class from which all models are derived.
    - `easyAIs.core.layer`: Definitions for network layers including `Layer`.
    - `easyAIs.layers`: Predefined layers such as `Dense`, `Input`, `Rec`, and `Conv`.
    - `easyAIs.core.optimizers`: Optimizers like `SGD` and `PLR` used for model training.

Usage:
    - `Perceptron`: Use this class for simple binary classification tasks. It consists of a single input layer and a single output layer.
    - `MLP`: Construct complex networks by defining a list of layers. Suitable for tasks requiring deep architectures.
    - `RNN`, `CNN`, `AutoEncoder`, `NN`: These classes are placeholders for future implementations of advanced neural network models.

Example:
    ```python
    # Example of using the Perceptron class
    perceptron = Perceptron(entries=3)
    perceptron.train(X=[0, 1, 0], Y=[1], epochs=10, learning_rate=0.01)
    prediction = perceptron([0, 1, 0])
    print(prediction)
    ```

Note:
    The RNN, CNN, AutoEncoder, and NN classes are placeholders and do not have implemented functionality in this module. 
"""

from typing import Tuple, Union, List
from easyAIs.utils.verifiers import verify_type, verify_components_type
from easyAIs.core.layer import Layer
from easyAIs.core.models import Model, Sequential
from easyAIs.layers import Dense, Input
from easyAIs.core.optimizers import Optimizer


class Perceptron(Sequential):
    """
    A single-layer feedforward neural network model, also known as a unitary layer model.

    The `Perceptron` class represents a basic neural network model with one input layer and one output layer.
    It is used for binary classification tasks and applies the Perceptron Learning Rule for training.

    Attributes:
        entries (int): The number of input features for the model.
    """

    def __init__(
        self,
        entries: int,
    ) -> None:
        """
        Initializes the Perceptron model with the specified number of input features and a single output layer.

        Args:
            entries (int): The number of input features (nodes) for the model.

        Keyword Args:
            activation (str, optional): Activation function for the output layer. Defaults to "step".
        """
        if not isinstance(entries, (float, int)):
            if not int(entries) == entries:
                raise TypeError("Expected entries to be a numeric (int) value.")

        super().__init__([Input(entries), Dense(1, activation="step")])
        self.optimizer = "plr"

    @property
    def neuron(self):
        """The neuron property."""
        return self.output_layer[0]

    def __call__(self, X: List[Union[int, float]]) -> float:
        """
        Performs a forward pass of the input data through the Perceptron model.

        Args:
            X (List[Union[int, float]]): Input data for the model.

        Returns:
            float: The output of the model after applying the forward pass.
        """
        return self.forward(X)

    def forward(self, input: List[Union[int, float]]) -> float:
        """
        Propagate input through the network and return the output of the model.

        Args:
            input (List[Union[int, float]]): Input data to be propagated through the network.

        Returns:
            List[float]: Output of the model after propagating through all layers.

        Raises:
            AssertionError: If the input size does not match the input layer's expected size.
            TypeError: If `input` is not a list of integers or floats.
        """
        verify_components_type(verify_type(input, list), (int, float))
        assert (
            len(input) == self.nx
        ), "Input size does not match the input layer expected size."

        for i, node in enumerate(self.input_layer):
            node.output = input[i]

        return self.neuron.output

    def train(
        self,
        X: List[Union[int, float]],
        Y: List[Union[int, float]],
        *,
        epochs: int = 1,
        learning_rate: Union[int, float] = 0.1,
        verbose: bool = False,
    ) -> None:
        """
        Trains the Perceptron model using the Perceptron Learning Rule.

        Args:
            X (List[Union[int, float]]): Input data for training.
            Y (List[Union[int, float]]): Target output data for training.

        Keyword Args:
            epochs (int, optional): Number of training epochs. Defaults to 1.
            learning_rate (Union[int, float], optional): Learning rate for the training. Defaults to 0.1.
            verbose (bool, optional): If True, prints training progress. Defaults to False.

        Raises:
            AssertionError: If the length of X and Y are not compatible or if X is smaller than the number of entries.
        """
        self.optimizer = self.optimizer._instanciate(learning_rate, verbose)

        epochs = verify_type(epochs, int)

        self.optimizer._fix_training_data(X, Y, self.nx, len(self.output_layer), epochs)

        self.optimizer.fit(X, Y)

        self.optimizer.optimize(self.forward, self.neuron)


class MLP(Sequential):
    """
    Represents a Multi-Layer Perceptron (MLP) model with customizable architecture.

    The `MLP` class allows for the construction of deep neural networks by specifying a list of layers.
    It supports various types of layers, such as input, dense, and activation layers, and can be used for complex tasks requiring deep architectures.

    Attributes:
        structure (List[Layer]): List of layers defining the network architecture.
    """

    def __init__(
        self,
        *structure: Union[List[Layer], Tuple[Layer], Layer],
    ) -> None:
        """
        Initializes the Multi-Layer Perceptron model with a specified network architecture.

        Args:
            structure (List[Layer]): List of layers that define the network architecture. Each layer should be an instance of Input, Dense, or similar classes.
        """
        verify_components_type(verify_type(structure, (list, tuple)), (Input, Dense))
        super().__init__(structure=structure)

    def __str__(self) -> str:
        """
        Returns a string representation of the MLP model, including layer details.

        Returns:
            str: A string representation of the model, including the layers.
        """
        return super().__str__() + f"\n{[layer for layer in self.layers]}"

    def __repr__(self) -> str:
        """
        Returns a string representation for debugging purposes.

        Returns:
            str: A detailed string representation of the model for debugging.
        """
        return super().__repr__() + f"\n{[layer for layer in self.layers]}"

    def forward(self, input: List[Union[int, float]]) -> List[float]:
        """
        Propagate input through the network and return the output of the model.

        Args:
            input (List[Union[int, float]]): Input data to be propagated through the network.

        Returns:
            List[float]: Output of the model after propagating through all layers.

        Raises:
            AssertionError: If the input size does not match the input layer's expected size.
            TypeError: If `input` is not a list of integers or floats.
        """
        verify_components_type(verify_type(input, list), (int, float))
        assert (
            len(input) == self.nx
        ), "Input size does not match the input layer expected size."

        for i, node in enumerate(self.input_layer):
            node.output = input[i]

        return [neuron.output for neuron in self.output_layer]

    def train(
        self,
        X: Union[List[int], List[float], List[Union[int, float]]],
        Y: Union[List[int], List[float], List[Union[int, float]]],
        *,
        loss: str = "mse",
        epochs: int = 10,
        optimizer: str = "sgd",
        learning_rate: Union[int, float] = 0.1,
        verbose: bool = False,
    ) -> None:
        """
        Trains the Multi-Layer Perceptron model using specified parameters.

        Args:
            X (Union[List[int], List[float], List[Union[int, float]]]): Input data for training.
            Y (Union[List[int], List[float], List[Union[int, float]]]): Target output data for training.

        Keyword Args:
            loss (str, optional): Loss function to be used during training. Defaults to "mse".
            epochs (int, optional): Number of training epochs. Defaults to 10.
            optimizer (str, optional): Optimizer to be used for training. Defaults to "sgd".
            learning_rate (Union[int, float], optional): Learning rate for the training. Defaults to 0.1.
            verbose (bool, optional): If True, prints training progress. Defaults to False.

        Returns:
            None
        """
        self.optimizer = optimizer

        # Now, the parameters to instanciate the optimizer class are given
        if not isinstance(self.optimizer, Optimizer):
            self.optimizer = self.optimizer._instanciate(
                learning_rate,
                loss,
            )

        self.loss = loss

        self.optimizer._fix_training_data(X, Y, self.nx, len(self.output_layer), epochs)

        self.optimizer.fit(X, Y)

        return self.optimizer.optimize(self.forward, self.layers)

class RNN(Model):
    """
    Represents a Recurrent Neural Network (RNN) model for sequence prediction tasks.

    The `RNN` class is designed for handling sequential data, such as time series.
    It is a placeholder for future implementations and is not yet functional.
    """

    pass


class CNN(Model):
    """
    Represents a Convolutional Neural Network (CNN) model for image and spatial data processing.

    The `CNN` class is designed for tasks involving grid-like data such as images.
    It is a placeholder for future implementations and is not yet functional.
    """

    pass


class NN(Model):
    """
    Represents a flexible neural network model for custom architectures.

    The `NN` class serves as a base class for building various custom neural network models.
    It is a placeholder for future implementations and is not yet functional.
    """

    pass
