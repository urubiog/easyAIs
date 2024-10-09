"""
Module: easyAIs.layers

This module defines various types of layers used in neural network models. Each layer represents a different
component of the network architecture, supporting various types of operations and activations.

Classes:
    - Input: Represents the input layer of the network. Each node in this layer corresponds to an input feature.
    - Dense: Represents a fully connected layer where each neuron is connected to every neuron in the previous layer.
    - Conv: Represents a convolutional layer, typically used for processing grid-like data such as images.
    - Rec: Represents a recurrent layer, designed for handling sequential data such as time series.

Dependencies:
    - `easyAIs.core.layer`: Base `Layer` class providing fundamental layer functionality.
    - `easyAIs.core.neural_components`: Definitions for neural components like `Node`.

Layer Descriptions:
    - `Input`: Initializes with a specified number of nodes. Each node represents an input feature, and the layer does not apply an activation function.
    - `Dense`: Initializes with a specified number of neurons and an optional activation function. Implements a fully connected layer where each neuron is connected to every neuron in the previous layer.
    - `Conv`: Initializes a convolutional layer for processing spatial data with optional activation. Intended to support convolution operations, useful for image and spatial data processing.
    - `Rec`: Initializes a recurrent layer for processing sequential data with optional activation. Suitable for tasks involving time series or sequences.

Usage:
    ```python
    # Example of using the Dense layer
    dense_layer = Dense(n=128, activation="relu")
    ```

Notes:
    - The `Conv` and `Rec` classes are designed to be used for convolutional and recurrent operations, respectively, and will be implemented to handle specific data processing tasks.
    - The `Input` layer serves as the entry point for data into the neural network and does not apply any activation functions, as its purpose is to handle raw input data.

"""

from typing import List, Union
from easyAIs.activations import ActivationFunction
from easyAIs.core.layer import Layer
from easyAIs.core.neural_components import Node


class Input(Layer):
    """
    Represents the input layer of a neural network.

    The `Input` layer initializes with a specified number of nodes, each corresponding to an input feature of the network.
    This layer serves as the entry point for raw data and does not apply any activation functions.
    """

    def __init__(self, n: int) -> None:
        """
        Initializes the `Input` layer with the given number of nodes.

        Args:
            n (int): Number of input nodes (features) in the layer.

        This initialization sets the activation function to "step" by default, which is then removed after setting up the layer.
        """
        super().__init__(n, activation_func="none")
        self._structure = [Node() for _ in range(self.n)]

        super()._set_indexes()


class Dense(Layer):
    """
    Represents a fully connected layer in a neural network.

    The `Dense` layer initializes with a specified number of neurons and supports an optional activation function.
    Each neuron in this layer is connected to every neuron in the previous layer, implementing a fully connected architecture.
    """

    def __init__(self, n: int, *, activation: Union[str, ActivationFunction] = "relu") -> None:
        """
        Initializes the `Dense` layer with the specified number of neurons and activation function.

        Args:
            n (int): Number of neurons in the layer.
            activation (str, optional): Activation function to be applied. Defaults to "relu".
        """
        super().__init__(n, activation_func=activation)

    @property
    def weights(self) -> List[List[Union[int, float]]]:
        """The weights property."""
        return [neuron.weights for neuron in self]

    @property
    def _ponderated_weights(self):
        """The ponderated_weights property."""
        return [neuron.z for neuron in self]

class Conv(Layer):
    """
    Represents a convolutional layer in a neural network.

    The `Conv` layer is designed for processing spatial data, such as images. It supports convolution operations and applies an optional activation function.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the `Conv` layer with the specified number of filters and activation function.

        Raises:
            NotImplementedError: This class is not yet implemented and will raise an exception if instantiated.
        """
        raise NotImplemented


class Rec(Layer):
    """
    Represents a recurrent layer in a neural network.

    The `Rec` layer is designed for handling sequential data, such as time series. It supports optional activation functions and processes sequences of data.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the `Rec` layer with the specified number of units and activation function.

        Raises:
            NotImplementedError: This class is not yet implemented and will raise an exception if instantiated.
        """
        raise NotImplemented
