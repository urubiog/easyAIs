"""
Module: easyAIs.core.neural_components

This module provides the definitions for two essential components of a neural network: 
`Neuron` and `Node`. 

Classes:
- `Neuron`: Models an artificial neuron including its inputs, weights, bias, and activation function. 
  Provides methods to compute the neuron's output based on its weighted inputs and activation function.

- `Node`: Represents an input node with a fixed value. Used as a source of input within the network, supplying values to connected neurons.

Key Features:
- `Neuron` class:
  - Handles connections (inputs and weights) and applies an activation function to compute its output.
  - Supports flexible bias and weight management.
  
- `Node` class:
  - Models input nodes with fixed values that provide input to neurons.
  - Allows setting and retrieving the node's value.

The module also includes type variables and utility functions for type verification to ensure the correct usage of these components within the neural network.

Type s:
- `T`: A type variable representing either `Node` or `Neuron`, used for generic programming.
"""

from collections.abc import Callable, Iterator
from random import random
from typing import List, Optional, Self, TypeVar, Union
from easyAIs.activations import ActivationFunction, NoneFunction
from easyAIs.utils.verifiers import verify_components_type, verify_type


class Node(object):
    """
    Class representing an input node in a neural network.

    This class models a node with a fixed value that serves as an input source within the network.

    Attributes:
        output (Union[int, float]): The value of the node.
    """

    def __init__(self, value: Union[int, float] = 0) -> None:
        """
        Initializes a Node object with a specified initial value.

        Args:
            value (Union[int, float]): The initial value of the node.
        """
        self._id: int = id(self)
        self.output: Union[int, float] = (verify_type(value, (int, float)))

    def __str__(self) -> str:
        """
        Return a string representation of the node.

        Returns:
            str: A string representation of the node.
        """
        return f"Node(): {self._id}"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the node.

        Returns:
            str: A detailed string representation of the node.
        """
        return f"Node(): {self._id}"


class Neuron(object):
    """
    Class representing an artificial neuron in a neural network.

    This class models a neuron with inputs, weights, a bias, and an activation function.
    It calculates the neuron's output based on the weighted sum of its inputs and applies the activation function.

    Attributes:
        activation (Callable): The activation function for the neuron.
        inputnodes (List[Union[Node, Neuron]]): List of input nodes or neurons connected to this neuron.
        bias (float): The bias value of the neuron.
        weights (List[float]): List of weights corresponding to the input connections.
    """

    def __init__(self, activation: Optional[ActivationFunction] = None) -> None:
        """
        Initializes a Neuron object with a specified activation function.

        Args:
            activation (Callable): The activation function for the neuron.
        """
        self._id: int = id(self)
        self._inputnodes: List[Union[Node, Self]] = []
        
        self.bias: float = random()
        self._weights: List[float] = []
        self.activation: ActivationFunction = activation if activation is not None else NoneFunction
        self.n: int = 0

    @property
    def weights(self) -> List[float]:
        """The weights property."""
        return self._weights

    @weights.setter
    def weights(self, value: List[float]) -> None:
        self._weights = verify_components_type(verify_type(value, list), (int, float))

    @property
    def z(self) -> float:
        """
        Calculate the preactivated weighted sum of inputs plus bias (z-value) for this neuron.

        Returns:
            float: The z-value of the neuron.
        """
        return sum(x * w for x, w in zip(self.inputs, self.weights)) + self.bias

    @property
    def output(self) -> float:
        """
        Calculate and return the output value of the neuron after applying the activation function.

        Returns:
            float: The output value of the neuron.
        """
        return self.activation(self.z)

    @property
    def inputs(self) -> List[float]:
        """
        Return the output values of the input nodes or neurons connected to this neuron.

        Returns:
            List[float]: The input values.
        """
        return [node.output for node in self.inputnodes]

    @property
    def inputnodes(self) -> List[Union[Node, Self]]:
        """
        Return the list of input nodes or neurons connected to this neuron.

        Returns:
            List[Union[Node, Neuron]]: The list of input nodes or neurons.
        """
        return self._inputnodes

    @inputnodes.setter
    def inputnodes(self, value: List[Union[Node, Self]]) -> None:
        """
        Set the list of input nodes or neurons connected to this neuron and initialize weights.

        Args:
            value (List[Union[Node, Neuron]]): The new list of input nodes or neurons.
        """
        self._inputnodes = verify_components_type(
            verify_type(value, list), (Node, Neuron)
        )
        
        # Give new values
        self.weights = [random() for _ in self._inputnodes]
        self.n = len(self._inputnodes)

    def __str__(self) -> str:
        """
        Return a string representation of the neuron.

        Returns:
            str: A string representation of the neuron.
        """
        return f"Neuron(): {self._id}"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the neuron.

        Returns:
            str: A detailed string representation of the neuron.
        """
        return f"Neuron(): {self._id}"

    def __hash__(self) -> int:
        """
        Rerturn the hash for the neuron performing the hash() function.

        The hash value is computed based on the neuron's weights and bias.
        This ensures that two neurons with identical weights and bias
        will have the same hash value, allowing the object to be used
        correctly in hash-based collections such as sets and dictionaries.

        Returns:
            int: The hash value representing the Nueron instance.
        """
        return hash((tuple(self.weights), self.bias))

    def __eq__(self, value: object, /) -> bool:
        """
        Determine if two Neuron instances are equal.

        This method compares the current Neuron instance with another object
        to determine equality. Two Neurons are considered equal if their
        hash values, which are based on their weights and bias, are identical.

        Args:
            value (object): The object to compare with the current Neuron instance.

        Returns:
            bool: True if the objects are considered equal, False otherwise.
        """
        if not hasattr(value, "__hash__"):
            return False

        return self.__hash__() == value.__hash__()

    def __iter__(self) -> Iterator[Union[Node, "Neuron"]]:
        """
        Initialize iteration over the inputnodes in the Neuron.

        Returns:
            Iterator[Union[Node, Neuron]]: An iterator over the inputnodes.
        """
        self._iter_index: int = 0
        return self

    def __next__(self) -> Union[Node, "Neuron"]:
        """
        Return the next neuron or node in the itertion.

        Returns:
            Union[Node, Neuron]: The next neuron or node in the inputnodes array.

        Raises:
            StopIteration: If there are no more neurons or nodes to iterate.
        """
        if self._iter_index < self.n:
            result = self.inputnodes[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration

    def __len__(self) -> int:
        """
        Return the amount of inputs for the Neuron.

        Returns:
            int: The total inputs amount.
        """
        return self.n


class RecNeuron(Neuron):
    """
    Class representing a recurrent neuron in a neural netwrok.

    This class serves it's porpouse for RNN networks, which are used for value iteration I/O.

    Attributes:
        activation (Callable): The activation function for the neuron.
        inputnodes (List[Union[Node, Neuron]]): List of input nodes or neurons connected to this neuron.
        bias (float): The bias value of the neuron.
        weights (List[float]): List of weights corresponding to the input connections.
    """

    def __init__(self, activation: Callable, iterations: int) -> None:
        raise NotImplemented
        super().__init__(activation)

        if verify_type(iterations, int) < 0:
            raise ValueError("Expected iterations to be greater than 0.")

        self._iterations = iterations
        self._saved_output

    # @property
    # def iterations(self) -> int:
    #     """The iterations property."""
    #     return self._iterations

    # @iterations.setter
    # def iterations(self, value: int) -> None:
    #     self._iterations = verify_type(value, int)

    # @property
    # def saved_output(self) -> Union[int, float]:
    #     """The saved_output property."""
    #     return self._saved_output

    # @saved_output.setter
    # def saved_output(self, value: Union[int, float]) -> None:
    #     self._saved_output = verify_type(value, (int, float))

    # @property
    # def z(self) -> float:
    #     raise NotImplemented
    #     """
    #     Calculate the weighted sum of inputs plus bias (z-value) for this neuron.

    #     Returns:
    #         float: The z-value of the neuron.
    #     """
    #     if self._iterations < 1:
    #         return sum(x * w for x, w in zip(self.inputs, self.weights)) + self.bias

    #     # Add here the logic for recurrence

Ntype = TypeVar("Ntype", Node, Neuron)

if __name__ == "__main__":
    pass
