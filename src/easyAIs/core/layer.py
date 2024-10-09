"""
Module: easyAIs.core.layer

This module defines the `Layer` class, which represents an abstract layer within a neural network. 
A `Layer` can be composed of neurons or nodes and includes methods to manage and manipulate these components. 

The `Layer` class provides functionalities to:
- Initialize a layer with a specified number of neurons or nodes and an activation function.
- Access and modify the number of neurons or nodes, the activation function, and the individual neurons or nodes.
- Iterate over the neurons or nodes.
- Add or remove neurons or nodes.
- Handle indexing operations to access specific neurons or nodes.

Key Components:
- `Layer[T]`: A generic class for layers containing either neurons or nodes, with methods for manipulation and access.
- `T`: A type variable representing either `Node` or `Neuron`, allowing the `Layer` class to handle different types of components.

The module ensures type safety and verifies the validity of the layer's structure through dedicated verification functions.
"""

from collections.abc import Callable
from typing import Iterator, List, Optional, Generic, Union
from easyAIs.core.neural_components import Neuron, Node, Ntype
from easyAIs.activations import ActivationFunction, activation_map
from easyAIs.utils.verifiers import verify_components_type, verify_type


class Layer(Generic[Ntype]):
    """
    Represents an abstract layer of neurons or nodes in a neural network.

    This class manages a collection of neurons or nodes and provides methods to manipulate and access them.
    It supports iteration, indexing, and modification of the layer's structure.

    Attributes:
        activation (Callable): The activation function applied to the neurons or nodes.
        structure (List[Ntype]): The collection of neurons or nodes in the layer.
    """

    def __init__(
        self,
        nodes: Union[int, List[Ntype]],
        activation_func: Union[str, ActivationFunction] = "none",
    ) -> None:
        """
        Initialize a Layer object with a specified number of neurons or nodes and an activation function.

        Args:
            nodes (Union[int, List[Ntype]]): Number of neurons or nodes, or a list of neurons or nodes.
            activation_func (Union[str, Callable]): The activation function for the neurons or nodes, either as a function or a string key for predefined functions.

        Raises:
            ValueError: If the activation function is neither callable nor a valid string key.
        """
        self._set_activationf(activation_func)
        self._set_struc(nodes)
        self._set_indexes()

    @property
    def structure(self) -> List[Ntype]:
        """
        Get the list of neurons or nodes in the layer.

        Returns:
            List[Ntype]: The list of nodes or neurons in the layer.
        """
        return self._structure

    @structure.setter
    def structure(self, value: List[Ntype]) -> None:
        """
        Set the structure of the layer to a new list of neurons or nodes.

        Args:
            value (List[Ntype]): The new list of nodes or neurons.

        Raises:
            ValueError: If the list contains invalid components.
        """
        self._structure = verify_components_type(
            verify_type(value, list), (Node, Neuron)
        )

    @property
    def n(self) -> int:
        """
        Get the number of neurons or nodes in the layer.

        Returns:
            int: The count of neurons or nodes.
        """
        return len(self._structure)

    @property
    def activationf(self) -> ActivationFunction:
        """
        Get the activation function used in the layer.

        Returns:
            Callable: The activation function applied to the neurons or nodes.
        """
        return self._activationf

    @activationf.setter
    def activationf(self, func: Union[str, ActivationFunction]) -> None:
        """
        Set a new activation function for the layer.

        Args:
            func (Union[str, Callable]): The new activation function, either as a callable or a string key.
        """
        self._set_activationf(func)
        self._set_struc(self.structure)

    @property
    def neta(self):
        """The neta property."""
        if type(self.structure[0]) == Neuron:
            return [n.z for n in self.structure]
        else:
            return

    @property
    def activation(self):
        """The activation property."""
        return [n.output for n in self.structure]

    def __str__(self) -> str:
        """
        Return a string representation of the layer.

        Returns:
            str: A string representation of the layer, including its type, number of components, and details.
        """
        return f"{type(self).__name__}({self.n}):\n\t{self._structure}\n"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the layer.

        Returns:
            str: A detailed string representation of the layer.
        """
        return f"{type(self).__name__}({self.n}):\n\t{self._structure}\n"

    def __eq__(self, value: object, /) -> bool:
        """
        Compare this layer with another for equality.

        Args:
            value (object): The other layer to compare with.

        Returns:
            bool: True if both layers are equal, False otherwise.
        """
        return self.__dict__ == value.__dict__

    def __ne__(self, value: object, /) -> bool:
        """
        Compare this layer with another for inequality.

        Args:
            value (object): The other layer to compare with.

        Returns:
            bool: True if both layers are not equal, False otherwise.
        """
        return not self.__eq__(value)

    def __len__(self) -> int:
        """
        Get the number of neurons or nodes in the layer.

        Returns:
            int: The count of neurons or nodes.
        """
        return self.n

    def __iter__(self) -> Iterator[Ntype]:
        """
        Initialize iteration over the neurons or nodes in the layer.

        Returns:
            Iterator[Ntype]: An iterator for the neurons or nodes.
        """
        self._iter_index = 0
        return self

    def __next__(self) -> Ntype:
        """
        Return the next neuron or node in the iteration.

        Returns:
            Ntype: The next neuron or node.

        Raises:
            StopIteration: If there are no more neurons or nodes to iterate over.
        """
        if self._iter_index < len(self._structure):
            result = self._structure[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, indx: int) -> Ntype:
        """
        Get the neuron or node at a specific index.

        Args:
            indx (int): The index of the neuron or node to retrieve.

        Returns:
            Ntype: The neuron or node at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        verify_type(indx, int)
        if indx >= 0:
            if indx < self.n:
                return self._structure[indx]
            raise IndexError(f"Index out of range: {indx}")
        else:
            if abs(indx) <= self.n:
                return self._structure[indx]
            raise IndexError(f"Index out of range: {indx}")

    def __setitem__(self, indx: int, val: Ntype) -> None:
        """
        Set the neuron or node at a specific index.

        Args:
            indx (int): The index to set.
            val (Ntype): The neuron or node to set at the index.

        Raises:
            TypeError: If the value is not of type `Node` or `Neuron`.
            IndexError: If the index is out of range.
        """
        verify_type(indx, int)
        verify_type(val, (Node, Neuron))

        if indx >= 0 and indx < self.n:
            self._structure[indx] = val
        elif indx < 0 and abs(indx) < len(self._structure):
            self._structure[indx] = val
        else:
            raise IndexError("Index out of range.")

    def __hash__(self) -> int:
        """
        Compute the hash value of the layer.

        Returns:
            int: The hash value of the layer.
        """
        return hash(str(self._structure) + str(self._activationf))

    def add_neuron(self, indx: Optional[int] = None) -> None:
        """
        Add a neuron to the layer at a specified index, or append it if no index is provided.

        Args:
            indx (Optional[int]): The index to insert the new neuron. If None, append to the end.
        """
        if indx is not None:
            verify_type(indx, int)
            self._structure.insert(indx, Neuron(self._activationf))
        else:
            self._structure.append(Neuron(self._activationf))

        self._set_indexes()

    def remove_neuron(self, indx: int) -> None:
        """
        Remove a neuron from the layer at a specific index.

        Args:
            indx (int): The index of the neuron to remove.

        Raises:
            IndexError: If the index is out of range.
        """
        verify_type(indx, int)
        self._structure.pop(indx)

        self._set_indexes()

    def _set_indexes(self) -> None:
        """
        Initialize the internal indexes for each neuron or node within the layer.
        """
        for i, n in enumerate(self._structure):
            n._ne_i = i

    def _set_struc(self, nodes: Union[int, List[Ntype]]) -> None:
        """
        Set the structure of the layer based on the `nodes` parameter provided during initialization.

        Args:
            nodes (Union[int, List[Ntype]]): The number of neurons or nodes, or a list of neurons or nodes.

        Raises:
            ValueError: If `nodes` is neither an integer nor a list of valid components.
        """
        if isinstance(nodes, int):
            if nodes < 1:
                raise ValueError("Expected at least '1' Neuron or Node for a Layer")
            self._structure = [Neuron(self._activationf) for _ in range(nodes)]
        elif isinstance(nodes, list):
            verify_components_type(nodes, (Node, Neuron))
            self._structure = nodes

            for node in nodes:
                node.activation = self.activationf
        else:
            raise ValueError(
                "Expected `nodes` to be either an integer or a list of `Node`/`Neuron` instances."
            )

    def _set_activationf(self, activationf: Union[str, ActivationFunction]) -> None:
        """
        Set the activation function for the layer.

        Args:
            activation (Union[str, Callable]): The activation function, either as a callable or a string key for predefined functions.

        Raises:
            ValueError: If `activation` is neither a callable nor a valid string key.
        """
        if activationf in activation_map:
            self._activationf = activation_map[activationf]
        elif isinstance(activationf, ActivationFunction):
            self._activationf = activationf
        else:
            raise ValueError(
                f"Expected activation to be callable or one of ({'/'.join(activation_map.keys())})"
            )


if __name__ == "__main__":
    pass
