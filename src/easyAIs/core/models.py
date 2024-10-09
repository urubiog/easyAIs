"""
Module: easyAIs.core.model

This module defines the `Model` class, which serves as an abstract base class for building various neural network architectures. It provides the fundamental framework for defining network layers, setting up training parameters, and executing the forward pass.

Classes:
    - `Model`:
        Abstract class representing neural network architectures. It lays the groundwork for constructing different types of models by defining essential attributes and methods.

Notes:

"""

from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Any, List, Union
from easyAIs.core.layer import Layer
from easyAIs.core.neural_components import Neuron, Node, Ntype
from easyAIs.core.optimizers import Optimizer, OptimizerType, optimizers_map
from easyAIs.layers import Input
from easyAIs.loss import LossFunction, TLossFunction, loss_map
from easyAIs.utils.instances import search_instnce_name
from easyAIs.utils.verifiers import verify_components_type, verify_len, verify_type


class Model(ABC):
    """Abstract base class representing neural network architectures."""

    def __init__(
        self,
        structure: List[Layer[Ntype]],
    ) -> None:
        """
        Initialize a Model object.

        Args:
            structure (List[Layer[Ntype]]): The structure of the neural network, consisting of a list of layers.

        Raises:
            ValueError: If `structure` is not a list of `Layer` objects.
        """
        self._name: str = type(self).__name__
        self.layers: List[Layer[Ntype]] = structure
        self._optimizer: Optimizer

    @property
    def input_layer(self) -> Layer[Node]:
        """Return the input layer of the model."""
        return self.layers[0]

    @property
    def hidden_layers(self) -> List[Layer[Neuron]]:
        """Return the hidden layers of the model."""
        return list(self.layers[1:-1])

    @property
    def output_layer(self) -> Layer[Neuron]:
        """Return the output layer of the model."""
        return self.layers[-1]

    @property
    def _output(self):
        """
        Return the output of the model based on the current inputs.

        Returns:
            List[float]: The output values from the output layer neurons.

        Notes:
            The current inputs are placeholder values; actual inputs should be provided via the `forward` method.
        """
        return [self.output_layer[i].output for i in range(len(self.output_layer))]

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Union[str, Optimizer]):
        """
        Set the optimizer of the model.

        Args:
            optimizer (Union[str, Optimizer]): The optimizer, either as a string key or an Optimizer instance.

        Raises:
            ValueError: If `optimizer` is an unknown string key.
            TypeError: If `optimizer` is not of a supported type.
        """
        if isinstance(optimizer, str):
            if optimizer in optimizers_map:
                self._optimizer = optimizers_map[optimizer]
            else:
                raise ValueError(
                    f"Expected optimizer to be one of ({'/'.join([k for k in optimizers_map.keys()])})"
                )

            # isinstance will accept an Optimizer subclass
        elif isinstance(optimizer, Optimizer):
            self._optimizer = optimizer
        else:
            raise TypeError("Unsupported type for optimizer.")

    @property
    def learning_rate(self) -> Union[int, float]:
        """Get or set the learning rate of the model."""
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: Union[int, float]) -> None:
        """
        Set the learning rate of the model.

        Args:
            value (Union[int, float]): The learning rate to set.
        """
        self.optimizer.learning_rate = verify_type(value, (int, float))

    @property
    def loss(self) -> LossFunction:
        """Get or set the loss function of the model."""
        return self.optimizer.loss

    @loss.setter
    def loss(self, value: Union[str, TLossFunction]) -> None:
        """
        Set the loss function of the model.

        Args:
            value (Union[str, Callable]): The loss function, either as a string key or a callable function.

        Raises:
            ValueError: If `value` is neither a known string key nor a callable.
        """
        if value in loss_map:
            self.optimizer.loss = loss_map[value]
        elif isinstance(value, LossFunction):
            self.optimizer.loss = value
        else:
            raise ValueError(
                f"Expected loss to be callable or one of ({'/'.join([k for k in loss_map.keys()])})"
            )

    @property
    def depth(self) -> int:
        """Return the depth of the model, i.e., the total number of layers."""
        return len(self.layers)

    @property
    def nx(self) -> int:
        """Return the number of neurons in the input layer."""
        return len(self.input_layer)

    def __repr__(self) -> str:
        """
        Return a string representation of the model for debugging.

        Returns:
            str: A string describing the model type and its layers.
        """
        return f"{self._name}({len(self._layers)}) object named {search_instnce_name(self)}"

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the model.

        Returns:
            str: A string describing the model type and its layers.
        """
        return f"{self._name}({len(self._layers)}) object named {search_instnce_name(self)}"

    @singledispatchmethod
    def __getitem__(self, _: Any, /) -> None:
        """
        Handle indexing operations with unsupported types.

        Raises:
            TypeError: If the type of `index` is not supported.
        """
        raise TypeError("Not supported type.")

    @__getitem__.register(tuple)
    def _(self, indx: tuple, /):
        """
        Get the Node/Neuron based on two indexes.

        Args:
            indx (tuple): A tuple with two indexes.

        Returns:
            Ntype: The Node or Neuron object at the specified indices.
        """
        verify_len(indx, 2)
        return self._layers[indx[0]][indx[1]]

    @__getitem__.register(float)
    def _(self, indx: float, /):
        """
        Get the layer based on the given floating-point index.

        Args:
            indx (float): The floating-point index.

        Returns:
            Layer: The layer corresponding to the index.
        """
        i: list = str(indx).split(".")
        verify_len(i, 2)
        return self._layers[int(i[0])][int(i[1])]

    @__getitem__.register(int)
    def _(self, indx: int, /):
        """
        Get the layer based on the given integer index.

        Args:
            indx (int): The integer index.

        Returns:
            Layer: The layer corresponding to the index.

        Raises:
            IndexError: If the index is out of range.
        """
        if len(self._layers) > indx >= 0 or len(self._layers) <= indx < 0:
            return self._layers[indx]
        raise IndexError("Index out of range.")

    @__getitem__.register(list)
    def _(self, indx: list[int], /):
        """
        Get multiple layers based on a list of indexes.

        Args:
            indx (list[int]): A list of indexes.

        Returns:
            List[Layer]: A list of layers corresponding to the indexes.

        Raises:
            IndexError: If any index is out of range.
        """
        l: list[Layer] = []
        for i in indx:
            verify_type(i, int)
            if i >= 0:
                if i > len(self._layers) - 1:
                    raise IndexError(f"Index out of range {i}.")
            else:
                if -i > len(self._layers):
                    raise IndexError(f"Index out of range: {i}.")
            l.append(self._layers[i])
        return l

    def __eq__(self, value: object, /) -> bool:
        """
        Check if two models are equal.

        Args:
            value (object): The model to compare with.

        Returns:
            bool: True if the models are equal, False otherwise.
        """
        return self.__dict__ == value.__dict__

    def __ne__(self, value: object, /) -> bool:
        """
        Check if two models are not equal.

        Args:
            value (object): The model to compare with.

        Returns:
            bool: True if the models are not equal, False otherwise.
        """
        return not self.__eq__(value)

    def __hash__(self) -> int:
        """
        Compute the hash of the model based on its layers.

        Returns:
            int: The hash value of the model.
        """
        return hash(tuple([l.__hash__() for l in self._layers]))

    def __call__(self, input: List[Union[int, float]]) -> List[float]:
        """
        Perform a forward pass with the given input.

        Args:
            input (List[Union[int, float]]): The input data to propagate through the network.

        Returns:
            List[float]: The output of the model after the forward pass.
        """
        return self.forward(input)

    def add(self, layer: Layer, indx: int = -1) -> None:
        """
        Add a layer to the model at the specified index.

        Args:
            layer (Layer): The layer to add.
            indx (int, optional): The index at which to add the layer. Defaults to -1 (append to the end).

        Raises:
            TypeError: If `layer` is not of type `Layer`.
            TypeError: If `indx` is not of type `int`.
        """
        verify_type(layer, Layer)
        verify_type(indx, int)
        self._layers.insert(indx, layer)

    def remove(self, indx: int = -1) -> None:
        """
        Remove a layer from the model at the specified index.

        Args:
            layer (Layer): The layer to remove.
            indx (int, optional): The index at which to remove the layer. Defaults to -1 (remove from the end).

        Raises:
            TypeError: If `layer` is not of type `Layer`.
            TypeError: If `indx` is not of type `int`.
        """
        verify_type(indx, int)
        self._layers.pop(indx)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def train(*args, **kwargs) -> Any:
        pass

    # TODO: Implement evaluate function
    def evaluate(self) -> None:
        """
        Evaluate the model's performance. Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # TODO: Implement save function
    def save(self) -> None:
        """
        Save the model to a file. Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # TODO: Implement load function
    def load(self) -> None:
        """
        Load the model from a file. Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # TODO: Implement summary function
    def summary(self) -> None:
        """
        Print a summary of the model. Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class Sequential(Model):
    def __init__(self, structure: List[Layer[Ntype]]) -> None:
        super().__init__(structure)

        self.__set_input_layer()
        self.__set_connections()

    def __set_input_layer(self) -> None:
        """
        Initialize the input layer with Node instances.
        """
        if not isinstance(self.input_layer, Input):
            self.input_layer.structure = Input(len(self.input_layer.structure))


    def __set_connections(self) -> None:
        """
        Set the connections between neurons in consecutive layers.
        """
        for i in range(1, self.depth):
            for n in self.layers[i]:
                n.inputnodes = [node for node in self.layers[i - 1]]
