"""
Optimization algorithms for training machine learning models.

This module offers a variety of optimization techniques essential for training machine learning models. The available optimizers are:

- `Optimizer`: An abstract base class that outlines the standard interface and attributes for all optimizer types.
- `PLR`: Perceptron Learning Rule optimizer, tailored for training Perceptron models.
- `SGD`: Stochastic Gradient Descent optimizer, enhanced with capabilities like momentum, Nesterov acceleration, and Adagrad.
- `Adam`: Adaptive Moment Estimation optimizer, recognized for its adaptive learning rates and momentum-based parameter updates.
- `RMSprop`: Root Mean Square Propagation optimizer, which modifies the learning rate based on a moving average of squared gradients.

Classes:
- `Optimizer` (ABC):
  The foundational abstract base class for all optimizers. It defines key attributes and includes the abstract method `fit` for training models.

- `PLR` (Optimizer):
  Implements the Perceptron Learning Rule, specifically designed for training Perceptron models.

- `SGD` (Optimizer):
  Realizes Stochastic Gradient Descent with additional functionalities such as momentum, Nesterov acceleration, and Adagrad.
  
  - Extras:
    - `momentum`: The momentum factor that accelerates SGD updates.
    - `batch_size`: The number of samples used per parameter update.
    - `nesterov`: Flag indicating whether to apply Nesterov Accelerated Gradient.
    - `adagrad`: Flag indicating whether to utilize Adagrad.

- `Adam` (Optimizer):
  Implements the Adam optimization algorithm, combining the strengths of both Adagrad and RMSprop.

- `RMSprop` (Optimizer):
  Provides RMSprop optimization, which adjusts learning rates based on a moving average of squared gradients.

Optimization Map:
- `optimizers_map`: A dictionary that maps optimizer names (e.g., "sgd", "adam") to their corresponding optimizer classes.
"""

from typing import Any, Callable, Dict, List, Tuple, Type, Union, TypeVar
from easyAIs.core.metrics import History
from easyAIs.core.neural_components import Neuron
from easyAIs.loss import LossFunction, TLossFunction
from easyAIs.core.layer import Layer
from easyAIs.activations import ActivationFunction
from easyAIs.utils.math import transpose
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.

    This class establishes a common interface and shared attributes for all optimization algorithms used
    in training machine learning models.

    Attributes:
        learning_rate (Union[int, float]): The rate at which the optimizer adjusts the model parameters.
        loss (LossFunction): The loss function utilized to evaluate model performance.
        verbose (Union[None, bool]): Controls the verbosity of the optimizer's output.
    """

    def __init__(
        self,
        learning_rate: Union[int, float],
        loss: LossFunction,
        verbose: Union[None, bool],
    ):
        """
        Initializes the optimizer with specified parameters.

        Args:
            learning_rate (Union[int, float]): The learning rate for parameter updates.
            loss (LossFunction): The loss function to be employed.
            verbose (Union[None, bool]): Determines if the optimizer should output verbose information.
        """
        self.learning_rate: Union[int, float] = learning_rate
        self.loss: LossFunction = loss
        self.verbose = verbose

    @abstractmethod
    def fit(*args, **kwargs) -> Any:
        """
        Fits the optimizer to the training data.

        This method should be implemented by all subclasses to handle the training process.
        """
        pass

    @abstractmethod
    def optimize(*args, **kwargs) -> Any:
        """
        Executes the optimization process.

        This method should be implemented by all subclasses to define how the model parameters are updated.
        """
        print(f"<optimize abstractmethod in {__name__}>")
        pass

    @abstractmethod
    def update_parameters(*args, **kwargs) -> Any:
        """
        Updates the model parameters based on computed gradients.

        This method should be implemented by all subclasses to specify the parameter update rules.
        """
        print(f"<update_parameters abstractmethod in {__name__}>")
        pass

    @abstractmethod
    def _fix_training_data(*args, **kwargs) -> Any:
        """
        Preprocesses and validates the training data.

        This method should be implemented by all subclasses to ensure that the training data meets required criteria.
        """
        print(f"<_fix_training_data abstractmethod in {__name__}>")
        return

    @classmethod
    def _instanciate(cls, *args):
        """
        Creates an instance of the optimizer.

        Args:
            *args: Variable length argument list for optimizer initialization.

        Returns:
            An instance of the optimizer.
        """
        return cls(*args)


class PLR(Optimizer):
    """
    Perceptron Learning Rule optimizer.

    This optimizer implements the Perceptron Learning Rule, specifically designed for training Perceptron models.

    Attributes:
        Inherits all attributes from `Optimizer`.
    """

    def __init__(self, learning_rate: Union[int, float], verbose: bool):
        """
        Initializes the Perceptron Learning Rule optimizer.

        Args:
            learning_rate (Union[int, float]): The learning rate for parameter updates.
            verbose (bool): Enables verbose output if set to True.
        """
        super().__init__(learning_rate=learning_rate, loss=None, verbose=verbose)
        del self.loss

    def update_parameters(
        self,
        inpts: List[Union[int, float]],
        expected_output: Union[int, float],
        model_output: Union[int, float],
        weights: List[Union[int, float]],
        bias: Union[int, float],
    ) -> Any:
        """
        Updates the Perceptron model parameters using the Perceptron Learning Rule.

        Args:
            inpts (List[Union[int, float]]): The input features.
            expected_output (Union[int, float]): The target output value.
            model_output (Union[int, float]): The output produced by the model.
            weights (List[Union[int, float]]): The current weights of the model.
            bias (Union[int, float]): The current bias of the model.

        Returns:
            Tuple[List[Union[int, float]], Union[int, float]]: The updated weights and bias.
        """
        if model_output != expected_output:
            for j in range(len(weights)):
                weights[j] += (
                    self.learning_rate * (expected_output - model_output) * inpts[j]
                )
            bias += self.learning_rate * (expected_output - model_output)

        return weights, bias

    def fit(self, X: List[List[Union[int, float]]], Y: List[Union[int, float]]) -> None:
        """
        Sets the training data for the optimizer.

        Args:
            X (List[Union[int, float]]): The input training data.
            Y (List[Union[int, float]]): The target training data.
        """
        self._X: List[List[Union[int, float]]] = X
        self._Y: List[Union[int, float]] = Y

        return

    def optimize(self, forward: Callable, neuron: Neuron):
        """
        Performs the optimization process on the given neuron.

        Args:
            forward (Callable): The forward pass function of the model.
            neuron (Neuron): The neuron to be optimized.
        """
        for epoch in range(self.epochs):
            for i, expected_output in enumerate(self._Y):
                # expected_output: num
                # inpts: numeric array
                # model_output: num
                inpts = self._X[i]
                model_output = forward(inpts)

                # Updating parameters (PLR)
                nweights, nbias = self.update_parameters(
                    inpts=inpts,
                    expected_output=expected_output,
                    model_output=model_output,
                    weights=neuron.weights,
                    bias=neuron.bias,
                )

                neuron.weights, neuron.bias = nweights, nbias

            if self.verbose:
                print(
                    f"Epoch {epoch}:\n\tModel output: {model_output}\n\tExpected output: {expected_output}"
                )
            else:
                pass
        else:
            pass

    def _fix_training_data(
        self,
        X: List[List[Union[int, float]]],
        Y: List[List[Union[int, float]]],
        inpts: int,
        outputs: int,
        epochs: int,
    ):
        """
        Validates and sets up the training data.

        Args:
            X (List[Union[int, float]]): The input training data.
            Y (List[Union[int, float]]): The target training data.
            inpts (int): Number of input features.
            outputs (int): Number of output neurons.
            epochs (int): Number of training epochs.

        Raises:
            AssertionError: If training data sizes are inconsistent or invalid.
        """
        if len(X) != len(Y):
            print("[!] Warning: X size and Y size do not correspond.")

        assert epochs >= 1, "Expected at least one epoch."

        assert (
            len(X[0]) == inpts
        ), "Uncorresponded X dim, expected X to be nx x m."
        assert (
            len(Y) >= outputs
        ), "Uncorresponded Y dim, expected X to be ny x m."

        self.nx = inpts
        self.ny = outputs
        self.epochs = epochs


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    This optimizer implements Stochastic Gradient Descent with additional features such as momentum,
    Nesterov acceleration, and Adagrad.

    Attributes:
        Inherits all attributes from `Optimizer`.
        momentum (Union[int, float]): The momentum factor to accelerate SGD.
        nesterov (bool): Flag to enable Nesterov Accelerated Gradient.
        adagrad (bool): Flag to enable Adagrad optimization.
    """

    def __init__(
        self,
        learning_rate: Union[int, float],
        loss: LossFunction,
        momentum: Union[int, float] = 0,
        nesterov: bool = False,
        adagrad: bool = False,
        verbose: bool = False,
    ):
        """
        Initializes the Stochastic Gradient Descent optimizer with optional enhancements.

        Args:
            learning_rate (Union[int, float]): The learning rate for parameter updates.
            loss (LossFunction): The loss function to be utilized.
            momentum (Union[int, float], optional): The momentum factor for acceleration. Defaults to 0.
            nesterov (bool, optional): Enables Nesterov Accelerated Gradient if set to True. Defaults to False.
            adagrad (bool, optional): Enables Adagrad optimization if set to True. Defaults to False.
            verbose (bool, optional): Enables verbose output if set to True. Defaults to False.
        """
        super().__init__(learning_rate, loss, verbose)
        self.momentum: Union[int, float] = momentum
        self.nesterov: bool = nesterov
        self.adagrad: bool = adagrad

    def _fix_training_data(self, X: List[List[Union[int, float]]], Y: List[List[Union[int, float]]], inpts: int, outputs: int, epochs: int):
        """
        Validates and configures the training data.

        Args:
            X (List[Union[int, float]]): The input training data.
            Y (List[Union[int, float]]): The target training data.
            inpts (int): Number of input features.
            outputs (int): Number of output neurons.
            epochs (int): Number of training epochs.

        Raises:
            AssertionError: If training data sizes are inconsistent or invalid.
        """
        if len(X) != len(Y):
            print("[!] Warning: X size and Y size do not correspond.")

        assert epochs >= 1, "Expected at least one epoch."

        assert (
            len(X[0]) == inpts
        ), "Uncorresponded X dim, expected X to be nx x m."
        assert (
            len(Y[0]) == outputs
        ), "Uncorresponded Y dim, expected X to be ny x m."

        self.nx = inpts
        self.ny = outputs
        self.epochs = epochs

    def fit(self, X: List[List[Union[int, float]]], Y: List[List[Union[int, float]]]) -> None:
        """
        Assigns the training data to the optimizer.

        Args:
            X (List[Union[int, float]]): The input training data.
            Y (List[Union[int, float]]): The target training data.
        """
        self.X: List[List[Union[int, float]]] = X
        self.Y: List[List[Union[int, float]]] = Y

    def optimize(self, forward: Callable, layers: List[Layer]):
        """
        Executes the optimization process across all layers of the model.

        Args:
            forward (Callable): The forward pass function of the model.
            layers (List[Layer]): A list of layers in the model.
        """
        epoch_size: int = len(self.Y)

        for e in range(self.epochs):
            epoch_loss: Union[int, float] = 0

            for i in range(len(self.Y)):
                # inpts, expected_output & model_output are numeric arrays
                inpts: List[Union[int, float]] = self.X[i]
                expected_output: List[Union[int, float]] = self.Y[i]
                model_output: List[Union[int, float]] = forward(inpts)

                loss: Union[int, float] = self.loss(expected_output, model_output)

                epoch_loss += loss

                gradients: List[float] = []

                # Todo: make activation come from layer as a vector
                # Compute gradients for the last layer
                for neuron_index, neuron in enumerate(layers[-1]):
                    gradient = self.compute_output_gradients(
                        neuron_index,
                        expected_output,
                        model_output,
                        neuron.activation,
                        neuron.z,
                    )

                    gradients.append(gradient)

                prev_activations: List[Union[int, float]] = [n.output for n in layers[-2]]

                self.update_parameters(gradients, layers[-1], prev_activations)  # Output layer parameters

                # Update parameters for the hidden layers
                for li in range(len(layers[1:-1]), 0, -1):
                    # Compute the gradients for a specific layer using weights from the next layer
                    layer: Layer = layers[li]
                    activationf: ActivationFunction = layer.activationf
                    weights: List[List[Union[int, float]]] = layers[li + 1].weights
                    ponderated_weights: List[Union[int, float]] = layer._ponderated_weights
                    prev_activations: List[Union[int, float]] = [n.output for n in layers[li - 1]]

                    # Compute gradients for hidden layers
                    gradients: List[float] = self.compute_layer_gradients(
                        weights, gradients, layer, activationf, ponderated_weights
                    )



                    self.update_parameters(gradients, layer, prev_activations)

            #     break
            # break
            print("Epoch", e + 1)
            print("Loss", epoch_loss / epoch_size)

    def update_parameters(self, grad: List[float], layer: Layer[Neuron], prev_activations: List[Union[int, float]]) -> None:
        """
        Updates the parameters of a given layer based on the gradients.

        Args:
            grad (List[float]): The list of gradients for each neuron in the layer.
            layer (Layer[Neuron]): The layer whose parameters are to be updated.
        """
        for i, neuron in enumerate(layer):
            # Update weights
            new_weights: List[float] = []

            for j, w in enumerate(neuron.weights):
                nw = w - (self.learning_rate * grad[i] * prev_activations[j])
                new_weights.append(nw)

            neuron.weights = new_weights

            # Update bias
            neuron.bias -= self.learning_rate * grad[i]

        return

    def compute_output_gradients(
        self,
        index: int,
        eoutput: List[Union[int, float]],
        output_vector: List[Union[int, float]],
        activation_func: ActivationFunction,
        ponderated: Union[int, float],
    ) -> float:
        """
        Calculates the gradient for a specific output neuron.

        Args:
            index (int): The index of the neuron.
            eoutput (List[Union[int, float]]): The expected outputs.
            output_vector (List[Union[int, float]]): The actual outputs from the model.
            activation_func (ActivationFunction): The activation function of the neuron.
            ponderated (Union[int, float]): The weighted input to the neuron.

        Returns:
            float: The computed gradient for the neuron.
        """
        dL_da = self.loss.deriv(eoutput, output_vector)
        da_dz = activation_func.deriv(ponderated)

        if isinstance(dL_da, list):
            dL_da = dL_da[index]

        return dL_da * da_dz

    def compute_layer_gradients(
        self,
        weights: List[List[Union[int, float]]],  # (J, I) shape
        gradients: List[float],  # (J) length
        layer: Layer,  # (I) length
        activation: ActivationFunction,
        zs: List[Union[int, float]],  # (I) length
    ) -> List[Union[int, float]]:
        """
        Computes the gradients for a specific layer based on the gradients from the subsequent layer.

        Args:
            weights (List[List[Union[int, float]]]): The weights of the next layer.
            gradients (List[float]): The gradients from the next layer.
            layer (Layer): The current layer for which gradients are being computed.
            activation (ActivationFunction): The activation function of the current layer.
            zs (List[Union[int, float]]): The weighted inputs to the neurons in the current layer.

        Returns:
            List[Union[int, float]]: The computed gradients for the current layer.
        """
        layer_gradients: List[Union[int, float]] = []

        # 'i' index is used for lth layer
        # 'j' index is used for (l+1)th layer

        # Transpose the weight matrix
        weights = transpose(weights)  # (I, J) shape

        # ith row for the weights matrix represents all the connections for the ith neuron in layer (l)
        # therefore, every column represents the jth neuron for the layer (l+1)

        # Compute the gradient for each neuron in the current layer
        for i in range(len(layer)):
            # Sum the product of transposed weights and gradients
            weightd_grds_sum: Union[int, float] = sum(
                weights[i][j] * gradients[j] for j in range(len(gradients))
            )

            # Gradient for the ith neuron in the current layer
            grad: Union[int, float] = weightd_grds_sum * activation.deriv(zs[i])

            layer_gradients.append(grad)

        return layer_gradients


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.

    This optimizer implements the Adam optimization algorithm, which combines the benefits of Adagrad and RMSprop.

    Attributes:
        Inherits all attributes from `Optimizer`.
    """

    pass

    @staticmethod
    def _fix_training_data(X, Y, inpts, outputs, epochs):
        """
        Validates and configures the training data for Adam optimizer.

        Args:
            X (List[Union[int, float]]): The input training data.
            Y (List[Union[int, float]]): The target training data.
            inpts (int): Number of input features.
            outputs (int): Number of output neurons.
            epochs (int): Number of training epochs.

        Raises:
            AssertionError: If training data sizes are inconsistent or invalid.
        """
        if len(X) % len(Y) != 0:
            print("[!] Warning: X size and Y size do not correspond.")

        assert epochs >= 1, "Expected at least one epoch."

        assert (
            len(X) >= inpts
        ), "Expected X to be at least equal to the number of entries."
        assert (
            len(Y) >= outputs
        ), "Length of Y must be at least as long as the output layer."


# Define a TypeVar that is bound to Optimizer, representing any subclass of Optimizer
OptimizerType = TypeVar("OptimizerType", bound=Optimizer)

# A dictionary mapping optimizer names to their corresponding optimizer classes.
optimizers_map: Dict[str, Type[OptimizerType]] = {
    "plr": PLR,
    "sgd": SGD,
    "adam": Adam,
}
