"""
Module: easyAIs.core.activations

This module provides a set of commonly used activation functions for neural network layers, as well as their derivatives.
Activation functions are crucial for introducing non-linearity into models, which allows them to learn complex patterns and representations.
Each activation function is implemented as a subclass of the `Function` metaclass, and includes a `__call__` method to apply the function,
and a `deriv` method to compute the derivative.

Functions:
    - `Step(x: Union[int, float], threshold: Union[int, float] = 0) -> Union[int, float]`:
        Computes the Heaviside step function, which returns 1 if the input value `x` is greater than or equal to the `threshold`, otherwise 0.
        - Derivative: 0 everywhere (non-classical derivative).

    - `Sigmoid(x: Union[int, float]) -> float`:
        Computes the Sigmoid function, mapping any real-valued number into the range (0, 1).
        - Derivative: `sigmoid(x) * (1 - sigmoid(x))`.

    - `ReLU(x: Union[int, float]) -> Union[int, float]`:
        Computes the Rectified Linear Unit (ReLU) function, which returns `x` if `x` is positive, otherwise returns 0.
        - Derivative: 1 if `x > 0`, otherwise 0.

    - `LeakyReLU(x: Union[int, float]) -> Union[int, float]`:
        Computes the Leaky Rectified Linear Unit (Leaky ReLU) function, allowing a small gradient when `x` is negative.
        - Derivative: 1 if `x >= 0`, otherwise 0.1.

    - `Tanh(x: Union[int, float]) -> float`:
        Computes the Hyperbolic Tangent (tanh) function, mapping input values to the range (-1, 1).
        - Derivative: `1 - tanh(x)^2`.

    - `Softmax(x: List[Union[int, float]], n: Optional[int] = None) -> List[float]`:
        Computes the Softmax function for a list of values, converting them into a probability distribution.
        Optionally, returns the value at index `n` if provided.
        - Derivative: Typically requires the Jacobian matrix, not fully implemented here.

    - `PReLU(x: Union[int, float], lp: Union[int, float]) -> Union[int, float]`:
        Computes the Parametric Rectified Linear Unit (PReLU) function with a learned parameter `lp` for the negative slope.
        - Derivative: `lp` if `x < 0`, otherwise 1.

    - `ELU(x: Union[int, float], alpha: Union[int, float]) -> Union[int, float]`:
        Computes the Exponential Linear Unit (ELU) function with a scaling parameter `alpha`.
        - Derivative: 1 if `x > 0`, otherwise `alpha * exp(x)`.

    - `Softplus(x: Union[int, float]) -> Union[int, float]`:
        Computes the Softplus function, a smooth approximation of the ReLU function.
        - Derivative: `sigmoid(x)`.

    - `NoneFunction(x: Union[int, float]) -> Union[int, float]`:
        Returns the input value as a float. This function is essentially the identity function.
        - Derivative: 1.

Activation Map:
    A dictionary mapping activation function names to their corresponding class implementations:
    - "step": Step
    - "sigmoid": Sigmoid
    - "relu": ReLU
    - "leaky_relu": LeakyReLU
    - "tanh": Tanh
    - "softmax": Softmax
    - "prelu": PReLU
    - "elu": ELU
    - "softplus": Softplus
    - "none": NoneFunction

Usage:
    Import the required activation functions from this module to use in defining neural network layers.

    Example:
    ```python
    from easyAIs.core.activations import Sigmoid, ReLU

    # Apply Sigmoid activation
    output = Sigmoid(0.5)

    # Apply ReLU activation
    output = ReLU(-1)
    ```

Notes:
    - Ensure to provide the correct input types as specified for each function.
    - For activation functions involving parameters (e.g., `PReLU`, `ELU`), make sure to provide all necessary arguments.
    - The derivative of the Softmax function typically requires additional implementation involving the Jacobian matrix, which is not provided in this module.
"""

from typing import Dict, Optional, Type, TypeVar, Union, List
from easyAIs.utils.verifiers import verify_type
from easyAIs.core.functions import Function
from math import log, exp, e


class ActivationFunction(metaclass=Function):
    @staticmethod
    def deriv(*args, **kwargs) -> None:
        raise NotImplementedError(
            "The deriv method is not implemented for this loss function."
        )


class Step(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(
        x: Union[int, float], threshold: Union[int, float] = 0
    ) -> Union[int, float]:
        """
        Compute the Heaviside step function.

        Args:
            x (Union[int, float]): The input value to be evaluated.
            threshold (Union[int, float], optional): The threshold value for comparison. Defaults to 0.

        Returns:
            int: Returns 1 if `x` is greater than or equal to the `threshold`, otherwise returns 0.
        """
        x = verify_type(x, (int, float))
        threshold = verify_type(threshold, (int, float))
        return 1 if x >= threshold else 0

    @staticmethod
    def deriv(
        x: Union[int, float], threshold: Union[int, float] = 0
    ) -> Union[int, float]:
        """
        The derivative of the Heaviside step function is not defined in a classical sense.
        It can be represented as an impulse function or zero everywhere except at the threshold.

        Args:
            x (Union[int, float]): The input value to be evaluated.
            threshold (Union[int, float], optional): The threshold value for comparison. Defaults to 0.

        Returns:
            int: Returns 0 as the derivative of Heaviside step function is zero almost everywhere.
        """
        return 0


class Sigmoid(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the Sigmoid function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The Sigmoid of `x`, which is calculated as 1 / (1 + exp(-x)).
        """
        x = verify_type(x, (int, float))
        return 1 / (1 + exp(-x))

    @staticmethod
    def deriv(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the derivative of the Sigmoid function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The derivative of Sigmoid of `x`, which is sigmoid(x) * (1 - sigmoid(x)).
        """
        sig = Sigmoid.__call__(x)
        return sig * (1 - sig)


class ReLU(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the Rectified Linear Unit (ReLU) function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The ReLU of `x`, which is max(0, x).
        """
        x = verify_type(x, (int, float))
        return max(0, x)

    @staticmethod
    def deriv(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the derivative of the ReLU function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The derivative of ReLU of `x`, which is 1 if x > 0, otherwise 0.
        """
        return 1 if x > 0 else 0


class LeakyReLU(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the Leaky Rectified Linear Unit (Leaky ReLU) function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The Leaky ReLU of `x`, which is x if x >= 0, otherwise x / 10.
        """
        x = verify_type(x, (int, float))
        return x if x >= 0 else x / 10

    @staticmethod
    def deriv(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the derivative of the Leaky ReLU function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The derivative of Leaky ReLU of `x`, which is 1 if x >= 0, otherwise 0.1.
        """
        return 1 if x >= 0 else 0.1


class Tanh(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the Hyperbolic Tangent (tanh) function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The tanh of `x`, which is calculated as (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
        """
        x = verify_type(x, (int, float))
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    @staticmethod
    def deriv(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the derivative of the tanh function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The derivative of tanh of `x`, which is 1 - tanh(x)^2.
        """
        tanh_x = Tanh.__call__(x)
        return 1 - tanh_x**2


class Softmax(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(x: List[Union[int, float]], n: Optional[int] = None) -> List[float]:
        """
        Compute the Softmax function.

        Args:
            x (List[float]): The input list of values to be transformed.
            n (int, optional): Index for a specific value in the Softmax distribution. If provided, returns the value at index `n`.

        Returns:
            List[float]: The Softmax distribution of the input values. If `n` is provided, returns a list containing only the value at index `n`.
        """
        x = verify_type(x, list)
        e_x = [exp(i) for i in x]
        sum_e_x = sum(e_x)
        distr = [i / sum_e_x for i in e_x]

        if n is not None:
            verify_type(n, int)
            return [distr[n]]

        return distr

    @staticmethod
    def deriv(x: List[Union[int, float]], n: Optional[int] = None) -> List[float]:
        """
        The derivative of the Softmax function is more complex and involves the Jacobian matrix.
        This method is a placeholder and does not provide a full implementation.

        Args:
            x (List[float]): The input list of values to be transformed.
            n (int, optional): Index for a specific value in the Softmax distribution.

        Returns:
            List[float]: Placeholder for the derivative of Softmax.
        """
        # Note: Computing the derivative of Softmax typically requires the Jacobian matrix,
        # which is not provided here due to its complexity.
        return []


class PReLU(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(x: Union[int, float], lp: Union[int, float]) -> Union[int, float]:
        """
        Compute the Parametric Rectified Linear Unit (PReLU) function.

        Args:
            x (Union[int, float]): The input value to be transformed.
            lp (Union[int, float]): The learned parameter for the negative slope.

        Returns:
            float: The PReLU of `x`, calculated as max(lp * x, x).
        """
        x = verify_type(x, (int, float))
        lp = verify_type(lp, (int, float))
        return max(lp * x, x)

    @staticmethod
    def deriv(x: Union[int, float], lp: Union[int, float]) -> Union[int, float]:
        """
        Compute the derivative of the PReLU function.

        Args:
            x (Union[int, float]): The input value to be transformed.
            lp (Union[int, float]): The learned parameter for the negative slope.

        Returns:
            float: The derivative of PReLU of `x`, which is lp if x < 0, otherwise 1.
        """
        x = verify_type(x, (int, float))
        lp = verify_type(lp, (int, float))
        return lp if x < 0 else 1


class ELU(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(x: Union[int, float], alpha: Union[int, float]) -> Union[int, float]:
        """
        Compute the Exponential Linear Unit (ELU) function.

        Args:
            x (Union[int, float]): The input value to be transformed.
            alpha (Union[int, float]): The scaling parameter for the function.

        Returns:
            float: The ELU of `x`, which is x if x > 0, otherwise alpha * (exp(x) - 1).
        """
        x = verify_type(x, (int, float))
        alpha = verify_type(alpha, (int, float))
        return x if x > 0 else alpha * (exp(x) - 1)

    @staticmethod
    def deriv(x: Union[int, float], alpha: Union[int, float]) -> Union[int, float]:
        """
        Compute the derivative of the ELU function.

        Args:
            x (Union[int, float]): The input value to be transformed.
            alpha (Union[int, float]): The scaling parameter for the function.

        Returns:
            float: The derivative of ELU of `x`, which is 1 if x > 0, otherwise alpha * exp(x).
        """
        x = verify_type(x, (int, float))
        alpha = verify_type(alpha, (int, float))
        return 1 if x > 0 else alpha * exp(x)


class Softplus(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the Softplus function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The Softplus of `x`, calculated as log(1 + exp(x)).
        """
        x = verify_type(x, (int, float))
        return log(1 + exp(x))

    @staticmethod
    def deriv(x: Union[int, float]) -> Union[int, float]:
        """
        Compute the derivative of the Softplus function.

        Args:
            x (Union[int, float]): The input value to be transformed.

        Returns:
            float: The derivative of Softplus of `x`, which is sigmoid(x).
        """
        return Sigmoid.__call__(x)


class NoneFunction(ActivationFunction, metaclass=Function):
    @staticmethod
    def __call__(x: Union[int, float]) -> Union[int, float]:
        """
        Return the input value as a float.

        Args:
            x (Union[int, float]): The input value to be returned.

        Returns:
            float: The input value converted to float.
        """
        x = verify_type(x, (int, float))
        return x

    @staticmethod
    def deriv(x: Union[int, float]) -> Union[int, float]:
        """
        The derivative of the NoneFunction is 1.

        Args:
            x (Union[int, float]): The input value to be evaluated.

        Returns:
            float: The derivative of NoneFunction, which is always 1.
        """
        return 1

# Definimos un TypeVar para cualquier clase que sea una subclase de ActivationFunction
TActivationFunction = TypeVar('TActivationFunction', bound=ActivationFunction)

activation_map: Dict[str, Type[TActivationFunction]] = {
    "step": Step,
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "tanh": Tanh,
    "softmax": Softmax,
    "prelu": PReLU,
    "elu": ELU,
    "softplus": Softplus,
    "none": NoneFunction,
}

if __name__ == "__main__":
    for k, v in activation_map.items():
        print(issubclass(ActivationFunction, v))
