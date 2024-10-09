"""
Module: easyAIs.core.loss

This module provides a set of commonly used loss functions for evaluating the performance of machine learning models. Loss functions measure the discrepancy between the true values and the predicted values from a model, guiding the optimization process during training.

Classes:
    - `MeanSquaredError`:
        Calculates the Mean Squared Error (MSE) between true labels and predicted values. Implements the `deriv` method to compute the derivative of the MSE with respect to the predictions.

    - `BinaryCrossEntropy`:
        Computes the Binary Cross-Entropy loss between binary true labels and predicted probabilities. Implements the `deriv` method to compute the derivative of the Binary Cross-Entropy with respect to the predicted probabilities.

    - `CategoricalCrossEntropy`:
        Calculates the Categorical Cross-Entropy loss for multi-class classification problems between one-hot encoded true labels and predicted probabilities. Implements the `deriv` method to compute the derivative of the Categorical Cross-Entropy with respect to the predicted probabilities.

    - `HuberLoss`:
        Computes the Huber loss, which is less sensitive to outliers than MSE, between true values and predicted values. Implements the `deriv` method to compute the derivative of the Huber Loss with respect to the predictions.

    - `KLDivergence`:
        Calculates the Kullback-Leibler (KL) divergence between two probability distributions, measuring how one probability distribution diverges from a second, expected probability distribution. Implements the `deriv` method to compute the derivative of the KL Divergence with respect to the second distribution.

Loss Map:
    A dictionary mapping loss function names to their corresponding class implementations:
    - "mse": MeanSquaredError
    - "cross-entropy": CategoricalCrossEntropy
    - "binary-cross-entropy": BinaryCrossEntropy
    - "huber": HuberLoss
    - "kl-divergence": KLDivergence

Usage:
    Import the required loss function classes from this module for use in model training and evaluation.

    Example:
    ```python
    from easyAIs.loss import MeanSquaredError, BinaryCrossEntropy

    # Compute Mean Squared Error
    mse = MeanSquaredError()(tags=[1, 2, 3], pred=[1.1, 2.1, 2.9])

    # Compute Binary Cross-Entropy
    bce = BinaryCrossEntropy()(y_true=[1, 0], y_pred=[0.8, 0.2])
    ```

Notes:
    - Ensure that the input types and shapes are consistent with the requirements of each loss function.
    - For loss functions involving probability distributions, ensure that the distributions are valid and properly normalized.
    - The `deriv` methods provide the derivatives of the loss functions with respect to their inputs and can be used for gradient computation during optimization.
"""

from typing import Dict, List, Optional, Union, Type, TypeVar
from math import log, log10
from easyAIs.core.functions import Function

class LossFunction(metaclass=Function):
    @staticmethod
    def deriv(*args, **kwargs):
        """
        Compute the derivative of the loss function.
        This method should be overridden by each specific loss function.
        """
        raise NotImplementedError(
            "The deriv method is not implemented for this loss function."
        )


class MeanSquaredError(LossFunction):
    @staticmethod
    def __call__(tags: List[Union[int, float]], pred: List[Union[int, float]]) -> float:
        """
        Calculates the mean squared error between true labels and predictions.

        Args:
            tags (List[Union[int, float]]): True labels.
            pred (List[Union[int, float]]): Predicted values.

        Returns:
            float: Mean squared error between tags and pred.
        """
        return sum([(t - p) ** 2 for t, p in zip(tags, pred)]) / len(tags)

    @staticmethod
    def deriv(
        tags: List[Union[int, float]], pred: List[Union[int, float]]
    ) -> List[float]:
        """
        Computes the derivative of Mean Squared Error with respect to predictions.

        Args:
            tags (List[Union[int, float]]): True labels.
            pred (List[Union[int, float]]): Predicted values.

        Returns:
            List[float]: The gradient of MSE with respect to predictions.
        """
        n = len(tags)

        return [2 * (p - t) / n for t, p in zip(tags, pred)]


class BinaryCrossEntropy(LossFunction):
    @staticmethod
    def __call__(
        y_true: List[Union[int, float]], y_pred: List[Union[int, float]]
    ) -> float:
        """
        Calculates the binary cross-entropy between the true values (y_true) and the predicted values (y_pred).

        Args:
            y_true (List[Union[int, float]]): A list of true values (0 or 1).
            y_pred (List[Union[int, float]]): A list of predicted probabilities (between 0 and 1).

        Returns:
            float: The binary cross-entropy.
        """
        assert len(y_true) == len(y_pred), "Both y_true and y_pred lists must be same size."

        epsilon = 1e-15
        y_pred = [max(epsilon, min(1 - epsilon, p)) for p in y_pred]

        bce = -sum(
            y_t * log(y_p) + (1 - y_t) * log(1 - y_p)
            for y_t, y_p in zip(y_true, y_pred)
        )

        return bce / len(y_true)

    @staticmethod
    def deriv(
        y_true: List[Union[int, float]], y_pred: List[Union[int, float]]
    ) -> List[float]:
        """
        Computes the derivative of Binary Cross-Entropy with respect to predictions.

        Args:
            y_true (List[Union[int, float]]): A list of true values (0 or 1).
            y_pred (List[Union[int, float]]): A list of predicted probabilities (between 0 and 1).

        Returns:
            List[float]: The gradient of BCE with respect to predictions.
        """
        assert len(y_true) == len(y_pred), "Both y_true and y_pred lists must be same size."

        epsilon = 1e-15
        y_pred = [max(epsilon, min(1 - epsilon, p)) for p in y_pred]
    
        return [(y_p - y_t) / (y_p * (1 - y_p)) for y_t, y_p in zip(y_true, y_pred)]


class CategoricalCrossEntropy(LossFunction):
    @staticmethod
    def __call__(
        y_true: List[Union[int, float]], y_pred: List[Union[int, float]]
    ) -> float:
        """
        Calculates categorical cross-entropy between true labels and predictions.

        Args:
            y_true (List[Union[int, float]]): List of true values encoded as one-hot.
            y_pred (List[Union[int, float]]): List of predicted probabilities for each class.

        Returns:
            float: Categorical cross-entropy between y_true and y_pred.
        """
        return -sum(true * log10(pred) for true, pred in zip(y_true, y_pred))

    @staticmethod
    def deriv(
        y_true: List[Union[int, float]], y_pred: List[Union[int, float]]
    ) -> List[float]:
        """
        Computes the derivative of Categorical Cross-Entropy with respect to predictions.

        Args:
            y_true (List[Union[int, float]]): List of true values encoded as one-hot.
            y_pred (List[Union[int, float]]): List of predicted probabilities for each class.

        Returns:
            List[float]: The gradient of Categorical Cross-Entropy with respect to predictions.
        """
        return [-true / pred for true, pred in zip(y_true, y_pred)]


class HuberLoss(LossFunction):
    @staticmethod
    def __call__(
        y_true: List[Union[int, float]],
        y_pred: List[Union[int, float]],
        delta: Union[int, float] = 1.0,
    ) -> float:
        """
        Calculates the Huber loss between true labels and predictions for each element in the lists.

        Args:
            y_true (List[Union[int, float]]): List of true values.
            y_pred (List[Union[int, float]]): List of predicted values.
            delta (Union[int, float], optional): Threshold parameter. Defaults to 1.0.

        Returns:
            float: Mean Huber loss between y_true and y_pred.
        """
        loss = 0.0
        for t, p in zip(y_true, y_pred):
            error = abs(t - p)
            if error <= delta:
                loss += 0.5 * error**2
            else:
                loss += delta * (error - 0.5 * delta)
        return loss / len(y_true)

    @staticmethod
    def deriv(
        y_true: List[Union[int, float]],
        y_pred: List[Union[int, float]],
        delta: Union[int, float] = 1.0,
    ) -> List[float]:
        """
        Computes the derivative of Huber Loss with respect to predictions for each element in the lists.

        Args:
            y_true (List[Union[int, float]]): List of true values.
            y_pred (List[Union[int, float]]): List of predicted values.
            delta (Union[int, float], optional): Threshold parameter. Defaults to 1.0.

        Returns:
            List[float]: The gradient of Huber Loss with respect to predictions.
        """
        gradients = []
        for t, p in zip(y_true, y_pred):
            error = t - p
            if abs(error) <= delta:
                gradients.append(-error)
            else:
                gradients.append(-delta * (1 if error > 0 else -1))
        return gradients


class KLDivergence(LossFunction):
    @staticmethod
    def __call__(p: List[Union[int, float]], q: List[Union[int, float]]) -> float:
        """
        Calculates the Kullback-Leibler divergence between two probability distributions p and q.

        Args:
            p (List[Union[int, float]]): Probability distribution p.
            q (List[Union[int, float]]): Probability distribution q.

        Returns:
            float: Kullback-Leibler divergence between p and q.
        """
        epsilon = 1e-15
        q = [max(epsilon, min(1 - epsilon, prob)) for prob in q]
        divergence = sum(p_i * log10(p_i / q_i) for p_i, q_i in zip(p, q) if p_i != 0)
        return divergence

    @staticmethod
    def deriv(p: List[Union[int, float]], q: List[Union[int, float]]) -> List[float]:
        """
        Computes the derivative of KL Divergence with respect to q.

        Args:
            p (List[Union[int, float]]): Probability distribution p.
            q (List[Union[int, float]]): Probability distribution q.

        Returns:
            List[float]: The gradient of KL Divergence with respect to q.
        """
        epsilon = 1e-15
        q = [max(epsilon, min(1 - epsilon, prob)) for prob in q]
        return [-(p_i / q_i) for p_i, q_i in zip(p, q)]


TLossFunction = TypeVar("TLossFunction", bound=LossFunction)

loss_map: Dict[str, Type[LossFunction]] = {
    "mse": MeanSquaredError,
    "cross-entropy": CategoricalCrossEntropy,
    "binary-cross-entropy": BinaryCrossEntropy,
    "huber": HuberLoss,
    "kl-divergence": KLDivergence,
}
