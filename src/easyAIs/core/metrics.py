"""
Module: easyAIs.core.metrics

This module provides utilities for tracking and storing performance metrics of a machine learning model during training.

Classes:
    - `History`:
        Represents a history object for tracking training progress. It stores various metrics such as loss and accuracy for both training and validation phases.

Notes:
    - The `update` method should be called at the end of each epoch to keep track of the training progress.
    - Metrics are stored as lists, allowing for easy retrieval and analysis of the model's performance over multiple epochs.
"""

class History:
    """Class representing a history object for tracking training progress."""

    def __init__(self):
        """
        Initializes a new `History` object with empty lists for storing training and validation metrics.
        """
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    def update(
        self, loss: float, accuracy: float, val_loss: float, val_accuracy: float
    ) -> None:
        """
        Updates the history with the latest metrics for the current epoch.

        Args:
            loss (float): The training loss for the current epoch.
            accuracy (float): The training accuracy for the current epoch.
            val_loss (float): The validation loss for the current epoch.
            val_accuracy (float): The validation accuracy for the current epoch.
        """
        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_accuracy)

