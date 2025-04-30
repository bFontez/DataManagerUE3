"""Metrics handling for regression and classification tasks.

This module provides dataclasses and functions for storing, computing and visualizing 
metrics for both regression and classification tasks.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


@dataclass
class RegressionMetrics:
    """Class to store training metrics for regression at each epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    epoch_time: float = 0.0  # Epoch execution time in seconds


@dataclass
class ClassificationMetrics:
    """Class to store training metrics for classification at each epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    epoch_time: float = 0.0  # Epoch execution time in seconds


def plot_regression_metrics_sequence(
    metrics_list: Sequence[RegressionMetrics],
    title: str = "Metrics Evolution",
    save_path: str | None = None,
) -> None:
    """Display the evolution of training metrics across epochs for regression models.

    Args:
        metrics_list: List of RegressionMetrics objects
        title: Chart title
        save_path: Path to save the chart (None = no saving)
    """
    epochs = np.array([m.epoch for m in metrics_list])
    train_losses = np.array([m.train_loss for m in metrics_list])
    val_losses = np.array([m.val_loss for m in metrics_list])
    epoch_times = np.array([m.epoch_time for m in metrics_list])

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [2, 1]}
    )

    # First subplot: training and validation losses
    ax1.plot(epochs, train_losses, label="Training Loss", color="blue", marker="o")
    ax1.plot(epochs, val_losses, label="Validation Loss", color="red", marker="x")
    ax1.set_title(title)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Second subplot: training time per epoch
    ax2.bar(epochs, epoch_times, color="green", alpha=0.7)
    ax2.set_title("Training Time per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Total and average training time
    total_time = float(np.sum(epoch_times))
    avg_time = total_time / len(epoch_times) if epoch_times.size > 0 else 0
    ax2.text(
        0.02,
        0.95,
        f"Total time: {total_time:.2f}s\nAverage time: {avg_time:.2f}s/epoch",
        transform=ax2.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Appearance improvement
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_classification_metrics_sequence(
    metrics_list: Sequence[ClassificationMetrics],
    title: str = "Metrics Evolution",
    save_path: str | None = None,
) -> None:
    """Display the evolution of training metrics across epochs for classification models.

    Args:
        metrics_list: List of ClassificationMetrics objects
        title: Chart title
        save_path: Path to save the chart (None = no saving)
    """
    epochs = np.array([m.epoch for m in metrics_list])
    train_losses = np.array([m.train_loss for m in metrics_list])
    val_losses = np.array([m.val_loss for m in metrics_list])
    val_accuracies = np.array([m.val_accuracy for m in metrics_list])
    epoch_times = np.array([m.epoch_time for m in metrics_list])

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 18), gridspec_kw={"height_ratios": [2, 2, 1]}
    )

    # First subplot: training and validation losses
    ax1.plot(epochs, train_losses, label="Training Loss", color="blue", marker="o")
    ax1.plot(epochs, val_losses, label="Validation Loss", color="red", marker="x")
    ax1.set_title(title)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Second subplot: validation accuracy
    ax2.plot(
        epochs, val_accuracies, label="Validation Accuracy", color="purple", marker="d"
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Third subplot: training time per epoch
    ax3.bar(epochs, epoch_times, color="green", alpha=0.7)
    ax3.set_title("Training Time per Epoch")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Time (seconds)")
    ax3.grid(True, linestyle="--", alpha=0.7)

    # Total and average training time
    total_time = float(np.sum(epoch_times))
    avg_time = total_time / len(epoch_times) if epoch_times.size > 0 else 0
    ax3.text(
        0.02,
        0.95,
        f"Total time: {total_time:.2f}s\nAverage time: {avg_time:.2f}s/epoch",
        transform=ax3.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Appearance improvement
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def print_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, set_name: str = "test"
) -> None:
    """Print standard regression metrics.

    Args:
        y_true: Actual values
        y_pred: Model predictions
        set_name: Name of the dataset for reporting (e.g., 'train', 'test')
    """
    # Calculate R² score
    r2 = r2_score(y_true, y_pred)
    print(f"\nR² score on {set_name} set: {r2:.4f}")

    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    print(f"MSE on {set_name} set: {mse:.4f}")

    # Calculate RMSE
    rmse = np.sqrt(mse)
    print(f"RMSE on {set_name} set: {rmse:.4f}")

    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE on {set_name} set: {mae:.4f}")


def print_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, set_name: str = "test"
) -> None:
    """Print standard classification metrics.

    Args:
        y_true: Actual values
        y_pred: Model predictions
        set_name: Name of the dataset for reporting (e.g., 'train', 'test')
    """
    # Calculate accuracy scores
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"\nMetrics for {set_name} set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Calculate and display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def plot_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual Values",
    save_path: str | None = None,
) -> None:
    """Plot regression predictions against actual values.

    Args:
        y_true: Actual values
        y_pred: Model predictions
        title: Plot title
        save_path: Path to save the plot (None = no saving)
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot(
        [float(np.min(y_true)), float(np.max(y_true))],
        [float(np.min(y_true)), float(np.max(y_true))],
        "r--",
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{title}\nR² = {r2_score(y_true, y_pred):.4f}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
    
def plot_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Classification Report",
    save_path: str | None = None,
) -> None:
    
    # Calculate accuracy scores
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    # Visualize confusion matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{title}\nAccuracy = {acc:.4f}, Balanced Accuracy = {balanced_acc:.4f}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
