"""Convolutional Neural Network for classification of num_classe.

This script implements a CNN for spectral data classification, with both
standard and data-augmented versions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time  # Import for measuring training time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import spmatrix



# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================
# Number of training epochs
N_EPOCHS = 15
# Random seed for reproducibility
RANDOM_SEED = 42
# Batch size for training
BATCH_SIZE = 32
# Learning rate for the optimizer
LEARNING_RATE = 0.0001
# Proportion of data for the test set
TEST_SIZE = 0.25
# Proportion of data for the validation set (relative to remaining data)
VAL_SIZE = 0.2
# Kernel size for convolutional layers
KERNEL_SIZE = 10
# Stride for convolutional layers
STRIDE = 1
# Padding for convolutional layers
PADDING = 2
# ==============================================================================

# Get current notebook path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.dataLoading import data_3cl, spectral_cols
from src.data_augmenter import DataAugmenter, AugmentationParams
from src.metrics import (
    ClassificationMetrics,
    print_classification_metrics,
    plot_classification_metrics_sequence,
)
# Custom Dataset class for our spectral data
class SpectralDataset(Dataset):
    """Dataset class to handle spectral data for PyTorch"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Initialize dataset with spectral data and labels.
        
        Args:
            X: Input features as numpy array
            y: Target labels as numpy array
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            
        # Reshape X for CNN input: [batch_size, channels, sequence_length]
        # For spectral data, we use 1 channel
        self.X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
        self.y = torch.LongTensor(y)  # LongTensor for classification targets

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# CNN model for classification
class CNNModel(nn.Module):
    """CNN model with two convolutional layers for spectral data classification

    Args:
        input_dim: Length of the spectral sequence
        output_dim: Number of classes for classification
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = KERNEL_SIZE,
        stride: int = STRIDE,
        padding: int = PADDING,
    ):
        super().__init__()

        # Calculate output size of first conv layer
        conv1_output_size = ((input_dim + 2 * padding - kernel_size) // stride) + 1

        # Calculate output size after pooling
        pool1_output_size = conv1_output_size // 2

        # Calculate output size of second conv layer
        conv2_output_size = (
            (pool1_output_size + 2 * padding - kernel_size) // stride
        ) + 1

        # Calculate output size after pooling
        pool2_output_size = conv2_output_size // 2

        # First convolutional layer: in_channels=1 (grayscale), out_channels=16
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Second convolutional layer: in_channels=16, out_channels=32
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Final output is a flattened tensor of size (batch_size, 32 * pool2_output_size)
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(32 * pool2_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch_size, 1, sequence_length)
        x = self.conv1(x)
        # Shape: (batch_size, 16, sequence_length/2)
        x = self.conv2(x)
        # Shape: (batch_size, 32, sequence_length/4)

        # Flatten: (batch_size, 32 * sequence_length/4)
        x = x.view(x.size(0), -1)

        # Output: (batch_size, output_dim)
        return self.fc(x)


# =============================================================================
# CNN for classification without data augmentation
# =============================================================================
print("\n" + "=" * 80)
print("CNN FOR CLASSIFICATION (NUM_CLASSE PREDICTION) WITHOUT DATA AUGMENTATION")
print("=" * 80)

# Preparing data for classification
indices = np.arange(len(data_3cl))
indices_train_temp, indices_test = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=data_3cl["num_classe"],
)

indices_train, indices_val = train_test_split(
    indices_train_temp,
    test_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    stratify=data_3cl.iloc[indices_train_temp]["num_classe"],
)

# Display set sizes
print(f"Training set: {len(indices_train)} samples")
print(f"Validation set: {len(indices_val)} samples")
print(f"Test set: {len(indices_test)} samples")

# Create datasets
train_data = data_3cl.iloc[indices_train].copy()
val_data = data_3cl.iloc[indices_val].copy()
test_data = data_3cl.iloc[indices_test].copy()

# Encode the categorical target variable
label_encoder = LabelEncoder()
label_encoder.fit(data_3cl["num_classe"])
num_classes = len(label_encoder.classes_)

print(f"Number of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

# Prepare features and targets for different sets
X_train_cls = np.array(train_data[spectral_cols])
y_train_cls = label_encoder.transform(train_data["num_classe"])

X_val_cls = np.array(val_data[spectral_cols])
y_val_cls = label_encoder.transform(val_data["num_classe"])

X_test_cls = np.array(test_data[spectral_cols])
y_test_cls = label_encoder.transform(test_data["num_classe"])

# Standardize features for classification
scaler_X_cls = StandardScaler()

X_train_cls = scaler_X_cls.fit_transform(X_train_cls)
X_val_cls = scaler_X_cls.transform(X_val_cls)
X_test_cls = scaler_X_cls.transform(X_test_cls)

# Create datasets and loaders for training and validation only
X_train_cls = np.array(X_train_cls)
y_train_cls = np.array(y_train_cls)
X_val_cls = np.array(X_val_cls)
y_val_cls = np.array(y_val_cls)

train_dataset = SpectralDataset(X_train_cls, y_train_cls)
val_dataset = SpectralDataset(X_val_cls, y_val_cls)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss function and optimizer
model = CNNModel(input_dim=len(spectral_cols), output_dim=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float("inf")
best_model_state = None

# List to store training metrics
metrics_history: List[ClassificationMetrics] = []

print("Starting CNN training for classification without augmentation...")
total_training_start = time.time()
for epoch in range(N_EPOCHS):
    epoch_start = time.time()
    # Set model to training mode
    model.train()
    # Initialize the total training loss for this epoch
    train_loss = 0.0
    # Iterate over mini-batches of training data
    for X_batch, y_batch in train_loader:
        # Reset gradients to zero before computing new gradients
        optimizer.zero_grad()
        # Forward pass: compute model predictions
        outputs = model(X_batch)
        # Compute the loss between predictions and true values
        loss = criterion(outputs, y_batch)
        # Backward pass: compute gradients of the loss
        loss.backward()
        # Update model parameters using the optimizer
        optimizer.step()
        # Accumulate the batch loss
        train_loss += loss.item()

    # Validation mode
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    # Calculate average losses and accuracy
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0.0

    # Calculate epoch execution time
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    # Store metrics for each epoch
    metrics_history.append(
        ClassificationMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_accuracy=val_accuracy,
            epoch_time=epoch_time,
        )
    )

    # Keep track of the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()

    # Print metrics every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{N_EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("-" * 50)

total_training_end = time.time()
total_training_time = total_training_end - total_training_start
print(f"Total training time: {total_training_time:.2f} seconds")
print(f"Average time per epoch: {total_training_time/N_EPOCHS:.2f} seconds")

# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(
        f"Best model loaded with validation loss of {best_val_loss/len(val_loader):.4f}"
    )

# Evaluation on test set
model.eval()
with torch.no_grad():
    # Convert test data to tensors directly
    X_test_tensor = torch.FloatTensor(X_test_cls).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.LongTensor(y_test_cls)

    # Get predictions for all test data at once
    test_logits = model(X_test_tensor)
    test_predictions = torch.argmax(test_logits, dim=1).detach().numpy()
    test_targets = y_test_tensor.detach().numpy()

# Print evaluation metrics
print_classification_metrics(test_targets, test_predictions, "test")

# Visualize the evolution of metrics
plot_classification_metrics_sequence(
    metrics_history,
    title="Evolution of Training Metrics for CNN Classification (Without Augmentation)",
    save_path=str(project_root) + "/plots/training_metrics_cnn_classification.png",
)


# =============================================================================
# CNN for classification with data augmentation
# =============================================================================
print("\n" + "=" * 80)
print("CNN FOR CLASSIFICATION (NUM_CLASSE PREDICTION) WITH DATA AUGMENTATION")
print("=" * 80)

# Re-use the same train-test-validation split
train_data = data_3cl.iloc[indices_train].copy()
val_data = data_3cl.iloc[indices_val].copy()
test_data = data_3cl.iloc[indices_test].copy()

# Create augmentation parameters
augmentation_params = AugmentationParams(
    mixup_alpha=0.4,  # Higher alpha for more diverse mixing
    gaussian_noise_std=0.03,  # Increased noise for better robustness
    jitter_factor=0.04,  # More intensity variation
    augmentation_probability=0.8,  # Higher probability of applying augmentation
    by=[
        "symptom",
        "variety",
        "plotLocation",
        "num_classe",
    ],  # Group by these columns, including class
    batch_size=100,  # Generate 100 samples per group
    exclude_columns=None,  # Don't exclude any columns to keep all spectral data
)

# Create augmenter and augment training data
augmenter = DataAugmenter(augmentation_params)
train_data_augmented = augmenter.augment(train_data)

print(f"Training set (original): {len(train_data)} samples")
print(f"Training set (after augmentation): {len(train_data_augmented)} samples")
print(f"Validation set: {len(val_data)} samples")
print(f"Test set: {len(test_data)} samples")

# Prepare features and targets for different sets with augmented data
X_train_cls = np.array(train_data_augmented[spectral_cols])
y_train_cls = label_encoder.transform(train_data_augmented["num_classe"])

X_val_cls = np.array(val_data[spectral_cols])
y_val_cls = label_encoder.transform(val_data["num_classe"])

X_test_cls = np.array(test_data[spectral_cols])
y_test_cls = label_encoder.transform(test_data["num_classe"])

# Standardize features for classification
scaler_X_cls = StandardScaler()

X_train_cls = scaler_X_cls.fit_transform(X_train_cls)
X_val_cls = scaler_X_cls.transform(X_val_cls)
X_test_cls = scaler_X_cls.transform(X_test_cls)

# Create datasets and loaders for training and validation only
X_train_cls = np.array(X_train_cls)
y_train_cls = np.array(y_train_cls)
X_val_cls = np.array(X_val_cls)
y_val_cls = np.array(y_val_cls)

train_dataset = SpectralDataset(X_train_cls, y_train_cls)
val_dataset = SpectralDataset(X_val_cls, y_val_cls)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss function and optimizer
model_aug = CNNModel(input_dim=len(spectral_cols), output_dim=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_aug.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float("inf")
best_model_state = None

# List to store training metrics
metrics_history_aug: List[ClassificationMetrics] = []

print("Starting CNN training for classification with augmentation...")
total_training_start = time.time()
for epoch in range(N_EPOCHS):
    epoch_start = time.time()
    # Set model to training mode
    model_aug.train()
    # Initialize the total training loss for this epoch
    train_loss = 0.0
    # Iterate over mini-batches of training data
    for X_batch, y_batch in train_loader:
        # Reset gradients to zero before computing new gradients
        optimizer.zero_grad()
        # Forward pass: compute model predictions
        outputs = model_aug(X_batch)
        # Compute the loss between predictions and true values
        loss = criterion(outputs, y_batch)
        # Backward pass: compute gradients of the loss
        loss.backward()
        # Update model parameters using the optimizer
        optimizer.step()
        # Accumulate the batch loss
        train_loss += loss.item()

    # Validation mode
    model_aug.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model_aug(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    # Calculate average losses and accuracy
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0.0

    # Calculate epoch execution time
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    # Store metrics for each epoch
    metrics_history_aug.append(
        ClassificationMetrics(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            val_accuracy=val_accuracy,
            epoch_time=epoch_time,
        )
    )

    # Keep track of the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model_aug.state_dict().copy()

    # Print metrics every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{N_EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print("-" * 50)

total_training_end = time.time()
total_training_time = total_training_end - total_training_start
print(f"Total training time: {total_training_time:.2f} seconds")
print(f"Average time per epoch: {total_training_time/N_EPOCHS:.2f} seconds")

# Load the best model
if best_model_state is not None:
    model_aug.load_state_dict(best_model_state)
    print(
        f"Best model loaded with validation loss of {best_val_loss/len(val_loader):.4f}"
    )

# Evaluation on test set
model_aug.eval()
with torch.no_grad():
    # Convert test data to tensors directly
    X_test_tensor = torch.FloatTensor(X_test_cls).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.LongTensor(y_test_cls)

    # Get predictions for all test data at once
    test_logits = model_aug(X_test_tensor)
    test_predictions = torch.argmax(test_logits, dim=1).detach().numpy()
    test_targets = y_test_tensor.detach().numpy()

# Print evaluation metrics
print_classification_metrics(test_targets, test_predictions, "test")

# Visualize the evolution of metrics
plot_classification_metrics_sequence(
    metrics_history_aug,
    title="Evolution of Training Metrics for CNN Classification (With Augmentation)")

# Compare results between models
print("\n" + "=" * 80)
print("COMPARISON BETWEEN MODELS (WITH VS WITHOUT AUGMENTATION)")
print("=" * 80)

# Compare the two models by evaluating both on the test set
model.eval()
model_aug.eval()

with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_cls).unsqueeze(1)

    # Get predictions from both models
    test_logits_standard = model(X_test_tensor)
    test_logits_augmented = model_aug(X_test_tensor)

    test_pred_standard = torch.argmax(test_logits_standard, dim=1).detach().numpy()
    test_pred_augmented = torch.argmax(test_logits_augmented, dim=1).detach().numpy()

# Calculate metrics for both models
accuracy_standard = accuracy_score(test_targets, test_pred_standard)
accuracy_augmented = accuracy_score(test_targets, test_pred_augmented)

precision_standard = precision_score(test_targets, test_pred_standard, average="macro")
precision_augmented = precision_score(
    test_targets, test_pred_augmented, average="macro"
)

recall_standard = recall_score(test_targets, test_pred_standard, average="macro")
recall_augmented = recall_score(test_targets, test_pred_augmented, average="macro")

f1_standard = f1_score(test_targets, test_pred_standard, average="macro")
f1_augmented = f1_score(test_targets, test_pred_augmented, average="macro")

# Print comparison
print(f"Accuracy (standard): {accuracy_standard:.4f}")
print(f"Accuracy (augmented): {accuracy_augmented:.4f}")
print(
    f"Improvement: {(accuracy_augmented - accuracy_standard) / accuracy_standard * 100:.2f}%"
)
print()

print(f"Precision (standard): {precision_standard:.4f}")
print(f"Precision (augmented): {precision_augmented:.4f}")
print(
    f"Improvement: {(precision_augmented - precision_standard) / precision_standard * 100:.2f}%"
)
print()

print(f"Recall (standard): {recall_standard:.4f}")
print(f"Recall (augmented): {recall_augmented:.4f}")
print(
    f"Improvement: {(recall_augmented - recall_standard) / recall_standard * 100:.2f}%"
)
print()

print(f"F1 Score (standard): {f1_standard:.4f}")
print(f"F1 Score (augmented): {f1_augmented:.4f}")
print(f"Improvement: {(f1_augmented - f1_standard) / f1_standard * 100:.2f}%")

# Print classification reports for both models
print("\nClassification Report (Standard Model):")
cls_report_standard = classification_report(test_targets, test_pred_standard, digits=4)
print(cls_report_standard)

print("\nClassification Report (Augmented Model):")
cls_report_augmented = classification_report(
    test_targets, test_pred_augmented, digits=4
)
print(cls_report_augmented)

# Visualize confusion matrices side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot confusion matrix for standard model
cm_standard = confusion_matrix(test_targets, test_pred_standard)
sns.heatmap(cm_standard, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_title(f"Standard Model\nAccuracy: {accuracy_standard:.4f}")
ax1.set_ylabel("True Label")
ax1.set_xlabel("Predicted Label")

# Plot confusion matrix for augmented model
cm_augmented = confusion_matrix(test_targets, test_pred_augmented)
sns.heatmap(cm_augmented, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_title(f"Augmented Model\nAccuracy: {accuracy_augmented:.4f}")
ax2.set_ylabel("True Label")
ax2.set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig(
    str(project_root) + "/plots/cnn_classification_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
