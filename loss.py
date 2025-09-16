"""
Loss Functions for RSNA 2025 Intracranial Aneurysm Detection

This module provides centralized loss function configuration for the competition.
Supports both training and validation loss functions with consistent behavior.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class LossConfig:
    """Configuration class for loss functions"""
    
    # Default loss function
    DEFAULT_LOSS = "BCEWithLogitsLoss"
    
    # Available loss functions
    AVAILABLE_LOSSES = {
        "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
        "BCELoss": nn.BCELoss,
        "CrossEntropyLoss": nn.CrossEntropyLoss,
    }
    
    # Loss-specific parameters
    LOSS_PARAMS = {
        "BCEWithLogitsLoss": {
            "reduction": "mean",
            "pos_weight": None,  # Can be set for class balancing
        },
        "BCELoss": {
            "reduction": "mean",
        },
        "CrossEntropyLoss": {
            "reduction": "mean",
            "ignore_index": -100,
        }
    }


def get_loss_function(loss_name: str = "BCEWithLogitsLoss", 
                     loss_params: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Get loss function by name with optional parameters
    
    Args:
        loss_name: Name of the loss function
        loss_params: Optional parameters for the loss function
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: If loss_name is not supported
    """
    if loss_name not in LossConfig.AVAILABLE_LOSSES:
        available = ", ".join(LossConfig.AVAILABLE_LOSSES.keys())
        raise ValueError(f"Unknown loss function '{loss_name}'. Available: {available}")
    
    # Get default parameters
    params = LossConfig.LOSS_PARAMS.get(loss_name, {}).copy()
    
    # Update with provided parameters
    if loss_params:
        params.update(loss_params)
    
    # Create loss function
    loss_class = LossConfig.AVAILABLE_LOSSES[loss_name]
    return loss_class(**params)


def get_training_loss(config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Get loss function for training
    
    Args:
        config: Optional configuration dictionary with 'loss_name' and 'loss_params'
        
    Returns:
        Loss function for training
    """
    if config is None:
        config = {}
    
    loss_name = config.get("loss_name", LossConfig.DEFAULT_LOSS)
    loss_params = config.get("loss_params", {})
    
    return get_loss_function(loss_name, loss_params)


def get_validation_loss(config: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Get loss function for validation (typically same as training)
    
    Args:
        config: Optional configuration dictionary with 'loss_name' and 'loss_params'
        
    Returns:
        Loss function for validation
    """
    return get_training_loss(config)


def create_class_weighted_loss(class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Create a loss function with class weighting for imbalanced datasets
    
    Args:
        class_weights: Tensor of weights for each class (shape: [num_classes])
        
    Returns:
        Weighted loss function
    """
    if class_weights is not None:
        # For BCEWithLogitsLoss, we use pos_weight for binary classification
        # For multilabel, we can use pos_weight for each class
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        return nn.BCEWithLogitsLoss()


def compute_class_weights(labels: torch.Tensor, method: str = "inverse_freq") -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        labels: True labels tensor (shape: [batch_size, num_classes])
        method: Method for computing weights ("inverse_freq", "balanced")
        
    Returns:
        Class weights tensor
    """
    if method == "inverse_freq":
        # Compute inverse frequency weights
        class_counts = labels.sum(dim=0).float()
        total_samples = labels.shape[0]
        class_weights = total_samples / (class_counts + 1e-8)  # Add small epsilon
        return class_weights
    
    elif method == "balanced":
        # Use sklearn-style balanced weights
        class_counts = labels.sum(dim=0).float()
        n_classes = labels.shape[1]
        class_weights = (labels.shape[0] / (n_classes * class_counts + 1e-8))
        return class_weights
    
    else:
        raise ValueError(f"Unknown method '{method}'. Available: 'inverse_freq', 'balanced'")


# Example usage and testing
if __name__ == "__main__":
    print("Loss Functions Module")
    print("=" * 40)
    
    # Test default loss
    loss_fn = get_training_loss()
    print(f"Default loss: {loss_fn}")
    
    # Test with custom parameters
    custom_config = {
        "loss_name": "BCEWithLogitsLoss",
        "loss_params": {"reduction": "mean"}
    }
    loss_fn = get_training_loss(custom_config)
    print(f"Custom loss: {loss_fn}")
    
    # Test class weighting
    dummy_labels = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.float32)
    class_weights = compute_class_weights(dummy_labels)
    weighted_loss = create_class_weighted_loss(class_weights)
    print(f"Weighted loss: {weighted_loss}")
    print(f"Class weights: {class_weights}")
    
    print("\n✅ All tests passed!")
