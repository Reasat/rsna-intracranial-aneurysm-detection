#!/usr/bin/env python3
"""
RSNA 2025 Intracranial Aneurysm Detection - Model Definitions

This module contains the model architectures for aneurysm detection.
Supports both binary and multiclass classification modes.

Models:
    - HybridAneurysmModel: Multiclass model with dual-stream (full image + ROI) + coordinates
    - BinaryAneurysmModel: Binary model with dual-stream (full image + ROI) + coordinates

Usage:
    # Multiclass model
    from model import HybridAneurysmModel, create_model
    model = create_model(config)
    
    # Binary model
    from model import BinaryAneurysmModel, create_binary_model
    model = create_binary_model(config)
    
    # By type
    from model import create_model_by_type
    model = create_model_by_type(config, 'binary')  # or 'multiclass'
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Union


class HybridAneurysmModel(nn.Module):
    """Hybrid dual-stream model for aneurysm detection"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone network
        self.backbone = timm.create_model(
            config.architecture, 
            in_chans=config.in_channels, 
            num_classes=0, 
            pretrained=config.pretrained
        )
        
        self.num_features = self.backbone.num_features
        
        # Coordinate processing
        self.coord_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features * 2 + 64, config.num_classes)
        )
    
    def forward(self, x_full: torch.Tensor, x_roi: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # Extract features from both streams
        f_full = self.backbone(x_full)
        f_roi = self.backbone(x_roi)
        
        # Coordinate dropout during training
        if self.training and torch.rand(()).item() < 0.3:
            coords = torch.zeros_like(coords)
        
        # Process coordinates
        f_coord = self.coord_fc(coords.float())
        
        # Combine features
        combined = torch.cat([f_full, f_roi, f_coord], dim=1)
        
        return self.classifier(combined)


class BinaryAneurysmModel(nn.Module):
    """Binary classification model for aneurysm detection (present/absent)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone network
        self.backbone = timm.create_model(
            config.architecture, 
            in_chans=config.in_channels, 
            num_classes=0, 
            pretrained=config.pretrained
        )
        
        self.num_features = self.backbone.num_features
        
        # Coordinate processing (same as hybrid)
        self.coord_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Binary classifier - uses full image + ROI + coordinates
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features * 2 + 64, config.num_classes)  # Should be 1 for binary
        )
    
    def forward(self, x_full: torch.Tensor, x_roi: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # Extract features from both streams
        f_full = self.backbone(x_full)
        f_roi = self.backbone(x_roi)
        
        # Coordinate dropout during training (same as hybrid)
        if self.training and torch.rand(()).item() < 0.3:
            coords = torch.zeros_like(coords)
        
        # Process coordinates
        f_coord = self.coord_fc(coords.float())
        
        # Combine features
        combined = torch.cat([f_full, f_roi, f_coord], dim=1)
        
        # Binary classification
        return self.classifier(combined)


def create_model(config) -> nn.Module:
    """
    Create model based on configuration
    
    Args:
        config: Configuration object containing model parameters
        
    Returns:
        PyTorch model instance
    """
    model = HybridAneurysmModel(config)
    return model


def create_binary_model(config) -> nn.Module:
    """
    Create binary classification model
    
    Args:
        config: Configuration object containing model parameters
        
    Returns:
        Binary classification model instance
    """
    model = BinaryAneurysmModel(config)
    return model


def create_model_by_type(config, model_type: str = 'multiclass') -> nn.Module:
    """
    Create model by type (binary or multiclass)
    
    Args:
        config: Configuration object containing model parameters
        model_type: 'binary' or 'multiclass'
        
    Returns:
        PyTorch model instance
    """
    if model_type == 'binary':
        return create_binary_model(config)
    elif model_type == 'multiclass':
        return create_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'binary' or 'multiclass'")


def load_model(model_path: str, config, device: str = 'cuda') -> nn.Module:
    """
    Load model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        device: Device to load model on
        
    Returns:
        Loaded PyTorch model
    """
    model = create_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_binary_model(model_path: str, config, device: str = 'cuda') -> nn.Module:
    """
    Load binary model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        device: Device to load model on
        
    Returns:
        Loaded binary PyTorch model
    """
    model = create_binary_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_model_by_type(model_path: str, config, model_type: str = 'multiclass', device: str = 'cuda') -> nn.Module:
    """
    Load model by type from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        model_type: 'binary' or 'multiclass'
        device: Device to load model on
        
    Returns:
        Loaded PyTorch model
    """
    if model_type == 'binary':
        return load_binary_model(model_path, config, device)
    elif model_type == 'multiclass':
        return load_model(model_path, config, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'binary' or 'multiclass'")
