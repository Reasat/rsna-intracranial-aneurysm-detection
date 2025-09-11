#!/usr/bin/env python3
"""
RSNA 2025 Intracranial Aneurysm Detection - 2.5D EfficientNet Training

Extracted from 2-5d-efficientnet-rsna-cta-training.ipynb with optimizations:
- Hybrid dual-stream model (full image + ROI)
- 5-slice window processing (2.5D)
- Cross-validation training
- Proper coordinate handling with train_localizers.csv
- Multi-GPU support
- Memory-efficient inference

Usage:
    python train.py --config configs/train_config.yaml
    python train.py --architecture tf_efficientnet_b0 --epochs 10 --batch_size 16
"""

import os
import sys
import argparse
import gc
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from matplotlib import patches
import yaml

# Import utilities
from utils import (
    LABEL_COLS, ID_COL, parse_coordinates, valid_coords, coords_to_px, 
    make_bbox_px, crop_and_resize_hwc, load_cached_volume, take_window,
    compute_weighted_auc, cleanup_memory, setup_device, setup_reproducibility,
    filter_available_series, create_coordinate_lookup
)

# ============================================================================
# Configuration and Constants
# ============================================================================

class Config:
    """Training configuration loaded from YAML file"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load from YAML file
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
        else:
            # Use default config if no file provided
            self.load_defaults()
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file"""
        print(f"📄 Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Model settings
        model_config = config.get('model', {})
        self.architecture = model_config.get('architecture', 'tf_efficientnet_b0')
        self.num_classes = model_config.get('num_classes', 14)
        self.in_channels = model_config.get('in_channels', 5)
        self.pretrained = model_config.get('pretrained', True)
        
        # Training settings
        training_config = config.get('training', {})
        self.epochs = training_config.get('epochs', 5)
        self.batch_size = training_config.get('batch_size', 8)
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.num_folds = training_config.get('num_folds', 5)
        self.use_cv = training_config.get('use_cv', True)
        self.mixed_precision = training_config.get('mixed_precision', True)
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        
        # Optimizer settings
        optimizer_config = training_config.get('optimizer', {})
        self.optimizer_type = optimizer_config.get('type', 'adam')
        self.weight_decay = optimizer_config.get('weight_decay', 0.01)
        self.betas = optimizer_config.get('betas', [0.9, 0.999])
        
        # Scheduler settings  
        scheduler_config = training_config.get('scheduler', {})
        self.scheduler_type = scheduler_config.get('type', 'cosine')
        self.warmup_epochs = scheduler_config.get('warmup_epochs', 1)
        self.min_lr = scheduler_config.get('min_lr', 1e-6)
        
        # Data settings
        data_config = config.get('data', {})
        self.img_size = data_config.get('img_size', 224)
        self.window_offsets = tuple(data_config.get('window_offsets', [-2, -1, 0, 1, 2]))
        self.roi_box_fraction = data_config.get('roi_box_fraction', 0.15)
        self.roi_min_pixels = data_config.get('roi_min_pixels', 24)
        self.modalities = data_config.get('modalities', ['CTA'])
        
        # Augmentation settings
        aug_config = data_config.get('augmentation', {})
        self.horizontal_flip_prob = aug_config.get('horizontal_flip', 0.5)
        self.affine_config = aug_config.get('affine', {})
        self.gaussian_noise_config = aug_config.get('gaussian_noise', {})
        self.motion_blur_config = aug_config.get('motion_blur', {})
        
        # Paths
        paths_config = config.get('paths', {})
        self.data_dir = paths_config.get('data_dir', "/mnt/storage/kaggle_data/rsna-intracranial-aneurysm-detection")
        self.cache_dir = paths_config.get('cache_dir', "/mnt/storage/kaggle_data/rsna-intracranial-aneurysm-detection/processed_data/2_5d_volumes_full")
        self.train_csv = paths_config.get('train_csv', "/mnt/storage/kaggle_data/rsna-intracranial-aneurysm-detection/train.csv")
        self.localizers_csv = paths_config.get('localizers_csv', "/mnt/storage/kaggle_data/rsna-intracranial-aneurysm-detection/train_localizers.csv")
        self.output_dir = paths_config.get('output_dir', "models")
        
        # System settings
        system_config = config.get('system', {})
        self.num_workers = system_config.get('num_workers', 4)
        self.device = system_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = system_config.get('seed', 42)
        
        # Validation settings
        validation_config = config.get('validation', {})
        self.metric = validation_config.get('metric', 'weighted_auc')
        self.save_best_only = validation_config.get('save_best_only', True)
        self.patience = validation_config.get('patience', 3)
        
        # Debug settings
        debug_config = config.get('debug', {})
        self.debug_mode = debug_config.get('enabled', False)
        self.subset_per_class = debug_config.get('subset_per_class', 50)
        self.fast_dev_run = debug_config.get('fast_dev_run', False)
        
        print(f"✅ Configuration loaded successfully")
        
    def load_defaults(self):
        """Load default configuration when no YAML file is provided"""
        print("⚠️  No config file provided, using defaults")
        
        # Model settings
        self.architecture = 'tf_efficientnet_b0'
        self.num_classes = 14
        self.in_channels = 5
        self.pretrained = True
        
        # Training settings
        self.epochs = 5
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.num_folds = 5
        self.use_cv = True
        self.mixed_precision = True
        self.gradient_accumulation_steps = 1
        
        # Optimizer settings
        self.optimizer_type = 'adam'
        self.weight_decay = 0.01
        self.betas = [0.9, 0.999]
        
        # Scheduler settings
        self.scheduler_type = 'cosine'
        self.warmup_epochs = 1
        self.min_lr = 1e-6
        
        # Data settings
        self.img_size = 224
        self.window_offsets = (-2, -1, 0, 1, 2)
        self.roi_box_fraction = 0.15
        self.roi_min_pixels = 24
        self.modalities = ['CTA']
        
        # Augmentation settings (defaults)
        self.horizontal_flip_prob = 0.5
        self.affine_config = {'scale': [0.95, 1.05], 'translate_percent': 0.05, 'rotate': [-10, 10], 'probability': 0.5}
        self.gaussian_noise_config = {'var_limit': [0.01, 0.05], 'probability': 0.2}
        self.motion_blur_config = {'blur_limit': 3, 'probability': 0.1}
        
        # Paths
        self.data_dir = "/mnt/storage/kaggle_data/rsna-intracranial-aneurysm-detection"
        self.cache_dir = "/mnt/storage/kaggle_data/rsna-intracranial-aneurysm-detection/processed_data/2_5d_volumes_full"
        self.train_csv = "/mnt/storage/kaggle_data/rsna-intracranial-aneurysm-detection/train.csv"
        self.localizers_csv = "/mnt/storage/kaggle_data/rsna-intracranial-aneurysm-detection/train_localizers.csv"
        self.output_dir = "models"
        
        # System settings
        self.num_workers = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        
        # Validation settings
        self.metric = 'weighted_auc'
        self.save_best_only = True
        self.patience = 3
        
        # Debug settings
        self.debug_mode = False
        self.subset_per_class = 50
        self.fast_dev_run = False
    
    def override_from_args(self, args):
        """Override configuration with command line arguments"""
        if args.architecture:
            self.architecture = args.architecture
        if args.epochs:
            self.epochs = args.epochs
        if args.batch_size:
            self.batch_size = args.batch_size
        if args.learning_rate:
            self.learning_rate = args.learning_rate
        if args.cache_dir:
            self.cache_dir = args.cache_dir
        if args.output_dir:
            self.output_dir = args.output_dir
        if args.debug:
            self.debug_mode = True
    
    def print_config(self):
        """Print current configuration"""
        print(f"\n🔧 Training Configuration:")
        print(f"  Architecture: {self.architecture}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Image size: {self.img_size}")
        print(f"  Window offsets: {self.window_offsets}")
        print(f"  Device: {self.device}")
        print(f"  Cache directory: {self.cache_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Debug mode: {self.debug_mode}")
    
    def save_config(self, save_path: str):
        """Save current configuration to YAML file"""
        config_dict = {
            'model': {
                'architecture': self.architecture,
                'num_classes': self.num_classes,
                'in_channels': self.in_channels,
                'pretrained': self.pretrained
            },
            'training': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_folds': self.num_folds,
                'use_cv': self.use_cv,
                'mixed_precision': self.mixed_precision
            },
            'data': {
                'img_size': self.img_size,
                'window_offsets': list(self.window_offsets),
                'roi_box_fraction': self.roi_box_fraction,
                'roi_min_pixels': self.roi_min_pixels,
                'modalities': self.modalities
            },
            'paths': {
                'data_dir': self.data_dir,
                'cache_dir': self.cache_dir,
                'train_csv': self.train_csv,
                'localizers_csv': self.localizers_csv,
                'output_dir': self.output_dir
            },
            'system': {
                'num_workers': self.num_workers,
                'device': self.device,
                'seed': self.seed
            }
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        print(f"💾 Configuration saved to: {save_path}")

# Note: Utility functions moved to utils.py

# ============================================================================
# Dataset Class
# ============================================================================

class HybridAneurysmDataset(Dataset):
    """Hybrid dataset for dual-stream training (full image + ROI)"""
    
    def __init__(self, df: pd.DataFrame, config: Config, localizers_df: Optional[pd.DataFrame] = None,
                 transform: Optional[A.Compose] = None, is_training: bool = True):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.localizers_df = localizers_df
        self.transform = transform
        self.is_training = is_training
        
        # Create coordinate lookup for faster access
        self.coord_lookup = create_coordinate_lookup(localizers_df)
        
        print(f"Dataset initialized with {len(self.df)} samples")
        if localizers_df is not None:
            print(f"Coordinates available for {len(self.coord_lookup)} series")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        sid = str(row[ID_COL])
        
        # Load cached volume
        volume_path = os.path.join(self.config.cache_dir, f"{sid}.npz")
        try:
            volume = load_cached_volume(volume_path)  # (N, H, W)
        except FileNotFoundError:
            # Create dummy volume if file not found
            volume = np.zeros((32, self.config.img_size, self.config.img_size), dtype=np.float32)
        
        # Get coordinates
        coords = self.coord_lookup.get(sid, np.array([0.0, 0.0], dtype=np.float32))
        
        # Take window from center of volume
        center_idx = volume.shape[0] // 2
        img_window = take_window(volume, center_idx, self.config.window_offsets)  # (5, H, W)
        
        # Convert to HWC for processing
        img_hwc = np.transpose(img_window, (1, 2, 0)).astype(np.float32)  # (H, W, 5)
        
        # Create ROI image
        if valid_coords(coords):
            cx, cy = coords_to_px(coords, self.config.img_size)
            x1, y1, x2, y2 = make_bbox_px(cx, cy, self.config.img_size, 
                                        self.config.roi_box_fraction, self.config.roi_min_pixels)
            img_roi_hwc = crop_and_resize_hwc(img_hwc, x1, y1, x2, y2, self.config.img_size)
        else:
            img_roi_hwc = img_hwc.copy()
        
        # Resize full image to target size
        img_full_hwc = cv2.resize(img_hwc, (self.config.img_size, self.config.img_size), 
                                 interpolation=cv2.INTER_AREA)
        
        # Apply augmentations if training
        if self.transform and self.is_training:
            # Apply same augmentation to both images
            augmented = self.transform(image=img_full_hwc, image2=img_roi_hwc)
            img_full_hwc = augmented["image"]
            img_roi_hwc = augmented["image2"]
        
        # Convert to CHW tensors
        x_full = torch.from_numpy(np.transpose(img_full_hwc, (2, 0, 1)).copy()).float()
        x_roi = torch.from_numpy(np.transpose(img_roi_hwc, (2, 0, 1)).copy()).float()
        
        # Labels
        vals = pd.to_numeric(row[LABEL_COLS], errors="coerce").values.astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0)
        y = torch.from_numpy(vals)
        
        # Coordinates
        c = torch.from_numpy(coords)
        
        return x_full, x_roi, y, c

# ============================================================================
# Model Definition
# ============================================================================

class HybridAneurysmModel(nn.Module):
    """Hybrid dual-stream model for aneurysm detection"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Backbone network
        self.backbone = timm.create_model(
            config.architecture, 
            in_chans=config.in_channels, 
            num_classes=0, 
            pretrained=True
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

# ============================================================================
# Training Functions
# ============================================================================

def create_transforms(config: Config) -> Tuple[A.Compose, Optional[A.Compose]]:
    """Create training and validation transforms from config"""
    
    transforms = [A.HorizontalFlip(p=config.horizontal_flip_prob)]
    
    # Add affine transforms if configured
    if hasattr(config, 'affine_config') and config.affine_config:
        affine_cfg = config.affine_config
        transforms.append(A.Affine(
            scale=affine_cfg.get('scale', (0.95, 1.05)),
            translate_percent={
                'x': (-affine_cfg.get('translate_percent', 0.05), affine_cfg.get('translate_percent', 0.05)),
                'y': (-affine_cfg.get('translate_percent', 0.05), affine_cfg.get('translate_percent', 0.05))
            },
            rotate=affine_cfg.get('rotate', (-10, 10)),
            p=affine_cfg.get('probability', 0.5)
        ))
    
    # Add noise if configured
    if hasattr(config, 'gaussian_noise_config') and config.gaussian_noise_config:
        noise_cfg = config.gaussian_noise_config
        transforms.append(A.GaussNoise(
            var_limit=noise_cfg.get('var_limit', (0.01, 0.05)),
            p=noise_cfg.get('probability', 0.2)
        ))
    
    # Add motion blur if configured
    if hasattr(config, 'motion_blur_config') and config.motion_blur_config:
        blur_cfg = config.motion_blur_config
        transforms.append(A.MotionBlur(
            blur_limit=blur_cfg.get('blur_limit', 3),
            p=blur_cfg.get('probability', 0.1)
        ))
    
    train_transform = A.Compose(transforms, additional_targets={'image2': 'image'})
    val_transform = None  # Keep original normalization
    
    return train_transform, val_transform

def create_data_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       config: Config, localizers_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    train_transform, val_transform = create_transforms(config)
    
    train_dataset = HybridAneurysmDataset(
        train_df, config, localizers_df, train_transform, is_training=True
    )
    val_dataset = HybridAneurysmDataset(
        val_df, config, localizers_df, val_transform, is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers // 2,
        pin_memory=True
    )
    
    return train_loader, val_loader

@torch.no_grad()
def validate_model(model: nn.Module, val_loader: DataLoader, config: Config) -> Dict[str, float]:
    """Validate model and compute metrics including validation loss"""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    all_probs = []
    all_targets = []
    val_losses = []
    
    for batch in tqdm(val_loader, desc="Validation", leave=False):
        x_full, x_roi, y, coords = [x.to(config.device) for x in batch]
        
        logits = model(x_full, x_roi, coords)
        
        # Calculate validation loss
        val_loss = criterion(logits, y)
        val_losses.append(val_loss.item())
        
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = y.cpu().numpy()
        
        all_probs.append(probs)
        all_targets.append(targets)
    
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    avg_val_loss = np.mean(val_losses)
    
    # Compute metrics
    weighted_auc, ap_auc, others_mean, per_class_auc, skipped = compute_weighted_auc(
        all_targets, all_probs, LABEL_COLS
    )
    
    return {
        'weighted_auc': weighted_auc,
        'aneurysm_present_auc': ap_auc,
        'others_mean_auc': others_mean,
        'per_class_auc': per_class_auc,
        'skipped_classes': skipped,
        'val_loss': avg_val_loss
    }

def train_fold(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame, 
               config: Config, localizers_df: pd.DataFrame, experiment_dir: str) -> str:
    """Train model for one fold with enhanced logging and saving"""
    
    print(f"\n🔄 Training Fold {fold}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_df, val_df, config, localizers_df)
    
    # Create model
    model = HybridAneurysmModel(config)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(config.device)
    
    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    # Loss history tracking
    loss_history = []
    best_weighted_auc = -1.0
    best_val_loss = float('inf')
    best_model_path = None
    final_model_path = None
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            x_full, x_roi, y, coords = [x.to(config.device) for x in batch]
            
            optimizer.zero_grad()
            
            if config.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = model(x_full, x_roi, coords)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_full, x_roi, coords)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            if (step + 1) % 10 == 0:
                pbar.set_postfix(loss=running_loss / (step + 1))
        
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        val_metrics = validate_model(model, val_loader, config)
        val_loss = val_metrics['val_loss']
        
        # Record loss history (both training and validation)
        loss_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': val_loss
        })
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"wAUC: {val_metrics['weighted_auc']:.4f} | "
              f"AP-AUC: {val_metrics['aneurysm_present_auc']:.4f}")
        
        # Save best model based on weighted AUC
        if val_metrics['weighted_auc'] > best_weighted_auc:
            best_weighted_auc = val_metrics['weighted_auc']
            best_val_loss = val_loss
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            # Create filename without loss value
            best_model_filename = f"{config.architecture}_fold{fold}_best.pth"
            best_model_path = os.path.join(experiment_dir, best_model_filename)
            torch.save(model_state, best_model_path)
        
        # Save loss history after each epoch
        loss_df = pd.DataFrame(loss_history)
        loss_history_path = os.path.join(experiment_dir, f"loss_history_fold{fold}.csv")
        loss_df.to_csv(loss_history_path, index=False)
        
        # Memory cleanup
        cleanup_memory()
    
    # Save final model 
    final_val_loss = loss_history[-1]['val_loss'] if loss_history else 0.0
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    final_model_filename = f"{config.architecture}_fold{fold}_final.pth"
    final_model_path = os.path.join(experiment_dir, final_model_filename)
    torch.save(model_state, final_model_path)
    
    print(f"💾 Saved best model: {best_model_path} (wAUC: {best_weighted_auc:.4f}, Val Loss: {best_val_loss:.4f})")
    print(f"💾 Saved final model: {final_model_path} (Val Loss: {final_val_loss:.4f})")
    print(f"📊 Loss history saved: {loss_history_path}")
    
    return best_model_path

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RSNA 2025 Aneurysm Detection Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--architecture", type=str, default="tf_efficientnet_b0", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--cache_dir", type=str, help="Cache directory path")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode with subset of data")
    
    args = parser.parse_args()
    
    # Create experiment timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"🕐 Experiment timestamp: {timestamp}")
    
    # Load configuration from YAML or use defaults
    config = Config(args.config)
    
    # Override with command line arguments
    config.override_from_args(args)
    
    # Create timestamped experiment directory
    experiment_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Print configuration
    print(f"🚀 Starting training...")
    print(f"📁 Experiment directory: {experiment_dir}")
    config.print_config()
    
    # Save configuration for reproducibility
    config_save_path = os.path.join(experiment_dir, "used_config.yaml")
    config.save_config(config_save_path)
    
    # Save experiment info
    experiment_info = {
        'timestamp': timestamp,
        'experiment_dir': experiment_dir,
        'architecture': config.architecture,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'debug_mode': config.debug_mode
    }
    experiment_info_path = os.path.join(experiment_dir, "experiment_info.yaml")
    with open(experiment_info_path, 'w') as f:
        yaml.dump(experiment_info, f, default_flow_style=False, indent=2)
    
    # Load data
    print("\n📊 Loading training data...")
    df = pd.read_csv(config.train_csv)
    
    # Filter by specified modalities
    if config.modalities and len(config.modalities) > 0:
        df = df[df['Modality'].isin(config.modalities)].reset_index(drop=True)
        modality_str = ", ".join(config.modalities)
        print(f"Filtered to modalities: {modality_str}")
    else:
        # Use all modalities if none specified
        modality_str = "ALL"
        print("Using all available modalities")
    
    # Load localizers
    localizers_df = pd.read_csv(config.localizers_csv)
    
    # Print dataset composition
    print(f"Loaded {len(df)} series across {df['Modality'].nunique()} modalities:")
    modality_counts = df['Modality'].value_counts()
    for modality, count in modality_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {modality}: {count} series ({percentage:.1f}%)")
    print(f"Localizer annotations: {len(localizers_df)}")
    
    # Debug mode: use subset
    if config.debug_mode:
        df = df.groupby('Aneurysm Present').apply(lambda x: x.head(50)).reset_index(drop=True)
        print(f"Debug mode: using {len(df)} samples")
    
    # Cross-validation setup
    if config.use_cv:
        df['fold'] = -1
        skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)
        for fold, (_, val_idx) in enumerate(skf.split(df, df['Aneurysm Present'])):
            df.loc[val_idx, 'fold'] = fold
    else:
        df['fold'] = 0
        config.num_folds = 1
    
    # Train each fold
    best_models = []
    for fold in range(config.num_folds):
        train_df = df[df['fold'] != fold].reset_index(drop=True)
        val_df = df[df['fold'] == fold].reset_index(drop=True)
        
        best_model_path = train_fold(fold, train_df, val_df, config, localizers_df, experiment_dir)
        best_models.append(best_model_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Experiment directory: {experiment_dir}")
    print(f"🕐 Timestamp: {timestamp}")
    print(f"💾 Best models saved:")
    for model_path in best_models:
        print(f"  {model_path}")
    
    print(f"\n📊 Loss history files:")
    for fold in range(config.num_folds):
        loss_history_path = os.path.join(experiment_dir, f"loss_history_fold{fold}.csv")
        if os.path.exists(loss_history_path):
            print(f"  {loss_history_path}")
    
    print(f"\n📋 Experiment summary saved in: {experiment_dir}")

if __name__ == "__main__":
    main()
