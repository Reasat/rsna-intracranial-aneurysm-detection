#!/usr/bin/env python3
"""
RSNA 2025 Intracranial Aneurysm Detection - Binary Classification Training

Binary classification training for aneurysm presence/absence detection:
- Dual-stream model (full image + ROI) + coordinates
- 5-slice window processing (2.5D)
- Cross-validation training
- Binary classification (aneurysm present/absent)
- Multi-GPU support
- Memory-efficient inference

Usage:
    python train_binary.py --config configs/binary_config.yaml
    python train_binary.py --config configs/binary_config.yaml --epochs 10 --batch_size 32
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
from loss import get_training_loss, get_validation_loss
from model import BinaryAneurysmModel, create_binary_model, create_model_by_type
from datamodules import BinaryAneurysmDataset, create_binary_data_loaders
from config import Config

# ============================================================================
# Configuration and Constants
# ============================================================================
# Config class moved to config.py

# ============================================================================
# Dataset Class
# ============================================================================
# Dataset classes moved to datamodules.py

# ============================================================================
# Model Definition
# ============================================================================
# Model classes moved to model.py

# ============================================================================
# Training Functions
# ============================================================================

# Data loading functions moved to datamodules.py

@torch.no_grad()
def validate_binary_model(model: nn.Module, val_loader: DataLoader, config: Config) -> Dict[str, float]:
    """Validate binary model and compute metrics"""
    model.eval()
    loss_config = {"loss_name": config.loss_name, "loss_params": config.loss_params}
    criterion = get_validation_loss(loss_config)
    
    all_probs = []
    all_targets = []
    val_losses = []
    
    for batch in tqdm(val_loader, desc="Validation", leave=False):
        x_full, x_roi, y, coords = [x.to(config.device) for x in batch]  # 4 items for binary
        
        logits = model(x_full, x_roi, coords)  # Binary model takes x_full, x_roi, coords
        
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
    
    # Binary evaluation - simple ROC AUC
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(all_targets.flatten(), all_probs.flatten())
    
    return {
        'auc': auc_score,
        'val_loss': avg_val_loss
    }

def train_binary_fold(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                     config: Config, experiment_dir: str, localizers_df: pd.DataFrame) -> str:
    """Train binary model for one fold"""
    
    print(f"\n🔄 Training Fold {fold}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Create binary data loaders
    train_loader, val_loader = create_binary_data_loaders(train_df, val_df, config, localizers_df)
    
    # Create binary model
    model = create_binary_model(config)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(config.device)
    
    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_config = {"loss_name": config.loss_name, "loss_params": config.loss_params}
    criterion = get_training_loss(loss_config)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    # Loss history tracking
    loss_history = []
    best_auc = -1.0
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            x_full, x_roi, y, coords = [x.to(config.device) for x in batch]  # 4 items for binary
            
            optimizer.zero_grad()
            
            if config.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = model(x_full, x_roi, coords)  # Binary model takes x_full, x_roi, coords
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x_full, x_roi, coords)  # Binary model takes x_full, x_roi, coords
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            if (step + 1) % 10 == 0:
                pbar.set_postfix(loss=running_loss / (step + 1))
        
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        val_metrics = validate_binary_model(model, val_loader, config)
        val_loss = val_metrics['val_loss']
        auc = val_metrics['auc']
        
        # Record loss history
        loss_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': val_loss
        })
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"AUC: {auc:.4f}")
        
        # Save best model based on AUC
        if auc > best_auc:
            best_auc = auc
            best_val_loss = val_loss
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            best_model_filename = f"{config.architecture}_fold{fold}_best.pth"
            best_model_path = os.path.join(experiment_dir, best_model_filename)
            torch.save(model_state, best_model_path)
        
        # Save loss history
        loss_df = pd.DataFrame(loss_history)
        loss_history_path = os.path.join(experiment_dir, f"loss_history_fold{fold}.csv")
        loss_df.to_csv(loss_history_path, index=False)
        
        # Memory cleanup
        cleanup_memory()
    
    print(f"💾 Saved best model: {best_model_path} (AUC: {best_auc:.4f})")
    print(f"📊 Loss history saved: {loss_history_path}")
    
    return best_model_path

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RSNA 2025 Aneurysm Detection - Binary Classification Training")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Create experiment timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"🕐 Experiment timestamp: {timestamp}")
    
    # Load configuration from YAML
    config = Config(args.config)
    
    # Force binary mode
    config.update_training_mode('binary')
    print(f"🔧 Training mode: Binary Classification")
    
    # Create timestamped experiment directory
    experiment_dir = os.path.join(config.output_dir, f"{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging to redirect stdout to log file
    log_file_path = config.setup_logging(experiment_dir)
    
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
        'training_mode': 'binary',
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
    
    # Load localizers data for coordinates
    localizers_df = pd.read_csv(config.localizers_csv)
    print(f"Loaded {len(localizers_df)} localizer coordinates")
    
    # Filter by specified modalities
    if config.modalities and len(config.modalities) > 0:
        df = df[df['Modality'].isin(config.modalities)].reset_index(drop=True)
        modality_str = ", ".join(config.modalities)
        print(f"Filtered to modalities: {modality_str}")
    else:
        # Use all modalities if none specified
        modality_str = "ALL"
        print("Using all available modalities")
    
    # Print dataset composition
    print(f"Loaded {len(df)} series across {df['Modality'].nunique()} modalities:")
    modality_counts = df['Modality'].value_counts()
    for modality, count in modality_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {modality}: {count} series ({percentage:.1f}%)")
    
    # Print binary class distribution
    aneurysm_present = df['Aneurysm Present'].sum()
    aneurysm_absent = len(df) - aneurysm_present
    print(f"Binary class distribution:")
    print(f"  Aneurysm Present: {aneurysm_present} ({aneurysm_present/len(df)*100:.1f}%)")
    print(f"  Aneurysm Absent: {aneurysm_absent} ({aneurysm_absent/len(df)*100:.1f}%)")
    
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
        
        best_model_path = train_binary_fold(fold, train_df, val_df, config, experiment_dir, localizers_df)
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
    
    # Close logging and restore stdout
    config.close_logging()

if __name__ == "__main__":
    main()
