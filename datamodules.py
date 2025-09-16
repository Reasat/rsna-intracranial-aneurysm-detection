#!/usr/bin/env python3
"""
RSNA 2025 Intracranial Aneurysm Detection - Data Modules

This module contains dataset classes and data loading utilities for aneurysm detection.
Supports both binary and multiclass classification modes.

Datasets:
    - HybridAneurysmDataset: Multiclass dataset with dual-stream (full image + ROI) + coordinates
    - BinaryAneurysmDataset: Binary dataset with dual-stream (full image + ROI) + coordinates

Usage:
    # Multiclass dataset
    from datamodules import HybridAneurysmDataset, create_data_loaders
    dataset = HybridAneurysmDataset(df, config, localizers_df)
    
    # Binary dataset
    from datamodules import BinaryAneurysmDataset, create_binary_data_loaders
    dataset = BinaryAneurysmDataset(df, config)
    
    # Data loaders
    from datamodules import create_data_loaders, create_binary_data_loaders
    # For multiclass: train_loader, val_loader = create_data_loaders(train_df, val_df, config, localizers_df)
    # For binary: train_loader, val_loader = create_binary_data_loaders(train_df, val_df, config, localizers_df)
"""

import os
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2

# Import utilities
from utils import (
    LABEL_COLS, ID_COL, parse_coordinates, valid_coords, coords_to_px, 
    make_bbox_px, crop_and_resize_hwc, load_cached_volume, take_window,
    create_coordinate_lookup
)


class HybridAneurysmDataset(Dataset):
    """Hybrid dataset for dual-stream training (full image + ROI)"""
    
    def __init__(self, df: pd.DataFrame, config, localizers_df: Optional[pd.DataFrame] = None,
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


class BinaryAneurysmDataset(Dataset):
    """Binary dataset for aneurysm presence/absence classification"""
    
    def __init__(self, df: pd.DataFrame, config, localizers_df: Optional[pd.DataFrame] = None,
                 transform: Optional[A.Compose] = None, is_training: bool = True):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.localizers_df = localizers_df
        self.transform = transform
        self.is_training = is_training
        
        # Create coordinate lookup for faster access
        self.coord_lookup = create_coordinate_lookup(localizers_df)
        
        print(f"Binary dataset initialized with {len(self.df)} samples")
        if localizers_df is not None:
            print(f"Coordinates available for {len(self.coord_lookup)} series")
        
        # Print class distribution
        aneurysm_present = self.df['Aneurysm Present'].sum()
        aneurysm_absent = len(self.df) - aneurysm_present
        print(f"  Aneurysm Present: {aneurysm_present} ({aneurysm_present/len(self.df)*100:.1f}%)")
        print(f"  Aneurysm Absent: {aneurysm_absent} ({aneurysm_absent/len(self.df)*100:.1f}%)")
    
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
        
        # Binary label (only "Aneurysm Present")
        y = torch.tensor([row['Aneurysm Present']], dtype=torch.float32)
        
        # Coordinates
        c = torch.from_numpy(coords)
        
        return x_full, x_roi, y, c


def create_transforms(config) -> Tuple[A.Compose, Optional[A.Compose]]:
    """Create training and validation transforms from config for both binary and multiclass"""
    
    transforms = [A.HorizontalFlip(p=config.horizontal_flip_prob)]
    
    # Add affine transforms if configured
    if hasattr(config, 'affine_config') and config.affine_config:
        affine_cfg = config.affine_config
        transforms.append(A.Affine(
            scale=affine_cfg['scale'],
            translate_percent={
                'x': (-affine_cfg['translate_percent'], affine_cfg['translate_percent']),
                'y': (-affine_cfg['translate_percent'], affine_cfg['translate_percent'])
            },
            rotate=affine_cfg['rotate'],
            p=affine_cfg['probability']
        ))
    
    # Add noise if configured
    if hasattr(config, 'gaussian_noise_config') and config.gaussian_noise_config:
        noise_cfg = config.gaussian_noise_config
        transforms.append(A.GaussNoise(
            var_limit=noise_cfg['var_limit'],
            p=noise_cfg['probability']
        ))
    
    # Add motion blur if configured
    if hasattr(config, 'motion_blur_config') and config.motion_blur_config:
        blur_cfg = config.motion_blur_config
        transforms.append(A.MotionBlur(
            blur_limit=blur_cfg['blur_limit'],
            p=blur_cfg['probability']
        ))
    
    train_transform = A.Compose(transforms, additional_targets={'image2': 'image'})
    val_transform = None  # Keep original normalization
    
    return train_transform, val_transform


def create_data_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       config, localizers_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
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


def create_binary_data_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                              config, localizers_df: pd.DataFrame = None) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders for binary classification"""
    
    train_transform, val_transform = create_transforms(config)
    
    train_dataset = BinaryAneurysmDataset(
        train_df, config, localizers_df, train_transform, is_training=True
    )
    val_dataset = BinaryAneurysmDataset(
        val_df, config, localizers_df, val_transform, is_training=False
    )
    
    # Use standard batch size
    batch_size = config.batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers // 2,
        pin_memory=True
    )
    
    return train_loader, val_loader


