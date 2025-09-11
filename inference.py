#!/usr/bin/env python3
"""
RSNA 2025 Intracranial Aneurysm Detection - Inference Script

Inference pipeline for the trained 2.5D EfficientNet models.
Supports single series inference and batch processing.
"""

import os
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2

from train import HybridAneurysmModel, Config
from utils import (
    LABEL_COLS, ID_COL, load_cached_volume, take_window, valid_coords, 
    coords_to_px, make_bbox_px, crop_and_resize_hwc, load_coordinate_cache
)

class AneurysmInference:
    """Inference pipeline for aneurysm detection"""
    
    def __init__(self, model_paths: List[str], config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.models = self._load_models(model_paths)
        
    def _load_models(self, model_paths: List[str]) -> List[nn.Module]:
        """Load trained models"""
        models = []
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"Warning: Model not found: {model_path}")
                continue
                
            model = HybridAneurysmModel(self.config)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            models.append(model)
            
        print(f"Loaded {len(models)} models for inference")
        return models
    
    def _prepare_windows(self, volume: np.ndarray, coords: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sliding windows for inference"""
        
        N, H, W = volume.shape
        all_windows_full = []
        all_windows_roi = []
        
        # Process all possible windows
        for center_idx in range(N):
            # Extract window
            window = take_window(volume, center_idx, self.config.window_offsets)  # (5, H, W)
            
            # Convert to HWC for processing
            img_hwc = np.transpose(window, (1, 2, 0)).astype(np.float32)  # (H, W, 5)
            
            # Create ROI image
            if valid_coords(coords):
                cx, cy = coords_to_px(coords, self.config.img_size)
                x1, y1, x2, y2 = make_bbox_px(cx, cy, self.config.img_size, 
                                            self.config.roi_box_fraction, self.config.roi_min_pixels)
                img_roi_hwc = crop_and_resize_hwc(img_hwc, x1, y1, x2, y2, self.config.img_size)
            else:
                img_roi_hwc = img_hwc.copy()
            
            # Resize full image
            img_full_hwc = cv2.resize(img_hwc, (self.config.img_size, self.config.img_size), 
                                     interpolation=cv2.INTER_AREA)
            
            # Convert to CHW tensors
            x_full = torch.from_numpy(np.transpose(img_full_hwc, (2, 0, 1))).float()
            x_roi = torch.from_numpy(np.transpose(img_roi_hwc, (2, 0, 1))).float()
            
            all_windows_full.append(x_full)
            all_windows_roi.append(x_roi)
        
        # Stack to tensors
        windows_full = torch.stack(all_windows_full)  # (N_windows, 5, H, W)
        windows_roi = torch.stack(all_windows_roi)    # (N_windows, 5, H, W)
        
        return windows_full, windows_roi
    
    @torch.no_grad()
    def infer_series(self, volume_path: str, coords: Optional[np.ndarray] = None, 
                    aggregate: str = "max") -> np.ndarray:
        """Infer on single series"""
        
        # Load volume
        try:
            volume = load_cached_volume(volume_path)  # (N, H, W)
        except:
            print(f"Warning: Could not load volume {volume_path}")
            return np.zeros(len(LABEL_COLS), dtype=np.float32)
        
        # Default coordinates if not provided
        if coords is None:
            coords = np.array([0.0, 0.0], dtype=np.float32)
        
        # Prepare windows
        windows_full, windows_roi = self._prepare_windows(volume, coords)
        
        # Process in batches
        batch_size = 16
        all_predictions = []
        
        for i in range(0, windows_full.shape[0], batch_size):
            batch_full = windows_full[i:i+batch_size].to(self.device)
            batch_roi = windows_roi[i:i+batch_size].to(self.device)
            batch_coords = torch.from_numpy(
                np.repeat(coords[None, :], batch_full.shape[0], axis=0)
            ).to(self.device)
            
            # Model ensemble predictions
            batch_preds = []
            for model in self.models:
                logits = model(batch_full, batch_roi, batch_coords)
                probs = torch.sigmoid(logits).cpu().numpy()
                batch_preds.append(probs)
            
            # Average ensemble predictions
            if batch_preds:
                avg_preds = np.mean(batch_preds, axis=0)
                all_predictions.append(avg_preds)
        
        if not all_predictions:
            return np.zeros(len(LABEL_COLS), dtype=np.float32)
        
        # Aggregate across all windows
        all_preds = np.vstack(all_predictions)  # (N_windows, N_classes)
        
        if aggregate == "max":
            series_pred = all_preds.max(axis=0)
        elif aggregate == "mean":
            series_pred = all_preds.mean(axis=0)
        elif aggregate == "topk_mean":
            k = max(1, all_preds.shape[0] // 5)
            series_pred = np.sort(all_preds, axis=0)[-k:].mean(axis=0)
        else:
            series_pred = all_preds.mean(axis=0)
        
        return series_pred
    
    def infer_batch(self, series_list: List[str], coord_dict: Optional[Dict[str, np.ndarray]] = None,
                   output_path: Optional[str] = None) -> pd.DataFrame:
        """Batch inference on multiple series"""
        
        results = []
        
        for series_id in tqdm(series_list, desc="Processing series"):
            volume_path = os.path.join(self.config.cache_dir, f"{series_id}.npz")
            coords = coord_dict.get(series_id) if coord_dict else None
            
            predictions = self.infer_series(volume_path, coords)
            
            result = {'SeriesInstanceUID': series_id}
            for i, label in enumerate(LABEL_COLS):
                result[label] = predictions[i]
            
            results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        
        return results_df

# Note: load_coordinate_cache function moved to utils.py

def main():
    parser = argparse.ArgumentParser(description="RSNA Aneurysm Detection Inference")
    parser.add_argument("--model_paths", nargs="+", required=True,
                       help="Paths to trained model files")
    parser.add_argument("--series_list", type=str,
                       help="File with list of SeriesInstanceUIDs to process")
    parser.add_argument("--series_id", type=str,
                       help="Single SeriesInstanceUID to process")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Directory with cached volume files")
    parser.add_argument("--coord_cache_dir", type=str,
                       help="Directory with coordinate cache files")
    parser.add_argument("--output_path", type=str, default="predictions.csv",
                       help="Output CSV file path")
    parser.add_argument("--aggregate", type=str, default="max",
                       choices=["max", "mean", "topk_mean"],
                       help="Aggregation method for window predictions")
    parser.add_argument("--architecture", type=str, default="tf_efficientnet_b0",
                       help="Model architecture")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = Config()
    config.cache_dir = args.cache_dir
    config.architecture = args.architecture
    
    # Load coordinate cache
    coord_dict = None
    if args.coord_cache_dir:
        coord_dict = load_coordinate_cache(args.coord_cache_dir)
    
    # Initialize inference pipeline
    inference = AneurysmInference(args.model_paths, config)
    
    if args.series_id:
        # Single series inference
        print(f"Processing single series: {args.series_id}")
        volume_path = os.path.join(args.cache_dir, f"{args.series_id}.npz")
        coords = coord_dict.get(args.series_id) if coord_dict else None
        
        predictions = inference.infer_series(volume_path, coords, args.aggregate)
        
        print(f"\nPredictions for {args.series_id}:")
        for i, label in enumerate(LABEL_COLS):
            print(f"  {label}: {predictions[i]:.4f}")
    
    elif args.series_list:
        # Batch inference
        print(f"Processing series list from: {args.series_list}")
        
        if args.series_list.endswith('.csv'):
            df = pd.read_csv(args.series_list)
            series_list = df['SeriesInstanceUID'].astype(str).tolist()
        else:
            with open(args.series_list, 'r') as f:
                series_list = [line.strip() for line in f if line.strip()]
        
        results_df = inference.infer_batch(series_list, coord_dict, args.output_path)
        
        print(f"\nProcessed {len(results_df)} series")
        print(f"Results shape: {results_df.shape}")
        
        # Show sample predictions
        print("\nSample predictions:")
        print(results_df[['SeriesInstanceUID', 'Aneurysm Present']].head())
    
    else:
        print("Error: Must specify either --series_id or --series_list")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
