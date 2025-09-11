#!/usr/bin/env python3
"""
RSNA 2025 Intracranial Aneurysm Detection - Utility Functions

Consolidated utility functions for data processing, coordinate handling,
ROI extraction, and training support.
"""

import os
import ast
import gc
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ============================================================================
# Constants and Configuration
# ============================================================================

# Label columns from the training CSV
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 'Basilar Tip',
    'Other Posterior Circulation', 'Aneurysm Present'
]

ID_COL = "SeriesInstanceUID"

# ============================================================================
# Coordinate Processing Functions
# ============================================================================

def parse_coordinates(coord_str: str) -> np.ndarray:
    """
    Parse coordinate string from train_localizers.csv
    
    Args:
        coord_str: String in format "{'x': 258.36, 'y': 261.36}"
    
    Returns:
        np.ndarray: [x, y] coordinates as float32
    """
    try:
        coord_dict = ast.literal_eval(coord_str)
        return np.array([coord_dict['x'], coord_dict['y']], dtype=np.float32)
    except Exception as e:
        print(f"Warning: Failed to parse coordinates '{coord_str}': {e}")
        return np.array([0.0, 0.0], dtype=np.float32)

def valid_coords(coords: np.ndarray) -> bool:
    """
    Check if coordinates are valid
    
    Args:
        coords: Coordinate array [x, y]
    
    Returns:
        bool: True if coordinates are valid
    """
    return np.all(np.isfinite(coords)) and not (coords[0] == 0.0 and coords[1] == 0.0)

def coords_to_px(coords: np.ndarray, img_size: int = 224) -> Tuple[int, int]:
    """
    Convert coordinates to pixel positions
    
    Args:
        coords: Coordinate array [x, y]
        img_size: Target image size
    
    Returns:
        Tuple[int, int]: Pixel coordinates (x, y)
    """
    x, y = float(coords[0]), float(coords[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= img_size
        y *= img_size
    return int(round(x)), int(round(y))

def make_bbox_px(x: int, y: int, img_size: int = 224, box_frac: float = 0.15, 
                min_px: int = 24, max_px: Optional[int] = None) -> Tuple[int, int, int, int]:
    """
    Create bounding box around center point
    
    Args:
        x, y: Center point coordinates
        img_size: Image size
        box_frac: Fraction of image size for box
        min_px: Minimum box size in pixels
        max_px: Maximum box size in pixels
    
    Returns:
        Tuple[int, int, int, int]: Bounding box (x1, y1, x2, y2)
    """
    r = max(min_px/2, box_frac * img_size / 2.0)
    if max_px is not None:
        r = min(r, max_px/2)
    x1 = int(np.clip(x - r, 0, img_size - 1))
    y1 = int(np.clip(y - r, 0, img_size - 1))
    x2 = int(np.clip(x + r, 0, img_size - 1))
    y2 = int(np.clip(y + r, 0, img_size - 1))
    # Enforce minimum width/height of 2 pixels
    if x2 <= x1: x2 = min(img_size - 1, x1 + 1)
    if y2 <= y1: y2 = min(img_size - 1, y1 + 1)
    return x1, y1, x2, y2

def create_coordinate_cache(localizers_csv_path: str, output_dir: str) -> Dict[str, np.ndarray]:
    """
    Create coordinate cache files for faster training access
    
    Args:
        localizers_csv_path: Path to train_localizers.csv
        output_dir: Directory to save coordinate cache files
    
    Returns:
        Dict mapping SeriesInstanceUID to coordinates
    """
    print(f"📍 Creating coordinate cache from {localizers_csv_path}")
    
    # Load localizers data
    localizers_df = pd.read_csv(localizers_csv_path)
    print(f"Loaded {len(localizers_df)} coordinate annotations")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    coord_cache = {}
    created_files = 0
    
    for _, row in tqdm(localizers_df.iterrows(), total=len(localizers_df), desc="Processing coordinates"):
        series_id = str(row['SeriesInstanceUID'])
        coordinates = parse_coordinates(row['coordinates'])
        
        # Store in memory cache
        coord_cache[series_id] = coordinates
        
        # Save individual coordinate file
        coord_file = os.path.join(output_dir, f"{series_id}_coords.npy")
        np.save(coord_file, coordinates)
        created_files += 1
    
    print(f"✅ Created {created_files} coordinate cache files in {output_dir}")
    return coord_cache

def validate_coordinates(coord_cache: Dict[str, np.ndarray]) -> Dict[str, int]:
    """
    Validate coordinate data and return statistics
    
    Args:
        coord_cache: Dictionary of SeriesInstanceUID -> coordinates
    
    Returns:
        Dict with validation statistics
    """
    stats = {
        'total': len(coord_cache),
        'valid': 0,
        'invalid': 0,
        'zero_coords': 0,
        'out_of_bounds': 0
    }
    
    for series_id, coords in coord_cache.items():
        x, y = coords[0], coords[1]
        
        # Check if coordinates are valid numbers
        if not (np.isfinite(x) and np.isfinite(y)):
            stats['invalid'] += 1
            continue
        
        # Check for zero coordinates (likely missing)
        if x == 0.0 and y == 0.0:
            stats['zero_coords'] += 1
            continue
        
        # Check for reasonable bounds (assuming max 1024x1024 images)
        if x < 0 or y < 0 or x > 1024 or y > 1024:
            stats['out_of_bounds'] += 1
            continue
        
        stats['valid'] += 1
    
    return stats

def update_coordinate_cache_from_processed_volumes(volume_dir: str, coord_cache_dir: str) -> int:
    """
    Create coordinate cache files for series that have processed volumes but missing coordinates
    
    Args:
        volume_dir: Directory containing processed .npz volume files
        coord_cache_dir: Directory for coordinate cache files
    
    Returns:
        Number of coordinate files created
    """
    print(f"🔄 Updating coordinate cache from volume directory: {volume_dir}")
    
    # Get list of processed series
    volume_files = [f for f in os.listdir(volume_dir) if f.endswith('.npz')]
    series_ids = [f.replace('.npz', '') for f in volume_files]
    
    # Check existing coordinate files
    existing_coords = set()
    if os.path.exists(coord_cache_dir):
        coord_files = [f for f in os.listdir(coord_cache_dir) if f.endswith('_coords.npy')]
        existing_coords = {f.replace('_coords.npy', '') for f in coord_files}
    
    # Create coordinate files for missing series (with zero coordinates)
    os.makedirs(coord_cache_dir, exist_ok=True)
    created_count = 0
    
    for series_id in tqdm(series_ids, desc="Creating missing coordinate files"):
        if series_id not in existing_coords:
            coord_file = os.path.join(coord_cache_dir, f"{series_id}_coords.npy")
            zero_coords = np.array([0.0, 0.0], dtype=np.float32)
            np.save(coord_file, zero_coords)
            created_count += 1
    
    print(f"✅ Created {created_count} missing coordinate files")
    return created_count

def analyze_coordinate_distribution(coord_cache: Dict[str, np.ndarray]) -> None:
    """
    Analyze and print coordinate distribution statistics
    
    Args:
        coord_cache: Dictionary of SeriesInstanceUID -> coordinates
    """
    coords_array = np.array(list(coord_cache.values()))
    valid_coords = coords_array[~((coords_array == 0).all(axis=1))]
    
    if len(valid_coords) == 0:
        print("No valid coordinates found for analysis")
        return
    
    print(f"\n📊 Coordinate Distribution Analysis:")
    print(f"Total coordinates: {len(coords_array)}")
    print(f"Valid coordinates: {len(valid_coords)} ({len(valid_coords)/len(coords_array)*100:.1f}%)")
    print(f"Zero coordinates: {len(coords_array) - len(valid_coords)} ({(len(coords_array) - len(valid_coords))/len(coords_array)*100:.1f}%)")
    
    if len(valid_coords) > 0:
        print(f"\nX coordinates:")
        print(f"  Range: [{valid_coords[:, 0].min():.1f}, {valid_coords[:, 0].max():.1f}]")
        print(f"  Mean: {valid_coords[:, 0].mean():.1f} ± {valid_coords[:, 0].std():.1f}")
        
        print(f"\nY coordinates:")
        print(f"  Range: [{valid_coords[:, 1].min():.1f}, {valid_coords[:, 1].max():.1f}]")
        print(f"  Mean: {valid_coords[:, 1].mean():.1f} ± {valid_coords[:, 1].std():.1f}")

def load_coordinate_cache(coord_cache_dir: str) -> Dict[str, np.ndarray]:
    """
    Load coordinate cache from directory
    
    Args:
        coord_cache_dir: Directory containing coordinate cache files
    
    Returns:
        Dict mapping SeriesInstanceUID to coordinates
    """
    coord_dict = {}
    
    if not os.path.exists(coord_cache_dir):
        print(f"Warning: Coordinate cache directory not found: {coord_cache_dir}")
        return coord_dict
    
    coord_files = [f for f in os.listdir(coord_cache_dir) if f.endswith('_coords.npy')]
    
    for coord_file in coord_files:
        series_id = coord_file.replace('_coords.npy', '')
        coord_path = os.path.join(coord_cache_dir, coord_file)
        coords = np.load(coord_path)
        coord_dict[series_id] = coords
    
    print(f"Loaded coordinates for {len(coord_dict)} series")
    return coord_dict

# ============================================================================
# Image Processing Functions
# ============================================================================

def crop_and_resize_hwc(img_hwc: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                       out_size: int) -> np.ndarray:
    """
    Crop and resize image region
    
    Args:
        img_hwc: Input image in HWC format
        x1, y1, x2, y2: Bounding box coordinates
        out_size: Output size
    
    Returns:
        np.ndarray: Cropped and resized image
    """
    crop = img_hwc[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        crop = img_hwc  # fallback to full image
    crop = crop.astype(np.float32, copy=False)
    crop = np.ascontiguousarray(crop)
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop

def crop_and_resize_chw(img_chw: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                       out_size: int) -> np.ndarray:
    """
    Crop and resize image region from CHW format
    
    Args:
        img_chw: Input image in CHW format
        x1, y1, x2, y2: Bounding box coordinates
        out_size: Output size
    
    Returns:
        np.ndarray: Cropped and resized image in CHW format
    """
    # Convert CHW to HWC
    img_hwc = np.transpose(np.asarray(img_chw), (1, 2, 0))
    crop = img_hwc[y1:y2, x1:x2]
    
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        # Fallback to full image
        full = img_hwc.astype(np.float32, copy=False)
        full = cv2.resize(full, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return np.transpose(full, (2, 0, 1))

    crop = crop.astype(np.float32, copy=False)
    crop = np.ascontiguousarray(crop)
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return np.transpose(crop, (2, 0, 1))

# ============================================================================
# Volume Processing Functions
# ============================================================================

def load_cached_volume(volume_path: str) -> np.ndarray:
    """
    Load cached volume from .npz file
    
    Args:
        volume_path: Path to .npz volume file
    
    Returns:
        np.ndarray: Volume data with shape (N, H, W)
    """
    data = np.load(volume_path)
    return data['volume']  # Shape: (N, H, W)

def load_cached_img(img_path: str) -> np.ndarray:
    """
    Load cached image from .npz or .npy file
    
    Args:
        img_path: Path to cached image file
    
    Returns:
        np.ndarray: Image data
    """
    if img_path.endswith(".npz"):
        return np.load(img_path)["arr_0"]          # [C_all, H, W]
    return np.load(img_path, mmap_mode="r")        # [C_all, H, W]

def take_window(img_nhw: np.ndarray, center_idx: int, offsets: Tuple = (-2, -1, 0, 1, 2)) -> np.ndarray:
    """
    Extract window of slices around center index
    
    Args:
        img_nhw: Input volume with shape (N, H, W)
        center_idx: Center slice index
        offsets: Offset indices for window
    
    Returns:
        np.ndarray: Window with shape (len(offsets), H, W)
    """
    N = img_nhw.shape[0]
    idxs = [min(max(0, center_idx + o), N - 1) for o in offsets]
    return img_nhw[idxs, :, :]  # [WINDOW_LEN, H, W]

# ============================================================================
# Training Utility Functions
# ============================================================================

def compute_weighted_auc(y_true: np.ndarray, y_prob: np.ndarray, 
                        class_names: List[str]) -> Tuple[float, float, float, Dict, List]:
    """
    Compute weighted AUC score as used in RSNA competition
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: List of class names
    
    Returns:
        Tuple of (weighted_auc, aneurysm_present_auc, others_mean, per_class_aucs, skipped_classes)
    """
    y_true = np.atleast_2d(y_true)
    y_prob = np.atleast_2d(y_prob)
    aucs, skipped = {}, []
    
    for i, name in enumerate(class_names):
        yi = y_true[:, i]
        if len(np.unique(yi)) < 2:
            skipped.append(name)
            continue
        aucs[name] = roc_auc_score(yi, y_prob[:, i])
    
    ap_name = "Aneurysm Present"
    ap_auc = aucs.get(ap_name, np.nan)
    others = [v for k, v in aucs.items() if k != ap_name]
    others_mean = np.mean(others) if others else np.nan
    weighted_auc = 0.5 * (ap_auc + others_mean)
    
    return weighted_auc, ap_auc, others_mean, aucs, skipped

def cleanup_memory():
    """Clean up GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def setup_device(device_name: str = 'auto') -> torch.device:
    """
    Setup computation device
    
    Args:
        device_name: Device name ('auto', 'cuda', 'cpu')
    
    Returns:
        torch.device: Configured device
    """
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("Using CPU")
    
    return device

def setup_reproducibility(seed: int = 42):
    """
    Setup reproducible training
    
    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# Data Loading Utilities
# ============================================================================

def filter_available_series(df: pd.DataFrame, cache_dir: str, id_col: str = ID_COL) -> pd.DataFrame:
    """
    Filter dataframe to only include series with available cached files
    
    Args:
        df: Input dataframe
        cache_dir: Directory containing cached files
        id_col: Column name for series IDs
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if not os.path.exists(cache_dir):
        print(f"Warning: Cache directory not found: {cache_dir}")
        return df.iloc[:0].copy()  # Return empty dataframe
    
    # Get available cached files
    cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.npz')]
    available_ids = {f.replace('.npz', '') for f in cached_files}
    
    # Filter dataframe
    original_count = len(df)
    df_filtered = df[df[id_col].astype(str).isin(available_ids)].reset_index(drop=True)
    filtered_count = len(df_filtered)
    
    print(f"Filtered dataset: {original_count} -> {filtered_count} series ({filtered_count/original_count*100:.1f}%)")
    return df_filtered

def create_coordinate_lookup(localizers_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Create coordinate lookup dictionary from localizers dataframe
    
    Args:
        localizers_df: Dataframe with coordinates
    
    Returns:
        Dict mapping SeriesInstanceUID to coordinates
    """
    coord_lookup = {}
    if localizers_df is not None:
        for _, row in localizers_df.iterrows():
            sid = str(row['SeriesInstanceUID'])
            coords = parse_coordinates(row['coordinates'])
            coord_lookup[sid] = coords
    
    return coord_lookup

# ============================================================================
# Validation and Testing Functions
# ============================================================================

def validate_training_data(df: pd.DataFrame, cache_dir: str, coord_cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate training data availability and consistency
    
    Args:
        df: Training dataframe
        cache_dir: Volume cache directory
        coord_cache_dir: Coordinate cache directory
    
    Returns:
        Dict with validation results
    """
    results = {
        'total_series': len(df),
        'available_volumes': 0,
        'available_coords': 0,
        'missing_volumes': [],
        'missing_coords': [],
        'validation_passed': False
    }
    
    # Check volume availability
    if os.path.exists(cache_dir):
        volume_files = {f.replace('.npz', '') for f in os.listdir(cache_dir) if f.endswith('.npz')}
        results['available_volumes'] = len(volume_files)
        
        for _, row in df.iterrows():
            sid = str(row[ID_COL])
            if sid not in volume_files:
                results['missing_volumes'].append(sid)
    
    # Check coordinate availability
    if coord_cache_dir and os.path.exists(coord_cache_dir):
        coord_files = {f.replace('_coords.npy', '') for f in os.listdir(coord_cache_dir) if f.endswith('_coords.npy')}
        results['available_coords'] = len(coord_files)
        
        for _, row in df.iterrows():
            sid = str(row[ID_COL])
            if sid not in coord_files:
                results['missing_coords'].append(sid)
    
    # Determine validation status
    volume_coverage = results['available_volumes'] / results['total_series'] if results['total_series'] > 0 else 0
    results['validation_passed'] = volume_coverage >= 0.9  # At least 90% coverage
    
    return results

def print_validation_summary(validation_results: Dict[str, Any]):
    """
    Print validation summary
    
    Args:
        validation_results: Results from validate_training_data
    """
    print(f"\n📊 Training Data Validation Summary:")
    print(f"Total series: {validation_results['total_series']}")
    print(f"Available volumes: {validation_results['available_volumes']}")
    print(f"Available coordinates: {validation_results['available_coords']}")
    print(f"Missing volumes: {len(validation_results['missing_volumes'])}")
    print(f"Missing coordinates: {len(validation_results['missing_coords'])}")
    
    if validation_results['validation_passed']:
        print("✅ Validation PASSED - Ready for training")
    else:
        print("❌ Validation FAILED - Check missing data")
        if validation_results['missing_volumes']:
            print(f"First 5 missing volumes: {validation_results['missing_volumes'][:5]}")

# ============================================================================
# Inference Utilities
# ============================================================================

def window_to_full_and_roi(win_chw: np.ndarray, coords: np.ndarray, img_size: int = 224,
                          roi_box_frac: float = 0.15, roi_min_px: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert window to full and ROI images for dual-stream inference
    
    Args:
        win_chw: Window in CHW format
        coords: Aneurysm coordinates
        img_size: Target image size
        roi_box_frac: ROI box fraction
        roi_min_px: Minimum ROI pixels
    
    Returns:
        Tuple of (full_image, roi_image) both in CHW format
    """
    # Convert to HWC for processing
    img_hwc = np.transpose(win_chw, (1, 2, 0)).astype(np.float32)
    
    # Create ROI if valid coordinates
    if valid_coords(coords):
        cx, cy = coords_to_px(coords, img_size)
        x1, y1, x2, y2 = make_bbox_px(cx, cy, img_size, roi_box_frac, roi_min_px)
        roi_hwc = crop_and_resize_hwc(img_hwc, x1, y1, x2, y2, img_size)
    else:
        roi_hwc = img_hwc.copy()
    
    # Resize full image
    full_hwc = cv2.resize(img_hwc, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # Convert back to CHW
    full_chw = np.transpose(full_hwc, (2, 0, 1))
    roi_chw = np.transpose(roi_hwc, (2, 0, 1))
    
    return full_chw, roi_chw
