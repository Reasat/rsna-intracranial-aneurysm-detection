"""
RSNA 2025 Intracranial Aneurysm Detection - Data Processing Pipeline

This module contains comprehensive DICOM preprocessing functionality extracted from the 
competition notebooks, including:
- 3D volume processing (32-channel approach)
- 2.5D slice window processing
- DICOM to PNG conversion
- Modality-specific normalization
- Robust DICOM sorting and metadata handling

Authors: Extracted from competition notebooks
"""

import os
import re
import gc
import warnings
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from multiprocessing import Pool, cpu_count

# Data handling
import numpy as np
import pandas as pd

# Medical imaging
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
import cv2
from scipy import ndimage

# Image processing
from PIL import Image

# Progress tracking
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# ====================================================
# Constants and Configuration
# ====================================================

class DataProcessingConfig:
    """Configuration for data processing"""
    
    # 3D Volume Processing (32-channel approach)
    DEFAULT_3D_SHAPE = (32, 384, 384)  # (depth, height, width)
    
    # 2.5D Processing
    DEFAULT_2D_SIZE = 224
    SLICE_WINDOW_SIZE = 5
    SLICE_OFFSETS = [-2, -1, 0, 1, 2]  # 5-slice window
    
    # PNG Processing
    DEFAULT_PNG_SIZE = 224
    
    # Normalization parameters
    CT_NORMALIZATION_RANGE = (0, 500)  # For statistical normalization
    CT_WINDOW_CENTER = 50
    CT_WINDOW_WIDTH = 350
    
    # Output formats
    OUTPUT_DTYPE = np.uint8
    OUTPUT_RANGE = (0, 255)

CFG = DataProcessingConfig()

# ====================================================
# DICOM Utilities
# ====================================================

def atoi(text: str) -> Union[int, str]:
    """Convert text to int if digit, otherwise return text"""
    return int(text) if text.isdigit() else text

def natural_keys(text: str) -> List[Union[int, str]]:
    """Natural sorting key function"""
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def extract_sort_key(path: str) -> Tuple[float, float, str]:
    """
    Extract sorting key for DICOM file based on:
    1. InstanceNumber (preferred)
    2. ImagePositionPatient's Z-axis (fallback)
    3. Defaults to (inf, inf) if neither is available
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        instance_number = getattr(ds, 'InstanceNumber', None)
        position = getattr(ds, 'ImagePositionPatient', [None, None, None])
        z = position[2] if position and len(position) == 3 else None

        if instance_number is not None:
            return (int(instance_number), 0, path)
        elif z is not None:
            return (float('inf'), float(z), path)
        else:
            return (float('inf'), float('inf'), path)
    except:
        return (float('inf'), float('inf'), path)

def sort_dicom_slices(filepaths: List[str]) -> List[pydicom.Dataset]:
    """Sort DICOM files and return loaded datasets"""
    dicoms = [pydicom.dcmread(fp, force=True) for fp in filepaths]
    try:
        dicoms.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    except Exception:
        dicoms.sort(key=lambda d: int(getattr(d, 'InstanceNumber', 0)))
    return dicoms

def fast_sort_dicom_paths(dcm_paths: List[str]) -> List[str]:
    """Fast sorting of DICOM paths using metadata"""
    sort_info = []
    for path in dcm_paths:
        sort_info.append(extract_sort_key(path))
    sort_info.sort()
    return [x[2] for x in sort_info]

def fast_sort_dicom_paths_parallel(dcm_paths: List[str], num_workers: Optional[int] = None) -> List[str]:
    """Parallel sorting of DICOM paths"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    with Pool(processes=num_workers) as pool:
        sort_info = pool.map(extract_sort_key, dcm_paths)
    
    sort_info.sort()
    return [x[2] for x in sort_info]

# ====================================================
# Intensity Processing and Normalization
# ====================================================

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    Apply DICOM windowing to enhance visibility of specific structures
    
    Args:
        img: Input image array
        window_center: Center of the intensity window
        window_width: Width of the intensity window
    
    Returns:
        Windowed image normalized to [0, 255]
    """
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    
    windowed = np.clip(img, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min + 1e-7)
    
    return (windowed * 255).astype(np.uint8)

def apply_statistical_normalization(img: np.ndarray, percentile_range: Tuple[float, float] = (1, 99)) -> np.ndarray:
    """
    Apply statistical normalization using percentiles
    
    Args:
        img: Input image array
        percentile_range: Tuple of (low_percentile, high_percentile)
    
    Returns:
        Normalized image in [0, 255] range
    """
    p_low, p_high = np.percentile(img, percentile_range)
    
    if p_high > p_low:
        normalized = np.clip(img, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low)
        return (normalized * 255).astype(np.uint8)
    else:
        # Fallback: min-max normalization
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            normalized = (img - img_min) / (img_max - img_min)
            return (normalized * 255).astype(np.uint8)
        else:
            return np.zeros_like(img, dtype=np.uint8)

def apply_ct_normalization(img: np.ndarray, fixed_range: Tuple[float, float] = CFG.CT_NORMALIZATION_RANGE) -> np.ndarray:
    """
    Apply CT-specific normalization with fixed range
    
    Args:
        img: Input CT image array
        fixed_range: Fixed intensity range for normalization
    
    Returns:
        Normalized image in [0, 255] range
    """
    r_min, r_max = fixed_range
    
    if r_max > r_min:
        normalized = np.clip(img, r_min, r_max)
        normalized = (normalized - r_min) / (r_max - r_min)
        return (normalized * 255).astype(np.uint8)
    else:
        return apply_statistical_normalization(img)

def get_modality_normalization(modality: str) -> callable:
    """
    Get appropriate normalization function based on modality
    
    Args:
        modality: DICOM modality ('CT' or 'MR')
    
    Returns:
        Normalization function
    """
    if modality == 'CT':
        return apply_ct_normalization
    else:  # MR modalities
        return apply_statistical_normalization

# ====================================================
# 3D Volume Processor (32-Channel Approach)
# ====================================================

class DICOMVolumeProcessor:
    """
    DICOM preprocessing system for 3D volume processing
    Handles both single 3D DICOM files and multiple 2D DICOM files
    """
    
    def __init__(self, target_shape: Tuple[int, int, int] = CFG.DEFAULT_3D_SHAPE):
        """
        Initialize processor
        
        Args:
            target_shape: Target volume size (depth, height, width)
        """
        self.target_depth, self.target_height, self.target_width = target_shape
    
    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        """Load DICOM series from directory"""
        series_path = Path(series_path)
        series_name = series_path.name
        
        # Search for DICOM files
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")
        
        # Load DICOM datasets
        datasets = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)
            except Exception as e:
                continue
        
        if not datasets:
            raise ValueError(f"No valid DICOM files in {series_path}")
        
        return datasets, series_name
    
    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        """Extract position information for each slice"""
        slice_info = []
        
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            
            # Get z-coordinate from ImagePositionPatient
            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                else:
                    info['z_position'] = float(info['instance_number'])
            except Exception:
                info['z_position'] = float(i)
            
            slice_info.append(info)
        
        return slice_info
    
    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        """Sort slices by z-coordinate"""
        return sorted(slice_info, key=lambda x: x['z_position'])
    
    def get_windowing_params(self, ds: pydicom.Dataset) -> Tuple[Optional[str], Optional[str]]:
        """Get windowing parameters based on modality"""
        modality = getattr(ds, 'Modality', 'CT')
        
        if modality == 'CT':
            return "CT", "CT"  # Marker for CT processing
        elif modality == 'MR':
            return None, None  # MR uses statistical normalization
        else:
            return None, None  # Default to statistical normalization
    
    def apply_windowing_or_normalize(self, img: np.ndarray, modality_marker: Optional[str], _: Optional[str]) -> np.ndarray:
        """Apply windowing or statistical normalization based on modality"""
        if modality_marker == "CT":
            # CT: Use fixed range statistical normalization
            return apply_ct_normalization(img)
        else:
            # MR: Use adaptive percentile normalization
            return apply_statistical_normalization(img)
    
    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """Extract 2D pixel array from DICOM and apply preprocessing"""
        # Get pixel data
        img = ds.pixel_array.astype(np.float32)
        
        # For 3D volume case (multiple frames) - select middle frame
        if img.ndim == 3:
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
        
        # Convert color image to grayscale
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Apply RescaleSlope and RescaleIntercept
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
        
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """Resize 3D volume to target size using interpolation"""
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        # 3D resizing using scipy.ndimage
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
        
        # Resize with linear interpolation
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        
        # Clip to exact size
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        
        # Padding if necessary
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        
        return resized_volume.astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        """Process DICOM series and return as 3D NumPy array"""
        try:
            # Load DICOM files
            datasets, series_name = self.load_dicom_series(series_path)
            
            # Check first DICOM to determine 3D/2D
            first_ds = datasets[0]
            first_img = first_ds.pixel_array
            
            if len(datasets) == 1 and first_img.ndim == 3:
                # Case 1: Single 3D DICOM file
                return self._process_single_3d_dicom(first_ds, series_name)
            else:
                # Case 2: Multiple 2D DICOM files
                return self._process_multiple_2d_dicoms(datasets, series_name)
                
        except Exception as e:
            raise RuntimeError(f"Failed to process series {series_path}: {e}")
    
    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        """Process single 3D DICOM file"""
        # Get pixel array
        volume = ds.pixel_array.astype(np.float32)
        
        # Apply RescaleSlope and RescaleIntercept
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        
        if slope != 1 or intercept != 0:
            volume = volume * float(slope) + float(intercept)
        
        # Get windowing settings
        window_center, window_width = self.get_windowing_params(ds)
        
        # Apply windowing to each slice
        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)
        
        volume = np.stack(processed_slices, axis=0)
        
        # 3D resize
        final_volume = self.resize_volume_3d(volume)
        
        return final_volume
    
    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset], series_name: str) -> np.ndarray:
        """Process multiple 2D DICOM files"""
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices_by_position(slice_info)
        first_img = self.extract_pixel_array(sorted_slices[0]['dataset'])
        window_center, window_width = self.get_windowing_params(sorted_slices[0]['dataset'])
        
        processed_slices = []
        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed_img = self.apply_windowing_or_normalize(img, window_center, window_width)
            resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))
            processed_slices.append(resized_img)
        
        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_volume_3d(volume)
        
        return final_volume

# ====================================================
# 2.5D Slice Window Processor
# ====================================================

class SliceWindowProcessor:
    """
    2.5D slice window processing for EfficientNet-based models
    Creates 5-slice windows with per-series normalization
    """
    
    def __init__(self, img_size: int = CFG.DEFAULT_2D_SIZE, window_offsets: List[int] = CFG.SLICE_OFFSETS):
        """
        Initialize processor
        
        Args:
            img_size: Target image size (square)
            window_offsets: Slice offsets for window creation
        """
        self.img_size = img_size
        self.offsets = window_offsets
    
    def series_to_tensor_chw(self, dicoms: List[pydicom.Dataset]) -> np.ndarray:
        """Convert DICOM series to normalized tensor with CHW format
        
        Handles both single-frame and multi-frame DICOMs:
        - Single-frame: Each DICOM contains one 2D slice
        - Multi-frame: Each DICOM contains multiple frames as 3D volume
        """
        # Resize all to img_size and per-series z-score normalization
        resized = []
        
        for d in dicoms:
            arr = d.pixel_array
            if arr is None or arr.size == 0:
                continue
            
            arr = arr.astype(np.float32)
            
            # Handle multi-frame vs single-frame DICOMs
            if arr.ndim == 3:
                # Multi-frame DICOM: Process each frame individually
                for frame_idx in range(arr.shape[0]):
                    frame = arr[frame_idx]  # Extract 2D frame
                    resized_frame = cv2.resize(frame, (self.img_size, self.img_size), 
                                             interpolation=cv2.INTER_AREA)
                    resized.append(resized_frame)
            elif arr.ndim == 2:
                # Single-frame DICOM: Process as single 2D slice
                resized_slice = cv2.resize(arr, (self.img_size, self.img_size), 
                                         interpolation=cv2.INTER_AREA)
                resized.append(resized_slice)
            else:
                # Unexpected dimensionality - skip with warning
                print(f"Warning: Unexpected pixel array dimensions: {arr.shape}")
                continue
        
        if len(resized) == 0:
            # Fallback to zeros to avoid crashes
            vol = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
        else:
            vol = np.stack(resized, axis=0)  # [N,H,W]
        
        # Per-series z-score normalization
        mean = float(vol.mean())
        std = float(vol.std()) + 1e-6
        vol = (vol - mean) / std
        
        return vol  # [N,H,W]
    
    def take_window_from_volume(self, vol_nhw: np.ndarray, center_idx: int) -> np.ndarray:
        """Extract window of slices around center index"""
        N = vol_nhw.shape[0]
        idxs = [min(max(0, center_idx + o), N - 1) for o in self.offsets]
        win = vol_nhw[idxs, :, :]  # [len(offsets), H, W]
        return win.astype(np.float32, copy=False)
    
    def process_series_to_windows(self, series_path: str) -> Tuple[np.ndarray, int]:
        """
        Process DICOM series to slice windows
        
        Args:
            series_path: Path to DICOM series
        
        Returns:
            Tuple of (volume, num_slices)
        """
        # Find and sort DICOM files
        filepaths = []
        for root, _, files in os.walk(series_path):
            for f in files:
                if f.endswith('.dcm'):
                    filepaths.append(os.path.join(root, f))
        
        dicoms = sort_dicom_slices(filepaths)
        
        # Convert to normalized volume
        vol = self.series_to_tensor_chw(dicoms)
        
        return vol, len(dicoms)

# ====================================================
# PNG Conversion Utilities
# ====================================================

class DICOMToPNGConverter:
    """
    Converter for DICOM files to PNG format with proper preprocessing
    """
    
    def __init__(self, output_size: int = CFG.DEFAULT_PNG_SIZE, apply_windowing: bool = True):
        """
        Initialize converter
        
        Args:
            output_size: Target PNG image size (square)
            apply_windowing: Whether to apply DICOM windowing
        """
        self.output_size = output_size
        self.apply_windowing = apply_windowing
    
    def process_dicom_to_png(self, dicom_path: str, output_path: str, window_center: Optional[float] = None, 
                           window_width: Optional[float] = None) -> bool:
        """
        Convert single DICOM file to PNG
        
        Args:
            dicom_path: Path to input DICOM file
            output_path: Path for output PNG file
            window_center: Window center for CT images
            window_width: Window width for CT images
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read DICOM
            ds = pydicom.dcmread(dicom_path, force=True)
            img = ds.pixel_array.astype(np.float32)
            
            # Handle multi-frame DICOM
            if img.ndim == 3:
                img = img[img.shape[0] // 2]  # Take middle frame
            
            # Handle color images
            if img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            
            # Apply RescaleSlope and RescaleIntercept
            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
            if slope != 1 or intercept != 0:
                img = img * float(slope) + float(intercept)
            
            # Apply windowing or normalization
            modality = getattr(ds, 'Modality', 'CT')
            
            if self.apply_windowing and modality == 'CT' and window_center is not None and window_width is not None:
                processed_img = apply_dicom_windowing(img, window_center, window_width)
            else:
                # Use appropriate normalization based on modality
                if modality == 'CT':
                    processed_img = apply_ct_normalization(img)
                else:
                    processed_img = apply_statistical_normalization(img)
            
            # Resize image
            if processed_img.shape[0] != self.output_size or processed_img.shape[1] != self.output_size:
                processed_img = cv2.resize(processed_img, (self.output_size, self.output_size), 
                                         interpolation=cv2.INTER_AREA)
            
            # Save as PNG
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, processed_img)
            
            return True
            
        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")
            return False
    
    def convert_series_to_pngs(self, series_path: str, output_dir: str, series_name: Optional[str] = None) -> List[str]:
        """
        Convert entire DICOM series to PNG files
        
        Args:
            series_path: Path to DICOM series directory
            output_dir: Output directory for PNG files
            series_name: Optional series name for output folder
        
        Returns:
            List of created PNG file paths
        """
        if series_name is None:
            series_name = Path(series_path).name
        
        # Find and sort DICOM files
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for f in files:
                if f.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, f))
        
        # Sort files properly
        sorted_files = fast_sort_dicom_paths(dicom_files)
        
        # Create output directory
        series_output_dir = os.path.join(output_dir, series_name)
        os.makedirs(series_output_dir, exist_ok=True)
        
        # Convert each file
        created_files = []
        for i, dicom_path in enumerate(sorted_files):
            png_filename = f"{i:04d}.png"
            png_path = os.path.join(series_output_dir, png_filename)
            
            if self.process_dicom_to_png(dicom_path, png_path):
                created_files.append(png_path)
        
        return created_files

# ====================================================
# High-Level Processing Functions
# ====================================================

def process_dicom_series_3d(series_path: str, target_shape: Tuple[int, int, int] = CFG.DEFAULT_3D_SHAPE) -> np.ndarray:
    """
    High-level function for 3D volume processing
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume size (depth, height, width)
    
    Returns:
        Processed 3D volume as NumPy array
    """
    processor = DICOMVolumeProcessor(target_shape=target_shape)
    return processor.process_series(series_path)

def process_dicom_series_2d(series_path: str, img_size: int = CFG.DEFAULT_2D_SIZE) -> Tuple[np.ndarray, int]:
    """
    High-level function for 2.5D slice window processing
    
    Args:
        series_path: Path to DICOM series
        img_size: Target image size
    
    Returns:
        Tuple of (normalized volume, number of slices)
    """
    processor = SliceWindowProcessor(img_size=img_size)
    return processor.process_series_to_windows(series_path)

def convert_dicom_series_to_png(series_path: str, output_dir: str, output_size: int = CFG.DEFAULT_PNG_SIZE) -> List[str]:
    """
    High-level function for DICOM to PNG conversion
    
    Args:
        series_path: Path to DICOM series
        output_dir: Output directory for PNG files
        output_size: Target PNG image size
    
    Returns:
        List of created PNG file paths
    """
    converter = DICOMToPNGConverter(output_size=output_size)
    return converter.convert_series_to_pngs(series_path, output_dir)

def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = CFG.DEFAULT_3D_SHAPE) -> np.ndarray:
    """
    Safe DICOM processing with memory cleanup
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume size
    
    Returns:
        Processed volume
    """
    try:
        volume = process_dicom_series_3d(series_path, target_shape)
        return volume
    finally:
        # Memory cleanup
        gc.collect()

# ====================================================
# Batch Processing Functions
# ====================================================

# ====================================================
# Multiprocessing Worker Functions
# ====================================================

def _process_single_series_to_2_5d(args):
    """Worker function for multiprocessing 2.5D volume processing"""
    series_name, input_dir, output_dir, img_size = args
    series_path = os.path.join(input_dir, series_name)
    
    try:
        start_time = time.time()
        
        # Process series to 2.5D volume
        volume, num_slices = process_dicom_series_2d(series_path, img_size)
        
        # Save processed volume
        output_path = os.path.join(output_dir, f"{series_name}.npz")
        np.savez_compressed(output_path, volume=volume, num_slices=num_slices)
        
        processing_time = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        
        return {
            'series_name': series_name,
            'output_path': output_path,
            'status': 'success',
            'processing_time': processing_time,
            'volume_shape': volume.shape,
            'num_slices': num_slices,
            'file_size_mb': file_size_mb
        }
        
    except Exception as e:
        return {
            'series_name': series_name,
            'output_path': None,
            'status': 'failed',
            'error': str(e),
            'processing_time': 0,
            'volume_shape': None,
            'num_slices': 0,
            'file_size_mb': 0
        }

def _process_single_series_to_2_5d_detailed(args):
    """Worker function for multiprocessing 2.5D volume processing with detailed logging"""
    series_name, input_dir, output_dir, img_size = args
    series_path = os.path.join(input_dir, series_name)
    
    try:
        start_time = time.time()
        print(f"🔄 Starting {series_name}...")
        
        # Process series to 2.5D volume
        volume, num_slices = process_dicom_series_2d(series_path, img_size)
        
        # Save processed volume
        output_path = os.path.join(output_dir, f"{series_name}.npz")
        np.savez_compressed(output_path, volume=volume, num_slices=num_slices)
        
        processing_time = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        
        print(f"✅ {series_name}: {volume.shape} -> {file_size_mb:.1f}MB in {processing_time:.2f}s")
        
        return {
            'series_name': series_name,
            'output_path': output_path,
            'status': 'success',
            'processing_time': processing_time,
            'volume_shape': volume.shape,
            'num_slices': num_slices,
            'file_size_mb': file_size_mb
        }
        
    except Exception as e:
        print(f"❌ {series_name}: FAILED - {str(e)}")
        return {
            'series_name': series_name,
            'output_path': None,
            'status': 'failed',
            'error': str(e),
            'processing_time': 0,
            'volume_shape': None,
            'num_slices': 0,
            'file_size_mb': 0
        }

# ====================================================
# Batch Processing Functions
# ====================================================

def batch_process_to_3d_volumes(input_dir: str, output_dir: str, target_shape: Tuple[int, int, int] = CFG.DEFAULT_3D_SHAPE,
                               save_format: str = 'npz') -> Dict[str, str]:
    """
    Batch process multiple DICOM series to 3D volumes
    
    Args:
        input_dir: Directory containing DICOM series folders
        output_dir: Output directory for processed volumes
        target_shape: Target volume shape
        save_format: Save format ('npz' or 'npy')
    
    Returns:
        Dictionary mapping series names to output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_files = {}
    
    for series_name in os.listdir(input_dir):
        series_path = os.path.join(input_dir, series_name)
        if not os.path.isdir(series_path):
            continue
        
        try:
            print(f"Processing series: {series_name}")
            volume = process_dicom_series_safe(series_path, target_shape)
            
            # Save processed volume
            if save_format == 'npz':
                output_path = os.path.join(output_dir, f"{series_name}.npz")
                np.savez_compressed(output_path, volume=volume)
            else:
                output_path = os.path.join(output_dir, f"{series_name}.npy")
                np.save(output_path, volume)
            
            processed_files[series_name] = output_path
            print(f"✓ Saved: {output_path}")
            
        except Exception as e:
            print(f"✗ Failed to process {series_name}: {e}")
    
    return processed_files

def batch_process_to_2_5d_volumes(input_dir: str, output_dir: str, img_size: int = CFG.DEFAULT_2D_SIZE,
                                parallel: bool = True, num_workers: Optional[int] = None) -> Dict[str, str]:
    """
    Batch process multiple DICOM series to 2.5D volumes with multiprocessing
    
    Args:
        input_dir: Directory containing DICOM series folders
        output_dir: Output directory for processed volumes
        img_size: Target image size
        parallel: Whether to use parallel processing
        num_workers: Number of parallel workers
    
    Returns:
        Dictionary mapping series names to output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of series directories
    series_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    total_series = len(series_dirs)
    
    print(f"🔄 Processing {total_series} series to 2.5D volumes...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖼️  Target size: {img_size}×{img_size}")
    
    if parallel and total_series > 1:
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        print(f"⚡ Using parallel processing with {num_workers} workers")
        
        # Prepare arguments for worker function
        worker_args = [(series_name, input_dir, output_dir, img_size) for series_name in series_dirs]
        
        # Parallel processing with progress tracking
        processed_files = {}
        successful_count = 0
        failed_count = 0
        total_processing_time = 0
        total_file_size = 0
        
        with Pool(processes=num_workers) as pool:
            # Use map with progress tracking
            results = []
            for result in tqdm(pool.imap(_process_single_series_to_2_5d, worker_args), 
                             total=total_series, desc="Processing series"):
                results.append(result)
                
                if result['status'] == 'success':
                    processed_files[result['series_name']] = result['output_path']
                    successful_count += 1
                    total_processing_time += result['processing_time']
                    total_file_size += result['file_size_mb']
                    
                    if successful_count % 100 == 0:  # Progress update every 100 series
                        avg_time = total_processing_time / successful_count
                        print(f"✓ Processed {successful_count}/{total_series} series "
                              f"(avg: {avg_time:.2f}s/series, total: {total_file_size:.1f}MB)")
                else:
                    failed_count += 1
                    print(f"✗ Failed: {result['series_name']} - {result.get('error', 'Unknown error')}")
    
    else:
        # Sequential processing
        print("🔄 Using sequential processing")
        processed_files = {}
        successful_count = 0
        failed_count = 0
        total_processing_time = 0
        total_file_size = 0
        
        for i, series_name in enumerate(tqdm(series_dirs, desc="Processing series")):
            series_path = os.path.join(input_dir, series_name)
            try:
                start_time = time.time()
                
                # Process series to 2.5D volume
                volume, num_slices = process_dicom_series_2d(series_path, img_size)
                
                # Save processed volume
                output_path = os.path.join(output_dir, f"{series_name}.npz")
                np.savez_compressed(output_path, volume=volume, num_slices=num_slices)
                
                processing_time = time.time() - start_time
                file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                
                processed_files[series_name] = output_path
                successful_count += 1
                total_processing_time += processing_time
                total_file_size += file_size_mb
                
                if (i + 1) % 100 == 0:
                    avg_time = total_processing_time / successful_count
                    print(f"✓ Processed {successful_count}/{total_series} series "
                          f"(avg: {avg_time:.2f}s/series, total: {total_file_size:.1f}MB)")
                    
            except Exception as e:
                failed_count += 1
                print(f"✗ Failed to process {series_name}: {e}")
    
    # Final summary
    avg_time = total_processing_time / max(successful_count, 1)
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"✅ Successful: {successful_count}/{total_series} series")
    print(f"❌ Failed: {failed_count}/{total_series} series")
    print(f"⏱️  Average time: {avg_time:.2f}s per series")
    print(f"💾 Total data: {total_file_size:.1f} MB")
    print(f"📁 Output directory: {output_dir}")
    
    return processed_files

def batch_process_to_2_5d_volumes_limited(input_dir: str, output_dir: str, img_size: int = CFG.DEFAULT_2D_SIZE,
                                         parallel: bool = True, num_workers: Optional[int] = None, 
                                         limit: int = 10) -> Dict[str, str]:
    """
    Test version: Process limited number of DICOM series to 2.5D volumes with detailed output
    
    Args:
        input_dir: Directory containing DICOM series folders
        output_dir: Output directory for processed volumes
        img_size: Target image size
        parallel: Whether to use parallel processing
        num_workers: Number of parallel workers
        limit: Maximum number of series to process
    
    Returns:
        Dictionary mapping series names to output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get limited list of series directories
    all_series = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    series_dirs = all_series[:limit]
    total_series = len(series_dirs)
    
    print(f"🧪 Test processing {total_series} series from {len(all_series)} total...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖼️  Target size: {img_size}×{img_size}")
    
    if parallel and total_series > 1:
        if num_workers is None:
            num_workers = min(4, max(1, cpu_count() - 1))  # Use fewer workers for test
        
        print(f"⚡ Using parallel processing with {num_workers} workers")
        
        # Prepare arguments for worker function
        worker_args = [(series_name, input_dir, output_dir, img_size) for series_name in series_dirs]
        
        # Parallel processing with detailed output
        processed_files = {}
        successful_count = 0
        failed_count = 0
        total_processing_time = 0
        total_file_size = 0
        
        with Pool(processes=num_workers) as pool:
            for result in pool.imap(_process_single_series_to_2_5d_detailed, worker_args):
                if result['status'] == 'success':
                    processed_files[result['series_name']] = result['output_path']
                    successful_count += 1
                    total_processing_time += result['processing_time']
                    total_file_size += result['file_size_mb']
                else:
                    failed_count += 1
    
    else:
        # Sequential processing with detailed output
        print("🔄 Using sequential processing")
        processed_files = {}
        successful_count = 0
        failed_count = 0
        total_processing_time = 0
        total_file_size = 0
        
        for series_name in series_dirs:
            series_path = os.path.join(input_dir, series_name)
            try:
                start_time = time.time()
                print(f"🔄 Starting {series_name}...")
                
                # Process series to 2.5D volume
                volume, num_slices = process_dicom_series_2d(series_path, img_size)
                
                # Save processed volume
                output_path = os.path.join(output_dir, f"{series_name}.npz")
                np.savez_compressed(output_path, volume=volume, num_slices=num_slices)
                
                processing_time = time.time() - start_time
                file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                
                processed_files[series_name] = output_path
                successful_count += 1
                total_processing_time += processing_time
                total_file_size += file_size_mb
                
                print(f"✅ {series_name}: {volume.shape} -> {file_size_mb:.1f}MB in {processing_time:.2f}s")
                    
            except Exception as e:
                failed_count += 1
                print(f"❌ {series_name}: FAILED - {str(e)}")
    
    # Test summary
    avg_time = total_processing_time / max(successful_count, 1)
    print(f"\n📊 TEST SUMMARY:")
    print(f"✅ Successful: {successful_count}/{total_series} series")
    print(f"❌ Failed: {failed_count}/{total_series} series")
    print(f"⏱️  Average time: {avg_time:.2f}s per series")
    print(f"💾 Total data: {total_file_size:.1f} MB")
    
    return processed_files

def batch_convert_to_png(input_dir: str, output_dir: str, output_size: int = CFG.DEFAULT_PNG_SIZE,
                        parallel: bool = True, num_workers: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Batch convert DICOM series to PNG files
    
    Args:
        input_dir: Directory containing DICOM series folders
        output_dir: Output directory for PNG files
        output_size: Target PNG image size
        parallel: Whether to use parallel processing
        num_workers: Number of parallel workers
    
    Returns:
        Dictionary mapping series names to lists of created PNG file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    converter = DICOMToPNGConverter(output_size=output_size)
    
    series_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    converted_files = {}
    
    if parallel and len(series_dirs) > 1:
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        def convert_series(series_name):
            series_path = os.path.join(input_dir, series_name)
            try:
                png_files = converter.convert_series_to_pngs(series_path, output_dir, series_name)
                return series_name, png_files
            except Exception as e:
                print(f"Error converting {series_name}: {e}")
                return series_name, []
        
        with Pool(processes=num_workers) as pool:
            results = pool.map(convert_series, series_dirs)
        
        converted_files = dict(results)
    else:
        # Sequential processing
        for series_name in tqdm(series_dirs, desc="Converting series"):
            series_path = os.path.join(input_dir, series_name)
            try:
                png_files = converter.convert_series_to_pngs(series_path, output_dir, series_name)
                converted_files[series_name] = png_files
                print(f"✓ Converted {series_name}: {len(png_files)} files")
            except Exception as e:
                print(f"✗ Failed to convert {series_name}: {e}")
                converted_files[series_name] = []
    
    return converted_files

# ====================================================
# Testing and Validation Functions
# ====================================================

def test_single_series_3d(series_path: str, target_shape: Tuple[int, int, int] = CFG.DEFAULT_3D_SHAPE) -> Optional[np.ndarray]:
    """
    Test 3D processing for single series
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume shape
    
    Returns:
        Processed volume or None if failed
    """
    try:
        print(f"Testing 3D processing: {series_path}")
        volume = process_dicom_series_safe(series_path, target_shape)
        
        print(f"✓ Successfully processed series")
        print(f"  Volume shape: {volume.shape}")
        print(f"  Volume dtype: {volume.dtype}")
        print(f"  Volume range: [{volume.min()}, {volume.max()}]")
        
        return volume
        
    except Exception as e:
        print(f"✗ Failed to process series: {e}")
        return None

def test_single_series_2d(series_path: str, img_size: int = CFG.DEFAULT_2D_SIZE) -> Optional[Tuple[np.ndarray, int]]:
    """
    Test 2.5D processing for single series
    
    Args:
        series_path: Path to DICOM series
        img_size: Target image size
    
    Returns:
        Tuple of (volume, num_slices) or None if failed
    """
    try:
        print(f"Testing 2.5D processing: {series_path}")
        volume, num_slices = process_dicom_series_2d(series_path, img_size)
        
        print(f"✓ Successfully processed series")
        print(f"  Volume shape: {volume.shape}")
        print(f"  Number of slices: {num_slices}")
        print(f"  Volume dtype: {volume.dtype}")
        print(f"  Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
        
        return volume, num_slices
        
    except Exception as e:
        print(f"✗ Failed to process series: {e}")
        return None

if __name__ == "__main__":
    """Comprehensive testing suite for data processing pipeline"""
    
    import pandas as pd
    import time
    import sys
    
    # Data paths
    DATA_DIR = "/mnt/storage/kaggle_data/rsna-intracranial-aneurysm-detection"
    SERIES_DIR = os.path.join(DATA_DIR, "series")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed_data")
    
    def get_sample_series(num_samples=3):
        """Get a few sample series for testing"""
        if not os.path.exists(SERIES_DIR):
            return []
            
        series_list = os.listdir(SERIES_DIR)
        # Filter out empty directories
        valid_series = []
        for series in series_list[:10]:  # Check first 10
            series_path = os.path.join(SERIES_DIR, series)
            if os.path.isdir(series_path):
                dcm_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
                if len(dcm_files) > 0:
                    valid_series.append(series)
                    if len(valid_series) >= num_samples:
                        break
        return valid_series

    def test_3d_volume_processing():
        """Test 3D volume processing (32-channel approach)"""
        print("\n" + "="*70)
        print("TESTING 3D VOLUME PROCESSING (32-Channel Approach)")
        print("="*70)
        
        sample_series = get_sample_series(2)
        
        for i, series_name in enumerate(sample_series):
            series_path = os.path.join(SERIES_DIR, series_name)
            print(f"\nTest {i+1}: Processing series {series_name[:50]}...")
            
            start_time = time.time()
            try:
                # Test with our high-level function
                volume = process_dicom_series_3d(series_path, target_shape=(32, 384, 384))
                
                processing_time = time.time() - start_time
                print(f"✓ SUCCESS - Processed in {processing_time:.2f}s")
                print(f"  Volume shape: {volume.shape}")
                print(f"  Volume dtype: {volume.dtype}")
                print(f"  Volume range: [{volume.min()}, {volume.max()}]")
                print(f"  Memory usage: {volume.nbytes / 1024 / 1024:.1f} MB")
                
                # Save processed volume
                output_dir = os.path.join(PROCESSED_DIR, "3d_volumes")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{series_name[:50]}.npz")
                np.savez_compressed(output_path, volume=volume)
                print(f"  Saved to: {output_path}")
                
            except Exception as e:
                print(f"✗ FAILED - {e}")

    def test_2d_slice_processing():
        """Test 2.5D slice window processing"""
        print("\n" + "="*70)
        print("TESTING 2.5D SLICE WINDOW PROCESSING")
        print("="*70)
        
        sample_series = get_sample_series(2)
        
        for i, series_name in enumerate(sample_series):
            series_path = os.path.join(SERIES_DIR, series_name)
            print(f"\nTest {i+1}: Processing series {series_name[:50]}...")
            
            start_time = time.time()
            try:
                # Test with our high-level function
                volume, num_slices = process_dicom_series_2d(series_path, img_size=224)
                
                processing_time = time.time() - start_time
                print(f"✓ SUCCESS - Processed in {processing_time:.2f}s")
                print(f"  Volume shape: {volume.shape}")
                print(f"  Number of slices: {num_slices}")
                print(f"  Volume dtype: {volume.dtype}")
                print(f"  Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
                
                # Test slice window extraction
                processor = SliceWindowProcessor(img_size=224)
                center_idx = num_slices // 2
                window = processor.take_window_from_volume(volume, center_idx)
                print(f"  Sample window shape: {window.shape}")
                
                # Save processed volume
                output_dir = os.path.join(PROCESSED_DIR, "2_5d_volumes")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{series_name[:50]}.npz")
                np.savez_compressed(output_path, volume=volume, num_slices=num_slices)
                print(f"  Saved to: {output_path}")
                
            except Exception as e:
                print(f"✗ FAILED - {e}")

    def test_png_conversion():
        """Test DICOM to PNG conversion"""
        print("\n" + "="*70)
        print("TESTING DICOM TO PNG CONVERSION")
        print("="*70)
        
        sample_series = get_sample_series(1)  # Just test one series
        
        for i, series_name in enumerate(sample_series):
            series_path = os.path.join(SERIES_DIR, series_name)
            print(f"\nTest {i+1}: Converting series {series_name[:50]}...")
            
            start_time = time.time()
            try:
                # Test PNG conversion
                output_dir = os.path.join(PROCESSED_DIR, "png_images")
                png_files = convert_dicom_series_to_png(series_path, output_dir, output_size=224)
                
                processing_time = time.time() - start_time
                print(f"✓ SUCCESS - Converted in {processing_time:.2f}s")
                print(f"  Created {len(png_files)} PNG files")
                print(f"  Output directory: {os.path.join(output_dir, series_name[:50])}")
                
                # Check first few PNG files
                if png_files:
                    print(f"  Sample files:")
                    for png_file in png_files[:3]:
                        print(f"    {os.path.basename(png_file)}")
                    if len(png_files) > 3:
                        print(f"    ... and {len(png_files) - 3} more")
                
            except Exception as e:
                print(f"✗ FAILED - {e}")

    def test_different_modalities():
        """Test processing on different modalities"""
        print("\n" + "="*70)
        print("TESTING DIFFERENT MODALITIES")
        print("="*70)
        
        # Check train.csv to find different modalities
        train_csv = os.path.join(DATA_DIR, "train.csv")
        
        if os.path.exists(train_csv):
            try:
                df = pd.read_csv(train_csv)
                
                # Get sample series for each modality
                modalities = df['Modality'].unique()
                print(f"Available modalities: {modalities}")
                
                for modality in modalities[:2]:  # Test first 2 modalities
                    modality_series = df[df['Modality'] == modality]['SeriesInstanceUID'].iloc[0]
                    series_path = os.path.join(SERIES_DIR, modality_series)
                    
                    if os.path.exists(series_path):
                        print(f"\nTesting {modality} modality:")
                        print(f"Series: {modality_series[:50]}...")
                        
                        try:
                            # Test 3D processing with smaller shape for speed
                            volume = process_dicom_series_3d(series_path, target_shape=(16, 256, 256))
                            print(f"  ✓ 3D processing: {volume.shape}, range [{volume.min()}, {volume.max()}]")
                            
                            # Test 2D processing  
                            volume_2d, num_slices = process_dicom_series_2d(series_path, img_size=224)
                            print(f"  ✓ 2D processing: {volume_2d.shape}, {num_slices} slices")
                            
                        except Exception as e:
                            print(f"  ✗ Failed: {e}")
                    else:
                        print(f"Series not found: {modality_series}")
            except Exception as e:
                print(f"Could not read train.csv: {e}")

    def test_edge_cases():
        """Test edge cases and error handling"""
        print("\n" + "="*70)
        print("TESTING EDGE CASES AND ERROR HANDLING")
        print("="*70)
        
        # Test with non-existent series
        print("\nTest 1: Non-existent series")
        try:
            volume = process_dicom_series_3d("/fake/path", target_shape=(32, 384, 384))
            print("✗ Should have failed!")
        except Exception as e:
            print(f"✓ Correctly failed: {type(e).__name__}")
        
        # Test with very small target shape
        print("\nTest 2: Very small target shape")
        sample_series = get_sample_series(1)
        if sample_series:
            series_path = os.path.join(SERIES_DIR, sample_series[0])
            try:
                volume = process_dicom_series_3d(series_path, target_shape=(4, 64, 64))
                print(f"✓ Small shape processing: {volume.shape}")
            except Exception as e:
                print(f"✗ Failed: {e}")

    def run_comprehensive_tests():
        """Run all tests"""
        print("RSNA 2025 Data Processing Pipeline - Test Suite")
        print("=" * 70)
        
        # Create processed data directory
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # Check data availability
        if not os.path.exists(SERIES_DIR):
            print(f"❌ Series directory not found: {SERIES_DIR}")
            print("\nModule loaded successfully. Available functions:")
            print("- process_dicom_series_3d(): 3D volume processing")
            print("- process_dicom_series_2d(): 2.5D slice window processing") 
            print("- convert_dicom_series_to_png(): DICOM to PNG conversion")
            print("- batch_process_to_3d_volumes(): Batch 3D processing")
            print("- batch_convert_to_png(): Batch PNG conversion")
            return
        
        series_count = len([d for d in os.listdir(SERIES_DIR) if os.path.isdir(os.path.join(SERIES_DIR, d))])
        print(f"📊 Found {series_count} series in dataset")
        
        # Run tests
        test_3d_volume_processing()
        test_2d_slice_processing() 
        test_png_conversion()
        test_different_modalities()
        test_edge_cases()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS COMPLETED!")
        print("="*70)
        print(f"Processed data saved to: {PROCESSED_DIR}")

    def run_2_5d_processing():
        """Run 2.5D volume processing on full dataset"""
        
        if len(sys.argv) > 1 and sys.argv[1] in ["--process-2-5d", "--test-2-5d"]:
            is_test = sys.argv[1] == "--test-2-5d"
            
            if is_test:
                print("🧪 TESTING 2.5D VOLUME PROCESSING (10 series)")
                output_suffix = "test"
                test_limit = 10
            else:
                print("🚀 STARTING 2.5D VOLUME PROCESSING (FULL DATASET)")
                output_suffix = "full"
                test_limit = None
                
            print("=" * 70)
            
            series_dir = os.path.join(DATA_DIR, "series")
            output_dir = os.path.join(PROCESSED_DIR, f"2_5d_volumes_{output_suffix}")
            
            if not os.path.exists(series_dir):
                print(f"❌ Series directory not found: {series_dir}")
                return
            
            print(f"📂 Input: {series_dir}")
            print(f"📁 Output: {output_dir}")
            
            # Check available space (only for full processing)
            if not is_test:
                import shutil
                total, used, free = shutil.disk_usage(os.path.dirname(output_dir))
                free_gb = free // (1024**3)
                print(f"💾 Available space: {free_gb} GB")
                
                if free_gb < 100:  # Need ~87GB for full dataset
                    print(f"⚠️  Warning: May need ~87GB for full dataset, only {free_gb}GB available")
                    response = input("Continue anyway? (y/N): ")
                    if response.lower() != 'y':
                        print("Cancelled.")
                        return
            
            # Start processing
            start_time = time.time()
            
            if is_test:
                # Test mode: process only limited series with detailed output
                processed_files = batch_process_to_2_5d_volumes_limited(
                    input_dir=series_dir,
                    output_dir=output_dir,
                    img_size=224,
                    parallel=True,
                    num_workers=4,  # Use fewer workers for test
                    limit=test_limit
                )
            else:
                # Full processing
                processed_files = batch_process_to_2_5d_volumes(
                    input_dir=series_dir,
                    output_dir=output_dir,
                    img_size=224,
                    parallel=True,
                    num_workers=None  # Auto-detect
                )
            
            total_time = time.time() - start_time
            hours = total_time // 3600
            minutes = (total_time % 3600) // 60
            
            if is_test:
                print(f"\n🧪 2.5D TEST COMPLETED!")
                avg_time = total_time / max(len(processed_files), 1)
                estimated_full = avg_time * 18000  # Estimate for ~18k series
                estimated_hours = estimated_full // 3600
                print(f"⏱️  Test time: {total_time:.1f}s ({len(processed_files)} series)")
                print(f"📊 Average: {avg_time:.2f}s per series")
                print(f"🔮 Estimated full dataset: ~{estimated_hours:.1f} hours")
            else:
                print(f"\n🎉 2.5D PROCESSING COMPLETED!")
                print(f"⏱️  Total time: {hours:.0f}h {minutes:.0f}m")
            
            print(f"📊 Processed {len(processed_files)} series")
            print(f"📁 Output directory: {output_dir}")
            
            return processed_files
        else:
            # Run the comprehensive test suite
            run_comprehensive_tests()
    
    # Check command line arguments
    run_2_5d_processing()
