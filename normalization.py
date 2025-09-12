"""
RSNA 2025 Intracranial Aneurysm Detection - Normalization Module

This module provides comprehensive normalization functionality for medical imaging data,
specifically designed for the RSNA 2025 Intracranial Aneurysm Detection competition.

Key Features:
- Modality-specific normalization (CT vs MR)
- DICOM RescaleSlope/RescaleIntercept handling
- Robust edge case handling
- Consistent output format (uint8 [0, 255])
- Training and inference consistency

Author: RSNA 2025 Competition Team
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union, List
import pydicom
from pathlib import Path


class NormalizationConfig:
    """Configuration for normalization parameters"""
    
    # CT normalization parameters
    CT_NORMALIZATION_RANGE = (0, 500)  # Fixed range for CT normalization
    CT_WINDOW_CENTER = 50
    CT_WINDOW_WIDTH = 350
    
    # MR normalization parameters
    MR_PERCENTILE_RANGE = (1, 99)  # Percentile range for MR normalization
    
    # Output parameters
    OUTPUT_DTYPE = np.uint8
    OUTPUT_RANGE = (0, 255)
    
    # DICOM parameters
    DEFAULT_RESCALE_SLOPE = 1.0
    DEFAULT_RESCALE_INTERCEPT = 0.0


# Global configuration instance
CFG = NormalizationConfig()


def apply_rescale_intercept_slope(arr: np.ndarray, slope: float = 1.0, intercept: float = 0.0) -> np.ndarray:
    """
    Apply DICOM RescaleSlope and RescaleIntercept to pixel array
    
    Args:
        arr: Input pixel array
        slope: RescaleSlope value from DICOM
        intercept: RescaleIntercept value from DICOM
    
    Returns:
        Rescaled pixel array
    """
    if slope != 1.0 or intercept != 0.0:
        return arr * float(slope) + float(intercept)
    return arr


def apply_statistical_normalization(img: np.ndarray, 
                                  percentile_range: Tuple[float, float] = CFG.MR_PERCENTILE_RANGE) -> np.ndarray:
    """
    Apply statistical normalization using percentiles (for MR modalities)
    
    This is the standard normalization method for MR modalities (MRA, MRI T2, MRI T1post)
    as their intensity values are arbitrary and vary significantly between scanners.
    
    Args:
        img: Input image array
        percentile_range: Tuple of (low_percentile, high_percentile)
    
    Returns:
        Normalized image in [0, 255] range as uint8
    """
    p_low, p_high = np.percentile(img, percentile_range)
    
    if p_high > p_low:
        normalized = np.clip(img, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low)
        return (normalized * 255).astype(np.uint8)
    else:
        # Fallback: min-max normalization for edge cases
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            normalized = (img - img_min) / (img_max - img_min)
            return (normalized * 255).astype(np.uint8)
        else:
            # All values are the same - return zeros
            return np.zeros_like(img, dtype=np.uint8)


def apply_ct_normalization(img: np.ndarray, 
                          fixed_range: Tuple[float, float] = CFG.CT_NORMALIZATION_RANGE) -> np.ndarray:
    """
    Apply CT-specific normalization with fixed range (for CT/CTA modalities)
    
    CT images use Hounsfield Units (HU) which have a well-defined scale.
    We use a fixed range [0, 500] to focus on soft tissue and blood vessels,
    excluding bone and air which are outside this range.
    
    Args:
        img: Input CT image array
        fixed_range: Fixed intensity range for normalization
    
    Returns:
        Normalized image in [0, 255] range as uint8
    """
    r_min, r_max = fixed_range
    
    if r_max > r_min:
        normalized = np.clip(img, r_min, r_max)
        normalized = (normalized - r_min) / (r_max - r_min)
        return (normalized * 255).astype(np.uint8)
    else:
        # Fallback to statistical normalization if range is invalid
        return apply_statistical_normalization(img)


def get_modality_normalization(modality: str) -> callable:
    """
    Get appropriate normalization function based on DICOM modality
    
    Args:
        modality: DICOM modality string ('CT' or 'MR')
    
    Returns:
        Normalization function (apply_ct_normalization or apply_statistical_normalization)
    """
    if modality == 'CT':
        return apply_ct_normalization
    else:  # MR modalities (MRA, MRI T2, MRI T1post)
        return apply_statistical_normalization


def normalize_dicom_series(dicoms: List[pydicom.Dataset], 
                          target_size: int = 224,
                          apply_rescale: bool = True) -> np.ndarray:
    """
    Normalize a series of DICOM files with modality-specific processing
    
    This is the main function used in both training and inference pipelines.
    It handles:
    - DICOM RescaleSlope/RescaleIntercept
    - Modality-specific normalization
    - Multi-frame DICOM handling
    - Image resizing
    - Edge case handling
    
    Args:
        dicoms: List of pydicom.Dataset objects
        target_size: Target image size for resizing
        apply_rescale: Whether to apply RescaleSlope/RescaleIntercept
    
    Returns:
        Normalized volume as [N, H, W] uint8 array
    """
    resized = []
    
    for d in dicoms:
        # Extract pixel array
        arr = d.pixel_array
        if arr is None or arr.size == 0:
            continue
            
        arr = arr.astype(np.float32)
        
        # Apply RescaleSlope and RescaleIntercept if requested
        if apply_rescale:
            slope = getattr(d, 'RescaleSlope', CFG.DEFAULT_RESCALE_SLOPE)
            intercept = getattr(d, 'RescaleIntercept', CFG.DEFAULT_RESCALE_INTERCEPT)
            arr = apply_rescale_intercept_slope(arr, slope, intercept)
        
        # Get modality and apply appropriate normalization
        modality = getattr(d, 'Modality', 'MR')
        normalization_func = get_modality_normalization(modality)
        arr = normalization_func(arr)
        
        # Handle multi-frame DICOMs
        if arr.ndim == 3:  # Multi-frame DICOM
            for frame in arr:
                frame_resized = cv2.resize(frame, (target_size, target_size), 
                                         interpolation=cv2.INTER_AREA)
                resized.append(frame_resized)
        else:  # Single-frame DICOM
            arr_resized = cv2.resize(arr, (target_size, target_size), 
                                   interpolation=cv2.INTER_AREA)
            resized.append(arr_resized)
    
    if len(resized) == 0:
        # Fallback to zeros if no valid frames
        return np.zeros((1, target_size, target_size), dtype=np.uint8)
    else:
        return np.stack(resized, axis=0)  # [N, H, W] uint8


def normalize_single_image(img: np.ndarray, 
                          modality: str,
                          apply_rescale: bool = False,
                          slope: float = 1.0,
                          intercept: float = 0.0) -> np.ndarray:
    """
    Normalize a single image with specified modality
    
    Args:
        img: Input image array
        modality: DICOM modality ('CT' or 'MR')
        apply_rescale: Whether to apply RescaleSlope/RescaleIntercept
        slope: RescaleSlope value
        intercept: RescaleIntercept value
    
    Returns:
        Normalized image as uint8 array
    """
    arr = img.astype(np.float32)
    
    # Apply rescaling if requested
    if apply_rescale:
        arr = apply_rescale_intercept_slope(arr, slope, intercept)
    
    # Apply modality-specific normalization
    normalization_func = get_modality_normalization(modality)
    return normalization_func(arr)


def verify_normalization_consistency(test_data: Optional[np.ndarray] = None) -> bool:
    """
    Verify that CT and MR normalization functions work consistently
    
    Args:
        test_data: Optional test data array. If None, generates synthetic data.
    
    Returns:
        True if all tests pass, False otherwise
    """
    if test_data is None:
        # Generate synthetic test data
        np.random.seed(42)
        ct_data = np.random.normal(50, 100, (32, 224, 224)).astype(np.float32)
        ct_data = np.clip(ct_data, -1000, 1000)
        
        mr_data = np.random.normal(1000, 300, (32, 224, 224)).astype(np.float32)
        mr_data = np.clip(mr_data, 0, 2000)
    else:
        ct_data = mr_data = test_data
    
    # Test CT normalization
    ct_normalized = apply_ct_normalization(ct_data)
    ct_valid = (ct_normalized.dtype == np.uint8 and 
                ct_normalized.min() >= 0 and 
                ct_normalized.max() <= 255)
    
    # Test MR normalization
    mr_normalized = apply_statistical_normalization(mr_data)
    mr_valid = (mr_normalized.dtype == np.uint8 and 
                mr_normalized.min() >= 0 and 
                mr_normalized.max() <= 255)
    
    # Test edge cases
    edge_cases = [
        np.zeros((10, 10), dtype=np.float32),
        np.full((10, 10), 100.0, dtype=np.float32),
        np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
    ]
    
    edge_valid = True
    for edge_case in edge_cases:
        ct_edge = apply_ct_normalization(edge_case)
        mr_edge = apply_statistical_normalization(edge_case)
        
        if (ct_edge.dtype != np.uint8 or mr_edge.dtype != np.uint8 or
            ct_edge.min() < 0 or ct_edge.max() > 255 or
            mr_edge.min() < 0 or mr_edge.max() > 255):
            edge_valid = False
            break
    
    return ct_valid and mr_valid and edge_valid


def get_normalization_summary() -> dict:
    """
    Get a summary of normalization parameters and methods
    
    Returns:
        Dictionary with normalization configuration
    """
    return {
        "ct_normalization": {
            "method": "Fixed range normalization",
            "range": CFG.CT_NORMALIZATION_RANGE,
            "output_range": CFG.OUTPUT_RANGE,
            "output_dtype": str(CFG.OUTPUT_DTYPE),
            "description": "CT/CTA: Fixed range [0, 500] → [0, 255] uint8"
        },
        "mr_normalization": {
            "method": "Adaptive percentile normalization", 
            "percentile_range": CFG.MR_PERCENTILE_RANGE,
            "output_range": CFG.OUTPUT_RANGE,
            "output_dtype": str(CFG.OUTPUT_DTYPE),
            "description": "MR modalities: [p1, p99] → [0, 255] uint8"
        },
        "supported_modalities": {
            "ct": ["CT", "CTA"],
            "mr": ["MRA", "MRI T2", "MRI T1post", "MR"]
        },
        "dicom_handling": {
            "rescale_slope": "Applied when present in DICOM",
            "rescale_intercept": "Applied when present in DICOM",
            "fallback_values": {
                "slope": CFG.DEFAULT_RESCALE_SLOPE,
                "intercept": CFG.DEFAULT_RESCALE_INTERCEPT
            }
        }
    }


# Example usage and testing
if __name__ == "__main__":
    print("RSNA 2025 Normalization Module")
    print("=" * 40)
    
    # Verify consistency
    print("Verifying normalization consistency...")
    if verify_normalization_consistency():
        print("✅ All normalization tests passed!")
    else:
        print("❌ Some normalization tests failed!")
    
    # Print summary
    print("\nNormalization Summary:")
    summary = get_normalization_summary()
    for key, value in summary.items():
        print(f"\n{key.upper()}:")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {value}")
