# RSNA 2025 Normalization Documentation

## Overview

This document explains the normalization techniques used in the RSNA 2025 Intracranial Aneurysm Detection project. The normalization system ensures consistent image preprocessing across training, inference, and analysis pipelines.

## Why Normalization is Critical

Medical imaging data comes from different scanners with varying intensity ranges and acquisition parameters. Without proper normalization:

- **CT images** have Hounsfield Units (HU) ranging from -1000 to +3000, but we only care about soft tissue (0-500 HU)
- **MR images** have arbitrary intensity values that vary significantly between scanners and sequences
- **Model performance** degrades when training and inference use different intensity ranges

## Normalization Techniques

### 1. CT/CTA Normalization (Fixed Range)

**What it does:** Maps CT intensity values to a fixed range for consistent processing.

**How it works:**
1. **Clip values** to [0, 500] Hounsfield Units (focuses on soft tissue and blood vessels)
2. **Scale linearly** from [0, 500] to [0, 255]
3. **Convert to uint8** for memory efficiency

**Why this range:** 
- 0 HU = Water (blood vessels, soft tissue)
- 500 HU = Bone (upper limit for vascular structures)
- Excludes air (-1000 HU) and dense bone (>500 HU)

**Code example:**
```python
def apply_ct_normalization(img, fixed_range=(0, 500)):
    r_min, r_max = fixed_range
    normalized = np.clip(img, r_min, r_max)
    normalized = (normalized - r_min) / (r_max - r_min)
    return (normalized * 255).astype(np.uint8)
```

### 2. MR Normalization (Adaptive Percentile)

**What it does:** Uses statistical percentiles to adapt to each image's intensity distribution.

**How it works:**
1. **Calculate percentiles** [1st, 99th] to find the main intensity range
2. **Clip values** to this range (removes outliers)
3. **Scale linearly** from [p1, p99] to [0, 255]
4. **Convert to uint8** for consistency

**Why percentiles:**
- MR intensities are arbitrary and vary by scanner/sequence
- Percentiles are robust to outliers (noise, artifacts)
- [1st, 99th] percentiles capture the main tissue signal

**Code example:**
```python
def apply_statistical_normalization(img, percentile_range=(1, 99)):
    p_low, p_high = np.percentile(img, percentile_range)
    if p_high > p_low:
        normalized = np.clip(img, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low)
        return (normalized * 255).astype(np.uint8)
    else:
        # Fallback to min-max if percentiles are equal
        return fallback_minmax_normalization(img)
```

### 3. DICOM Rescaling

**What it does:** Applies DICOM RescaleSlope and RescaleIntercept to convert raw pixel values to meaningful units.

**Why needed:** DICOM files store raw pixel values that need to be converted to actual intensity units.

**How it works:**
```python
def apply_rescale_intercept_slope(arr, slope=1.0, intercept=0.0):
    if slope != 1.0 or intercept != 0.0:
        return arr * float(slope) + float(intercept)
    return arr
```

## Script Architecture

### Central Module: `normalization.py`

**Purpose:** Single source of truth for all normalization functions.

**Key Functions:**
- `apply_ct_normalization()` - CT/CTA fixed range normalization
- `apply_statistical_normalization()` - MR adaptive percentile normalization
- `normalize_dicom_series()` - Main function for processing DICOM series
- `get_modality_normalization()` - Returns appropriate function based on modality

**Configuration:**
```python
class NormalizationConfig:
    CT_NORMALIZATION_RANGE = (0, 500)  # Fixed range for CT
    MR_PERCENTILE_RANGE = (1, 99)      # Percentile range for MR
    OUTPUT_DTYPE = np.uint8            # Consistent output type
    OUTPUT_RANGE = (0, 255)            # Consistent output range
```

### Training Pipeline: `data_processing.py`

**How it works:**
1. **Imports** normalization functions from `normalization.py`
2. **Processes** DICOM files during data preparation
3. **Applies** modality-specific normalization
4. **Saves** preprocessed volumes as `.npz` files

**Integration:**
```python
from normalization import (
    apply_statistical_normalization,
    apply_ct_normalization, 
    get_modality_normalization
)

# Use in processing pipeline
if modality == 'CT':
    processed_img = apply_ct_normalization(img)
else:
    processed_img = apply_statistical_normalization(img)
```

### Inference Pipeline: `kaggle_inference.ipynb`

**How it works:**
1. **Imports** `normalize_dicom_series` from `normalization.py`
2. **Processes** DICOM files in real-time during inference
3. **Applies** identical normalization as training
4. **Ensures** consistency between training and inference

**Integration:**
```python
from normalization import normalize_dicom_series

def series_to_tensor_chw(dicoms) -> np.ndarray:
    """
    Convert DICOM series to normalized tensor using centralized normalization
    """
    return normalize_dicom_series(dicoms, target_size=IMG_SIZE, apply_rescale=True)
```

## Data Flow

### Training Data Flow
```
Raw DICOM Files
    ↓
DICOM Rescaling (RescaleSlope/RescaleIntercept)
    ↓
Modality Detection (CT vs MR)
    ↓
Modality-Specific Normalization
    ↓
Resize to Target Size
    ↓
Save as .npz (uint8 [0, 255])
```

### Inference Data Flow
```
Raw DICOM Files
    ↓
DICOM Rescaling (RescaleSlope/RescaleIntercept)
    ↓
Modality Detection (CT vs MR)
    ↓
Modality-Specific Normalization (IDENTICAL to training)
    ↓
Resize to Target Size
    ↓
Model Input (uint8 [0, 255])
```

### Normalization Decision Tree
```
                    Raw DICOM Image
                           │
                           ▼
                Apply RescaleSlope/RescaleIntercept
                           │
                           ▼
                    Check Modality
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
               Modality = CT   Modality = MR
                    │             │
                    ▼             ▼
            Fixed Range [0,500]  Percentile [p1,p99]
                    │             │
                    ▼             ▼
            Scale to [0,255]  Scale to [0,255]
                    │             │
                    └──────┬──────┘
                           │
                           ▼
                    Convert to uint8
                           │
                           ▼
                    Final Output [0,255]
```

## Modality-Specific Processing

### CT/CTA Modalities
- **Input Range:** -1000 to +3000 HU (raw DICOM)
- **Processing Range:** 0 to 500 HU (soft tissue focus)
- **Output Range:** 0 to 255 (uint8)
- **Method:** Fixed range normalization
- **Rationale:** CT has well-defined Hounsfield Units

### MR Modalities (MRA, MRI T2, MRI T1post)
- **Input Range:** Arbitrary units (varies by scanner)
- **Processing Range:** [p1, p99] percentiles (adaptive)
- **Output Range:** 0 to 255 (uint8)
- **Method:** Adaptive percentile normalization
- **Rationale:** MR intensities are scanner-dependent

## Edge Case Handling

### 1. All Zeros
- **CT:** Returns zeros (no valid tissue)
- **MR:** Returns zeros (no signal)

### 2. All Same Value
- **CT:** Returns zeros (no contrast)
- **MR:** Returns zeros (no contrast)

### 3. Very Small Range
- **Both:** Fallback to min-max normalization
- **Ensures:** Always produces valid [0, 255] output

### 4. Multi-frame DICOMs
- **Process:** Each frame individually
- **Resize:** All frames to target size
- **Stack:** Into volume array

## Verification and Testing

### Built-in Verification
```python
from normalization import verify_normalization_consistency

# Test that CT and MR normalization work correctly
if verify_normalization_consistency():
    print("✅ All normalization tests passed!")
```

### Consistency Check
- **Training vs Inference:** Identical normalization methods
- **Edge Cases:** Robust handling of unusual data
- **Output Format:** Consistent uint8 [0, 255] range

## Configuration Summary

| Parameter | CT/CTA | MR Modalities |
|-----------|--------|---------------|
| **Method** | Fixed range | Adaptive percentile |
| **Range** | [0, 500] HU | [p1, p99] |
| **Output** | uint8 [0, 255] | uint8 [0, 255] |
| **Rationale** | Hounsfield Units | Scanner variability |

## Usage Examples

### Basic Usage
```python
from normalization import apply_ct_normalization, apply_statistical_normalization

# CT normalization
ct_image = load_ct_image()  # Raw DICOM data
ct_normalized = apply_ct_normalization(ct_image)

# MR normalization  
mr_image = load_mr_image()  # Raw DICOM data
mr_normalized = apply_statistical_normalization(mr_image)
```

### DICOM Series Processing
```python
from normalization import normalize_dicom_series

# Process entire DICOM series
dicoms = load_dicom_series(series_path)
normalized_volume = normalize_dicom_series(
    dicoms, 
    target_size=224, 
    apply_rescale=True
)
```

### Configuration Access
```python
from normalization import get_normalization_summary

# Get current configuration
config = get_normalization_summary()
print(config['ct_normalization']['description'])
# Output: "CT/CTA: Fixed range [0, 500] → [0, 255] uint8"
```

## Script Integration

### File Dependencies
```
normalization.py (Central Module)
    ├── data_processing.py (Training Pipeline)
    ├── kaggle_inference.ipynb (Inference Pipeline)
    └── result_analysis.ipynb (Analysis - uses preprocessed data)
```

### Import Structure
```python
# In data_processing.py
from normalization import (
    apply_statistical_normalization,
    apply_ct_normalization, 
    get_modality_normalization
)

# In kaggle_inference.ipynb
from normalization import normalize_dicom_series, apply_rescale_intercept_slope
```

### Execution Flow
1. **Training:** `data_processing.py` → `normalization.py` → Preprocessed `.npz` files
2. **Inference:** `kaggle_inference.ipynb` → `normalization.py` → Model input
3. **Analysis:** `result_analysis.ipynb` → Uses preprocessed data (no normalization)

## Benefits of Centralized Normalization

1. **Consistency:** Identical processing across all pipelines
2. **Maintainability:** Single place to update normalization logic
3. **Testability:** Built-in verification and testing functions
4. **Documentation:** Comprehensive docstrings and examples
5. **Reusability:** Can be imported by any component
6. **Performance:** Optimized for medical imaging data
7. **Version Control:** Single file to track normalization changes

## Troubleshooting

### Common Issues

1. **Import Errors:** Ensure `normalization.py` is in the Python path
2. **DICOM Errors:** Check RescaleSlope/RescaleIntercept values
3. **Memory Issues:** Use uint8 output format for efficiency
4. **Edge Cases:** Built-in fallbacks handle unusual data

### Debug Mode
```python
from normalization import verify_normalization_consistency

# Run comprehensive tests
if not verify_normalization_consistency():
    print("❌ Normalization issues detected!")
    # Check configuration and data
```

## Conclusion

The centralized normalization system ensures consistent, robust image preprocessing across the entire RSNA 2025 pipeline. By using modality-specific techniques and handling edge cases properly, it maximizes model performance while maintaining data integrity.

For questions or issues, refer to the `normalization.py` module documentation or run the built-in verification functions.
