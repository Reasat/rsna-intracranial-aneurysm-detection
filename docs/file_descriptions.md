# RSNA 2025 Intracranial Aneurysm Detection - File Structure and Data Organization

## Overview

This document describes the complete data structure and file organization for the RSNA 2025 Intracranial Aneurysm Detection project. The data is organized in a hierarchical structure where each row in the training CSV represents a **DICOM series** (not individual DICOM files), and each series contains multiple DICOM slices that form a 3D volume.

## Dataset Structure

### Root Directory Structure
```
../../Datasets/rsna-intracranial-aneurysm-detection/
├── train.csv                    # Main training labels (4,349 series)
├── train_localizers.csv         # Localizer series labels
├── series/                      # Raw DICOM series directories
│   ├── {SeriesInstanceUID}/     # One directory per series
│   │   ├── {InstanceUID}.dcm    # Individual DICOM slices
│   │   └── ...
│   └── ...
├── segmentations/               # Ground truth segmentation masks
├── processed_data/              # Preprocessed data
│   ├── 2_5d_volumes_full/       # 2.5D processed volumes (.npz)
│   ├── 2_5d_volumes_test/       # Test set 2.5D volumes
│   ├── 2d_volumes/              # 2D processed volumes
│   ├── 3d_volumes/              # 3D processed volumes
│   └── png_images/              # PNG converted images
└── kaggle_evaluation/           # Kaggle submission evaluation
```

## Data Organization Details

### 1. Training CSV Structure (`train.csv`)

**Each row represents a DICOM series, not individual DICOM files.**

| Column | Description | Example |
|--------|-------------|---------|
| `SeriesInstanceUID` | Unique identifier for the DICOM series | `1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647` |
| `PatientAge` | Patient age in years | `64` |
| `PatientSex` | Patient gender | `Female` |
| `Modality` | Imaging modality | `MRA`, `CTA`, `MRI T1post`, `MRI T2` |
| `Left Infraclinoid Internal Carotid Artery` | Binary label (0/1) | `0` |
| `Right Infraclinoid Internal Carotid Artery` | Binary label (0/1) | `0` |
| `Left Supraclinoid Internal Carotid Artery` | Binary label (0/1) | `0` |
| `Right Supraclinoid Internal Carotid Artery` | Binary label (0/1) | `0` |
| `Left Middle Cerebral Artery` | Binary label (0/1) | `0` |
| `Right Middle Cerebral Artery` | Binary label (0/1) | `0` |
| `Anterior Communicating Artery` | Binary label (0/1) | `0` |
| `Left Anterior Cerebral Artery` | Binary label (0/1) | `0` |
| `Right Anterior Cerebral Artery` | Binary label (0/1) | `0` |
| `Left Posterior Communicating Artery` | Binary label (0/1) | `0` |
| `Right Posterior Communicating Artery` | Binary label (0/1) | `0` |
| `Basilar Tip` | Binary label (0/1) | `0` |
| `Other Posterior Circulation` | Binary label (0/1) | `0` |
| `Aneurysm Present` | Overall aneurysm presence (0/1) | `0` |

**Key Points:**
- **4,349 total series** (4,348 data rows + 1 header row)
- Each series contains **multiple DICOM slices** (typically 50-300 slices)
- Labels are **series-level**, not slice-level
- `Aneurysm Present` = 1 if any anatomical location has an aneurysm

### 2. DICOM Series Structure (`series/`)

Each series is stored in its own directory named by the `SeriesInstanceUID`:

```
series/
├── 1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647/
│   ├── 1.2.826.0.1.3680043.8.498.10124807242473374136099471315028464450.dcm
│   ├── 1.2.826.0.1.3680043.8.498.10138383895715496920719014209752366343.dcm
│   ├── 1.2.826.0.1.3680043.8.498.10163629202066490350525656863994550563.dcm
│   └── ... (typically 50-300 DICOM files per series)
└── ...
```

**Key Points:**
- Each directory contains **all DICOM slices** for that series
- Slices are typically sorted by `ImagePositionPatient[2]` (Z-coordinate)
- Each series forms a **3D volume** when stacked
- DICOM files contain metadata (patient info, imaging parameters, etc.)

### 3. Processed Data Structure (`processed_data/`)

#### 2.5D Volumes (`2_5d_volumes_full/`)
- **Format**: `.npz` files (NumPy compressed arrays)
- **Content**: Preprocessed 2.5D sliding window data
- **Naming**: `{SeriesInstanceUID}.npz`
- **Usage**: Primary input for the 2.5D EfficientNet model

#### 3D Volumes (`3d_volumes/`)
- **Format**: `.npz` files
- **Content**: Full 3D volumes processed for 3D CNN models
- **Usage**: Alternative processing approach

#### 2D Volumes (`2d_volumes/`)
- **Format**: `.npz` files
- **Content**: 2D slice data for 2D CNN models
- **Usage**: Traditional 2D approach

#### PNG Images (`png_images/`)
- **Format**: PNG files
- **Content**: Converted DICOM slices as PNG images
- **Usage**: Visualization and debugging

## Data Processing Pipeline

### 1. Raw DICOM Processing
```python
# From data_processing.py
def process_dicom_series(series_path: str) -> np.ndarray:
    """
    Process a DICOM series into a 3D volume
    - Load all DICOM files in the series
    - Sort by Z-coordinate (ImagePositionPatient[2])
    - Apply modality-specific normalization
    - Return 3D numpy array (N, H, W)
    """
```

### 2. 2.5D Window Processing
```python
# From data_processing.py
def create_2_5d_windows(volume: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Create 2.5D sliding windows from 3D volume
    - Extract overlapping windows of 5 consecutive slices
    - Each window becomes a 5-channel input
    - Process all possible windows in the volume
    """
```

### 3. Model Input Preparation
```python
# From inference.py and analysis.py
def prepare_windows(volume: np.ndarray, coords: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare windows for model inference
    - Extract sliding windows from volume
    - Create full image and ROI versions
    - Resize to model input size (224x224)
    - Convert to PyTorch tensors
    """
```

## Data Flow in the Project

### Training Phase
1. **Load Series**: Read `train.csv` to get series list and labels
2. **Process DICOM**: Convert each series to 3D volume
3. **Create Windows**: Generate 2.5D sliding windows
4. **Cache Data**: Save processed volumes as `.npz` files
5. **Train Model**: Use cached data for model training

### Inference Phase
1. **Load Cached Data**: Read preprocessed `.npz` files
2. **Create Windows**: Generate sliding windows for inference
3. **Model Prediction**: Run inference on each window
4. **Aggregate Results**: Combine window predictions to series-level prediction

### Analysis Phase
1. **Load OOF Predictions**: Use out-of-fold predictions from training
2. **Analyze Misclassifications**: Identify hard samples and error patterns
3. **Visualize Results**: Create comprehensive analysis plots

## Key Data Characteristics

### Volume Dimensions
- **Typical Series Size**: 50-300 slices per series
- **Slice Dimensions**: Varies by series (typically 256x256 to 512x512)
- **2.5D Window Size**: 5 consecutive slices
- **Model Input Size**: 224x224 pixels

### Modality Distribution
- **MRA**: Magnetic Resonance Angiography
- **CTA**: Computed Tomography Angiography  
- **MRI T1post**: T1-weighted MRI with contrast
- **MRI T2**: T2-weighted MRI

### Label Distribution
- **14 Anatomical Locations**: Each can have an aneurysm (0/1)
- **Overall Aneurysm Present**: 1 if any location has an aneurysm
- **Class Imbalance**: Most series have no aneurysms (negative class)

## File Naming Conventions

### DICOM Files
- **SeriesInstanceUID**: Unique identifier for the entire series
- **InstanceUID**: Unique identifier for individual DICOM slices
- **Format**: `{InstanceUID}.dcm`

### Processed Files
- **Volume Files**: `{SeriesInstanceUID}.npz`
- **Model Checkpoints**: `tf_efficientnet_b0_fold{fold}_best.pth`
- **Cache Files**: `{SeriesInstanceUID}.npz` (in cache directory)

## Usage Examples

### Loading a Series
```python
# Load series metadata
df = pd.read_csv('train.csv')
series_id = df['SeriesInstanceUID'].iloc[0]

# Load raw DICOM series
series_path = f'series/{series_id}/'
volume = process_dicom_series(series_path)  # Shape: (N, H, W)

# Load processed volume
volume = load_cached_volume(f'processed_data/2_5d_volumes_full/{series_id}.npz')
```

### Creating Windows for Inference
```python
# Create 2.5D windows
windows = []
for center_idx in range(volume.shape[0]):
    window = take_window(volume, center_idx, offsets=(-2, -1, 0, 1, 2))
    windows.append(window)  # Shape: (5, H, W)
```

## Important Notes

1. **Series-Level Labels**: All labels are at the series level, not individual slice level
2. **3D Context**: The model uses 3D context through sliding windows
3. **Modality-Specific Processing**: Different imaging modalities require different normalization
4. **Memory Management**: Large volumes are cached to avoid reprocessing
5. **Cross-Validation**: Data is split by series, not individual slices

This structure allows for efficient processing of 3D medical imaging data while maintaining the relationship between individual DICOM slices and their corresponding labels.
