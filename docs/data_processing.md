# Data Processing Guide

This document provides comprehensive information about data processing for the RSNA 2025 Intracranial Aneurysm Detection dataset.

## 📊 Dataset Statistics

### Patient Demographics
- **Total Patients**: 4,348 unique patients
- **Patients with Aneurysms**: 1,864 (42.9%)
- **Control Patients (No Aneurysms)**: 2,484 (57.1%)
- **Test Set**: ~2,500 series (approximate)
- **⚠️ Important**: **1:1 Patient-to-Series ratio** - each patient has exactly ONE imaging series from ONE modality (no multi-modal data per patient)

### Series and Instance Statistics
- **Total Training Series**: 4,348 series
- **Total DICOM Instances**: 1,028,811 individual DICOM files
- **DICOM Format Distribution**:
  - Single-frame DICOMs: 4,026 series (92.6%)
  - Multi-frame DICOMs: 322 series (7.4%)

## 🔧 Multi-Frame DICOM Handling

The dataset contains two distinct DICOM file structures that require different processing approaches:

### Single-Frame DICOMs (92.6% of series)
- **Structure**: Each slice stored as separate DICOM file
- **Processing**: Load each file individually, stack slices
- **Advantages**: Widely compatible, easier per-slice handling
- **File count**: Multiple files per series (avg: ~236 files/series)

### Multi-Frame DICOMs (7.4% of series)  
- **Structure**: All slices stored in single DICOM file as 3D volume
- **Processing**: `pixel_array` returns 3D array with shape `(N, H, W)`
- **Advantages**: Faster I/O, reduced file count, shared metadata
- **Challenge**: Requires special handling to avoid resize errors

### Processing Strategies by Use Case
| Processing Type | Single-Frame Strategy | Multi-Frame Strategy |
|----------------|----------------------|---------------------|
| **3D Volume Processing** | Load all files, stack slices | Use entire 3D volume directly |
| **2.5D Slice Windows** | Load files, extract windows | Extract all frames, then windows |
| **2D Slice Extraction** | Load individual file | Extract specific frame from volume |
| **PNG Conversion** | Process each file separately | Process each frame from volume |

### Critical Implementation Note
Multi-frame DICOMs require frame-by-frame processing for 2D operations:
```python
# ❌ INCORRECT - Will cause OpenCV resize error
arr = dicom.pixel_array  # Shape: (150, 528, 528)
resized = cv2.resize(arr, (224, 224))  # ERROR: 3D array not supported

# ✅ CORRECT - Process each frame individually  
if arr.ndim == 3:  # Multi-frame DICOM
    frames = [cv2.resize(frame, (224, 224)) for frame in arr]
    resized = np.stack(frames, axis=0)
else:  # Single-frame DICOM
    resized = cv2.resize(arr, (224, 224))
```

## 🩻 Imaging Modalities

| Modality | Count | Percentage | Typical Frames/Series |
|----------|-------|------------|----------------------|
| CTA (CT Angiography) | 1,804 | 41.5% | ~400 |
| MRA (MR Angiography) | 1,252 | 28.8% | ~175 |
| MRI T2 | 982 | 22.6% | ~150 |
| MRI T1 post-contrast | 310 | 7.0% | ~35 |

### Image Properties
- **Most Common Image Size**: 512 × 512 pixels
- **Image Planes**:
  - Axial: 4,197 series (96.5%)
  - Coronal: 87 series (2.0%)
  - Sagittal: 64 series (1.5%)

## 📍 Localization Data

### Train_Localizers.csv - Aneurysm Center Point Annotations

The `train_localizers.csv` file provides precise aneurysm center coordinates essential for ROI-based training approaches, particularly the 2.5D dual-stream models.

**File Structure:**
```csv
SeriesInstanceUID, SOPInstanceUID, coordinates, location
1.2.826...317, 1.2.826...346, "{'x': 258.36, 'y': 261.36}", "Other Posterior Circulation"
```

**Coverage Analysis:**
| Metric | Count | Coverage |
|--------|-------|----------|
| **Total series in dataset** | 4,348 | 100% |
| **Series with aneurysms** | 1,864 | 42.9% |
| **Series with localizers** | 1,862 | **99.9%** of aneurysm cases |
| **Series without localizers** | 2,486 | All negative cases + 2 missing |

**Key Characteristics:**
- **Center Points Only**: Single (x, y) coordinates marking aneurysm centers, not bounding boxes
- **Positive Cases Only**: Localizers exist only for series with confirmed aneurysms
- **Multiple Aneurysms**: 292 series have multiple aneurysms (up to 5 per series)
- **Anatomical Distribution**: Matches overall aneurysm location patterns

**Coordinate Format and Usage:**

*Training Phase (WITH coordinates):*
```python
# Raw coordinate from train_localizers.csv
center_coords = "{'x': 258.36, 'y': 261.36}"
parsed_coords = ast.literal_eval(center_coords)  # {'x': 258.36, 'y': 261.36}

# Generate bounding box around center point
def make_bbox_px(x, y, img_size=224, box_frac=0.15):
    r = box_frac * img_size / 2  # 15% of image = ~16.8 pixels radius
    x1, y1 = max(0, x-r), max(0, y-r)
    x2, y2 = min(img_size-1, x+r), min(img_size-1, y+r)
    return x1, y1, x2, y2  # Bounding box: (241, 244, 275, 278)

# Dual-stream model training
full_image = entire_slice_window     # [5, 224, 224] - global context
roi_image = crop_around_center(bbox) # [5, 224, 224] - focused region  
prediction = model(full_image, roi_image, center_coords)
```

*Inference Phase (WITHOUT coordinates):*
```python
# No coordinates available during inference
coords = np.zeros((N, 2), dtype=np.float32)  # All zeros

# Both streams get identical input
def window_to_full_and_roi(win_chw, coords):
    if np.any(coords != 0):
        # Never executes during inference
        roi = crop_around_coords(win_chw, coords)
        return win_chw, roi
    # Identical streams during inference
    return win_chw, win_chw

# Model inference with identical streams
prediction = model(full_image, full_image, zeros)
```

## 🏗️ Processing Architectures

### 3D Volume Processing (32-Channel Approach)
The 32-channel approach addresses the dramatic variation in series lengths by compressing entire volumes to a standardized format:

**Compression Strategy:**
- **Variable Input**: 35-400 slices per series (depending on modality)
- **Fixed Output**: 32 standardized slices via 3D linear interpolation
- **Spatial Normalization**: 512×512 → 384×384 pixels

**Compression Ratios by Modality:**
| Modality | Original Slices | Compression Ratio | Information Loss |
|----------|----------------|-------------------|------------------|
| **CTA** | ~400 slices | 12.5:1 | High compression, preserves global structure |
| **MRA** | ~175 slices | 5.5:1 | Moderate compression, good detail retention |
| **MRI T2** | ~150 slices | 4.7:1 | Moderate compression |
| **MRI T1** | ~35 slices | 1.1:1 | Minimal compression, preserves fine detail |

**Advantages:**
- **Global context**: Analyzes entire volume in single pass
- **Memory efficient**: Fixed 32×384×384 format (4.5MB)
- **Standardized input**: Consistent shape for neural network training
- **Preserves 3D relationships**: Linear interpolation maintains anatomical continuity

### 2.5D Slice Window Processing
The 2.5D approach creates pseudo-3D inputs by combining adjacent slices into multi-channel windows:

**Window Strategy:**
- **Window Size**: 5 slices (center ± 2 adjacent slices)
- **Sliding Windows**: Extract window for every slice position
- **Significant Overlap**: 4/5 slices overlap between adjacent windows
- **Target Size**: 224×224 pixels per slice

**Example Window Extraction:**
```
Slice 10 window: [8, 9, 10, 11, 12]  ← center slice 10
Slice 11 window: [9, 10, 11, 12, 13] ← center slice 11
                  ↑ 80% overlap
```

**Advantages:**
- **High resolution**: Preserves original slice detail (224×224)
- **Local context**: 5-slice window captures aneurysm span
- **Robust detection**: Multiple overlapping windows detect same aneurysm
- **Memory scalable**: Process one window at a time

**Medical Rationale:**
- **Aneurysm size**: Small aneurysms (1-2mm) visible across 2-5 adjacent slices
- **Spatial continuity**: Adjacent slices provide critical anatomical context
- **Detection redundancy**: Overlapping windows increase detection sensitivity

### Complementary Approaches
Both strategies capture different aspects of 3D brain anatomy:

| Aspect | 32-Channel 3D | 2.5D Windows |
|--------|---------------|--------------|
| **Context** | Global volume structure | Local slice detail |
| **Resolution** | Compressed (32 slices) | Original (all slices) |
| **Memory** | Fixed 4.5MB | Variable by series length |
| **Detection** | Volume-level patterns | Slice-level features |
| **Best For** | Large aneurysms, vessel topology | Small aneurysms, fine detail |

## ⚡ Processing Performance and Time Estimates

Based on comprehensive testing on real RSNA data, here are the processing performance metrics and full dataset estimates:

**Note**: Initial testing revealed a 90% success rate due to multi-frame DICOM processing issues. The 10% failure rate matches the expected 7.4% multi-frame DICOMs in the dataset, indicating the need for proper multi-frame handling.

### Test Performance Results
| Processing Type | Test Series 1 (188 slices) | Test Series 2 (147 slices) | Average Time |
|----------------|---------------------------|---------------------------|--------------|
| **3D Volume Processing** | 5.98s | 3.53s | **4.76s** |
| **2.5D Slice Processing** | 2.38s | 1.24s | **1.81s** |
| **PNG Conversion** | 3.59s | - | **3.59s** |

### Full Dataset Processing Estimates (4,348 Series)

**Sequential Processing (Single-threaded):**
| Processing Type | Total Time | Storage Required |
|----------------|------------|------------------|
| **3D Volumes (32×384×384)** | **≈ 5.7 hours** | 19.6 GB |
| **2.5D Volumes (N×224×224)** | **≈ 2.4 hours** | 87 GB |
| **PNG Conversion (224×224)** | **≈ 4.3 hours** | 200+ GB |

**Parallel Processing (8 cores):**
| Processing Type | Total Time | Speedup |
|----------------|------------|---------|
| **3D Volumes** | **≈ 43 minutes** | 8x faster |
| **2.5D Volumes** | **≈ 18 minutes** | 8x faster |
| **PNG Conversion** | **≈ 32 minutes** | 8x faster |

### Modality-Weighted Time Estimates
Processing time varies significantly by modality due to different series lengths:

| Modality | Series Count | Avg. Slices | Est. Time (3D) | Est. Time (2.5D) |
|----------|--------------|-------------|----------------|------------------|
| **CTA** | 1,804 (41.5%) | ~400 | 3.0 hours | 1.3 hours |
| **MRA** | 1,252 (28.8%) | ~175 | 1.4 hours | 0.6 hours |
| **MRI T2** | 982 (22.6%) | ~150 | 1.1 hours | 0.4 hours |
| **MRI T1** | 310 (7.0%) | ~35 | 0.2 hours | 0.1 hours |

### Recommended Processing Strategy

**Option 1: Memory-Efficient (Recommended)**
```python
# Start with 3D volumes - most efficient representation
batch_process_to_3d_volumes(series_dir, output_dir, parallel=True, num_workers=8)
# Time: ~43 minutes, Storage: 19.6 GB
```

**Option 2: High-Resolution**
```python
# 2.5D processing for maximum detail retention
batch_process_to_2_5d_volumes(series_dir, output_dir, parallel=True)
# Time: ~18 minutes, Storage: 87 GB
```

**Option 3: Visualization**
```python
# PNG conversion for analysis and visualization
batch_convert_to_png(series_dir, output_dir, parallel=True)
# Time: ~32 minutes, Storage: 200+ GB
```

### Performance Optimization Notes
- **CTA series** (400 slices) take longest to process but compress most efficiently
- **MRI T1** series (35 slices) process fastest with minimal compression
- **Parallel processing** provides near-linear speedup up to CPU core count
- **Memory usage** peaks during 3D interpolation (~1-2GB per series)
- **I/O optimization** benefits from SSD storage for large datasets

## 🎨 Modality-Specific Intensity Processing

Each imaging modality requires different intensity processing due to varying physical properties and intensity characteristics:

### 🩻 CTA (CT Angiography) - 41.5% of data
- **Intensity Units**: Calibrated Hounsfield Units (HU), range: -1000 to +3000
- **Windowing Parameters**: Center=50, Width=350 (optimized for blood vessel visualization)
- **Processing Strategy**:
  ```
  Traditional: Window range [-125, +225] HU
  Implemented: Statistical normalization [0, 500] → [0, 255]
  ```
- **Clinical Rationale**: Wide window (350 HU) captures contrast-enhanced vessels

### 🧲 MRA (MR Angiography) - 28.8% of data  
- **Intensity Units**: Arbitrary units (not calibrated like CT)
- **Windowing**: None - uses adaptive statistical normalization
- **Processing Strategy**:
  ```
  Adaptive percentile normalization: [p1, p99] → [0, 255]
  p1, p99 = np.percentile(img, [1, 99])
  ```
- **Clinical Rationale**: MR intensities vary by scanner/sequence parameters

### 🧠 MRI T2 - 22.6% of data
- **Intensity Units**: Arbitrary units (T2-weighted contrast)
- **Processing**: Same as MRA - adaptive statistical normalization
- **Characteristics**: Higher signal from CSF and edema
- **Range**: Highly variable depending on scanner and sequence

### 🧠 MRI T1 Post-contrast - 7.0% of data
- **Intensity Units**: Arbitrary units (gadolinium-enhanced)
- **Processing**: Statistical normalization with adaptive percentiles
- **Characteristics**: Enhanced signal from vascularized structures
- **Special Considerations**: Intensity distribution affected by contrast timing

### Unified Processing Pipeline
1. **DICOM Loading**: Extract pixel arrays with RescaleSlope/Intercept handling
2. **Modality Detection**: Based on DICOM `Modality` tag ('CT' vs 'MR')
3. **Intensity Normalization**:
   - **CT modalities**: Statistical normalization with fixed range [0, 500]
   - **MR modalities**: Adaptive percentile normalization [p1, p99]
4. **Final Output**: All modalities normalized to [0, 255] uint8 range
5. **Volume Resizing**: 3D interpolation to target shapes (32×384×384 or 224×224)

### Processing Parameters Summary
| Modality | Units | Windowing | Processing Range | Final Output |
|----------|-------|-----------|------------------|--------------|
| **CTA** | Hounsfield Units | Center=50, Width=350 | [0, 500] → [0, 255] | uint8 |
| **MRA** | Arbitrary | None | [p1, p99] → [0, 255] | uint8 |
| **MRI T2** | Arbitrary | None | [p1, p99] → [0, 255] | uint8 |
| **MRI T1post** | Arbitrary | None | [p1, p99] → [0, 255] | uint8 |

This approach ensures robust processing across all modalities while preserving the clinical characteristics essential for aneurysm detection.

## 🔧 Technical Implementation

### Multi-Frame DICOM Processing in Practice
Different preprocessing approaches have been implemented across the notebooks to handle the 7.4% multi-frame DICOMs:

#### 32-Channel Inference Notebook (3D Processing)
```python
# Adaptive strategy based on series structure
if len(datasets) == 1 and first_img.ndim == 3:
    # Single multi-frame DICOM: Process all frames as 3D volume
    return self._process_single_3d_dicom(first_ds, series_name)
else:
    # Multiple single-frame DICOMs: Extract and stack 2D slices
    return self._process_multiple_2d_dicoms(datasets, series_name)
```

#### EDA Notebook (Complete Frame Processing)
```python
# Process all frames from multi-frame DICOMs
if hasattr(dcm, 'NumberOfFrames'):
    frames = dcm.pixel_array  # 3D array (N, H, W)
    resized = np.stack([cv2.resize(f, target_shape) for f in frames], axis=0)
```

#### DICOM-PNG Notebook (Frame Extraction)
```python
# Convert multi-frame to single-channel grayscale images
if img.ndim == 3:
    # Handle multi-channel/multi-frame → single image conversion
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```

#### Current Data Processing Script
- **Issue**: Original code assumed all DICOMs are single-frame
- **Fix Required**: Implement frame-by-frame processing for multi-frame DICOMs
- **Expected Success Rate**: ~92.6% (single-frame) without fix, 100% with proper handling

### Data Processing Pipeline
- **Input Formats**: DICOM (single-frame and multi-frame)
- **Preprocessing**: Modality-specific windowing and normalization
- **Target Sizes**: 224×224 (2.5D) and 384×384 (32-channel)
- **Augmentation**: Spatial transforms, intensity variations

## 📁 Repository Structure

```
rsna-intracranial-aneurysm-detection/
├── data_processing.py                           # Data preprocessing pipeline
├── coordinate_utils.py                          # Coordinate processing utilities
├── processed_data/
│   ├── 2_5d_volumes_full/                     # Processed 2.5D volumes
│   └── coordinates/                           # Coordinate cache files
└── docs/
    └── data_processing.md                     # This comprehensive guide
```

## ⚠️ Important Notes

- **Single Modality Per Patient**: Each patient has only ONE imaging series from ONE modality - no multi-modal fusion possible
- **Left/Right Distinction**: Critical for anatomical accuracy (no horizontal flipping in TTA)
- **Modality-Specific Processing**: Different windowing for CT vs MR
- **Memory Management**: Large medical volumes require careful memory handling
- **Clinical Context**: Understanding vascular anatomy essential for model design

---

*This guide provides comprehensive information for processing the RSNA 2025 Intracranial Aneurysm Detection dataset, covering all technical aspects from DICOM handling to modality-specific preprocessing.*
