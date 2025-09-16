# RSNA 2025 Intracranial Aneurysm Detection - Evaluation Metrics

This document describes the evaluation metrics used in the RSNA 2025 Intracranial Aneurysm Detection competition, including detailed explanations and numerical examples.

## Overview

The competition uses a **Weighted AUC (wAUC)** metric that combines:
1. **Aneurysm Present AUC** (50% weight) - Overall aneurysm detection performance
2. **Anatomical Location AUCs** (50% weight) - Specific location classification performance

## Metric Components

### 1. Aneurysm Present (AP)
- **Purpose**: Binary classification - "Does this patient have any aneurysm?"
- **Weight**: 50% of final score
- **Calculation**: Standard ROC AUC for binary classification

### 2. Anatomical Location Classes (13 locations)
- **Purpose**: Multilabel classification - "Which specific locations have aneurysms?"
- **Weight**: 50% of final score (averaged across all 13 locations)
- **Classes**:
  1. Left Infraclinoid Internal Carotid Artery
  2. Right Infraclinoid Internal Carotid Artery
  3. Left Supraclinoid Internal Carotid Artery
  4. Right Supraclinoid Internal Carotid Artery
  5. Left Middle Cerebral Artery
  6. Right Middle Cerebral Artery
  7. Anterior Communicating Artery
  8. Left Anterior Cerebral Artery
  9. Right Anterior Cerebral Artery
  10. Left Posterior Communicating Artery
  11. Right Posterior Communicating Artery
  12. Basilar Tip
  13. Other Posterior Circulation

## Weighted AUC Formula

```
wAUC = 0.5 × AP_AUC + 0.5 × Others_Mean_AUC
```

Where:
- `AP_AUC` = ROC AUC for "Aneurysm Present" class
- `Others_Mean_AUC` = Average ROC AUC across all 13 anatomical location classes

## Detailed Calculation Process

### Step 1: Individual Class AUCs
For each of the 14 classes, compute ROC AUC:
```python
for i, class_name in enumerate(class_names):
    yi = y_true[:, i]  # True labels for class i
    pi = y_prob[:, i]  # Predicted probabilities for class i
    
    if len(np.unique(yi)) >= 2:  # Skip if only one class present
        aucs[class_name] = roc_auc_score(yi, pi)
    else:
        skipped.append(class_name)  # Skip classes with insufficient data
```

### Step 2: Separate AP and Location AUCs
```python
ap_auc = aucs.get("Aneurysm Present", np.nan)
others = [v for k, v in aucs.items() if k != "Aneurysm Present"]
others_mean = np.mean(others) if others else np.nan
```

### Step 3: Compute Weighted AUC
```python
weighted_auc = 0.5 * (ap_auc + others_mean)
```

## Numerical Examples

### Example 1: Perfect Predictions

**Input Data:**
```python
# True labels (3 patients, 14 classes)
y_true = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Patient 1: No aneurysm
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Patient 2: Left Supraclinoid ICA + AP
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]   # Patient 3: Left MCA + AP
])

# Predicted probabilities (perfect predictions)
y_prob = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
])
```

**Calculation:**
```python
# Individual AUCs (all perfect = 1.0)
per_class_aucs = {
    'Left Infraclinoid Internal Carotid Artery': 1.0,
    'Right Infraclinoid Internal Carotid Artery': 1.0,
    'Left Supraclinoid Internal Carotid Artery': 1.0,
    'Right Supraclinoid Internal Carotid Artery': 1.0,
    'Left Middle Cerebral Artery': 1.0,
    'Right Middle Cerebral Artery': 1.0,
    'Anterior Communicating Artery': 1.0,
    'Left Anterior Cerebral Artery': 1.0,
    'Right Anterior Cerebral Artery': 1.0,
    'Left Posterior Communicating Artery': 1.0,
    'Right Posterior Communicating Artery': 1.0,
    'Basilar Tip': 1.0,
    'Other Posterior Circulation': 1.0,
    'Aneurysm Present': 1.0
}

# Separate components
ap_auc = 1.0
others = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
others_mean = np.mean(others) = 1.0

# Final weighted AUC
weighted_auc = 0.5 × 1.0 + 0.5 × 1.0 = 1.0
```

**Result: wAUC = 1.0 (Perfect Score)**

### Example 2: Realistic Predictions

**Input Data:**
```python
# True labels (4 patients)
y_true = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Patient 1: No aneurysm
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Patient 2: Left Supraclinoid ICA
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Patient 3: Left MCA
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]   # Patient 4: Anterior Communicating
])

# Predicted probabilities (realistic model predictions)
y_prob = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.9],
    [0.2, 0.2, 0.2, 0.2, 0.7, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.7]
])
```

**Calculation:**
```python
# Individual AUCs (simplified for demonstration)
per_class_aucs = {
    'Left Infraclinoid Internal Carotid Artery': 0.5,    # Random
    'Right Infraclinoid Internal Carotid Artery': 0.5,   # Random
    'Left Supraclinoid Internal Carotid Artery': 1.0,    # Perfect
    'Right Supraclinoid Internal Carotid Artery': 0.5,   # Random
    'Left Middle Cerebral Artery': 1.0,                  # Perfect
    'Right Middle Cerebral Artery': 0.5,                 # Random
    'Anterior Communicating Artery': 1.0,                # Perfect
    'Left Anterior Cerebral Artery': 0.5,                # Random
    'Right Anterior Cerebral Artery': 0.5,               # Random
    'Left Posterior Communicating Artery': 0.5,          # Random
    'Right Posterior Communicating Artery': 0.5,         # Random
    'Basilar Tip': 0.5,                                  # Random
    'Other Posterior Circulation': 0.5,                  # Random
    'Aneurysm Present': 0.9                              # Good
}

# Separate components
ap_auc = 0.9
others = [0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
others_mean = np.mean(others) = 0.577

# Final weighted AUC
weighted_auc = 0.5 × 0.9 + 0.5 × 0.577 = 0.45 + 0.2885 = 0.7385
```

**Result: wAUC = 0.7385**

### Example 3: Handling Edge Cases

**Scenario**: Some classes have insufficient data (only one class present)

```python
# True labels with some classes having only one label
y_true = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Patient 1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Patient 2
])

y_prob = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.9]
])
```

**Calculation:**
```python
# Classes with insufficient data are skipped
skipped_classes = [
    'Right Infraclinoid Internal Carotid Artery',  # Only 0s
    'Right Supraclinoid Internal Carotid Artery',  # Only 0s
    'Left Middle Cerebral Artery',                 # Only 0s
    # ... (most classes skipped due to insufficient data)
]

# Only classes with both 0s and 1s get AUC calculated
per_class_aucs = {
    'Left Supraclinoid Internal Carotid Artery': 1.0,  # Has both 0 and 1
    'Aneurysm Present': 1.0                            # Has both 0 and 1
}

# Separate components
ap_auc = 1.0
others = [1.0]  # Only one class with sufficient data
others_mean = 1.0

# Final weighted AUC
weighted_auc = 0.5 × 1.0 + 0.5 × 1.0 = 1.0
```

**Result: wAUC = 1.0 (Only classes with sufficient data are evaluated)**

## Implementation Details

### Robust Handling
The implementation includes several robust features:

1. **Shape Alignment**: Handles mismatched dimensions between predictions and targets
2. **Insufficient Data**: Skips classes with only one unique label
3. **Constant Predictions**: Skips classes where all predictions are identical
4. **Missing Components**: Handles cases where either AP or location AUCs are missing

### Error Handling
```python
# Skip classes with insufficient data
if len(np.unique(yi)) < 2:
    skipped.append(name)
    continue

# Skip classes with constant predictions
if np.allclose(pi, pi[0]):
    skipped.append(name)
    continue

# Handle missing components
if np.isnan(ap_auc) and np.isnan(others_mean):
    weighted_auc = np.nan
elif np.isnan(ap_auc):
    weighted_auc = others_mean
elif np.isnan(others_mean):
    weighted_auc = ap_auc
else:
    weighted_auc = 0.5 * (ap_auc + others_mean)
```

## Interpretation Guidelines

### Score Ranges
- **0.5**: Random guessing (worst possible)
- **0.6-0.7**: Poor performance
- **0.7-0.8**: Moderate performance
- **0.8-0.9**: Good performance
- **0.9-1.0**: Excellent performance
- **1.0**: Perfect performance

### Competition Context
- **Leaderboard scores** typically range from 0.65 to 0.85
- **Top performers** achieve wAUC > 0.80
- **Baseline models** usually score around 0.70

### Model Optimization
Focus on improving both components:
1. **Aneurysm Present**: Overall detection capability
2. **Location Classification**: Specific anatomical accuracy

The 50-50 weighting ensures balanced optimization of both global and local aneurysm detection capabilities.

## References

- [RSNA 2025 Competition Page](https://www.kaggle.com/competitions/rsna-2025-intracranial-aneurysm-detection)
- [Competition Evaluation Details](https://www.kaggle.com/competitions/rsna-2025-intracranial-aneurysm-detection/overview/evaluation)
- [ROC AUC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
