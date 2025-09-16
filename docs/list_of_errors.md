# List of Errors and Solutions

## Error: Incorrect Out-of-Fold (OOF) Predictions Logic

### **Date:** 2025-01-27

### **Error Description:**
The result analysis notebook implemented incorrect out-of-fold prediction logic that created data leakage:

- **Incorrect Logic:** Each sample was predicted by 4 models (from folds that "didn't train on this sample")
- **Problem:** These 4 models actually DID train on this sample in their respective folds
- **Result:** Data leakage - models were predicting on samples they had seen during training

### **Root Cause:**
Misunderstanding of how K-fold cross-validation works:
- In standard K-fold CV, each sample belongs to exactly one validation fold
- That sample should only be predicted by the model trained on the other 4 folds
- The sample should NOT be predicted by models from folds where it was in the training set

### **Solution:**
1. **Fixed `collect_oof_predictions`:** Each sample now gets exactly one prediction from the model where it was in the validation fold
2. **Updated `create_oof_ensemble`:** Simplified to return single prediction instead of ensemble
3. **Modified analysis methods:** Updated to work with single predictions instead of multiple
4. **Removed fold agreement analysis:** Not applicable with true OOF predictions

### **Key Changes:**
```python
# Before (incorrect):
for fold in range(NUM_FOLDS):
    if fold != sample_fold:  # This was wrong!
        model = self.fold_models[fold]
        prediction = self.predict_sample(model, sample_id)

# After (correct):
if sample_fold in self.fold_models:
    model = self.fold_models[sample_fold]  # Use validation fold model
    prediction = self.predict_sample(model, sample_id)
```

### **Impact:**
- Eliminated data leakage
- Ensured statistically valid results
- Maintained all analysis functionality
- Results are now truly out-of-fold and unbiased

### **Lesson Learned:**
Always verify that out-of-fold predictions come from models that never saw the sample during training. In K-fold CV, each sample has exactly one OOF prediction from the model where it was in the validation fold.

---

## Error: Inconsistent Modality-Specific Normalization Between Training and Inference

### **Date:** 2025-01-27

### **Error Description:**
The modality-specific normalization logic was not consistently applied between training and inference phases:

- **Training Phase:** Used modality-specific normalization (CTA: statistical normalization [0, 500] → [0, 255], MR: adaptive percentile normalization [p1, p99] → [0, 255])
- **Inference Phase:** May have used different or default normalization
- **Result:** Model performance degradation due to distribution mismatch between training and inference data

### **Root Cause:**
- Normalization logic was implemented in training pipeline but not properly replicated in inference code
- Different preprocessing pipelines between training and inference phases
- Missing modality detection and normalization consistency checks

### **Solution:**
1. **Standardize normalization pipeline:** Ensure identical normalization logic in both training and inference
2. **Create shared normalization functions:** Extract normalization logic into reusable utility functions
3. **Add modality detection:** Implement consistent modality detection across all phases
4. **Validation checks:** Add assertions to verify normalization consistency

### **Key Changes:**
```python
# Create shared normalization function
def normalize_by_modality(image, modality):
    if modality in ['CTA']:
        # CT: Statistical normalization [0, 500] → [0, 255]
        return statistical_normalize(image, min_val=0, max_val=500)
    else:  # MR modalities
        # MR: Adaptive percentile normalization [p1, p99] → [0, 255]
        return adaptive_percentile_normalize(image, percentiles=[1, 99])

# Use in both training and inference
normalized_image = normalize_by_modality(image, detected_modality)
```

### **Impact:**
- Ensures consistent data preprocessing across all phases
- Prevents model performance degradation due to distribution shifts
- Maintains clinical accuracy of modality-specific processing
- Improves model reliability and reproducibility

### **Lesson Learned:**
Always ensure preprocessing consistency between training and inference phases. Modality-specific normalization must be identical across all data processing pipelines to maintain model performance and clinical validity.
