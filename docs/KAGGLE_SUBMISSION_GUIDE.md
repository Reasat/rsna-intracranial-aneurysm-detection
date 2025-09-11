# Kaggle Submission Guide for RSNA 2025

This guide explains the streamlined workflow for submitting your trained models to the RSNA 2025 Intracranial Aneurysm Detection competition on Kaggle.

## Prerequisites

1. **Kaggle CLI installed and configured**:
   ```bash
   pip install kaggle
   ```

2. **API credentials configured**:
   - Download `kaggle.json` from your Kaggle account settings
   - Place it at `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Environment variables configured**:
   Create a `.env` file in the project root:
   ```bash
   KAGGLE_USERNAME=your-kaggle-username
   ```

## 🚀 **Quick Start: Two-Step Submission Process**

### **Step 1: Upload Models as Dataset**

Upload your trained models as a Kaggle dataset:

```bash
python upload_models_to_kaggle.py --model_dir models/2025-09-10-08-33-25
```

**Result**: Creates dataset `2025-09-10-08-33-25` containing all `.pth` files and configs.

### **Step 2: Upload Inference Notebook**

Upload a notebook with the same timestamp as your model:

```bash
python upload_notebook_to_kaggle.py --dataset_name 2025-09-10-08-33-25 --message "EfficientNet-B0 ensemble, 5-fold CV"
```

**Result**: Creates notebook `rsna-2025-09-10-08-33-25` that references your model dataset.

### **Step 3: Manual Submission (Web Interface)**

1. **Visit the generated notebook URL** (displayed in script output)
2. **Click "Save & Run All"** to execute the inference
3. **Click "Submit to Competition"** button in the notebook interface
4. **Add submission message** describing your approach

## ✅ **Timestamp Consistency**

The workflow maintains perfect timestamp consistency:
- **Model directory**: `models/2025-09-10-08-33-25/`
- **Dataset name**: `2025-09-10-08-33-25`
- **Notebook name**: `rsna-2025-09-10-08-33-25`

This ensures perfect traceability: Model ↔ Dataset ↔ Notebook all share the same timestamp.

## 📋 **Script Options**

### Model Upload Options
```bash
# Basic upload with auto-generated dataset name
python upload_models_to_kaggle.py --model_dir models/2025-09-10-08-33-25

# Custom dataset name
python upload_models_to_kaggle.py --model_dir models/2025-09-10-08-33-25 --title "custom-name"

# Dry run (prepare but don't upload)
python upload_models_to_kaggle.py --model_dir models/2025-09-10-08-33-25 --dry_run
```

### Notebook Upload Options
```bash
# Basic upload
python upload_notebook_to_kaggle.py --dataset_name 2025-09-10-08-33-25

# With custom message
python upload_notebook_to_kaggle.py --dataset_name 2025-09-10-08-33-25 --message "My submission description"

# Custom base notebook
python upload_notebook_to_kaggle.py --dataset_name 2025-09-10-08-33-25 --base_notebook my_custom_notebook.ipynb
```

## 📁 **File Structure**

After running the workflow:

```
rsna-intracranial-aneurysm-detection/
├── models/
│   └── 2025-09-10-08-33-25/          # Your training timestamp
│       ├── *.pth                      # Model files (cleaned)
│       └── used_config.yaml           # Config
├── kaggle_inference.ipynb             # Base notebook template
├── upload_models_to_kaggle.py         # Model upload script
├── upload_notebook_to_kaggle.py       # Notebook upload script
└── .env                               # Environment variables
```

---

## 📚 **Appendix: Detailed Information**

### **Why Consistent Timestamping?**

Each submission uses the **same timestamp as the model** because:
- ✅ **Perfect traceability**: Model ↔ Dataset ↔ Notebook all share same timestamp
- ✅ **No confusion**: Easy to identify which notebook uses which model
- ✅ **Version consistency**: Training and inference are clearly linked
- ✅ **Clean organization**: Timestamp-based naming throughout the workflow
- ✅ **Kaggle requirement**: Code competitions need separate notebook submissions

### **Configuration Files**

Both scripts automatically handle:
- **Model files**: All `.pth` files in the model directory
- **Config files**: `used_config.yaml`, `experiment_info.yaml`, `config.yaml`
- **Metadata**: Auto-generated dataset and notebook metadata
- **README**: Includes usage instructions and file descriptions

### **Competition Submission Flow**

```
Train Models → Upload Dataset → Upload Notebook → Execute on Kaggle → Submit to Competition
     ↓              ↓              ↓                    ↓                 ↓
 train.py    upload_models.py  upload_notebook.py   "Save & Run All"   Web Interface
```

### **Workflow Benefits**

#### Separation of Concerns
- **Model upload**: Focuses only on dataset creation
- **Notebook upload**: Handles timestamped submission notebooks
- **Clean interface**: Simple command-line options for each step

#### Flexibility
- **Reuse datasets**: Upload models once, create multiple notebook submissions
- **Version control**: Each submission maintains timestamp consistency with its model
- **Experimentation**: Easy to test different inference approaches with same models

#### Automation
- **Smart naming**: Automatic timestamp consistency across model directories, datasets, and notebooks
- **Dependency handling**: Scripts automatically update notebook references
- **Error handling**: Clear feedback for common issues

### **Timestamp Consistency Benefits**

The updated workflow ensures perfect timestamp consistency:

#### **Before (Inconsistent)**
- Model: `models/2025-09-10-08-33-25/`
- Dataset: `2025-09-10-08-33-25`
- Notebook: `rsna-aneurysm-inference-2025-01-15-14-22-30` ❌ (different timestamp)

#### **After (Consistent)**
- Model: `models/2025-09-10-08-33-25/`
- Dataset: `2025-09-10-08-33-25`
- Notebook: `rsna-2025-09-10-08-33-25` ✅ (same timestamp)

#### **Key Improvements**
- **Perfect traceability**: Model ↔ Dataset ↔ Notebook all share the same timestamp
- **No confusion**: Easy to identify which notebook uses which model
- **Clean organization**: Timestamp-based naming throughout the entire workflow
- **Version control**: Training and inference are clearly linked

---

## 🔧 **Appendix: Troubleshooting & Error Fixes**

### **Authentication Issues**
```bash
# Test Kaggle CLI authentication
kaggle competitions list

# Check credentials file
ls -la ~/.kaggle/kaggle.json
```

### **Environment Setup**
```bash
# Verify environment variables
cat .env

# Should contain:
# KAGGLE_USERNAME=your-actual-username
```

### **Dataset Issues**
- **"Category already exists"**: Usually a warning, upload often succeeds
- **No model files found**: Ensure `.pth` files exist in model directory
- **Permission denied**: Check `~/.kaggle/kaggle.json` permissions (`chmod 600`)

### **Notebook Issues**
- **Dataset not found**: Verify dataset uploaded successfully on Kaggle website
- **Metadata errors**: Check that `KAGGLE_USERNAME` matches your actual Kaggle username

### **Tips for Success**

1. **Clean model directories first**: Remove unnecessary checkpoint files to reduce upload time
2. **Use descriptive messages**: Help track different experimental approaches
3. **Monitor dataset uploads**: Check Kaggle website to confirm successful dataset creation
4. **Test with dry runs**: Use `--dry_run` to verify files before uploading
5. **Keep base notebook updated**: Ensure `kaggle_inference.ipynb` has your latest inference logic

### **Code Competition Workflow Understanding**

Kaggle code competitions work differently than file competitions:

1. **Push/Commit**: Notebook runs on **small test set** first
2. **Validation**: Code must complete successfully on small test
3. **Submission**: Web interface submission automatically uses **full test set**
4. **Scoring**: Full test set results determine final score

**CLI submission** (`kaggle competitions submit`) may not work for all code competitions - **web interface submission is the standard method**.

### **Common Issues & Solutions**

#### **"Failed to load any checkpoints"**
- **Cause**: Model discovery function can't find files
- **Check**: Verify dataset path `/kaggle/input/your-dataset-name/`
- **Verify**: Model files exist with correct naming pattern

#### **"Missing key(s) in state_dict"**
- **Cause**: Model architecture mismatch between training and inference
- **Solution**: Add layer name mapping in model loading function
- **Example**: `classifier.` → `fc.` mapping

#### **"No kernel name found"**
- **Cause**: Missing kernelspec metadata in notebook
- **Solution**: Add kernelspec to notebook metadata before upload

#### **CLI Submission 400 Errors**
- **Cause**: Code competitions may not support CLI submission
- **Solution**: Use web interface submission instead

### **Critical Fixes Applied**

#### **Layer Name Mismatch Fix**
The main issue was a model architecture mismatch:
- **Problem**: Models saved with `classifier.1.weight` but notebook expected `fc.1.weight`
- **Solution**: Added layer name mapping in `load_hybrid_model()`:
```python
# Fix layer name mismatch: classifier -> fc
if isinstance(state, dict):
    state = {k.replace('classifier.', 'fc.') if k.startswith('classifier.') else k: v for k, v in state.items()}
```

#### **Kernelspec Metadata Fix**
- **Problem**: Missing `kernelspec` metadata caused "No kernel name found" error
- **Solution**: Added proper kernelspec to notebook metadata:
```python
nb['metadata']['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python', 
    'name': 'python3'
}
```

This workflow eliminates the complexity of the previous all-in-one script while providing better control and traceability for your submissions.