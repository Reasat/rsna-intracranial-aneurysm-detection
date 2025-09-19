# %% [markdown]
# # RSNA 2025 Intracranial Aneurysm Detection - Binary Result Analysis
# 
# This notebook performs comprehensive cross-fold analysis of misclassifications from the 5-fold CV training using binary models with a modular engine-based architecture.
# 
# ## Architecture
# - **InferenceEngine**: Handles data loading, binary model loading, and prediction collection
# - **AnalysisEngine**: Performs binary misclassification analysis and hard sample identification
# - **VisualizationEngine**: Creates comprehensive visualizations and plots for binary classification
# 
# ## Analysis Framework
# - **Out-of-Fold (OOF) Predictions**: Uses true OOF predictions for each sample (no data leakage)
# - **Binary Classification Analysis**: Focuses on aneurysm present/absent classification
# - **Hard Sample Identification**: Identifies challenging cases for binary classification
# 
# ## Key Features
# - Modular, separated engine architecture
# - Comprehensive binary misclassification analysis
# - Per-modality error breakdown
# - Hard sample case studies
# - Interactive visualizations
# - Actionable insights for binary model improvement
# 

# %%
# Import required libraries
import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from tqdm import tqdm
import cv2

# Add project root to path
sys.path.append('..')
from model import BinaryAneurysmModel, create_binary_model
from config import Config
from utils import ID_COL, load_cached_volume, take_window, valid_coords
from analysis import InferenceEngine, AnalysisEngine, VisualizationEngine

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# %% [markdown]
# ## Configuration and Setup
# 

# %%
# Configuration for binary models
EXPERIMENT_DIR = "models/2025-09-15-05-27-19"  # Binary model experiment directory
NUM_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load experiment configuration
with open(f"{EXPERIMENT_DIR}/used_config.yaml", 'r') as f:
    import yaml
    config_dict = yaml.safe_load(f)

# Create Config object
path = f"configs/train_config_binary.yaml"
config = Config(path) # load binary config

# Override with experiment config
config.architecture = config_dict['model']['architecture']
config.img_size = config_dict['data']['img_size']
config.window_offsets = config_dict['data']['window_offsets']
config.roi_box_fraction = config_dict['data']['roi_box_fraction']
config.cache_dir = config_dict['paths']['cache_dir']
config.device = DEVICE

# CRITICAL: Override num_classes for binary classification
config.num_classes = 1  # Binary classification (aneurysm present/absent)

print(f"Binary Experiment: {EXPERIMENT_DIR}")
print(f"Architecture: {config.architecture}")
print(f"Image size: {config.img_size}")
print(f"Num classes: {config.num_classes}")
print(f"Device: {DEVICE}")


# %% [markdown]
# ## 🔧 Aggregation Method Configuration
# 
# The inference engine now supports two aggregation methods for combining predictions across all slices in a volume:
# 
# ### **MAX Aggregation (Default)**
# - **Method**: Takes the maximum prediction across all slices
# - **Use Case**: High sensitivity - captures any slice with high aneurysm probability
# - **Best For**: Detection tasks where missing an aneurysm is costly
# - **Example**: For predictions [0.1, 0.3, 0.7, 0.2, 0.9] → Result: 0.9
# 
# ### **MEAN Aggregation**
# - **Method**: Takes the average prediction across all slices
# - **Use Case**: Balanced approach - considers overall volume characteristics
# - **Best For**: Classification tasks where overall volume characteristics matter
# - **Example**: For predictions [0.1, 0.3, 0.7, 0.2, 0.9] → Result: 0.44
# 
# ### **Configuration**
# Change the `aggregation_method` variable in the next cell to switch between methods:
# - `aggregation_method = "max"` for maximum aggregation
# - `aggregation_method = "mean"` for mean aggregation
# 

# %% [markdown]
# ## 🔧 Binary Inference Engine Execution
# 
# The following section handles data loading, binary model loading, and prediction collection using the InferenceEngine.
# 

# %% [markdown]
# ## Main Execution
# 

# %%
# Create a custom InferenceEngine for binary models
class BinaryInferenceEngine(InferenceEngine):
    """Custom InferenceEngine for binary models"""
    
    def __init__(self, experiment_dir: str, config: Config, aggregation: str = "max"):
        super().__init__(experiment_dir, config, aggregation)
        self.num_classes = 1  # Binary classification
    
    def load_fold_models(self):
        """Load all fold models for binary classification"""
        for fold in range(5):
            model_path = f"{self.experiment_dir}/tf_efficientnet_b0_fold{fold}_best.pth"
            if os.path.exists(model_path):
                # Create binary model instead of hybrid model
                model = create_binary_model(self.config)
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.fold_models[fold] = model
                print(f"Loaded binary fold {fold} model")
            else:
                print(f"Warning: Binary model not found for fold {fold}")

# Create a custom AnalysisEngine for binary models
class BinaryAnalysisEngine(AnalysisEngine):
    """Custom AnalysisEngine for binary models"""
    
    def __init__(self, inference_engine: BinaryInferenceEngine):
        super().__init__(inference_engine)
        self.num_classes = 1  # Binary classification
    
    def analyze_binary_misclassifications(self, true_labels_df: pd.DataFrame):
        """Analyze binary misclassifications for aneurysm present/absent classification"""
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        class_name = 'Aneurysm Present'
        class_analysis = {
            'class_name': class_name,
            'total_samples': 0,
            'positive_samples': 0,
            'misclassified_samples': [],
            'false_positives': [],
            'false_negatives': [],
            'true_positives': [],
            'true_negatives': [],
            'confidence_distribution': {'correct': [], 'incorrect': []}
        }
        
        # Collect predictions and true labels
        predictions = []
        true_labels = []
        
        for sample_id in self.inference_engine.oof_predictions.keys():
            if sample_id not in true_labels_df[ID_COL].values:
                continue
                
            true_label = true_labels_df[true_labels_df[ID_COL] == sample_id][class_name].iloc[0]
            oof_pred = self.inference_engine.oof_predictions[sample_id]  # Binary prediction [1]
            
            # Binary prediction is a 1-element array, so use index 0
            pred_prob = oof_pred[0] if len(oof_pred) > 0 else 0.0
            pred_binary = 1 if pred_prob >= 0.5 else 0
            
            class_analysis['total_samples'] += 1
            if true_label == 1:
                class_analysis['positive_samples'] += 1
            
            # Categorize predictions
            if true_label == 1 and pred_binary == 1:
                class_analysis['true_positives'].append({
                    'sample_id': sample_id,
                    'true_label': true_label,
                    'prediction': pred_prob,
                    'fold_predictions': [pred_prob]
                })
            elif true_label == 0 and pred_binary == 0:
                class_analysis['true_negatives'].append({
                    'sample_id': sample_id,
                    'true_label': true_label,
                    'prediction': pred_prob,
                    'fold_predictions': [pred_prob]
                })
            elif true_label == 1 and pred_binary == 0:
                class_analysis['false_negatives'].append({
                    'sample_id': sample_id,
                    'true_label': true_label,
                    'prediction': pred_prob,
                    'fold_predictions': [pred_prob]
                })
            elif true_label == 0 and pred_binary == 1:
                class_analysis['false_positives'].append({
                    'sample_id': sample_id,
                    'true_label': true_label,
                    'prediction': pred_prob,
                    'fold_predictions': [pred_prob]
                })
            
            # Track confidence distribution
            if (true_label == 1 and pred_binary == 1) or (true_label == 0 and pred_binary == 0):
                class_analysis['confidence_distribution']['correct'].append(pred_prob)
            else:
                class_analysis['confidence_distribution']['incorrect'].append(pred_prob)
            
            predictions.append(pred_prob)
            true_labels.append(true_label)
        
        # Calculate metrics
        if len(predictions) > 0 and len(set(true_labels)) > 1:
            class_analysis['auc'] = roc_auc_score(true_labels, predictions)
            
            # Calculate precision, recall, F1 with binary predictions
            binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
            class_analysis['precision'] = precision_score(true_labels, binary_preds, zero_division=0)
            class_analysis['recall'] = recall_score(true_labels, binary_preds, zero_division=0)
            class_analysis['f1_score'] = f1_score(true_labels, binary_preds, zero_division=0)
        else:
            class_analysis['auc'] = 0.0
            class_analysis['precision'] = 0.0
            class_analysis['recall'] = 0.0
            class_analysis['f1_score'] = 0.0
        
        return {'Aneurysm Present': class_analysis}
    
    def identify_binary_hard_samples(self, true_labels_df: pd.DataFrame):
        """Identify hard samples for binary classification"""
        hard_samples = {
            'high_confidence_errors': [],
            'low_confidence_correct': [],
            'borderline_predictions': [],
            'modality_specific_errors': []
        }
        
        for sample_id in self.inference_engine.oof_predictions.keys():
            if sample_id not in true_labels_df[ID_COL].values:
                continue
                
            # Get true label for "Aneurysm Present"
            true_label = true_labels_df[true_labels_df[ID_COL] == sample_id]['Aneurysm Present'].iloc[0]
            oof_pred = self.inference_engine.oof_predictions[sample_id]  # Binary prediction [1]
            
            # Binary prediction is a 1-element array, so use index 0
            pred_prob = oof_pred[0] if len(oof_pred) > 0 else 0.0
            pred_binary = 1 if pred_prob >= 0.5 else 0
            
            # Determine if prediction is correct
            is_correct = (true_label == 1) == (pred_binary == 1)
            
            # Categorize hard samples
            if not is_correct:
                # High confidence errors (wrong prediction with high confidence)
                if pred_prob > 0.8 or pred_prob < 0.2:
                    hard_samples['high_confidence_errors'].append({
                        'sample_id': sample_id,
                        'true_label': true_label,
                        'prediction': pred_prob,
                        'confidence': max(pred_prob, 1 - pred_prob)
                    })
            else:
                # Low confidence correct predictions
                if 0.4 <= pred_prob <= 0.6:
                    hard_samples['low_confidence_correct'].append({
                        'sample_id': sample_id,
                        'true_label': true_label,
                        'prediction': pred_prob,
                        'confidence': min(pred_prob, 1 - pred_prob)
                    })
            
            # Borderline predictions (close to threshold)
            if 0.3 <= pred_prob <= 0.7:
                hard_samples['borderline_predictions'].append({
                    'sample_id': sample_id,
                    'true_label': true_label,
                    'prediction': pred_prob,
                    'is_correct': is_correct
                })
        
        return hard_samples
    
    def analyze_modality_binary_performance(self, true_labels_df: pd.DataFrame, sample_modalities: dict):
        """Analyze binary model performance across different modalities"""
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        modality_results = {}
        modalities = list(set(sample_modalities.values()))
        
        for modality in modalities:
            modality_analysis = {
                'modality': modality,
                'total_samples': 0,
                'aneurysm_present_samples': 0,
                'per_class_analysis': {},
                'overall_metrics': {},
                'confusion_matrix': {'true_positives': 0, 'true_negatives': 0, 'false_positives': 0, 'false_negatives': 0}
            }
            
            # Filter samples by modality
            modality_samples = [sid for sid, mod in sample_modalities.items() 
                              if mod == modality and sid in self.inference_engine.oof_predictions]
            
            modality_analysis['total_samples'] = len(modality_samples)
            
            # Analyze "Aneurysm Present" class for this modality
            class_name = 'Aneurysm Present'
            class_analysis = {
                'class_name': class_name,
                'total_samples': 0,
                'positive_samples': 0,
                'predictions': [],
                'true_labels': [],
                'false_positives': [],
                'false_negatives': [],
                'true_positives': [],
                'true_negatives': []
            }
            
            for sample_id in modality_samples:
                if sample_id not in true_labels_df[ID_COL].values:
                    continue
                
                true_label = true_labels_df[true_labels_df[ID_COL] == sample_id][class_name].iloc[0]
                oof_pred = self.inference_engine.oof_predictions[sample_id]  # Binary prediction [1]
                
                # Binary prediction is a 1-element array, so use index 0
                pred_score = oof_pred[0] if len(oof_pred) > 0 else 0.0
                pred_binary = 1 if pred_score >= 0.5 else 0
                
                class_analysis['total_samples'] += 1
                class_analysis['predictions'].append(pred_score)
                class_analysis['true_labels'].append(true_label)
                
                if true_label == 1:
                    class_analysis['positive_samples'] += 1
                    modality_analysis['aneurysm_present_samples'] += 1
                    
                    if pred_binary == 1:
                        class_analysis['true_positives'].append(sample_id)
                        modality_analysis['confusion_matrix']['true_positives'] += 1
                    else:
                        class_analysis['false_negatives'].append(sample_id)
                        modality_analysis['confusion_matrix']['false_negatives'] += 1
                else:
                    if pred_binary == 1:
                        class_analysis['false_positives'].append(sample_id)
                        modality_analysis['confusion_matrix']['false_positives'] += 1
                    else:
                        class_analysis['true_negatives'].append(sample_id)
                        modality_analysis['confusion_matrix']['true_negatives'] += 1
            
            # Calculate metrics for this modality
            if len(class_analysis['predictions']) > 0 and len(set(class_analysis['true_labels'])) > 1:
                class_analysis['auc'] = roc_auc_score(class_analysis['true_labels'], class_analysis['predictions'])
                
                binary_preds = [1 if p >= 0.5 else 0 for p in class_analysis['predictions']]
                class_analysis['precision'] = precision_score(class_analysis['true_labels'], binary_preds, zero_division=0)
                class_analysis['recall'] = recall_score(class_analysis['true_labels'], binary_preds, zero_division=0)
                class_analysis['f1_score'] = f1_score(class_analysis['true_labels'], binary_preds, zero_division=0)
                
                # Overall metrics for modality
                modality_analysis['overall_metrics'] = {
                    'auc': class_analysis['auc'],
                    'precision': class_analysis['precision'],
                    'recall': class_analysis['recall'],
                    'f1': class_analysis['f1_score']
                }
            else:
                class_analysis['auc'] = 0.0
                class_analysis['precision'] = 0.0
                class_analysis['recall'] = 0.0
                class_analysis['f1_score'] = 0.0
                
                modality_analysis['overall_metrics'] = {
                    'auc': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
            
            modality_analysis['per_class_analysis'][class_name] = class_analysis
            modality_results[modality] = modality_analysis
        
        return modality_results
    
    def identify_modality_binary_hard_samples(self, true_labels_df: pd.DataFrame, sample_modalities: dict):
        """Identify modality-specific hard samples for binary classification"""
        modality_hard_samples = {}
        modalities = list(set(sample_modalities.values()))
        
        for modality in modalities:
            modality_hard_samples[modality] = {
                'high_confidence_errors': [],
                'low_confidence_correct': [],
                'borderline_predictions': []
            }
            
            # Filter samples by modality
            modality_samples = [sid for sid, mod in sample_modalities.items() 
                              if mod == modality and sid in self.inference_engine.oof_predictions]
            
            for sample_id in modality_samples:
                if sample_id not in true_labels_df[ID_COL].values:
                    continue
                
                true_label = true_labels_df[true_labels_df[ID_COL] == sample_id]['Aneurysm Present'].iloc[0]
                oof_pred = self.inference_engine.oof_predictions[sample_id]  # Binary prediction [1]
                
                # Binary prediction is a 1-element array, so use index 0
                pred_prob = oof_pred[0] if len(oof_pred) > 0 else 0.0
                pred_binary = 1 if pred_prob >= 0.5 else 0
                
                # Determine if prediction is correct
                is_correct = (true_label == 1) == (pred_binary == 1)
                
                # Categorize hard samples
                if not is_correct:
                    # High confidence errors
                    if pred_prob > 0.8 or pred_prob < 0.2:
                        modality_hard_samples[modality]['high_confidence_errors'].append({
                            'sample_id': sample_id,
                            'true_label': true_label,
                            'prediction': pred_prob,
                            'confidence': max(pred_prob, 1 - pred_prob)
                        })
                else:
                    # Low confidence correct predictions
                    if 0.4 <= pred_prob <= 0.6:
                        modality_hard_samples[modality]['low_confidence_correct'].append({
                            'sample_id': sample_id,
                            'true_label': true_label,
                            'prediction': pred_prob,
                            'confidence': min(pred_prob, 1 - pred_prob)
                        })
                
                # Borderline predictions
                if 0.3 <= pred_prob <= 0.7:
                    modality_hard_samples[modality]['borderline_predictions'].append({
                        'sample_id': sample_id,
                        'true_label': true_label,
                        'prediction': pred_prob,
                        'is_correct': is_correct
                    })
        
        return modality_hard_samples

# Initialize engines (adapted for binary models)
# Choose aggregation method: "max" or "mean"
aggregation_method = "mean"  # Change to "mean" for mean aggregation
print(f"Using {aggregation_method} aggregation for inference")

inference_engine = BinaryInferenceEngine(EXPERIMENT_DIR, config, aggregation=aggregation_method)
analysis_engine = BinaryAnalysisEngine(inference_engine)
visualization_engine = VisualizationEngine(inference_engine)
# Override num_classes for binary classification
visualization_engine.num_classes = 1

# Load fold assignments
train_csv_path = config_dict['paths']['train_csv']
inference_engine.load_fold_assignments(train_csv_path)

# Load binary models
inference_engine.load_fold_models()

# Get sample IDs for analysis (use a subset for testing)
all_sample_ids = list(inference_engine.fold_assignments.keys())
print(f"Total samples available: {len(all_sample_ids)}")

# For testing, use a subset (remove this for full analysis)
test_sample_ids = np.random.RandomState(42).choice(all_sample_ids, size=1000, replace=False).tolist()
print(f"Using {len(test_sample_ids)} samples for binary analysis")


# %%
# Collect OOF predictions for binary classification
inference_engine.collect_oof_predictions(test_sample_ids)

print(f"Collected OOF predictions for {len(inference_engine.oof_predictions)} samples")
print(f"Average predictions per sample: {np.mean([len(preds) for preds in inference_engine.oof_predictions.values()]):.1f}")

# Load true labels for analysis
true_labels_df = pd.read_csv(train_csv_path)
print(f"Loaded true labels for {len(true_labels_df)} samples")


# %% [markdown]
# ## 📊 Binary Analysis Engine Execution
# 
# The following section uses the AnalysisEngine to perform binary misclassification analysis and identify hard samples.
# 

# %%
# Run comprehensive binary analysis
print("🔍 Starting Binary Cross-Fold Analysis...")

# 1. Binary misclassification analysis
print("📊 Analyzing binary misclassifications...")
binary_analysis = analysis_engine.analyze_binary_misclassifications(true_labels_df)

# 2. Hard sample identification for binary classification
print("🎯 Identifying binary hard samples...")
hard_samples = analysis_engine.identify_binary_hard_samples(true_labels_df)

print("✅ Binary analysis complete!")


# %%
# Print binary analysis results (focusing on "Aneurysm Present" class)
aneurysm_present_analysis = binary_analysis['Aneurysm Present']
print("Binary Analysis Results (Aneurysm Present class):")
print(f"Total samples: {aneurysm_present_analysis['total_samples']}")
print(f"Positive samples (aneurysm present): {aneurysm_present_analysis['positive_samples']}")
print(f"Negative samples (no aneurysm): {aneurysm_present_analysis['total_samples'] - aneurysm_present_analysis['positive_samples']}")
print(f"False Negatives: {len(aneurysm_present_analysis['false_negatives'])}")
print(f"False Positives: {len(aneurysm_present_analysis['false_positives'])}")
print(f"True Positives: {len(aneurysm_present_analysis['true_positives'])}")
print(f"True Negatives: {len(aneurysm_present_analysis['true_negatives'])}")
print(f"Overall AUC: {aneurysm_present_analysis['auc']:.3f}")
print(f"Precision: {aneurysm_present_analysis['precision']:.3f}")
print(f"Recall: {aneurysm_present_analysis['recall']:.3f}")
print(f"F1 Score: {aneurysm_present_analysis['f1_score']:.3f}")


# %% [markdown]
# ## 📈 Binary Visualization Engine Execution
# 
# The following section uses the VisualizationEngine to create comprehensive visualizations of the binary analysis results.
# 

# %%
# Generate binary visualizations (moved to 4-modality section below)
print("📈 Binary visualizations will be generated in the 4-modality analysis section...")


# %% [markdown]
# ## Results Summary
# 

# %%
# Print comprehensive binary results summary
print("=" * 80)
print("📋 BINARY CROSS-FOLD ANALYSIS RESULTS SUMMARY")
print("=" * 80)

# Overall statistics
total_samples = len(inference_engine.oof_predictions)
print(f"\n📊 Overall Statistics:")
print(f"  • Total samples analyzed: {total_samples}")
print(f"  • Average OOF predictions per sample: {np.mean([len(preds) for preds in inference_engine.oof_predictions.values()]):.1f}")

# Binary classification summary (focusing on Aneurysm Present class)
aneurysm_analysis = binary_analysis['Aneurysm Present']
print(f"\n🎯 Binary Classification Summary (Aneurysm Present):")
print(f"  • Total samples: {aneurysm_analysis['total_samples']}")
print(f"  • Positive samples (aneurysm present): {aneurysm_analysis['positive_samples']} ({aneurysm_analysis['positive_samples']/aneurysm_analysis['total_samples']*100:.1f}%)")
print(f"  • Negative samples (no aneurysm): {aneurysm_analysis['total_samples'] - aneurysm_analysis['positive_samples']} ({(aneurysm_analysis['total_samples'] - aneurysm_analysis['positive_samples'])/aneurysm_analysis['total_samples']*100:.1f}%)")
print(f"  • False Negatives: {len(aneurysm_analysis['false_negatives'])}")
print(f"  • False Positives: {len(aneurysm_analysis['false_positives'])}")
print(f"  • True Positives: {len(aneurysm_analysis['true_positives'])}")
print(f"  • True Negatives: {len(aneurysm_analysis['true_negatives'])}")

# Performance metrics
print(f"\n📈 Performance Metrics:")
print(f"  • Overall AUC: {aneurysm_analysis['auc']:.3f}")
print(f"  • Precision: {aneurysm_analysis['precision']:.3f}")
print(f"  • Recall: {aneurysm_analysis['recall']:.3f}")
print(f"  • F1 Score: {aneurysm_analysis['f1_score']:.3f}")

# Hard samples summary
print(f"\n🎯 Hard Sample Summary:")
for hard_type, samples in hard_samples.items():
    print(f"  • {hard_type}: {len(samples)} samples")

print("\n" + "=" * 80)
print("✅ Binary Analysis Complete! Check visualizations above for detailed insights.")
print("=" * 80)


# %% [markdown]
# 
# 

# %%



# %% [markdown]
# ## 🔬 4-Modality Binary Analysis
# 
# This section demonstrates the 4-modality analysis capability for binary classification:
# - **CTA** (CT Angiography)
# - **MRA** (MR Angiography) 
# - **MRI T2** (T2-weighted MRI)
# - **MRI T1post** (T1-weighted post-contrast MRI)
# 
# The analysis extracts modality information from CSV and performs comprehensive 4-way comparisons for binary classification.
# 

# %%
# Extract modality mapping for 4-modality binary analysis
from analysis import extract_modality_mapping

# Get sample IDs that have predictions
sample_ids = list(inference_engine.oof_predictions.keys())
print(f"Analyzing {len(sample_ids)} samples with binary predictions")

# Extract modality mapping from CSV
modality_mapping = extract_modality_mapping(sample_ids, train_csv_path)

# Show the 4-modality distribution
print(f"\n4-Modality Distribution:")
for modality, count in sorted(modality_mapping.items(), key=lambda x: x[1], reverse=True):
    print(f"  {modality}: {count} samples")


# %%
# Run 4-modality binary analysis
print("🔬 Starting 4-Modality Binary Analysis...")

# 1. Modality-specific binary performance analysis
print("📊 Analyzing binary performance across 4 modalities...")
modality_analysis = analysis_engine.analyze_modality_binary_performance(
    true_labels_df, 
    sample_modalities=modality_mapping
)

# 2. Modality-specific binary hard sample identification
print("🎯 Identifying modality-specific binary hard samples...")
modality_hard_samples = analysis_engine.identify_modality_binary_hard_samples(
    true_labels_df,
    sample_modalities=modality_mapping
)

print("✅ 4-Modality binary analysis complete!")


# %%
# Print 4-modality binary performance summary (focusing on Aneurysm Present class)
print("=" * 80)
print("📋 4-MODALITY BINARY ANALYSIS RESULTS SUMMARY")
print("=" * 80)

for modality, analysis in modality_analysis.items():
    total_samples = analysis['total_samples']
    aneurysm_samples = analysis['aneurysm_present_samples']
    
    # Get Aneurysm Present class analysis
    aneurysm_class_analysis = analysis['per_class_analysis']['Aneurysm Present']
    auc = aneurysm_class_analysis.get('auc', 0.0)
    precision = aneurysm_class_analysis.get('precision', 0.0)
    recall = aneurysm_class_analysis.get('recall', 0.0)
    f1 = aneurysm_class_analysis.get('f1_score', 0.0)
    
    print(f"\n🔬 {modality}:")
    print(f"  • Total samples: {total_samples}")
    print(f"  • Aneurysm present: {aneurysm_samples} ({aneurysm_samples/total_samples*100:.1f}%)")
    print(f"  • AUC: {auc:.3f}")
    print(f"  • Precision: {precision:.3f}")
    print(f"  • Recall: {recall:.3f}")
    print(f"  • F1 Score: {f1:.3f}")
    
    # Show confusion matrix summary for Aneurysm Present class
    tn = len(aneurysm_class_analysis['true_negatives'])
    fp = len(aneurysm_class_analysis['false_positives'])
    fn = len(aneurysm_class_analysis['false_negatives'])
    tp = len(aneurysm_class_analysis['true_positives'])
    print(f"  • Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\n" + "=" * 80)


# %%
# Plot per-modality AUCs
print("📊 Creating per-modality AUC plots...")

def plot_modality_aucs(modality_analysis):
    """Plot per-modality AUCs for binary classification"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract modality names and AUCs
    modalities = list(modality_analysis.keys())
    aucs = []
    
    for mod in modalities:
        aneurysm_class_analysis = modality_analysis[mod]['per_class_analysis']['Aneurysm Present']
        auc = aneurysm_class_analysis.get('auc', 0.0)
        aucs.append(auc)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    bars = plt.bar(modalities, aucs, alpha=0.7, color='lightcoral', edgecolor='darkred', linewidth=1.5)
    
    # Customize the plot
    plt.title('Per-Modality AUC Performance (Binary Classification)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Modality', fontsize=12, fontweight='bold')
    plt.ylabel('AUC Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add AUC labels on bars
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add horizontal line at 0.5 (random classifier)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Classifier')
    
    # Add legend
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 Per-Modality AUC Summary:")
    print(f"  • Best performing modality: {modalities[aucs.index(max(aucs))]} (AUC: {max(aucs):.3f})")
    print(f"  • Worst performing modality: {modalities[aucs.index(min(aucs))]} (AUC: {min(aucs):.3f})")
    print(f"  • Average AUC across modalities: {np.mean(aucs):.3f}")
    print(f"  • AUC standard deviation: {np.std(aucs):.3f}")

# Generate the plot
plot_modality_aucs(modality_analysis)


# %%
# Generate 4-modality binary visualizations
print("📈 Creating 4-modality binary visualizations...")
visualization_engine.create_visualizations(
    per_class_analysis=binary_analysis,  # Use the per-class analysis
    hard_samples=hard_samples,
    true_labels_df=true_labels_df,
    modality_analysis=modality_analysis,
    modality_hard_samples=modality_hard_samples,
    sample_modalities=modality_mapping
)


# %% [markdown]
# ## 🔍 Detailed Binary Analysis
# 
# This section provides detailed analysis of binary classification performance, including threshold analysis and error case studies.
# 

# %%
# Detailed binary analysis
print("🔍 Starting Detailed Binary Analysis...")

# Note: The current analysis engine doesn't have threshold_analysis and error_analysis methods
# These would need to be implemented in the analysis.py file for full binary analysis
print("📊 Threshold analysis and error case analysis methods not yet implemented in analysis engine")
print("🎯 Using available per-class analysis for binary classification insights")

print("✅ Detailed binary analysis complete!")


# %%
# Print detailed analysis results (using available data)
print("=" * 60)
print("📊 DETAILED BINARY ANALYSIS RESULTS")
print("=" * 60)

# Use the Aneurysm Present class analysis for detailed insights
aneurysm_analysis = binary_analysis['Aneurysm Present']
print(f"Binary Classification Performance:")
print(f"  • AUC: {aneurysm_analysis['auc']:.3f}")
print(f"  • Precision: {aneurysm_analysis['precision']:.3f}")
print(f"  • Recall: {aneurysm_analysis['recall']:.3f}")
print(f"  • F1 Score: {aneurysm_analysis['f1_score']:.3f}")

print("\n" + "=" * 60)
print("🎯 ERROR CASE ANALYSIS")
print("=" * 60)
print(f"False Negative cases: {len(aneurysm_analysis['false_negatives'])}")
print(f"False Positive cases: {len(aneurysm_analysis['false_positives'])}")
print(f"True Positive cases: {len(aneurysm_analysis['true_positives'])}")
print(f"True Negative cases: {len(aneurysm_analysis['true_negatives'])}")

print("\n" + "=" * 60)


# %% [markdown]
# ## 📋 Final Summary and Recommendations
# 
# This section provides a comprehensive summary of the binary analysis results and actionable recommendations for model improvement.
# 

# %%
# Final comprehensive summary
print("=" * 80)
print("📋 COMPREHENSIVE BINARY ANALYSIS SUMMARY")
print("=" * 80)

# Overall performance (using Aneurysm Present class)
aneurysm_analysis = binary_analysis['Aneurysm Present']
print("\n🎯 Overall Binary Performance (Aneurysm Present):")
print(f"  • AUC: {aneurysm_analysis['auc']:.3f}")
print(f"  • F1 Score: {aneurysm_analysis['f1_score']:.3f}")
print(f"  • Precision: {aneurysm_analysis['precision']:.3f}")
print(f"  • Recall: {aneurysm_analysis['recall']:.3f}")

# Modality performance ranking (by AUC for Aneurysm Present class)
print("\n🔬 Modality Performance Ranking (by AUC for Aneurysm Present):")
modality_aucs = []
for mod, analysis in modality_analysis.items():
    aneurysm_class_analysis = analysis['per_class_analysis']['Aneurysm Present']
    auc = aneurysm_class_analysis.get('auc', 0.0)
    modality_aucs.append((mod, auc))
modality_aucs.sort(key=lambda x: x[1], reverse=True)
for i, (modality, auc) in enumerate(modality_aucs, 1):
    print(f"  {i}. {modality}: {auc:.3f}")

# Error analysis summary
print("\n❌ Error Analysis Summary:")
print(f"  • False Negatives: {len(aneurysm_analysis['false_negatives'])} (missed aneurysms)")
print(f"  • False Positives: {len(aneurysm_analysis['false_positives'])} (false alarms)")
print(f"  • Error Rate: {(len(aneurysm_analysis['false_negatives']) + len(aneurysm_analysis['false_positives'])) / aneurysm_analysis['total_samples'] * 100:.1f}%")

# Recommendations
print("\n💡 Recommendations for Model Improvement:")
print("  1. Focus on reducing false negatives (missed aneurysms)")
print("  2. Consider modality-specific training strategies")
print("  3. Analyze hard samples for data augmentation opportunities")
print("  4. Optimize threshold based on clinical requirements")
print("  5. Consider ensemble approaches with 14-class models")

print("\n" + "=" * 80)
print("✅ Binary Analysis Complete!")
print("=" * 80)



