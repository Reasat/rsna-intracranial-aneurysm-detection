"""
Analysis and Visualization Engine for RSNA Intracranial Aneurysm Detection

This module provides comprehensive analysis and visualization capabilities for analyzing
model predictions and misclassifications from cross-fold validation.

Includes:
- InferenceEngine: Data loading, model loading, and prediction collection
- AnalysisEngine: Misclassification analysis and hard sample identification
- VisualizationEngine: Comprehensive plotting and visualization
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import cv2
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Import from utils to get LABEL_COLS and ID_COL
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils import LABEL_COLS, ID_COL, load_cached_volume, take_window
from train import HybridAneurysmModel, Config


def extract_modality_mapping(sample_ids: List[str], csv_path: str) -> Dict[str, str]:
    """
    Extract modality mapping for a list of sample IDs from CSV file
    
    Args:
        sample_ids: List of sample IDs to get modality info for
        csv_path: Path to the training CSV file
        
    Returns:
        Dictionary mapping sample_id to modality string
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Create modality mapping for the requested samples
    modality_mapping = {}
    for sample_id in sample_ids:
        if sample_id in df[ID_COL].values:
            modality = df[df[ID_COL] == sample_id]['Modality'].iloc[0]
            modality_mapping[sample_id] = modality
        else:
            print(f"Warning: Sample {sample_id} not found in CSV file")
    
    print(f"Extracted modality mapping for {len(modality_mapping)} samples")
    modality_counts = {}
    for modality in modality_mapping.values():
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    print(f"Modality distribution: {modality_counts}")
    
    return modality_mapping


class InferenceEngine:
    """Handles data loading, model loading, and prediction collection for cross-fold analysis"""
    
    def __init__(self, experiment_dir: str, config: Config):
        self.experiment_dir = experiment_dir
        self.config = config
        self.device = config.device
        self.num_classes = len(LABEL_COLS)
        
        # Data storage
        self.fold_assignments = {}
        self.oof_predictions = {}  # {sample_id: prediction_array}
        self.true_labels = {}      # {sample_id: true_label_vector}
        self.sample_modalities = {}  # {sample_id: modality}
        self.fold_models = {}
        
    def load_fold_assignments(self, train_csv_path: str):
        """Recreate fold assignments using the same StratifiedKFold logic as in train.py"""
        
        # Load the training data
        df = pd.read_csv(train_csv_path)
        
        # Filter by modalities if specified in config
        if hasattr(self.config, 'modalities') and self.config.modalities and len(self.config.modalities) > 0:
            df = df[df['Modality'].isin(self.config.modalities)].reset_index(drop=True)
            print(f"Filtered to modalities: {', '.join(self.config.modalities)}")
        
        # Recreate fold assignments using the same logic as train.py
        df['fold'] = -1
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (_, val_idx) in enumerate(skf.split(df, df['Aneurysm Present'])):
            df.loc[val_idx, 'fold'] = fold
        
        # Create fold assignments dictionary
        self.fold_assignments = dict(zip(df[ID_COL], df['fold']))
        # Store modality information for each sample
        self.sample_modalities = dict(zip(df[ID_COL], df['Modality']))
        print(f"Recreated fold assignments for {len(self.fold_assignments)} samples")
        print(f"Fold distribution: {df['fold'].value_counts().sort_index().to_dict()}")
        print(f"Modality distribution: {df['Modality'].value_counts().to_dict()}")
        
    def load_fold_models(self):
        """Load all fold models"""
        for fold in range(5):
            model_path = f"{self.experiment_dir}/tf_efficientnet_b0_fold{fold}_best.pth"
            if os.path.exists(model_path):
                model = HybridAneurysmModel(self.config)
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.fold_models[fold] = model
                print(f"Loaded fold {fold} model")
            else:
                print(f"Warning: Model not found for fold {fold}")
        
    def predict_sample(self, model, sample_id: str, batch_size: int = 128) -> np.ndarray:
        """Predict single sample using a model"""
        try:
            # Load cached volume
            volume_path = f"{self.config.cache_dir}/{sample_id}.npz"
            volume = load_cached_volume(volume_path)  # (N, H, W)
            
            # Prepare windows
            N = volume.shape[0]
            all_predictions = []
            
            # Process in batches
            for i in range(0, N, batch_size):
                batch_windows = []
                batch_coords = []
                
                for center_idx in range(i, min(i + batch_size, N)):
                    # Extract window
                    window = take_window(volume, center_idx, self.config.window_offsets)
                    
                    # Convert to HWC and resize
                    img_hwc = np.transpose(window, (1, 2, 0)).astype(np.float32)
                    img_resized = cv2.resize(img_hwc, (self.config.img_size, self.config.img_size))
                    
                    # Convert to CHW tensor
                    x_full = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).float()
                    x_roi = x_full.clone()  # Same for both streams (no coords available)
                    coords = torch.zeros(2).float()  # No coordinates available
                    
                    batch_windows.append((x_full, x_roi, coords))
                
                # Stack and predict
                if batch_windows:
                    x_full_batch = torch.stack([x[0] for x in batch_windows]).to(self.device)
                    x_roi_batch = torch.stack([x[1] for x in batch_windows]).to(self.device)
                    coords_batch = torch.stack([x[2] for x in batch_windows]).to(self.device)
                    
                    with torch.no_grad():
                        logits = model(x_full_batch, x_roi_batch, coords_batch)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        all_predictions.append(probs)
            
            if all_predictions:
                # Aggregate across windows (max aggregation)
                all_preds = np.vstack(all_predictions)
                series_pred = all_preds.max(axis=0)
                return series_pred
            else:
                return np.zeros(self.num_classes, dtype=np.float32)
                
        except Exception as e:
            print(f"Error predicting {sample_id}: {e}")
            return np.zeros(self.num_classes, dtype=np.float32)
    
    def collect_oof_predictions(self, sample_ids: List[str]):
        """Collect true out-of-fold predictions for all samples"""
        print(f"Collecting OOF predictions for {len(sample_ids)} samples...")
        
        for sample_id in tqdm(sample_ids):
            sample_fold = self.fold_assignments.get(sample_id, -1)
            if sample_fold == -1:
                continue
                
            # True OOF: Get prediction from the model where this sample was in validation fold
            # This model was trained on the other 4 folds and never saw this sample
            if sample_fold in self.fold_models:
                model = self.fold_models[sample_fold]
                prediction = self.predict_sample(model, sample_id)
                # Store as single prediction (not a list)
                self.oof_predictions[sample_id] = prediction
            else:
                print(f"Warning: Model not found for fold {sample_fold}")
    
    def create_oof_ensemble(self, sample_id: str, method: str = "mean") -> np.ndarray:
        """Get the single OOF prediction for a sample"""
        # Since we now use true OOF, each sample has exactly one prediction
        # The method parameter is kept for compatibility but not used
        return self.oof_predictions[sample_id]


class AnalysisEngine:
    """Handles analysis of predictions and identification of hard samples"""
    
    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine
        self.num_classes = len(LABEL_COLS)
    
    def analyze_per_class_misclassifications(self, true_labels_df: pd.DataFrame):
        """Detailed analysis of misclassifications per class"""
        results = {}
        
        for class_idx, class_name in enumerate(LABEL_COLS):
            class_analysis = {
                'class_name': class_name,
                'total_samples': 0,
                'positive_samples': 0,
                'misclassified_samples': [],
                'false_positives': [],
                'false_negatives': [],
                'confidence_distribution': {'correct': [], 'incorrect': []}
            }
            
            for sample_id in self.inference_engine.oof_predictions.keys():
                if sample_id not in true_labels_df[ID_COL].values:
                    continue
                    
                true_label = true_labels_df[true_labels_df[ID_COL] == sample_id][class_name].iloc[0]
                oof_pred = self.inference_engine.oof_predictions[sample_id]  # Single prediction now
                ensemble_pred = self.inference_engine.create_oof_ensemble(sample_id)
                
                class_analysis['total_samples'] += 1
                
                if true_label == 1:  # Positive sample
                    class_analysis['positive_samples'] += 1
                    
                    # Check if misclassified
                    if ensemble_pred[class_idx] < 0.5:
                        class_analysis['false_negatives'].append({
                            'sample_id': sample_id,
                            'true_label': true_label,
                            'prediction': ensemble_pred[class_idx],
                            'fold_predictions': [oof_pred[class_idx]]  # Single prediction
                        })
                
                else:  # Negative sample
                    if ensemble_pred[class_idx] >= 0.5:
                        class_analysis['false_positives'].append({
                            'sample_id': sample_id,
                            'true_label': true_label,
                            'prediction': ensemble_pred[class_idx],
                            'fold_predictions': [oof_pred[class_idx]]  # Single prediction
                        })
                
                # Track confidence distribution
                is_correct = (true_label == 1) == (ensemble_pred[class_idx] >= 0.5)
                class_analysis['confidence_distribution']['correct' if is_correct else 'incorrect'].append(
                    ensemble_pred[class_idx]
                )
            
            results[class_name] = class_analysis
        
        return results

    def identify_hard_samples(self, true_labels_df: pd.DataFrame):
        """Identify different types of hard samples"""
        hard_samples = {
            'high_confidence_wrong': [],      # Model very confident but wrong
            'low_confidence_correct': [],     # Model uncertain but correct
            'ambiguous_boundary': [],         # Near decision boundary
            'rare_class_misclassified': []    # Misclassified rare class samples
        }
        
        for sample_id in self.inference_engine.oof_predictions.keys():
            if sample_id not in true_labels_df[ID_COL].values:
                continue
                
            oof_pred = self.inference_engine.oof_predictions[sample_id]  # Single prediction now
            ensemble_pred = self.inference_engine.create_oof_ensemble(sample_id)
            true_labels = true_labels_df[true_labels_df[ID_COL] == sample_id][LABEL_COLS].iloc[0].values
            
            # Check each class
            for class_idx, class_name in enumerate(LABEL_COLS):
                true_label = true_labels[class_idx]
                pred_score = ensemble_pred[class_idx]
                fold_scores = [oof_pred[class_idx]]  # Single prediction
                
                is_correct = (true_label == 1) == (pred_score >= 0.5)
                confidence = max(pred_score, 1 - pred_score)
                
                sample_info = {
                    'sample_id': sample_id,
                    'class_name': class_name,
                    'true_label': true_label,
                    'prediction': pred_score,
                    'confidence': confidence,
                    'fold_scores': fold_scores
                }
                
                # Categorize hard samples
                if not is_correct and confidence > 0.8:
                    hard_samples['high_confidence_wrong'].append(sample_info)
                elif is_correct and confidence < 0.3:
                    hard_samples['low_confidence_correct'].append(sample_info)
                elif 0.4 <= pred_score <= 0.6:  # Near decision boundary
                    hard_samples['ambiguous_boundary'].append(sample_info)
        
        return hard_samples

    def analyze_modality_performance(self, true_labels_df: pd.DataFrame, sample_modalities: Optional[Dict[str, str]] = None):
        """
        Analyze model performance across different modalities
        
        Args:
            true_labels_df: DataFrame with true labels
            sample_modalities: Optional modality mapping {sample_id: modality}. 
                              If None, uses self.inference_engine.sample_modalities
        """
        modality_results = {}
        
        # Use provided modality mapping or fall back to inference engine
        if sample_modalities is not None:
            modality_mapping = sample_modalities
        else:
            modality_mapping = self.inference_engine.sample_modalities
        
        # Get unique modalities
        modalities = list(set(modality_mapping.values()))
        
        for modality in modalities:
            modality_analysis = {
                'modality': modality,
                'total_samples': 0,
                'aneurysm_present_samples': 0,
                'per_class_analysis': {},
                'overall_metrics': {},
                'misclassifications': {'false_positives': [], 'false_negatives': []}
            }
            
            # Filter samples by modality
            modality_samples = [sid for sid, mod in modality_mapping.items() 
                              if mod == modality and sid in self.inference_engine.oof_predictions]
            
            modality_analysis['total_samples'] = len(modality_samples)
            
            # Analyze each class for this modality
            for class_idx, class_name in enumerate(LABEL_COLS):
                class_analysis = {
                    'class_name': class_name,
                    'total_samples': 0,
                    'positive_samples': 0,
                    'correct_predictions': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'predictions': [],
                    'true_labels': []
                }
                
                for sample_id in modality_samples:
                    if sample_id not in true_labels_df[ID_COL].values:
                        continue
                    
                    true_label = true_labels_df[true_labels_df[ID_COL] == sample_id][class_name].iloc[0]
                    ensemble_pred = self.inference_engine.create_oof_ensemble(sample_id)
                    pred_score = ensemble_pred[class_idx]
                    pred_binary = 1 if pred_score >= 0.5 else 0
                    
                    class_analysis['total_samples'] += 1
                    class_analysis['predictions'].append(pred_score)
                    class_analysis['true_labels'].append(true_label)
                    
                    if true_label == 1:
                        class_analysis['positive_samples'] += 1
                        if pred_binary == 1:
                            class_analysis['correct_predictions'] += 1
                        else:
                            class_analysis['false_negatives'] += 1
                            modality_analysis['misclassifications']['false_negatives'].append({
                                'sample_id': sample_id,
                                'class_name': class_name,
                                'true_label': true_label,
                                'prediction': pred_score
                            })
                    else:
                        if pred_binary == 1:
                            class_analysis['false_positives'] += 1
                            modality_analysis['misclassifications']['false_positives'].append({
                                'sample_id': sample_id,
                                'class_name': class_name,
                                'true_label': true_label,
                                'prediction': pred_score
                            })
                        else:
                            class_analysis['correct_predictions'] += 1
                
                # Calculate metrics for this class
                if class_analysis['total_samples'] > 0:
                    class_analysis['accuracy'] = class_analysis['correct_predictions'] / class_analysis['total_samples']
                    class_analysis['precision'] = (class_analysis['correct_predictions'] / 
                                                 (class_analysis['correct_predictions'] + class_analysis['false_positives'])) if (class_analysis['correct_predictions'] + class_analysis['false_positives']) > 0 else 0
                    class_analysis['recall'] = (class_analysis['correct_predictions'] / 
                                              class_analysis['positive_samples']) if class_analysis['positive_samples'] > 0 else 0
                    class_analysis['f1_score'] = (2 * class_analysis['precision'] * class_analysis['recall'] / 
                                                (class_analysis['precision'] + class_analysis['recall'])) if (class_analysis['precision'] + class_analysis['recall']) > 0 else 0
                
                modality_analysis['per_class_analysis'][class_name] = class_analysis
            
            # Calculate overall metrics for this modality
            all_predictions = []
            all_true_labels = []
            for class_name in LABEL_COLS:
                if class_name in modality_analysis['per_class_analysis']:
                    all_predictions.extend(modality_analysis['per_class_analysis'][class_name]['predictions'])
                    all_true_labels.extend(modality_analysis['per_class_analysis'][class_name]['true_labels'])
            
            if len(set(all_true_labels)) > 1:  # Both classes present
                try:
                    modality_analysis['overall_metrics']['auc'] = roc_auc_score(all_true_labels, all_predictions)
                except:
                    modality_analysis['overall_metrics']['auc'] = 0.0
            else:
                modality_analysis['overall_metrics']['auc'] = 0.0
            
            # Count aneurysm present samples
            modality_analysis['aneurysm_present_samples'] = sum(1 for sample_id in modality_samples 
                                                               if sample_id in true_labels_df[ID_COL].values and 
                                                               true_labels_df[true_labels_df[ID_COL] == sample_id]['Aneurysm Present'].iloc[0] == 1)
            
            modality_results[modality] = modality_analysis
        
        return modality_results

    def identify_modality_hard_samples(self, true_labels_df: pd.DataFrame, sample_modalities: Optional[Dict[str, str]] = None):
        """
        Identify hard samples specific to each modality
        
        Args:
            true_labels_df: DataFrame with true labels
            sample_modalities: Optional modality mapping {sample_id: modality}. 
                              If None, uses self.inference_engine.sample_modalities
        """
        modality_hard_samples = {}
        
        # Use provided modality mapping or fall back to inference engine
        if sample_modalities is not None:
            modality_mapping = sample_modalities
        else:
            modality_mapping = self.inference_engine.sample_modalities
        
        modalities = list(set(modality_mapping.values()))
        
        for modality in modalities:
            modality_samples = [sid for sid, mod in modality_mapping.items() 
                              if mod == modality and sid in self.inference_engine.oof_predictions]
            
            hard_samples = {
                'high_confidence_wrong': [],
                'low_confidence_correct': [],
                'ambiguous_boundary': [],
                'modality_specific_errors': []
            }
            
            for sample_id in modality_samples:
                if sample_id not in true_labels_df[ID_COL].values:
                    continue
                
                oof_pred = self.inference_engine.oof_predictions[sample_id]
                ensemble_pred = self.inference_engine.create_oof_ensemble(sample_id)
                true_labels = true_labels_df[true_labels_df[ID_COL] == sample_id][LABEL_COLS].iloc[0].values
                
                # Check each class
                for class_idx, class_name in enumerate(LABEL_COLS):
                    true_label = true_labels[class_idx]
                    pred_score = ensemble_pred[class_idx]
                    
                    is_correct = (true_label == 1) == (pred_score >= 0.5)
                    confidence = max(pred_score, 1 - pred_score)
                    
                    sample_info = {
                        'sample_id': sample_id,
                        'class_name': class_name,
                        'true_label': true_label,
                        'prediction': pred_score,
                        'confidence': confidence,
                        'modality': modality
                    }
                    
                    # Categorize hard samples
                    if not is_correct and confidence > 0.8:
                        hard_samples['high_confidence_wrong'].append(sample_info)
                    elif is_correct and confidence < 0.3:
                        hard_samples['low_confidence_correct'].append(sample_info)
                    elif 0.4 <= pred_score <= 0.6:  # Near decision boundary
                        hard_samples['ambiguous_boundary'].append(sample_info)
                    
                    # Modality-specific error analysis
                    if not is_correct:
                        hard_samples['modality_specific_errors'].append(sample_info)
            
            modality_hard_samples[modality] = hard_samples
        
        return modality_hard_samples


class VisualizationEngine:
    """Handles all visualization and plotting functionality"""
    
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.num_classes = len(LABEL_COLS)
    
    def create_visualizations(self, per_class_analysis, hard_samples, true_labels_df, modality_analysis=None, modality_hard_samples=None, sample_modalities=None):
        """
        Create comprehensive visualizations
        
        Args:
            per_class_analysis: Per-class analysis results
            hard_samples: Hard sample analysis results
            true_labels_df: DataFrame with true labels
            modality_analysis: Optional modality analysis results
            modality_hard_samples: Optional modality-specific hard samples
            sample_modalities: Optional modality mapping for additional analysis
        """
        
        # 1. Per-class error summary
        self._plot_per_class_error_summary(per_class_analysis)
        
        # 2. Confidence distribution plots
        self._plot_confidence_distributions(per_class_analysis)
        
        # 3. Hard sample analysis
        self._plot_hard_sample_analysis(hard_samples)
        
        # 4. ROC curves per class
        self._plot_roc_curves_per_class(per_class_analysis, true_labels_df)
        
        # 5. Modality-based analysis (if provided)
        if modality_analysis is not None:
            self._plot_modality_performance_comparison(modality_analysis)
            self._plot_modality_class_performance(modality_analysis)
            self._plot_modality_confidence_distributions(modality_analysis)
            
        if modality_hard_samples is not None:
            self._plot_modality_hard_samples(modality_hard_samples)
            self._plot_modality_error_analysis(modality_hard_samples)
    
    def _plot_per_class_error_summary(self, per_class_analysis):
        """Plot per-class error summary"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error counts
        class_names = list(per_class_analysis.keys())
        false_negatives = [len(per_class_analysis[cls]['false_negatives']) for cls in class_names]
        false_positives = [len(per_class_analysis[cls]['false_positives']) for cls in class_names]
        total_errors = [fn + fp for fn, fp in zip(false_negatives, false_positives)]
        
        # Plot 1: Total errors per class
        axes[0, 0].bar(range(len(class_names)), total_errors)
        axes[0, 0].set_title('Total Misclassifications per Class')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: False Negatives vs False Positives
        x = np.arange(len(class_names))
        width = 0.35
        axes[0, 1].bar(x - width/2, false_negatives, width, label='False Negatives', alpha=0.8)
        axes[0, 1].bar(x + width/2, false_positives, width, label='False Positives', alpha=0.8)
        axes[0, 1].set_title('False Negatives vs False Positives')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Error rate per class
        error_rates = []
        for cls in class_names:
            total_samples = per_class_analysis[cls]['total_samples']
            errors = len(per_class_analysis[cls]['false_negatives']) + len(per_class_analysis[cls]['false_positives'])
            error_rate = errors / total_samples if total_samples > 0 else 0
            error_rates.append(error_rate)
        
        axes[1, 0].bar(range(len(class_names)), error_rates)
        axes[1, 0].set_title('Error Rate per Class')
        axes[1, 0].set_ylabel('Error Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Positive sample distribution
        positive_counts = [per_class_analysis[cls]['positive_samples'] for cls in class_names]
        axes[1, 1].bar(range(len(class_names)), positive_counts)
        axes[1, 1].set_title('Positive Samples per Class')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    def _plot_confidence_distributions(self, per_class_analysis):
        """Plot confidence distributions for correct vs incorrect predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select 4 classes to plot
        selected_classes = list(per_class_analysis.keys())[:4]
        
        for i, class_name in enumerate(selected_classes):
            if i >= 4:
                break
                
            correct_conf = per_class_analysis[class_name]['confidence_distribution']['correct']
            incorrect_conf = per_class_analysis[class_name]['confidence_distribution']['incorrect']
            
            axes[i].hist(correct_conf, bins=20, alpha=0.7, label='Correct', density=True)
            axes[i].hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', density=True)
            axes[i].set_title(f'{class_name}\nConfidence Distribution')
            axes[i].set_xlabel('Prediction Confidence')
            axes[i].set_ylabel('Density')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()

    def _plot_hard_sample_analysis(self, hard_samples):
        """Plot hard sample analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Count hard samples by type
        hard_sample_counts = {
            'High Confidence Wrong': len(hard_samples['high_confidence_wrong']),
            'Low Confidence Correct': len(hard_samples['low_confidence_correct']),
            'Ambiguous Boundary': len(hard_samples['ambiguous_boundary'])
        }
        
        # Plot 1: Hard sample counts
        axes[0, 0].bar(hard_sample_counts.keys(), hard_sample_counts.values())
        axes[0, 0].set_title('Hard Sample Counts by Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Hard samples by class
        class_hard_counts = {}
        for hard_type, samples in hard_samples.items():
            for sample in samples:
                class_name = sample['class_name']
                if class_name not in class_hard_counts:
                    class_hard_counts[class_name] = 0
                class_hard_counts[class_name] += 1
        
        if class_hard_counts:
            axes[0, 1].bar(class_hard_counts.keys(), class_hard_counts.values())
            axes[0, 1].set_title('Hard Samples by Class')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Confidence distribution for hard samples
        all_confidences = []
        all_types = []
        for hard_type, samples in hard_samples.items():
            for sample in samples:
                all_confidences.append(sample['confidence'])
                all_types.append(hard_type)
        
        if all_confidences:
            for hard_type in set(all_types):
                type_confidences = [conf for conf, t in zip(all_confidences, all_types) if t == hard_type]
                axes[1, 0].hist(type_confidences, alpha=0.6, label=hard_type, bins=15)
            axes[1, 0].set_title('Confidence Distribution by Hard Sample Type')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Plot 4: Prediction confidence distribution for hard samples
        all_confidences = []
        for hard_type, samples in hard_samples.items():
            for sample in samples:
                all_confidences.append(sample['confidence'])
        
        if all_confidences:
            axes[1, 1].hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Prediction Confidence Distribution for Hard Samples')
            axes[1, 1].set_xlabel('Prediction Confidence')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

    def _plot_roc_curves_per_class(self, per_class_analysis, true_labels_df):
        """Plot ROC curves for each class"""
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, class_name in enumerate(LABEL_COLS):
            if i >= len(axes):
                break
                
            # Collect true labels and predictions for this class
            y_true = []
            y_pred = []
            
            for sample_id in self.inference_engine.oof_predictions.keys():
                if sample_id in true_labels_df[ID_COL].values:
                    true_label = true_labels_df[true_labels_df[ID_COL] == sample_id][class_name].iloc[0]
                    ensemble_pred = self.inference_engine.create_oof_ensemble(sample_id)
                    pred_score = ensemble_pred[i]
                    
                    y_true.append(true_label)
                    y_pred.append(pred_score)
            
            if len(set(y_true)) > 1:  # Only plot if we have both classes
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred)
                
                axes[i].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
                axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[i].set_title(f'{class_name}\nAUC: {auc:.3f}')
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].legend()
            else:
                axes[i].text(0.5, 0.5, 'Insufficient data', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(class_name)
        
        # Hide unused subplots
        for i in range(len(LABEL_COLS), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def _plot_modality_performance_comparison(self, modality_analysis):
        """Plot performance comparison across modalities"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        modalities = list(modality_analysis.keys())
        
        # Plot 1: Sample distribution by modality
        sample_counts = [modality_analysis[mod]['total_samples'] for mod in modalities]
        axes[0, 0].bar(modalities, sample_counts, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Sample Distribution by Modality')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(sample_counts):
            axes[0, 0].text(i, count + max(sample_counts) * 0.01, str(count), 
                           ha='center', va='bottom')
        
        # Plot 2: Aneurysm present rate by modality
        aneurysm_rates = []
        for mod in modalities:
            total = modality_analysis[mod]['total_samples']
            aneurysm_count = modality_analysis[mod]['aneurysm_present_samples']
            rate = (aneurysm_count / total) * 100 if total > 0 else 0
            aneurysm_rates.append(rate)
        
        axes[0, 1].bar(modalities, aneurysm_rates, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Aneurysm Present Rate by Modality')
        axes[0, 1].set_ylabel('Percentage (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        for i, rate in enumerate(aneurysm_rates):
            axes[0, 1].text(i, rate + max(aneurysm_rates) * 0.01, f'{rate:.1f}%', 
                           ha='center', va='bottom')
        
        # Plot 3: Overall AUC by modality
        aucs = []
        for mod in modalities:
            auc = modality_analysis[mod]['overall_metrics'].get('auc', 0.0)
            aucs.append(auc)
        
        bars = axes[1, 0].bar(modalities, aucs, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Overall AUC by Modality')
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add AUC labels on bars
        for i, auc in enumerate(aucs):
            axes[1, 0].text(i, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom')
        
        # Plot 4: Error rate by modality
        error_rates = []
        for mod in modalities:
            total_errors = (len(modality_analysis[mod]['misclassifications']['false_positives']) + 
                           len(modality_analysis[mod]['misclassifications']['false_negatives']))
            total_samples = modality_analysis[mod]['total_samples']
            error_rate = (total_errors / total_samples) * 100 if total_samples > 0 else 0
            error_rates.append(error_rate)
        
        axes[1, 1].bar(modalities, error_rates, alpha=0.7, color='orange')
        axes[1, 1].set_title('Error Rate by Modality')
        axes[1, 1].set_ylabel('Error Rate (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add error rate labels on bars
        for i, rate in enumerate(error_rates):
            axes[1, 1].text(i, rate + max(error_rates) * 0.01, f'{rate:.1f}%', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def _plot_modality_class_performance(self, modality_analysis):
        """Plot per-class performance across modalities"""
        modalities = list(modality_analysis.keys())
        class_names = LABEL_COLS
        
        # Create a comprehensive heatmap
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: Accuracy heatmap
        accuracy_matrix = np.zeros((len(modalities), len(class_names)))
        for i, mod in enumerate(modalities):
            for j, class_name in enumerate(class_names):
                if class_name in modality_analysis[mod]['per_class_analysis']:
                    accuracy_matrix[i, j] = modality_analysis[mod]['per_class_analysis'][class_name].get('accuracy', 0)
        
        im1 = axes[0, 0].imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_title('Accuracy by Modality and Class')
        axes[0, 0].set_xticks(range(len(class_names)))
        axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0, 0].set_yticks(range(len(modalities)))
        axes[0, 0].set_yticklabels(modalities)
        axes[0, 0].set_xlabel('Classes')
        axes[0, 0].set_ylabel('Modalities')
        
        # Add text annotations
        for i in range(len(modalities)):
            for j in range(len(class_names)):
                text = axes[0, 0].text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Plot 2: F1 Score heatmap
        f1_matrix = np.zeros((len(modalities), len(class_names)))
        for i, mod in enumerate(modalities):
            for j, class_name in enumerate(class_names):
                if class_name in modality_analysis[mod]['per_class_analysis']:
                    f1_matrix[i, j] = modality_analysis[mod]['per_class_analysis'][class_name].get('f1_score', 0)
        
        im2 = axes[0, 1].imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0, 1].set_title('F1 Score by Modality and Class')
        axes[0, 1].set_xticks(range(len(class_names)))
        axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0, 1].set_yticks(range(len(modalities)))
        axes[0, 1].set_yticklabels(modalities)
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Modalities')
        
        # Add text annotations
        for i in range(len(modalities)):
            for j in range(len(class_names)):
                text = axes[0, 1].text(j, i, f'{f1_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Plot 3: Precision heatmap
        precision_matrix = np.zeros((len(modalities), len(class_names)))
        for i, mod in enumerate(modalities):
            for j, class_name in enumerate(class_names):
                if class_name in modality_analysis[mod]['per_class_analysis']:
                    precision_matrix[i, j] = modality_analysis[mod]['per_class_analysis'][class_name].get('precision', 0)
        
        im3 = axes[1, 0].imshow(precision_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_title('Precision by Modality and Class')
        axes[1, 0].set_xticks(range(len(class_names)))
        axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 0].set_yticks(range(len(modalities)))
        axes[1, 0].set_yticklabels(modalities)
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_ylabel('Modalities')
        
        # Add text annotations
        for i in range(len(modalities)):
            for j in range(len(class_names)):
                text = axes[1, 0].text(j, i, f'{precision_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Plot 4: Recall heatmap
        recall_matrix = np.zeros((len(modalities), len(class_names)))
        for i, mod in enumerate(modalities):
            for j, class_name in enumerate(class_names):
                if class_name in modality_analysis[mod]['per_class_analysis']:
                    recall_matrix[i, j] = modality_analysis[mod]['per_class_analysis'][class_name].get('recall', 0)
        
        im4 = axes[1, 1].imshow(recall_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 1].set_title('Recall by Modality and Class')
        axes[1, 1].set_xticks(range(len(class_names)))
        axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 1].set_yticks(range(len(modalities)))
        axes[1, 1].set_yticklabels(modalities)
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Modalities')
        
        # Add text annotations
        for i in range(len(modalities)):
            for j in range(len(class_names)):
                text = axes[1, 1].text(j, i, f'{recall_matrix[i, j]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()

    def _plot_modality_confidence_distributions(self, modality_analysis):
        """Plot confidence distributions by modality"""
        modalities = list(modality_analysis.keys())
        
        # Select 4 classes to plot
        selected_classes = LABEL_COLS[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, class_name in enumerate(selected_classes):
            if i >= 4:
                break
            
            # Collect confidence data for each modality
            modality_confidences = {}
            for mod in modalities:
                if class_name in modality_analysis[mod]['per_class_analysis']:
                    predictions = modality_analysis[mod]['per_class_analysis'][class_name]['predictions']
                    modality_confidences[mod] = predictions
            
            # Plot confidence distributions
            for mod, confidences in modality_confidences.items():
                if confidences:  # Only plot if we have data
                    axes[i].hist(confidences, bins=20, alpha=0.6, label=mod, density=True)
            
            axes[i].set_title(f'{class_name}\nConfidence Distribution by Modality')
            axes[i].set_xlabel('Prediction Confidence')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _plot_modality_hard_samples(self, modality_hard_samples):
        """Plot hard sample analysis by modality"""
        modalities = list(modality_hard_samples.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Hard sample counts by modality
        hard_sample_counts = {}
        for mod in modalities:
            counts = {
                'High Confidence Wrong': len(modality_hard_samples[mod]['high_confidence_wrong']),
                'Low Confidence Correct': len(modality_hard_samples[mod]['low_confidence_correct']),
                'Ambiguous Boundary': len(modality_hard_samples[mod]['ambiguous_boundary']),
                'Modality Specific Errors': len(modality_hard_samples[mod]['modality_specific_errors'])
            }
            hard_sample_counts[mod] = counts
        
        # Create grouped bar chart
        x = np.arange(len(modalities))
        width = 0.2
        
        categories = ['High Confidence Wrong', 'Low Confidence Correct', 'Ambiguous Boundary', 'Modality Specific Errors']
        colors = ['red', 'blue', 'orange', 'purple']
        
        for i, (category, color) in enumerate(zip(categories, colors)):
            values = [hard_sample_counts[mod][category] for mod in modalities]
            axes[0, 0].bar(x + i * width, values, width, label=category, color=color, alpha=0.7)
        
        axes[0, 0].set_title('Hard Sample Counts by Modality')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(modalities, rotation=45)
        axes[0, 0].legend()
        
        # Plot 2: Hard sample rate by modality
        total_samples = {}
        for mod in modalities:
            # Get total samples for this modality
            total_samples[mod] = len([sid for sid, mod_name in self.inference_engine.sample_modalities.items() 
                                    if mod_name == mod and sid in self.inference_engine.oof_predictions])
        
        hard_sample_rates = {}
        for mod in modalities:
            total = total_samples[mod]
            if total > 0:
                hard_sample_rates[mod] = {
                    'High Confidence Wrong': len(modality_hard_samples[mod]['high_confidence_wrong']) / total * 100,
                    'Low Confidence Correct': len(modality_hard_samples[mod]['low_confidence_correct']) / total * 100,
                    'Ambiguous Boundary': len(modality_hard_samples[mod]['ambiguous_boundary']) / total * 100,
                    'Modality Specific Errors': len(modality_hard_samples[mod]['modality_specific_errors']) / total * 100
                }
            else:
                hard_sample_rates[mod] = {cat: 0 for cat in categories}
        
        for i, (category, color) in enumerate(zip(categories, colors)):
            values = [hard_sample_rates[mod][category] for mod in modalities]
            axes[0, 1].bar(x + i * width, values, width, label=category, color=color, alpha=0.7)
        
        axes[0, 1].set_title('Hard Sample Rate by Modality (%)')
        axes[0, 1].set_ylabel('Rate (%)')
        axes[0, 1].set_xticks(x + width * 1.5)
        axes[0, 1].set_xticklabels(modalities, rotation=45)
        axes[0, 1].legend()
        
        # Plot 3: Confidence distribution for hard samples by modality
        for mod in modalities:
            all_confidences = []
            for category in ['high_confidence_wrong', 'low_confidence_correct', 'ambiguous_boundary']:
                for sample in modality_hard_samples[mod][category]:
                    all_confidences.append(sample['confidence'])
            
            if all_confidences:
                axes[1, 0].hist(all_confidences, alpha=0.6, label=mod, bins=15, density=True)
        
        axes[1, 0].set_title('Confidence Distribution for Hard Samples by Modality')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Error distribution by class and modality
        class_error_counts = {}
        for mod in modalities:
            class_errors = {}
            for sample in modality_hard_samples[mod]['modality_specific_errors']:
                class_name = sample['class_name']
                class_errors[class_name] = class_errors.get(class_name, 0) + 1
            class_error_counts[mod] = class_errors
        
        # Create stacked bar chart
        all_classes = set()
        for mod_errors in class_error_counts.values():
            all_classes.update(mod_errors.keys())
        all_classes = sorted(list(all_classes))
        
        bottom = np.zeros(len(modalities))
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_classes)))
        
        for i, class_name in enumerate(all_classes):
            values = [class_error_counts[mod].get(class_name, 0) for mod in modalities]
            axes[1, 1].bar(modalities, values, bottom=bottom, label=class_name, color=colors[i], alpha=0.7)
            bottom += values
        
        axes[1, 1].set_title('Error Distribution by Class and Modality')
        axes[1, 1].set_ylabel('Error Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

    def _plot_modality_error_analysis(self, modality_hard_samples):
        """Plot detailed error analysis by modality"""
        modalities = list(modality_hard_samples.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: False positive vs false negative rates by modality
        fp_rates = []
        fn_rates = []
        
        for mod in modalities:
            # Count false positives and false negatives
            fp_count = len([s for s in modality_hard_samples[mod]['modality_specific_errors'] 
                           if s['true_label'] == 0])
            fn_count = len([s for s in modality_hard_samples[mod]['modality_specific_errors'] 
                           if s['true_label'] == 1])
            
            # Get total samples for this modality
            total_samples = len([sid for sid, mod_name in self.inference_engine.sample_modalities.items() 
                               if mod_name == mod and sid in self.inference_engine.oof_predictions])
            
            if total_samples > 0:
                fp_rates.append(fp_count / total_samples * 100)
                fn_rates.append(fn_count / total_samples * 100)
            else:
                fp_rates.append(0)
                fn_rates.append(0)
        
        x = np.arange(len(modalities))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, fp_rates, width, label='False Positive Rate', alpha=0.7, color='red')
        axes[0, 0].bar(x + width/2, fn_rates, width, label='False Negative Rate', alpha=0.7, color='blue')
        axes[0, 0].set_title('False Positive vs False Negative Rates by Modality')
        axes[0, 0].set_ylabel('Rate (%)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(modalities, rotation=45)
        axes[0, 0].legend()
        
        # Plot 2: Error confidence distribution by modality
        for mod in modalities:
            error_confidences = [s['confidence'] for s in modality_hard_samples[mod]['modality_specific_errors']]
            if error_confidences:
                axes[0, 1].hist(error_confidences, alpha=0.6, label=mod, bins=15, density=True)
        
        axes[0, 1].set_title('Error Confidence Distribution by Modality')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Prediction score distribution for errors by modality
        for mod in modalities:
            error_predictions = [s['prediction'] for s in modality_hard_samples[mod]['modality_specific_errors']]
            if error_predictions:
                axes[1, 0].hist(error_predictions, alpha=0.6, label=mod, bins=15, density=True)
        
        axes[1, 0].set_title('Error Prediction Score Distribution by Modality')
        axes[1, 0].set_xlabel('Prediction Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Error count by class and modality (heatmap)
        error_matrix = np.zeros((len(modalities), len(LABEL_COLS)))
        for i, mod in enumerate(modalities):
            for j, class_name in enumerate(LABEL_COLS):
                count = len([s for s in modality_hard_samples[mod]['modality_specific_errors'] 
                           if s['class_name'] == class_name])
                error_matrix[i, j] = count
        
        im = axes[1, 1].imshow(error_matrix, cmap='Reds', aspect='auto')
        axes[1, 1].set_title('Error Count Heatmap by Modality and Class')
        axes[1, 1].set_xticks(range(len(LABEL_COLS)))
        axes[1, 1].set_xticklabels(LABEL_COLS, rotation=45, ha='right')
        axes[1, 1].set_yticks(range(len(modalities)))
        axes[1, 1].set_yticklabels(modalities)
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Modalities')
        
        # Add text annotations
        for i in range(len(modalities)):
            for j in range(len(LABEL_COLS)):
                text = axes[1, 1].text(j, i, f'{int(error_matrix[i, j])}',
                                     ha="center", va="center", color="white" if error_matrix[i, j] > error_matrix.max()/2 else "black")
        
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
