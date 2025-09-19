#!/usr/bin/env python3
"""
RSNA 2025 Intracranial Aneurysm Detection - Sample Visualization Script

This script visualizes random samples from the binary classification model results.
It creates montages of 5-slice windows converted to RGB for easy visualization.

Features:
- Random sample visualization
- Prediction type-based visualization (TP, TN, FP, FN)
- 3x3 montage of 5-slice windows converted to RGB
- Ground truth and prediction display
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Import required modules
from model import create_binary_model
from config import Config
from utils import ID_COL, load_cached_volume, take_window, parse_coordinates, create_coordinate_lookup
from analysis import (
    InferenceEngine, AnalysisEngine, VisualizationEngine,
    create_sample_montage, visualize_random_samples, visualize_sample_by_prediction_type,
    compute_window_predictions
)

# Set random seed for reproducibility
np.random.seed(42)

def analyze_localizer_window_predictions(localizers_df, inference_engine, true_labels_df):
    """
    Analyze window predictions at exact localizer coordinates
    
    Args:
        localizers_df: DataFrame with localizer annotations
        inference_engine: InferenceEngine with loaded models and predictions
        true_labels_df: DataFrame with true labels
    
    Returns:
        Dict with location-based analysis results
    """
    print("📍 Analyzing window predictions at localizer coordinates...")
    
    location_analysis = {}
    processed_samples = 0
    failed_samples = 0
    
    for idx, row in localizers_df.iterrows():
        series_id = row['SeriesInstanceUID']
        coordinates = parse_coordinates(row['coordinates'])
        location = row['location']
        
        # Skip if we don't have predictions for this sample
        if series_id not in inference_engine.oof_predictions:
            continue
            
        try:
            # Get the model for this sample (OOF model)
            sample_fold = inference_engine.fold_assignments.get(series_id, -1)
            model = inference_engine.fold_models.get(sample_fold, None)
            
            if model is None:
                print(f"Warning: No model found for sample {series_id}")
                failed_samples += 1
                continue
            
            # Load volume
            volume = load_cached_volume(series_id, inference_engine.config.cache_dir)
            if volume is None:
                print(f"Warning: Could not load volume for sample {series_id}")
                failed_samples += 1
                continue
            
            # Compute window predictions for all slices
            window_predictions = compute_window_predictions(volume, model, inference_engine.config.window_offsets)
            
            if not window_predictions:
                print(f"Warning: No window predictions for sample {series_id}")
                failed_samples += 1
                continue
            
            # Find the slice with highest prediction (proxy for localizer location)
            # In a more sophisticated implementation, you would map 3D coordinates to slice indices
            best_slice = max(window_predictions.keys(), key=lambda k: window_predictions[k])
            best_prediction = window_predictions[best_slice]
            
            # Get true label
            true_label = true_labels_df[true_labels_df[ID_COL] == series_id]['Aneurysm Present'].iloc[0]
            
            # Initialize location analysis if needed
            if location not in location_analysis:
                location_analysis[location] = {
                    'predictions': [],
                    'coordinates': [],
                    'slices': [],
                    'true_labels': [],
                    'sample_ids': [],
                    'window_predictions': []
                }
            
            # Store results
            location_analysis[location]['predictions'].append(best_prediction)
            location_analysis[location]['coordinates'].append(coordinates)
            location_analysis[location]['slices'].append(best_slice)
            location_analysis[location]['true_labels'].append(true_label)
            location_analysis[location]['sample_ids'].append(series_id)
            location_analysis[location]['window_predictions'].append(window_predictions)
            
            processed_samples += 1
            
        except Exception as e:
            print(f"Error processing sample {series_id}: {e}")
            failed_samples += 1
            continue
    
    print(f"✅ Processed {processed_samples} localizer samples successfully")
    print(f"❌ Failed to process {failed_samples} localizer samples")
    
    # Calculate summary statistics for each location
    for location, data in location_analysis.items():
        predictions = np.array(data['predictions'])
        true_labels = np.array(data['true_labels'])
        
        # Calculate metrics
        binary_predictions = (predictions >= 0.5).astype(int)
        accuracy = np.mean(binary_predictions == true_labels)
        
        # Separate by true label
        pos_predictions = predictions[true_labels == 1]
        neg_predictions = predictions[true_labels == 0]
        
        data['summary'] = {
            'total_samples': len(predictions),
            'positive_samples': np.sum(true_labels),
            'negative_samples': np.sum(true_labels == 0),
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'mean_positive_prediction': np.mean(pos_predictions) if len(pos_predictions) > 0 else 0.0,
            'mean_negative_prediction': np.mean(neg_predictions) if len(neg_predictions) > 0 else 0.0,
            'accuracy': accuracy,
            'false_negatives': np.sum((true_labels == 1) & (binary_predictions == 0)),
            'false_positives': np.sum((true_labels == 0) & (binary_predictions == 1))
        }
    
    return location_analysis

def main():
    """Main function to run sample visualization"""
    
    print("🔍 RSNA Intracranial Aneurysm Detection - Sample Visualization")
    print("=" * 70)
    
    # Configuration
    EXPERIMENT_DIR = "models/2025-09-15-05-27-19"  # Binary model experiment directory
    DEVICE = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    
    print(f"📁 Experiment Directory: {EXPERIMENT_DIR}")
    print(f"🖥️  Device: {DEVICE}")
    
    # Load experiment configuration
    with open(f"{EXPERIMENT_DIR}/used_config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create Config object
    config = Config('configs/train_config_binary.yaml')
    
    # Override with experiment config
    config.architecture = config_dict['model']['architecture']
    config.img_size = config_dict['data']['img_size']
    config.window_offsets = config_dict['data']['window_offsets']
    config.roi_box_fraction = config_dict['data']['roi_box_fraction']
    config.cache_dir = config_dict['paths']['cache_dir']
    config.device = DEVICE
    config.num_classes = 1  # Binary classification
    
    print(f"🏗️  Architecture: {config.architecture}")
    print(f"📏 Image size: {config.img_size}")
    print(f"🔢 Num classes: {config.num_classes}")
    print(f"💾 Cache dir: {config.cache_dir}")
    
    # Create custom InferenceEngine for binary models
    class BinaryInferenceEngine(InferenceEngine):
        """Custom InferenceEngine for binary models"""
        
        def __init__(self, experiment_dir: str, config: Config, aggregation: str = "mean"):
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
                    print(f"✅ Loaded binary fold {fold} model")
                else:
                    print(f"⚠️  Warning: Binary model not found for fold {fold}")
    
    # Initialize engines
    print("\n🔧 Initializing engines...")
    inference_engine = BinaryInferenceEngine(EXPERIMENT_DIR, config, aggregation="mean")
    
    # Load fold assignments and models
    train_csv_path = config_dict['paths']['train_csv']
    inference_engine.load_fold_assignments(train_csv_path)
    inference_engine.load_fold_models()
    
    # Load ground truth
    print(f"\n📊 Loading ground truth from: {train_csv_path}")
    true_labels_df = pd.read_csv(train_csv_path)
    print(f"✅ Loaded ground truth for {len(true_labels_df)} samples")
    
    # Get sample IDs for analysis (use a subset for testing)
    all_sample_ids = list(inference_engine.fold_assignments.keys())
    print(f"📋 Total samples available: {len(all_sample_ids)}")
    
    # Use a subset for testing
    test_sample_ids = np.random.choice(all_sample_ids, size=min(10, len(all_sample_ids)), replace=False).tolist()
    print(f"🎯 Using {len(test_sample_ids)} samples for visualization")
    
    # Collect OOF predictions
    print(f"\n🔄 Collecting OOF predictions...")
    inference_engine.collect_oof_predictions(test_sample_ids)
    print(f"✅ Collected OOF predictions for {len(inference_engine.oof_predictions)} samples")
    
    # Visualize random samples
    print(f"\n🎨 Visualizing random samples...")
    visualize_random_samples(inference_engine, true_labels_df, num_samples=3)
    
    # Visualize samples by prediction type
    print(f"\n🎯 Visualizing false negatives (missed aneurysms)...")
    visualize_sample_by_prediction_type(inference_engine, true_labels_df, 
                                      prediction_type="false_negatives", num_samples=2)
    
    # Localizer-based analysis
    print(f"\n📍 Starting localizer-based window prediction analysis...")
    localizers_csv_path = "/workspace/Datasets/rsna-intracranial-aneurysm-detection/train_localizers.csv"
    
    try:
        localizers_df = pd.read_csv(localizers_csv_path)
        print(f"✅ Loaded {len(localizers_df)} localizer annotations")
        
        # Filter localizers for samples we have predictions for
        available_samples = set(inference_engine.oof_predictions.keys())
        localizers_filtered = localizers_df[localizers_df['SeriesInstanceUID'].isin(available_samples)]
        print(f"🎯 Found {len(localizers_filtered)} localizers for analyzed samples")
        
        if len(localizers_filtered) > 0:
            location_analysis = analyze_localizer_window_predictions(localizers_filtered, inference_engine, true_labels_df)
            
            # Print summary results
            print("\n" + "=" * 80)
            print("📍 LOCALIZER-BASED WINDOW PREDICTION ANALYSIS RESULTS")
            print("=" * 80)
            
            for location, data in location_analysis.items():
                summary = data['summary']
                print(f"\n🔍 {location}:")
                print(f"  • Total samples: {summary['total_samples']}")
                print(f"  • Positive samples: {summary['positive_samples']}")
                print(f"  • Negative samples: {summary['negative_samples']}")
                print(f"  • Mean prediction: {summary['mean_prediction']:.3f} ± {summary['std_prediction']:.3f}")
                print(f"  • Mean positive prediction: {summary['mean_positive_prediction']:.3f}")
                print(f"  • Mean negative prediction: {summary['mean_negative_prediction']:.3f}")
                print(f"  • Accuracy: {summary['accuracy']:.3f}")
                print(f"  • False negatives: {summary['false_negatives']}")
                print(f"  • False positives: {summary['false_positives']}")
            
            print("\n" + "=" * 80)
        else:
            print("⚠️ No localizer data available for analyzed samples")
            
    except FileNotFoundError:
        print(f"⚠️ Localizer CSV file not found at: {localizers_csv_path}")
        print("Skipping localizer analysis...")
    except Exception as e:
        print(f"❌ Error during localizer analysis: {e}")
    
    print(f"\n✅ Sample visualization and localizer analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
