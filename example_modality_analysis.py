#!/usr/bin/env python3
"""
Example script demonstrating modality-based analysis capabilities

This script shows how to use the enhanced analysis engine with modality-specific
analysis and visualization features.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from analysis import InferenceEngine, AnalysisEngine, VisualizationEngine
from train import Config

def main():
    """Example usage of modality-based analysis"""
    
    # Example configuration (adjust paths as needed)
    experiment_dir = "models/experiment_20241201_120000"
    train_csv_path = "/workspace/Datasets/rsna-intracranial-aneurysm-detection/train.csv"
    
    # Create config (adjust as needed)
    config = Config()
    config.device = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    config.cache_dir = "/workspace/Datasets/rsna-intracranial-aneurysm-detection/processed_data/2_5d_volumes_full"
    config.img_size = 224
    config.window_offsets = [-2, -1, 0, 1, 2]
    config.modalities = ["CTA", "MRA", "MRI T2", "MRI T1post"]
    
    print("🔍 Initializing Modality-Based Analysis Engine")
    print("=" * 60)
    
    # Initialize engines
    inference_engine = InferenceEngine(experiment_dir, config)
    analysis_engine = AnalysisEngine(inference_engine)
    viz_engine = VisualizationEngine(inference_engine)
    
    # Load data and models
    print("\n📊 Loading fold assignments and models...")
    inference_engine.load_fold_assignments(train_csv_path)
    inference_engine.load_fold_models()
    
    # Collect OOF predictions
    print("\n🔮 Collecting out-of-fold predictions...")
    sample_ids = list(inference_engine.fold_assignments.keys())
    inference_engine.collect_oof_predictions(sample_ids)
    
    # Load true labels
    print("\n📋 Loading true labels...")
    import pandas as pd
    true_labels_df = pd.read_csv(train_csv_path)
    
    # Perform modality-based analysis
    print("\n🔬 Performing modality-based analysis...")
    modality_analysis = analysis_engine.analyze_modality_performance(true_labels_df)
    modality_hard_samples = analysis_engine.identify_modality_hard_samples(true_labels_df)
    
    # Print modality analysis summary
    print("\n📈 Modality Analysis Summary")
    print("=" * 40)
    for modality, analysis in modality_analysis.items():
        print(f"\n{modality}:")
        print(f"  Total samples: {analysis['total_samples']}")
        print(f"  Aneurysm present: {analysis['aneurysm_present_samples']}")
        print(f"  Overall AUC: {analysis['overall_metrics'].get('auc', 0.0):.3f}")
        
        # Show top performing classes
        class_performances = []
        for class_name, class_analysis in analysis['per_class_analysis'].items():
            if class_analysis['total_samples'] > 0:
                f1 = class_analysis.get('f1_score', 0)
                class_performances.append((class_name, f1))
        
        class_performances.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top 3 classes by F1-score:")
        for i, (class_name, f1) in enumerate(class_performances[:3]):
            print(f"    {i+1}. {class_name}: {f1:.3f}")
    
    # Print hard sample analysis
    print("\n🎯 Hard Sample Analysis by Modality")
    print("=" * 40)
    for modality, hard_samples in modality_hard_samples.items():
        print(f"\n{modality}:")
        print(f"  High confidence wrong: {len(hard_samples['high_confidence_wrong'])}")
        print(f"  Low confidence correct: {len(hard_samples['low_confidence_correct'])}")
        print(f"  Ambiguous boundary: {len(hard_samples['ambiguous_boundary'])}")
        print(f"  Total errors: {len(hard_samples['modality_specific_errors'])}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Standard analysis (for comparison)
    per_class_analysis = analysis_engine.analyze_per_class_misclassifications(true_labels_df)
    hard_samples = analysis_engine.identify_hard_samples(true_labels_df)
    
    # Create comprehensive visualizations including modality analysis
    viz_engine.create_visualizations(
        per_class_analysis=per_class_analysis,
        hard_samples=hard_samples,
        true_labels_df=true_labels_df,
        modality_analysis=modality_analysis,
        modality_hard_samples=modality_hard_samples
    )
    
    print("\n✅ Modality-based analysis completed!")
    print("\nGenerated visualizations:")
    print("  - Per-class error summary")
    print("  - Confidence distributions")
    print("  - Hard sample analysis")
    print("  - ROC curves per class")
    print("  - Modality performance comparison")
    print("  - Modality class performance heatmaps")
    print("  - Modality confidence distributions")
    print("  - Modality hard sample analysis")
    print("  - Modality error analysis")

if __name__ == "__main__":
    main()
