#!/usr/bin/env python3
"""
Example demonstrating the new aggregation capabilities in analysis.py

This script shows how to use both "max" and "mean" aggregation methods
for inference in the binary classification pipeline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from analysis import InferenceEngine
from train_binary import Config

def demonstrate_aggregation_methods():
    """Demonstrate different aggregation methods"""
    
    # Load configuration
    config_path = "configs/train_config.yaml"
    config = Config(config_path)
    
    # Example 1: MAX Aggregation (default)
    print("🔍 Example 1: MAX Aggregation")
    print("=" * 50)
    print("Method: Takes the maximum prediction across all slices")
    print("Use case: High sensitivity - captures any slice with high aneurysm probability")
    print("Code: inference_engine = InferenceEngine(experiment_dir, config, aggregation='max')")
    print()
    
    # Example 2: MEAN Aggregation
    print("🔍 Example 2: MEAN Aggregation")
    print("=" * 50)
    print("Method: Takes the average prediction across all slices")
    print("Use case: Balanced approach - considers overall volume characteristics")
    print("Code: inference_engine = InferenceEngine(experiment_dir, config, aggregation='mean')")
    print()
    
    # Example 3: Comparison
    print("📊 Aggregation Method Comparison")
    print("=" * 50)
    print("MAX Aggregation:")
    print("  • Pros: High sensitivity, captures rare but strong signals")
    print("  • Cons: May be overly sensitive to noise, single slice can dominate")
    print("  • Best for: Detection tasks where missing an aneurysm is costly")
    print()
    print("MEAN Aggregation:")
    print("  • Pros: More stable, considers entire volume context")
    print("  • Cons: May miss subtle signals, could average out important features")
    print("  • Best for: Classification tasks where overall volume characteristics matter")
    print()
    
    # Example 4: Usage in Binary Analysis
    print("💻 Usage in Binary Analysis")
    print("=" * 50)
    print("# In result_analysis_binary.py:")
    print("aggregation_method = 'max'  # or 'mean'")
    print("inference_engine = BinaryInferenceEngine(EXPERIMENT_DIR, config, aggregation=aggregation_method)")
    print()
    print("# The aggregation method affects how predictions are combined:")
    print("# For a volume with predictions [0.1, 0.3, 0.7, 0.2, 0.9]:")
    print("# MAX: 0.9 (highest confidence)")
    print("# MEAN: 0.44 (average confidence)")

if __name__ == "__main__":
    demonstrate_aggregation_methods()
