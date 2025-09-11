#!/usr/bin/env python3
"""
Loss History Plotting Script for RSNA Aneurysm Detection Training
Plots training and validation loss curves for all folds in an experiment
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

def plot_loss_history(experiment_dir):
    """
    Plot training and validation loss history for all folds
    
    Args:
        experiment_dir (str): Path to experiment directory containing loss_history_fold*.csv files
    """
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return
    
    # Find all loss history files
    loss_files = list(experiment_path.glob("loss_history_fold*.csv"))
    
    if not loss_files:
        print(f"❌ No loss history files found in: {experiment_dir}")
        return
    
    print(f"📊 Found {len(loss_files)} loss history files")
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Color schemes for folds
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    all_train_losses = []
    all_val_losses = []
    
    # Plot each fold
    for i, loss_file in enumerate(sorted(loss_files)):
        fold_num = int(loss_file.stem.split('fold')[-1])
        
        try:
            # Read loss history
            df = pd.read_csv(loss_file)
            
            if 'train_loss' not in df.columns or 'val_loss' not in df.columns:
                print(f"⚠️  Missing columns in {loss_file.name}, skipping...")
                continue
                
            epochs = df['epoch'].values
            train_loss = df['train_loss'].values
            val_loss = df['val_loss'].values
            
            color = colors[i % len(colors)]
            
            # Plot training loss (solid line)
            plt.plot(epochs, train_loss, 
                    color=color, linestyle='-', linewidth=2, alpha=0.8,
                    label=f'Fold {fold_num} - Train')
            
            # Plot validation loss (dashed line)
            plt.plot(epochs, val_loss, 
                    color=color, linestyle='--', linewidth=2, alpha=0.8,
                    label=f'Fold {fold_num} - Val')
            
            all_train_losses.extend(train_loss)
            all_val_losses.extend(val_loss)
            
            print(f"✅ Plotted Fold {fold_num}: Train {train_loss[-1]:.4f}, Val {val_loss[-1]:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {loss_file.name}: {e}")
            continue
    
    if not all_train_losses:
        print("❌ No valid loss data found to plot")
        return
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training and Validation Loss History - All Folds', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set y-axis limits for better visualization
    all_losses = all_train_losses + all_val_losses
    y_min = max(0, min(all_losses) * 0.95)
    y_max = max(all_losses) * 1.05
    plt.ylim(y_min, y_max)
    
    # Add summary statistics
    avg_final_train = np.mean([df[df['epoch'] == df['epoch'].max()]['train_loss'].iloc[0] 
                              for loss_file in sorted(loss_files)
                              for df in [pd.read_csv(loss_file)] 
                              if 'train_loss' in df.columns])
    
    avg_final_val = np.mean([df[df['epoch'] == df['epoch'].max()]['val_loss'].iloc[0] 
                            for loss_file in sorted(loss_files)
                            for df in [pd.read_csv(loss_file)] 
                            if 'val_loss' in df.columns])
    
    # Add text box with summary
    textstr = f'Final Loss (Avg)\nTrain: {avg_final_train:.4f}\nVal: {avg_final_val:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    timestamp = experiment_path.name
    plot_filename = f"loss_history_all_folds.png"
    plot_path = experiment_path / plot_filename
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n💾 Plot saved: {plot_path}")
    print(f"📊 Summary:")
    print(f"   Average Final Train Loss: {avg_final_train:.4f}")
    print(f"   Average Final Val Loss: {avg_final_val:.4f}")
    print(f"   Number of Folds: {len(loss_files)}")

def main():
    parser = argparse.ArgumentParser(description="Plot loss history for RSNA training experiment")
    parser.add_argument("experiment_dir", type=str, 
                       help="Path to experiment directory (e.g., models/2025-09-10-08-33-25)")
    parser.add_argument("--show", action="store_true", 
                       help="Show plot in addition to saving")
    
    args = parser.parse_args()
    
    print(f"🎨 Plotting loss history for experiment: {args.experiment_dir}")
    
    # Plot loss history
    plot_loss_history(args.experiment_dir)
    
    if args.show:
        # Show the saved plot
        plot_path = Path(args.experiment_dir) / "loss_history_all_folds.png"
        if plot_path.exists():
            import subprocess
            try:
                subprocess.run(['xdg-open', str(plot_path)], check=True)
            except:
                print(f"📊 Plot saved but couldn't open automatically: {plot_path}")

if __name__ == "__main__":
    main()
