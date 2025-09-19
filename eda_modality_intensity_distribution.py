#!/usr/bin/env python3
"""
EDA: Modality Intensity Distribution Analysis

This script analyzes the intensity distribution of normalized values across different modalities.
It randomly samples 10 series from each modality, extracts windows, applies proper normalization,
and creates distribution plots to visualize how different normalization strategies affect intensity values.

Author: AI Assistant
Date: January 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

# Import required modules
from utils import ID_COL, load_cached_volume_with_renormalization, take_window
from normalization import get_modality_normalization, apply_ct_normalization, apply_statistical_normalization
from config import Config

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
NUM_SAMPLES_PER_MODALITY = 10
WINDOW_OFFSETS = (-2, -1, 0, 1, 2)  # 5-slice windows
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (15, 10)
MAX_PIXELS_PER_MODALITY = 75000  # Limit pixels for efficient histogram creation

class ModalityIntensityAnalyzer:
    """Analyze intensity distributions across modalities"""
    
    def __init__(self, cache_dir: str, train_csv_path: str):
        """
        Initialize analyzer
        
        Args:
            cache_dir: Directory containing cached volumes
            train_csv_path: Path to train.csv with modality information
        """
        self.cache_dir = cache_dir
        self.train_csv_path = train_csv_path
        self.df = None
        self.intensity_data = defaultdict(list)
        self.modality_stats = {}
        
    def load_data(self):
        """Load train.csv and verify cache directory"""
        print("📊 Loading data...")
        
        # Load train.csv
        if not os.path.exists(self.train_csv_path):
            raise FileNotFoundError(f"Train CSV not found: {self.train_csv_path}")
        
        self.df = pd.read_csv(self.train_csv_path)
        print(f"✅ Loaded train.csv with {len(self.df)} samples")
        
        # Verify cache directory
        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")
        
        # Check available modalities
        modalities = self.df['Modality'].value_counts()
        print(f"📋 Available modalities: {dict(modalities)}")
        
        return True
    
    def sample_series_by_modality(self):
        """Randomly sample series from each modality"""
        print(f"\n🎯 Sampling {NUM_SAMPLES_PER_MODALITY} series from each modality...")
        
        sampled_series = {}
        modalities = ['CTA', 'MRA', 'MRI T2', 'MRI T1post']
        
        for modality in modalities:
            modality_df = self.df[self.df['Modality'] == modality]
            
            if len(modality_df) == 0:
                print(f"⚠️  No samples found for {modality}")
                continue
            
            # Randomly sample
            n_samples = min(NUM_SAMPLES_PER_MODALITY, len(modality_df))
            sampled = modality_df.sample(n=n_samples, random_state=42)
            sampled_series[modality] = sampled['SeriesInstanceUID'].tolist()
            
            print(f"  {modality}: {len(sampled_series[modality])} samples")
        
        return sampled_series
    
    def extract_windows_from_volume(self, volume: np.ndarray, max_pixels: int = 10000) -> np.ndarray:
        """Extract random windows from volume with pixel limit"""
        N = volume.shape[0]
        
        if N < 5:  # Need at least 5 slices for window
            return np.array([])
        
        # Calculate how many windows we can extract to stay under pixel limit
        pixels_per_window = 5 * 224 * 224  # 5 slices × 224 × 224
        max_windows = max(1, max_pixels // pixels_per_window)
        
        # Select random center indices for windows
        max_center = N - 2  # Leave room for window offsets
        min_center = 2
        available_windows = max_center - min_center
        num_windows = min(max_windows, available_windows)
        
        if num_windows <= 0:
            return np.array([])
        
        center_indices = np.random.choice(
            range(min_center, max_center), 
            size=num_windows, 
            replace=False
        )
        
        windows = []
        for center_idx in center_indices:
            window = take_window(volume, center_idx, WINDOW_OFFSETS)
            if window is not None:
                windows.append(window.flatten())  # Flatten for intensity analysis
        
        return np.concatenate(windows) if windows else np.array([])
    
    def analyze_modality_intensities(self, sampled_series: dict):
        """Analyze intensity distributions for each modality"""
        print(f"\n🔍 Analyzing intensity distributions...")
        print(f"📊 Limiting to {MAX_PIXELS_PER_MODALITY:,} pixels per modality for efficient visualization")
        
        for modality, series_list in sampled_series.items():
            print(f"\n  Processing {modality}...")
            modality_intensities = []
            pixels_per_series = MAX_PIXELS_PER_MODALITY // len(series_list)
            
            for i, series_id in enumerate(series_list):
                try:
                    # Load volume with proper normalization
                    volume_path = os.path.join(self.cache_dir, f"{series_id}.npz")
                    
                    if not os.path.exists(volume_path):
                        print(f"    ⚠️  Volume not found: {series_id}")
                        continue
                    
                    # Load with re-normalization
                    volume = load_cached_volume_with_renormalization(volume_path, modality)
                    
                    # Extract windows with pixel limit
                    window_intensities = self.extract_windows_from_volume(volume, max_pixels=pixels_per_series)
                    
                    if len(window_intensities) > 0:
                        modality_intensities.extend(window_intensities)
                        print(f"    ✅ {series_id}: {len(window_intensities):,} intensity values")
                    else:
                        print(f"    ⚠️  {series_id}: No valid windows extracted")
                        
                except Exception as e:
                    print(f"    ❌ Error processing {series_id}: {e}")
                    continue
            
            # Store results
            if modality_intensities:
                # Convert to numpy array and apply final random sampling if needed
                modality_intensities = np.array(modality_intensities)
                
                # If we still have too many pixels, randomly sample
                if len(modality_intensities) > MAX_PIXELS_PER_MODALITY:
                    np.random.seed(42)  # Ensure reproducibility
                    indices = np.random.choice(len(modality_intensities), 
                                             size=MAX_PIXELS_PER_MODALITY, 
                                             replace=False)
                    modality_intensities = modality_intensities[indices]
                    print(f"    🔄 Randomly sampled to {len(modality_intensities):,} pixels")
                
                self.intensity_data[modality] = modality_intensities
                self.modality_stats[modality] = {
                    'count': len(modality_intensities),
                    'mean': np.mean(modality_intensities),
                    'std': np.std(modality_intensities),
                    'min': np.min(modality_intensities),
                    'max': np.max(modality_intensities),
                    'median': np.median(modality_intensities),
                    'q25': np.percentile(modality_intensities, 25),
                    'q75': np.percentile(modality_intensities, 75)
                }
                print(f"    📊 Final intensity values: {len(modality_intensities):,}")
            else:
                print(f"    ❌ No intensity data collected for {modality}")
    
    def create_distribution_plots(self):
        """Create comprehensive distribution plots"""
        print(f"\n📈 Creating distribution plots...")
        
        # Set up the plotting style
        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        fig.suptitle('Modality Intensity Distribution Analysis\n(Normalized Values from 5-Slice Windows)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Colors for each modality
        colors = {
            'CTA': '#FF6B6B',      # Red
            'MRA': '#4ECDC4',      # Teal  
            'MRI T2': '#45B7D1',   # Blue
            'MRI T1post': '#96CEB4' # Green
        }
        
        # 1. Histogram comparison
        ax1 = axes[0, 0]
        for modality, intensities in self.intensity_data.items():
            ax1.hist(intensities, bins=50, alpha=0.7, label=modality, 
                    color=colors.get(modality, 'gray'), density=True)
        ax1.set_title('Intensity Distribution (Histogram)', fontweight='bold')
        ax1.set_xlabel('Normalized Intensity Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot comparison
        ax2 = axes[0, 1]
        box_data = [self.intensity_data[modality] for modality in self.intensity_data.keys()]
        box_labels = list(self.intensity_data.keys())
        box_colors = [colors.get(modality, 'gray') for modality in box_labels]
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('Intensity Distribution (Box Plot)', fontweight='bold')
        ax2.set_ylabel('Normalized Intensity Value')
        ax2.grid(True, alpha=0.3)
        
        # 3. Violin plot
        ax3 = axes[1, 0]
        violin_data = []
        violin_labels = []
        for modality, intensities in self.intensity_data.items():
            violin_data.append(intensities)
            violin_labels.append(modality)
        
        parts = ax3.violinplot(violin_data, positions=range(len(violin_labels)), 
                              showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors.get(violin_labels[i], 'gray'))
            pc.set_alpha(0.7)
        
        ax3.set_xticks(range(len(violin_labels)))
        ax3.set_xticklabels(violin_labels)
        ax3.set_title('Intensity Distribution (Violin Plot)', fontweight='bold')
        ax3.set_ylabel('Normalized Intensity Value')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative distribution
        ax4 = axes[1, 1]
        for modality, intensities in self.intensity_data.items():
            sorted_intensities = np.sort(intensities)
            cumulative = np.arange(1, len(sorted_intensities) + 1) / len(sorted_intensities)
            ax4.plot(sorted_intensities, cumulative, label=modality, 
                    color=colors.get(modality, 'gray'), linewidth=2)
        
        ax4.set_title('Cumulative Distribution Function', fontweight='bold')
        ax4.set_xlabel('Normalized Intensity Value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_statistics(self):
        """Print detailed statistics for each modality"""
        print(f"\n📊 Detailed Statistics:")
        print("=" * 80)
        
        # Create summary table
        stats_data = []
        for modality, stats in self.modality_stats.items():
            stats_data.append({
                'Modality': modality,
                'Count': f"{stats['count']:,}",
                'Mean': f"{stats['mean']:.2f}",
                'Std': f"{stats['std']:.2f}",
                'Min': f"{stats['min']:.2f}",
                'Max': f"{stats['max']:.2f}",
                'Median': f"{stats['median']:.2f}",
                'Q25': f"{stats['q25']:.2f}",
                'Q75': f"{stats['q75']:.2f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        print(stats_df.to_string(index=False))
        
        # Print normalization method info
        print(f"\n🔧 Normalization Methods:")
        print("-" * 40)
        print("CTA: CT normalization (fixed range [0, 500] → [0, 255])")
        print("MRA, MRI T2, MRI T1post: MR normalization (adaptive percentiles → [0, 255])")
    
    def run_analysis(self):
        """Run complete analysis"""
        print("🔍 Modality Intensity Distribution Analysis")
        print("=" * 60)
        
        try:
            # Load data
            self.load_data()
            
            # Sample series
            sampled_series = self.sample_series_by_modality()
            
            # Analyze intensities
            self.analyze_modality_intensities(sampled_series)
            
            # Print statistics
            self.print_statistics()
            
            # Create plots
            fig = self.create_distribution_plots()
            
            # Save plot
            output_path = "modality_intensity_distribution_analysis.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Plot saved to: {output_path}")
            
            # Show plot
            plt.show()
            
            print(f"\n✅ Analysis completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            raise


def main():
    """Main function"""
    # Configuration
    cache_dir = "/workspace/Datasets/rsna-intracranial-aneurysm-detection/processed_data/2_5d_volumes_full"
    train_csv_path = "/workspace/Datasets/rsna-intracranial-aneurysm-detection/train.csv"
    
    # Create analyzer and run analysis
    analyzer = ModalityIntensityAnalyzer(cache_dir, train_csv_path)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
