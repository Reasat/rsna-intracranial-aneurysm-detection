#!/usr/bin/env python3
"""
Simple script to upload trained models to Kaggle as a dataset
"""

import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_dataset_metadata(model_dir: str, title: str, subtitle: str) -> Dict[str, Any]:
    """Create dataset metadata for Kaggle"""
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    if not kaggle_username or kaggle_username == 'your-kaggle-username-here':
        raise ValueError("Please set KAGGLE_USERNAME in .env file")
    
    return {
        "title": title,
        "id": f"{kaggle_username}/{title.lower().replace(' ', '-')}",
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": subtitle,
        "description": f"Trained models from {model_dir} for RSNA 2025 Intracranial Aneurysm Detection",
        "keywords": ["medical", "aneurysm", "detection", "pytorch", "efficientnet"],
        "collaborators": [],
        "data": []
    }

def copy_model_files(model_dir: str, temp_dir: str) -> List[str]:
    """Copy model files to temporary directory"""
    model_files = []
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Copy .pth files
    for pth_file in model_path.glob("*.pth"):
        dest_file = os.path.join(temp_dir, pth_file.name)
        shutil.copy2(pth_file, dest_file)
        model_files.append(pth_file.name)
        print(f"Copied: {pth_file.name}")
    
    # Copy config file if exists
    config_files = ["used_config.yaml", "config.yaml", "experiment_info.yaml"]
    for config_file in config_files:
        config_path = model_path / config_file
        if config_path.exists():
            dest_file = os.path.join(temp_dir, config_file)
            shutil.copy2(config_path, dest_file)
            model_files.append(config_file)
            print(f"Copied: {config_file}")
    
    return model_files

def create_kaggle_dataset(model_dir: str, dataset_title: str = None) -> str:
    """Create and upload Kaggle dataset with models"""
    
    timestamp = os.path.basename(model_dir)
    if dataset_title is None:
        dataset_title = f"{timestamp}"
    
    subtitle = f"PyTorch models trained for RSNA 2025 competition"
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating dataset in temporary directory: {temp_dir}")
        
        # Copy model files
        try:
            model_files = copy_model_files(model_dir, temp_dir)
            if not model_files:
                print("No model files found to upload!")
                return False
        except Exception as e:
            print(f"Error copying files: {e}")
            return False
        
        # Create dataset metadata
        metadata = create_dataset_metadata(model_dir, dataset_title, subtitle)
        
        # Write metadata file
        metadata_file = os.path.join(temp_dir, "dataset-metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created metadata file with {len(model_files)} files")
        
        # Create README
        readme_content = f"""# {dataset_title}

This dataset contains trained PyTorch models for the RSNA 2025 Intracranial Aneurysm Detection competition.

## Model Details
- Architecture: tf_efficientnet_b0
- Training: 5-fold cross-validation
- Input: 2.5D windows (5-slice)
- Dual-stream: Full image + ROI processing

## Files Included
{chr(10).join([f"- {f}" for f in model_files])}

## Usage
Load models in your Kaggle notebook:
```python
import torch
model = torch.load('/kaggle/input/your-dataset-name/model.pth')
```

Generated from: {model_dir}
"""
        
        readme_file = os.path.join(temp_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Try to create dataset using Kaggle CLI
        try:
            print("Creating Kaggle dataset...")
            result = subprocess.run([
                "kaggle", "datasets", "create", "-p", temp_dir
            ], capture_output=True, text=True, cwd=temp_dir)
            
            if result.returncode == 0:
                print("✅ Dataset created successfully!")
                print(result.stdout)
                return dataset_title
            else:
                print("❌ Failed to create dataset")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                raise RuntimeError("Dataset creation failed")
                
        except FileNotFoundError:
            print("❌ Kaggle CLI not found. Please install with: pip install kaggle")
            print("📁 Files prepared in temporary directory (will be deleted)")
            print("🔧 Manual upload required:")
            print(f"   1. Zip contents of: {temp_dir}")
            print(f"   2. Upload to Kaggle manually")
            raise RuntimeError("Manual upload required")
        except Exception as e:
            print(f"❌ Error running Kaggle CLI: {e}")
            raise RuntimeError(f"Dataset upload failed: {e}")

def main():
    """Main function to upload models"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload trained models to Kaggle")
    parser.add_argument("--model_dir", required=True, help="Directory containing model files")
    parser.add_argument("--title", help="Dataset title (auto-generated if not provided)")
    parser.add_argument("--dry_run", action="store_true", help="Prepare files but don't upload")
    
    args = parser.parse_args()
    
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return 1
    
    print(f"📦 Preparing to upload models from: {model_dir}")
    
    if args.dry_run:
        print("🧪 DRY RUN MODE - files will be prepared but not uploaded")
        # Just list files that would be uploaded
        try:
            model_files = []
            for pth_file in Path(model_dir).glob("*.pth"):
                model_files.append(pth_file.name)
            config_files = ["used_config.yaml", "config.yaml", "experiment_info.yaml"]
            for config_file in config_files:
                if (Path(model_dir) / config_file).exists():
                    model_files.append(config_file)
            
            print(f"📋 Would upload {len(model_files)} files:")
            for f in model_files:
                print(f"   - {f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    else:
        # Actually upload
        try:
            dataset_name = create_kaggle_dataset(model_dir, args.title)
            print(f"✅ Dataset created: {dataset_name}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return 1
    
    print("✅ Done!")
    return 0

if __name__ == "__main__":
    exit(main())
