#!/usr/bin/env python3
"""
Automated script to upload timestamped inference notebooks to Kaggle
"""

import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_notebook_metadata(notebook_name: str, dataset_name: str, competition: str = "rsna-intracranial-aneurysm-detection") -> Dict[str, Any]:
    """Create notebook metadata for Kaggle submission"""
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    if not kaggle_username or kaggle_username == 'your-kaggle-username-here':
        raise ValueError("Please set KAGGLE_USERNAME in .env file")
    
    return {
        "id": f"{kaggle_username}/{notebook_name}",
        "title": notebook_name,  # Use notebook name as title for slug consistency
        "code_file": "kaggle_inference.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "dataset_sources": [
            f"{kaggle_username}/{dataset_name}"    # Your model dataset in username/slug format
        ],
        "competition_sources": [competition],
        "kernel_sources": []
    }

def update_notebook_dataset_reference(notebook_path: str, dataset_name: str) -> str:
    """Update notebook to reference the correct dataset and return updated content"""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Find and update the model directory reference
    updated = False
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                if 'CANDIDATE_MODEL_DIRS' in line:
                    # Update the first directory to point to our dataset
                    for j in range(i, min(i+10, len(source))):  # Look in next few lines
                        if '"/kaggle/input/' in source[j] and not updated:
                            source[j] = f'    "/kaggle/input/{dataset_name}",\n'
                            updated = True
                            print(f"✅ Updated notebook to use dataset: {dataset_name}")
                            break
                    if updated:
                        break
    
    if not updated:
        print("⚠️  Warning: Could not find CANDIDATE_MODEL_DIRS to update")
    
    return json.dumps(notebook, indent=1)

def upload_timestamped_notebook(dataset_name: str, base_notebook: str = "kaggle_inference.ipynb", 
                               submission_message: str = None) -> str:
    """Upload a timestamped inference notebook to Kaggle"""
    
    if not os.path.exists(base_notebook):
        raise FileNotFoundError(f"Base notebook not found: {base_notebook}")
    
    # Use model timestamp for notebook name (maintains consistency)
    notebook_name = f"rsna-{dataset_name}"
    
    if not submission_message:
        submission_message = f"Inference submission with models from {dataset_name}"
    
    print(f"📝 Creating timestamped notebook: {notebook_name}")
    
    # Create temporary directory for submission
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update notebook content with correct dataset reference
        updated_notebook_content = update_notebook_dataset_reference(base_notebook, dataset_name)
        
        # Write updated notebook to temp directory
        temp_notebook = os.path.join(temp_dir, "kaggle_inference.ipynb")
        with open(temp_notebook, 'w') as f:
            f.write(updated_notebook_content)
        
        # Create metadata file
        metadata = create_notebook_metadata(notebook_name, dataset_name)
        metadata_file = os.path.join(temp_dir, "kernel-metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📁 Prepared files in: {temp_dir}")
        print(f"📋 Notebook ID: {metadata['id']}")
        
        try:
            # Push kernel to Kaggle
            print("📤 Uploading notebook to Kaggle...")
            result = subprocess.run([
                "kaggle", "kernels", "push", "-p", temp_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Notebook uploaded successfully!")
                print(result.stdout)
                
                # Extract notebook URL from output
                for line in result.stdout.split('\n'):
                    if 'Your Kernel has been' in line or 'successfully' in line:
                        print(f"🔗 Notebook URL: https://www.kaggle.com/code/{metadata['id']}")
                        
                return notebook_name
            else:
                print("❌ Failed to upload notebook")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                print("Return code:", result.returncode)
                raise RuntimeError(f"Notebook upload failed: {result.stderr}")
                
        except FileNotFoundError:
            print("❌ Kaggle CLI not found. Please install with: pip install kaggle")
            raise RuntimeError("Kaggle CLI not available")
        except Exception as e:
            print(f"❌ Error during notebook upload: {e}")
            raise RuntimeError(f"Notebook upload failed: {e}")

def main():
    """CLI interface for notebook upload"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload timestamped inference notebook to Kaggle")
    parser.add_argument("--dataset_name", required=True, 
                       help="Name of the model dataset to reference")
    parser.add_argument("--base_notebook", default="kaggle_inference.ipynb",
                       help="Base notebook file to upload (default: kaggle_inference.ipynb)")
    parser.add_argument("--message", 
                       help="Submission message")
    
    args = parser.parse_args()
    
    try:
        # Check Kaggle authentication
        result = subprocess.run(["kaggle", "competitions", "list"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Kaggle authentication failed")
            print("💡 Make sure ~/.kaggle/kaggle.json exists with your API credentials")
            return 1
        
        # Upload notebook
        notebook_name = upload_timestamped_notebook(
            dataset_name=args.dataset_name,
            base_notebook=args.base_notebook,
            submission_message=args.message
        )
        
        print(f"\n🎉 Success! Notebook uploaded as: {notebook_name}")
        print(f"🔗 View at: https://www.kaggle.com/code/{os.getenv('KAGGLE_USERNAME')}/{notebook_name}")
        print("\n📝 Next steps:")
        print("1. Visit the notebook URL above")
        print("2. Click 'Save & Run All' to execute")
        print("3. Submit to competition via the notebook interface")
        
        return 0
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
