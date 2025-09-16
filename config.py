#!/usr/bin/env python3
"""
RSNA 2025 Intracranial Aneurysm Detection - Configuration Management

This module contains configuration classes for training and inference.
Supports both binary and multiclass classification modes.

Usage:
    from config import Config
    config = Config('configs/train_config.yaml')
"""

import os
import sys
import yaml
from typing import Dict, Any, Optional
from datetime import datetime


class Config:
    """Training configuration loaded from YAML file"""
    
    def __init__(self, config_path: str):
        # Require config file - no defaults
        if not config_path:
            raise ValueError("Config file path is required")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.load_from_yaml(config_path)
    
    def load_from_yaml(self, config_path: str):
        """Load configuration from YAML file - all values must be provided"""
        print(f"📄 Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            self.original_config = yaml.safe_load(f)
            config = self.original_config
        
        # Model settings - all required
        model_config = config['model']
        self.architecture = model_config['architecture']
        self.num_classes = model_config['num_classes']
        self.in_channels = model_config['in_channels']
        self.pretrained = model_config['pretrained']
        
        # Training settings - all required
        training_config = config['training']
        self.epochs = training_config['epochs']
        self.batch_size = training_config['batch_size']
        self.learning_rate = training_config['learning_rate']
        self.num_folds = training_config['num_folds']
        self.use_cv = training_config['use_cv']
        self.mixed_precision = training_config['mixed_precision']
        self.gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        
        # Loss settings - optional with defaults
        loss_config = training_config.get('loss', {})
        self.loss_name = loss_config.get('name', 'BCEWithLogitsLoss')
        self.loss_params = loss_config.get('params', {})
        
        # Optimizer settings - all required
        optimizer_config = training_config['optimizer']
        self.optimizer_type = optimizer_config['type']
        self.weight_decay = optimizer_config['weight_decay']
        self.betas = optimizer_config['betas']
        
        # Scheduler settings - all required
        scheduler_config = training_config['scheduler']
        self.scheduler_type = scheduler_config['type']
        self.warmup_epochs = scheduler_config['warmup_epochs']
        self.min_lr = scheduler_config['min_lr']
        
        # Data settings - all required
        data_config = config['data']
        self.img_size = data_config['img_size']
        self.window_offsets = tuple(data_config['window_offsets'])
        self.roi_box_fraction = data_config['roi_box_fraction']
        self.roi_min_pixels = data_config['roi_min_pixels']
        self.modalities = data_config['modalities']
        
        # Augmentation settings - all required
        aug_config = data_config['augmentation']
        self.horizontal_flip_prob = aug_config['horizontal_flip']
        self.affine_config = aug_config['affine']
        self.gaussian_noise_config = aug_config['gaussian_noise']
        self.motion_blur_config = aug_config['motion_blur']
        
        # Paths - all required
        paths_config = config['paths']
        self.data_dir = paths_config['data_dir']
        self.cache_dir = paths_config['cache_dir']
        self.train_csv = paths_config['train_csv']
        self.localizers_csv = paths_config['localizers_csv']
        self.output_dir = paths_config['output_dir']
        
        # System settings - all required
        system_config = config['system']
        self.num_workers = system_config['num_workers']
        self.device = system_config['device']
        self.seed = system_config['seed']
        
        # Validation settings - all required
        validation_config = config['validation']
        self.metric = validation_config['metric']
        self.save_best_only = validation_config['save_best_only']
        self.patience = validation_config['patience']
        
        # Debug settings - all required
        debug_config = config['debug']
        self.debug_mode = debug_config['enabled']
        self.subset_per_class = debug_config['subset_per_class']
        self.fast_dev_run = debug_config['fast_dev_run']
        
        # Logging settings - optional with defaults
        logging_config = config.get('logging', {})
        self.log_level = logging_config.get('log_level', 'INFO')
        self.log_file = logging_config.get('log_file', 'training.log')
        self.wandb_enabled = logging_config.get('wandb', {}).get('enabled', False)
        self.wandb_project = logging_config.get('wandb', {}).get('project', '')
        self.wandb_entity = logging_config.get('wandb', {}).get('entity', None)
        
        # Training mode - optional with default
        self.training_mode = config.get('training_mode', 'multiclass')
        
        # Binary-specific settings - optional
        
        print(f"✅ Configuration loaded successfully")
        
    def setup_logging(self, experiment_dir: str = None):
        """Setup logging to redirect stdout to a log file in the experiment directory"""
        if experiment_dir:
            # Create log file path in experiment directory
            log_file_path = os.path.join(experiment_dir, self.log_file)
        else:
            # Use current directory if no experiment directory provided
            log_file_path = self.log_file
        
        # Create a custom class to redirect stdout to both console and file
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # Open log file for writing
        self.log_file_handle = open(log_file_path, 'w')
        
        # Redirect stdout to both console and log file
        self.original_stdout = sys.stdout
        sys.stdout = TeeOutput(sys.stdout, self.log_file_handle)
        
        print(f"📝 Logging enabled: {log_file_path}")
        print(f"🕐 Log started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return log_file_path
    
    def close_logging(self):
        """Close logging and restore stdout"""
        if hasattr(self, 'log_file_handle'):
            self.log_file_handle.close()
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
    
    def print_config(self):
        """Print current configuration"""
        print(f"\n🔧 Training Configuration:")
        print(f"  Training Mode: {self.training_mode}")
        print(f"  Architecture: {self.architecture}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Image size: {self.img_size}")
        print(f"  Window offsets: {self.window_offsets}")
        print(f"  Device: {self.device}")
        print(f"  Cache directory: {self.cache_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Debug mode: {self.debug_mode}")
    
    def save_config(self, save_path: str):
        """Save original configuration to YAML file"""
        with open(save_path, 'w') as f:
            yaml.dump(self.original_config, f, default_flow_style=False, indent=2)
        print(f"💾 Configuration saved to: {save_path}")
    
    def update_training_mode(self, mode: str):
        """Update training mode and related settings"""
        self.training_mode = mode
        if mode == 'binary':
            self.num_classes = 1
        elif mode == 'multiclass':
            self.num_classes = 14
        else:
            raise ValueError(f"Unknown training mode: {mode}. Use 'binary' or 'multiclass'")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            'architecture': self.architecture,
            'num_classes': self.num_classes,
            'in_channels': self.in_channels,
            'pretrained': self.pretrained,
            'training_mode': self.training_mode
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-specific configuration"""
        return {
            'img_size': self.img_size,
            'window_offsets': self.window_offsets,
            'roi_box_fraction': self.roi_box_fraction,
            'roi_min_pixels': self.roi_min_pixels,
            'modalities': self.modalities,
            'horizontal_flip_prob': self.horizontal_flip_prob,
            'affine_config': self.affine_config,
            'gaussian_noise_config': self.gaussian_noise_config,
            'motion_blur_config': self.motion_blur_config
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_folds': self.num_folds,
            'use_cv': self.use_cv,
            'mixed_precision': self.mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'loss_name': self.loss_name,
            'loss_params': self.loss_params,
            'optimizer_type': self.optimizer_type,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'scheduler_type': self.scheduler_type,
            'warmup_epochs': self.warmup_epochs,
            'min_lr': self.min_lr
        }
