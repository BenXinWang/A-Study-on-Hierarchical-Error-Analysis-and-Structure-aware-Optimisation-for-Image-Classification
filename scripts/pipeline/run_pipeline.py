#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import time
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.models as models

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import project modules
from scripts.data.prepare_dataset import prepare_cifar100_data
from scripts.features.extract_features import modify_resnet50, extract_features, save_data
from scripts.features.prototype_calculator import (
    ModifiedResNet50, calculate_confidences, calculate_class_prototypes,
    calculate_superclass_prototypes, SUPERCLASS_MAPPING
)
from scripts.visualization.prototype_visualizer import visualize_prototypes
from scripts.data.add_superclass_labels import add_superclass_labels
from scripts.features.verify_labels import verify_labels

class Pipeline:
    def __init__(self):
        self.config = {
            'data_dir': project_root / 'data',
            'features_dir': project_root / 'features',
            'results_dir': project_root / 'results',
            'batch_size': 32,
            'num_workers': 4,
            'device': 'mps' if torch.backends.mps.is_available() else 'cpu',  # 优先使用 MPS
            'feature_dim': 2048,
            'image_size': 32,
        }
        
        # Ensure directories exist
        for dir_path in [self.config['data_dir'], 
                        self.config['features_dir'], 
                        self.config['results_dir']]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.config['results_dir'] / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def create_dataloader(self):
        """Create dataloader for feature extraction"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        test_dataset = CIFAR100(
            root=str(self.config['data_dir']),
            train=False,
            transform=transform,
            download=False  # 已经在prepare_dataset步骤下载了
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return test_loader
        
    def run_step(self, step_name, func, *args, **kwargs):
        """Run a pipeline step with proper logging and error handling"""
        logging.info(f"Starting step: {step_name}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logging.info(f"Completed step: {step_name} in {duration:.2f} seconds")
            return result
        except Exception as e:
            logging.error(f"Error in step {step_name}: {str(e)}")
            raise
            
    def save_execution_status(self, status):
        """Save execution status to a JSON file"""
        status_file = self.config['results_dir'] / 'pipeline_status.json'
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=4)
            
    def run(self):
        """Execute the complete pipeline"""
        status = {
            'start_time': datetime.now().isoformat(),
            'steps': []
        }
        
        try:
            # Step 1: Prepare dataset
            self.run_step(
                'prepare_dataset',
                prepare_cifar100_data,
                self.config['data_dir']
            )
            status['steps'].append({'name': 'prepare_dataset', 'status': 'success'})
            
            # Step 2: Extract features
            # Create dataloader
            test_loader = self.create_dataloader()
            
            # First modify the model
            model = self.run_step(
                'modify_resnet',
                modify_resnet50
            )
            model = model.to(self.config['device'])
            
            # Then extract features
            features, labels, filenames = self.run_step(
                'extract_features',
                extract_features,
                model=model,
                dataloader=test_loader,
                device=self.config['device']
            )
            
            # Save the extracted features
            self.run_step(
                'save_features',
                save_data,
                features=features,
                labels=labels,
                filenames=filenames,
                save_dir=self.config['features_dir']
            )
            status['steps'].append({'name': 'extract_features', 'status': 'success'})
            
            # Step 3: Verify features (skipped - file not available)
            logging.info("Skipping verify_features step - file not available")
            status['steps'].append({'name': 'verify_features', 'status': 'skipped'})
            
            # Step 4: Calculate prototypes
            # Create model
            model = ModifiedResNet50().to(self.config['device'])
            
            # Calculate confidences
            confidences = self.run_step(
                'calculate_confidences',
                calculate_confidences,
                model=model,
                dataloader=test_loader,
                device=self.config['device']
            )
            
            # Calculate class prototypes
            class_prototypes = self.run_step(
                'calculate_class_prototypes',
                calculate_class_prototypes,
                features=features,
                labels=labels,
                confidences=confidences
            )
            
            # Calculate superclass prototypes
            superclass_prototypes = self.run_step(
                'calculate_superclass_prototypes',
                calculate_superclass_prototypes,
                class_prototypes=class_prototypes
            )
            
            # Save prototypes
            output = {
                'class': class_prototypes,
                'super': superclass_prototypes,
                'mapping': SUPERCLASS_MAPPING
            }
            torch.save(output, self.config['features_dir'] / 'prototypes.pt')
            status['steps'].append({'name': 'calculate_prototypes', 'status': 'success'})
            
            # Step 5: Visualize prototypes
            self.run_step(
                'visualize_prototypes',
                visualize_prototypes,
                self.config['features_dir'],
                self.config['results_dir']
            )
            status['steps'].append({'name': 'visualize_prototypes', 'status': 'success'})
            
            # Step 6: Add superclass labels
            self.run_step(
                'add_superclass_labels',
                add_superclass_labels,
                metadata_path=self.config['features_dir'] / 'metadata.csv',
                output_path=self.config['data_dir'] / 'testset_with_superclass.csv',
                data_root=self.config['data_dir']
            )
            status['steps'].append({'name': 'add_superclass_labels', 'status': 'success'})
            
            # Step 7: Verify labels
            self.run_step(
                'verify_labels',
                verify_labels,
                self.config['data_dir']
            )
            status['steps'].append({'name': 'verify_labels', 'status': 'success'})
            
            # Step 8: Check vehicle mapping (skipped - file not available)
            logging.info("Skipping check_vehicle_mapping step - file not available")
            status['steps'].append({'name': 'check_vehicle_mapping', 'status': 'skipped'})
            
            status['end_time'] = datetime.now().isoformat()
            status['overall_status'] = 'success'
            logging.info("Pipeline completed successfully!")
            
        except Exception as e:
            status['end_time'] = datetime.now().isoformat()
            status['overall_status'] = 'failed'
            status['error'] = str(e)
            logging.error(f"Pipeline failed: {str(e)}")
            
        finally:
            self.save_execution_status(status)
            
def main():
    pipeline = Pipeline()
    pipeline.run()

if __name__ == '__main__':
    main() 