#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify CIFAR-100 label mappings.
"""

import os
import pickle
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def load_cifar100_meta(data_root='../../data'):
    """Load CIFAR-100 metadata."""
    try:
        meta_file = os.path.join(data_root, 'cifar-100-python', 'meta')
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
            return meta['fine_label_names'], meta['coarse_label_names']
    except Exception as e:
        logging.error(f"Error loading CIFAR-100 metadata: {e}")
        raise

def load_cifar100_test(data_root='../../data'):
    """Load CIFAR-100 test set data."""
    try:
        test_file = os.path.join(data_root, 'cifar-100-python', 'test')
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')
            return test_data[b'fine_labels'], test_data[b'coarse_labels']
    except Exception as e:
        logging.error(f"Error loading CIFAR-100 test data: {e}")
        raise

def verify_labels(data_dir):
    """验证标签映射
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        dict: 验证结果
    """
    try:
        data_dir = Path(data_dir)
        
        # Load metadata
        fine_names, coarse_names = load_cifar100_meta(data_dir)
        logging.info(f"\nFound {len(fine_names)} fine classes and {len(coarse_names)} coarse classes")
        
        # Load test data
        fine_labels, coarse_labels = load_cifar100_test(data_dir)
        
        # Load our generated metadata
        df = pd.read_csv(data_dir / 'testset_with_superclass.csv')
        logging.info(f"\nGenerated metadata columns: {df.columns.tolist()}")
        
        # Verify the new column structure
        expected_columns = ['filename', 'fine_grained_class_index', 'subclass_name', 'coarse_grained_class_index', 'superclass_name']
        if not all(col in df.columns for col in expected_columns):
            logging.error(f"Missing expected columns. Found: {df.columns.tolist()}")
            raise ValueError("CSV file does not have expected column structure")
        
        # Verify pickup truck mapping
        pickup_idx = fine_names.index('pickup_truck')
        pickup_coarse = coarse_names[coarse_labels[fine_labels.index(pickup_idx)]]
        logging.info(f"\nPickup truck (ID: {pickup_idx}) belongs to superclass: {pickup_coarse}")
        
        # Get sample mappings
        sample_mappings = []
        for i in range(5):
            fine_idx = fine_labels[i]
            coarse_idx = coarse_labels[i]
            sample_mappings.append({
                'fine_label': fine_names[fine_idx],
                'fine_id': fine_idx,
                'coarse_label': coarse_names[coarse_idx],
                'coarse_id': coarse_idx
            })
            logging.info(f"Fine label: {fine_names[fine_idx]} (ID: {fine_idx})")
            logging.info(f"Coarse label: {coarse_names[coarse_idx]} (ID: {coarse_idx})\n")
            
        return {
            'num_fine_classes': len(fine_names),
            'num_coarse_classes': len(coarse_names),
            'metadata_columns': df.columns.tolist(),
            'pickup_truck_superclass': pickup_coarse,
            'sample_mappings': sample_mappings
        }

    except Exception as e:
        logging.error(f"Verification failed: {e}")
        raise

def main():
    """Main function."""
    verify_labels('./data')

if __name__ == '__main__':
    main() 