#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add superclass labels to CIFAR-100 test set metadata.
"""

import os
import pickle
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def load_cifar100_meta(data_root='../data'):
    """Load CIFAR-100 metadata."""
    try:
        meta_file = os.path.join(data_root, 'cifar-100-python', 'meta')
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
            return meta['fine_label_names'], meta['coarse_label_names']
    except Exception as e:
        logging.error(f"Error loading CIFAR-100 metadata: {e}")
        raise

def load_cifar100_test(data_root='../data'):
    """Load CIFAR-100 test set data."""
    try:
        test_file = os.path.join(data_root, 'cifar-100-python', 'test')
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')
            return test_data[b'fine_labels'], test_data[b'coarse_labels']
    except Exception as e:
        logging.error(f"Error loading CIFAR-100 test data: {e}")
        raise

def add_superclass_labels(metadata_path, output_path, data_root='../data'):
    """Add superclass labels to the metadata CSV file."""
    try:
        # Get class names and test set labels
        fine_names, coarse_names = load_cifar100_meta(data_root)
        fine_labels, coarse_labels = load_cifar100_test(data_root)
        logging.info(f"Found {len(coarse_names)} superclasses")

        # Read metadata
        df_test = pd.read_csv(metadata_path)
        logging.info(f"Metadata columns: {df_test.columns.tolist()}")
        logging.info(f"Read metadata file with {len(df_test)} entries")
        
        # Print sample of metadata and fine names
        logging.info("\nSample of metadata labels:")
        logging.info(df_test['label'].head())
        logging.info("\nSample of CIFAR-100 fine names:")
        logging.info(fine_names[:5])

        # Create mapping from fine label names to coarse labels
        fine_to_coarse = {}
        for fine_idx, coarse_idx in zip(fine_labels, coarse_labels):
            fine_name = fine_names[fine_idx]
            fine_to_coarse[fine_name] = coarse_idx

        # Add labels with new column names only
        df_test['fine_grained_class_index'] = df_test['label'].astype(int)  # Convert label to integer index
        df_test['subclass_name'] = df_test['fine_grained_class_index'].map(lambda x: fine_names[x])
        df_test['coarse_grained_class_index'] = df_test['fine_grained_class_index'].map(lambda x: coarse_labels[fine_labels.index(x)])
        df_test['superclass_name'] = df_test['coarse_grained_class_index'].map(lambda x: coarse_names[x])

        # Verify pickup truck mapping
        pickup_idx = fine_names.index('pickup_truck')
        pickup_entries = df_test[df_test['fine_grained_class_index'] == pickup_idx]
        if not pickup_entries.empty:
            logging.info(f"\nPickup truck mapping verification:")
            logging.info(f"Label: pickup_truck (ID: {pickup_idx})")
            logging.info(f"Coarse ID: {pickup_entries.iloc[0]['coarse_grained_class_index']}")
            logging.info(f"Superclass: {pickup_entries.iloc[0]['superclass_name']}")

        # Select only the required 5 columns and save
        df_final = df_test[['filename', 'fine_grained_class_index', 'subclass_name', 'coarse_grained_class_index', 'superclass_name']]
        df_final.to_csv(output_path, index=False)
        logging.info(f"Saved updated metadata to {output_path}")

        # Validation
        sample = df_test[['filename', 'fine_grained_class_index', 'subclass_name', 'coarse_grained_class_index', 'superclass_name']].drop_duplicates().sample(5)
        logging.info("\nRandom sample validation:\n" + str(sample))

    except Exception as e:
        logging.error(f"Error processing metadata: {e}")
        raise

def main():
    """Main function."""
    metadata_path = '../features/metadata.csv'
    output_path = '../data/testset_with_superclass.csv'

    try:
        add_superclass_labels(metadata_path, output_path)
        logging.info("Successfully added superclass labels")
    except Exception as e:
        logging.error(f"Failed to add superclass labels: {e}")
        raise

if __name__ == '__main__':
    main() 