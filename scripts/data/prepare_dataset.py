#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def prepare_cifar100_data(data_dir='../data'):
    """Prepare CIFAR-100 dataset
    
    Args:
        data_dir: Directory to save the dataset
        
    Returns:
        tuple: (trainset, testset) containing training set and test set
    """
    # Ensure data directory exists
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),  # CIFAR-100 mean
            (0.2675, 0.2565, 0.2761)   # CIFAR-100 std
        )
    ])
    
    # Download and prepare training set
    print("Downloading and preparing CIFAR-100 training set...")
    trainset = torchvision.datasets.CIFAR100(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and prepare test set
    print("Downloading and preparing CIFAR-100 test set...")
    testset = torchvision.datasets.CIFAR100(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )
    
    print("Dataset preparation completed!")
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    
    return trainset, testset

if __name__ == "__main__":
    prepare_cifar100_data() 