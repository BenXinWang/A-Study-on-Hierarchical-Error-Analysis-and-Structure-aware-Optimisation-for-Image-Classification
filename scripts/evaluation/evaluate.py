#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
from datetime import datetime

def get_device():
    """Configure device, prioritize MPS"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend for acceleration")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    return device

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.resnet import ModifiedResNet50

class SuperclassDataset:
    """Superclass dataset wrapper."""
    def __init__(self, original_dataset, fine_to_coarse_mapping):
        self.dataset = original_dataset
        self.fine_to_coarse = fine_to_coarse_mapping
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, fine_label = self.dataset[idx]
        coarse_label = self.fine_to_coarse[fine_label]
        return image, coarse_label

def load_cifar100_superclass_mapping(data_root='./data'):
    """Load CIFAR-100 superclass mapping."""
    import pickle
    
    # Ensure absolute path
    if not os.path.isabs(data_root):
        # Get project root relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        data_root = os.path.join(project_root, 'data')
    
    meta_file = os.path.join(data_root, 'cifar-100-python', 'meta')
    
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
        fine_names = meta['fine_label_names']
        coarse_names = meta['coarse_label_names']
    
    # Build mapping from training data
    train_file = os.path.join(data_root, 'cifar-100-python', 'train')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
        fine_labels = train_data[b'fine_labels']
        coarse_labels = train_data[b'coarse_labels']
    
    # Create fine-to-coarse mapping
    fine_to_coarse = {}
    for fine_idx, coarse_idx in zip(fine_labels, coarse_labels):
        if fine_idx not in fine_to_coarse:
            fine_to_coarse[fine_idx] = coarse_idx
    
    print(f"Loaded {len(fine_names)} fine classes and {len(coarse_names)} superclasses")
    
    return fine_to_coarse, coarse_names

def print_checkpoint_info(checkpoint):
    """Print checkpoint information"""
    print('\nCheckpoint Information:')
    print('-' * 30)
    
    if 'epoch' in checkpoint:
        print(f'Training epochs: {checkpoint["epoch"]}')
    
    if 'val_acc' in checkpoint:
        print(f'Validation accuracy at save: {checkpoint["val_acc"]*100:.2f}%')
    
    if 'history' in checkpoint:
        history = checkpoint['history']
        if 'val_acc' in history and len(history['val_acc']) > 0:
            best_val_acc = max(history['val_acc'])
            print(f'Best validation accuracy: {best_val_acc*100:.2f}%')
            print(f'Last 5 epochs validation accuracy: {[f"{acc*100:.2f}%" for acc in history["val_acc"][-5:]]}')
    
    print('-' * 30)

def save_predictions(predictions_df, save_path):
    """Save prediction results to CSV file
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing prediction results
        save_path (Path): Save path
    """
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save prediction results
    predictions_df.to_csv(save_path, index=False)
    print(f"\nPrediction results saved to: {save_path}")

def plot_confusion_matrix(cm, classes, save_path):
    """Plot and save confusion matrix
    
    Args:
        cm (np.ndarray): Confusion matrix
        classes (list): List of class names
        save_path (Path): Save path
    """
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(20, 20))
    
    # Plot confusion matrix
    sns.heatmap(
        cm,
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    
    # Set labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to: {save_path}")

def main(args):
    # 1. Setup device
    device = get_device()
    
    # 2. Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load original test dataset
    original_test_dataset = CIFAR100(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    
    # 3. Load trained model
    checkpoint = torch.load(args.checkpoint_path)
    print_checkpoint_info(checkpoint)  # Print checkpoint information
    
    # Detect model type from checkpoint
    model_state_dict = checkpoint['model_state_dict']
    is_superclass = any('superclass_head' in key for key in model_state_dict.keys())
    
    if is_superclass:
        print("🔍 Detected superclass model, loading in superclass_only mode...")
        model = ModifiedResNet50(mode='superclass_only')
        # Load superclass mapping if available
        if 'fine_to_coarse_mapping' in checkpoint:
            fine_to_coarse = checkpoint['fine_to_coarse_mapping']
        else:
            # Load default mapping
            fine_to_coarse, coarse_names = load_cifar100_superclass_mapping()
        
        # Create superclass dataset wrapper for testing
        test_dataset = SuperclassDataset(original_test_dataset, fine_to_coarse)
        print(f"Using superclass labels (0-19) for evaluation")
    else:
        print("🔍 Detected subclass model, loading in subclass_only mode...")
        model = ModifiedResNet50(mode='subclass_only')
        test_dataset = original_test_dataset
    
    # Create DataLoader with the correct dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # 4. Make predictions on test set
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():  # Don't compute gradients
        for images, targets in tqdm(test_loader, desc="Testing"):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass to get predictions
            outputs = model(images)
            probabilities = torch.softmax(outputs['logits'], dim=1)
            predicted = probabilities.argmax(dim=1)
            
            # Collect results
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.max(dim=1)[0].cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # 5. Calculate evaluation metrics
    # 5.1 Calculate accuracy
    accuracy = 100. * (all_predictions == all_targets).mean()
    print('\n' + '='*50)
    print(f'Test set accuracy: {accuracy:.2f}%')
    print('='*50)
    
    # 5.2 Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # 6. Save results
    if args.save_dir:
        save_dir = Path(args.save_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 6.1 Save prediction results
        predictions_df = pd.DataFrame({
            'true_label': all_targets,
            'predicted_label': all_predictions,
            'confidence': all_probabilities,
            'correct': all_targets == all_predictions
        })
        save_predictions(
            predictions_df,
            save_dir / 'predictions' / f'predictions_{timestamp}.csv'
        )
        
        # 6.2 Save confusion matrix visualization
        # Get class names
        if is_superclass:
            # Use superclass names for superclass model
            class_names = coarse_names if 'coarse_names' in locals() else [str(i) for i in range(20)]
        else:
            # Use subclass names for subclass model
            class_names = test_dataset.classes if hasattr(test_dataset, 'classes') else \
                         [str(i) for i in range(100)]
        
        plot_confusion_matrix(
            cm,
            class_names,
            save_dir / 'visualizations' / f'confusion_matrix_{timestamp}.png'
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model performance on CIFAR-100 test set')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default='results',
                      help='Directory path to save results')
    args = parser.parse_args()
    main(args) 