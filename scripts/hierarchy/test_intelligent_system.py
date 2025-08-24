#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import numpy as np
from datetime import datetime
import argparse

# Import intelligent hierarchical classifier
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hierarchy.intelligent_classifier import create_intelligent_hierarchical_classifier
from models.resnet import ModifiedResNet50

def create_test_loader(batch_size=128):
    """Create CIFAR-100 test dataloader"""
    # Test-time transforms (match training)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load CIFAR-100 test set
    test_dataset = datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return test_loader

def evaluate_baseline_model():
    """Evaluate baseline subclass model"""
    print("=" * 50)
    print("Evaluate baseline subclass model")
    print("=" * 50)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load baseline model
    checkpoint = torch.load("checkpoints/best_models/subclass_best.pth", map_location='cpu')
    model = ModifiedResNet50(mode='subclass_only', dropout_rate=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create test dataloader
    test_loader = create_test_loader()
    
    # Evaluate performance
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            # Handle dict/tensor outputs
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            _, predicted = torch.max(logits, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Baseline test accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_intelligent_system():
    """Evaluate intelligent hierarchical system"""
    print("=" * 50)
    print("Evaluate intelligent hierarchical system")
    print("=" * 50)
    
    # Create intelligent hierarchical classifier
    classifier = create_intelligent_hierarchical_classifier()
    
    # Create test dataloader
    test_loader = create_test_loader()
    
    # Evaluate multiple thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(f"Thresholds: {thresholds}")
    
    results = classifier.evaluate_threshold_hierarchical(test_loader, thresholds)
    
    print("\n" + "=" * 140)
    print("Intelligent system performance analysis")
    print("=" * 140)
    print(f"{'Threshold':<12} {'Hierarchical':<18} {'Traditional':<18} {'Exact':<14} {'Partial':<14} {'Subclass':<14} {'Intelligent':<18}")
    print(f"{'':<12} {'Accuracy':<18} {'Accuracy':<18} {'Correct':<14} {'Correct':<14} {'Usage':<14} {'Degradation':<18}")
    print("-" * 140)
    
    best_threshold = None
    best_hierarchical_acc = 0
    best_traditional_acc = 0
    
    for threshold in thresholds:
        metrics = results[threshold]
        
        print(f"{threshold:<12.2f} {metrics['hierarchical_accuracy']:<18.4f} "
              f"{metrics['traditional_accuracy']:<18.4f} {metrics['exact_accuracy']:<14.4f} "
              f"{metrics['partial_accuracy']:<14.4f} {metrics['subclass_usage']:<14.4f} "
              f"{metrics['intelligent_degradation_rate']:<18.4f}")
        
        if metrics['hierarchical_accuracy'] > best_hierarchical_acc:
            best_hierarchical_acc = metrics['hierarchical_accuracy']
            best_threshold = threshold
        
        if metrics['traditional_accuracy'] > best_traditional_acc:
            best_traditional_acc = metrics['traditional_accuracy']
    
    print("-" * 140)
    print(f"Best hierarchical accuracy: {best_hierarchical_acc:.4f} (thr: {best_threshold})")
    print(f"Best traditional accuracy: {best_traditional_acc:.4f}")
    
    return results, best_threshold, best_hierarchical_acc

def test_high_threshold_performance():
    """Test high-threshold performance (near pure subclass)"""
    print("=" * 50)
    print("High-threshold performance (near pure subclass)")
    print("=" * 50)
    
    classifier = create_intelligent_hierarchical_classifier()
    test_loader = create_test_loader()
    
    # High thresholds often choose subclass predictions
    high_thresholds = [0.95, 0.99, 0.999]
    results = classifier.evaluate_threshold_hierarchical(test_loader, high_thresholds)
    
    for threshold in high_thresholds:
        metrics = results[threshold]
        print(f"thr {threshold}: trad_acc {metrics['traditional_accuracy']:.4f}, "
              f"sub_usage {metrics['subclass_usage']:.4f}")
    
    return results

def compare_with_baseline():
    """Detailed comparison with baseline"""
    print("\n" + "=" * 60)
    print("Baseline comparison analysis")
    print("=" * 60)
    
    # Evaluate baseline
    baseline_acc = evaluate_baseline_model()
    
    # Evaluate intelligent system
    intelligent_results, best_threshold, best_hierarchical_acc = evaluate_intelligent_system()
    
    # Test high threshold
    high_threshold_results = test_high_threshold_performance()
    
    print("\n" + "=" * 60)
    print("Final comparison")
    print("=" * 60)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    print(f"Best hierarchical accuracy: {best_hierarchical_acc*100:.2f}% (thr: {best_threshold})")
    
    # Check traditional accuracy under high threshold
    high_threshold_acc = high_threshold_results[0.99]['traditional_accuracy'] * 100
    print(f"High-threshold (0.99) traditional accuracy: {high_threshold_acc:.2f}%")
    
    # Analyze differences
    diff_hierarchical = best_hierarchical_acc * 100 - baseline_acc
    diff_high_threshold = high_threshold_acc - baseline_acc
    
    print(f"\nPerformance differences:")
    print(f"Hierarchical vs baseline: {diff_hierarchical:+.2f}%")
    print(f"High-threshold trad acc vs baseline: {diff_high_threshold:+.2f}%")
    
    if diff_high_threshold < -1:
        print("\n⚠️  Warning: Even at high threshold, traditional accuracy is below baseline!")
        print("This may indicate issues in model loading or evaluation logic.")
    elif diff_hierarchical > 0:
        print("\n✅ Success: Hierarchical accuracy exceeds baseline!")
    else:
        print("\n❌ Issue: Hierarchical system did not surpass baseline performance.")
    
    return {
        'baseline_accuracy': baseline_acc,
        'best_hierarchical_accuracy': best_hierarchical_acc * 100,
        'best_threshold': best_threshold,
        'high_threshold_accuracy': high_threshold_acc,
        'hierarchical_improvement': diff_hierarchical,
        'high_threshold_difference': diff_high_threshold
    }

def main():
    parser = argparse.ArgumentParser(description='Test intelligent hierarchical classification system')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['baseline', 'intelligent', 'high_threshold', 'full'],
                       help='test mode')
    
    args = parser.parse_args()
    
    print(f"Intelligent hierarchical classification system test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'baseline':
        evaluate_baseline_model()
    elif args.mode == 'intelligent':
        evaluate_intelligent_system()
    elif args.mode == 'high_threshold':
        test_high_threshold_performance()
    else:  # full
        results = compare_with_baseline()
        
        # Save results
        import json
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results/intelligent_system_comparison_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main() 