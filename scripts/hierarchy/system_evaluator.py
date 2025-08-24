#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import Dict, List, Tuple
from scipy import stats
from tqdm import tqdm

# Import modules
from models.resnet import ModifiedResNet50

from evaluation.error_analyzer import analyze_errors  # Reuse existing error analysis
from hierarchy.intelligent_classifier import IntelligentHierarchicalClassifier, create_intelligent_hierarchical_classifier

class SystemEvaluator:
    """Hierarchical system evaluator - compares baseline and hierarchical systems."""
    
    def __init__(self, test_loader: data.DataLoader):
        """Initialize evaluator.
        
        Args:
            test_loader: Test dataloader
        """
        self.test_loader = test_loader
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load baseline model
        self.baseline_model = self._load_baseline_model()
        
        # Load hierarchical classifier
        self.hierarchical_classifier = self._load_hierarchical_classifier()
        
        print(f"System evaluator initialized. Device: {self.device}")
        print(f"Test set size: {len(test_loader.dataset)}")
    
    def _load_baseline_model(self) -> ModifiedResNet50:
        """Load baseline subclass model"""
        print("Loading baseline model...")
        model = ModifiedResNet50(mode='subclass_only', dropout_rate=0.4)
        checkpoint = torch.load('checkpoints/best_models/subclass_best.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        print(f"Baseline model loaded. Val accuracy: {checkpoint.get('val_acc', 'N/A')}")
        return model
    
    def _load_hierarchical_classifier(self) -> IntelligentHierarchicalClassifier:
        """Load intelligent hierarchical classifier"""
        print("Loading intelligent hierarchical classifier...")
        classifier = create_intelligent_hierarchical_classifier(
            'checkpoints/best_models/subclass_best.pth',
            'checkpoints/best_models/superclass_best.pth'
        )
        return classifier
    
    @torch.no_grad()
    def evaluate_baseline(self) -> Dict[str, float]:
        """Evaluate baseline model performance."""
        print("Evaluating baseline model...")
        self.baseline_model.eval()
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        for images, targets in tqdm(self.test_loader, desc="Baseline Evaluation"):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            outputs = self.baseline_model(images)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            
            # Collect results
            predictions = probs.argmax(dim=1)
            confidences = probs.max(dim=1)[0]
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_confidences.append(confidences.cpu())
        
        # Merge results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_confidences = torch.cat(all_confidences)
        
        # Metrics
        accuracy = (all_predictions == all_targets).float().mean().item()
        
        # Severe error rate
        severe_error_rate = self._calculate_severe_error_rate(
            all_targets.numpy(), all_predictions.numpy()
        )
        
        # Confidence stats
        avg_confidence = all_confidences.mean().item()
        confidence_std = all_confidences.std().item()
        
        baseline_metrics = {
            'accuracy': accuracy,
            'severe_error_rate': severe_error_rate,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'predictions': all_predictions.numpy(),
            'targets': all_targets.numpy(),
            'confidences': all_confidences.numpy()
        }
        
        print(f"Baseline performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Severe error rate: {severe_error_rate:.4f}")
        print(f"  Avg confidence: {avg_confidence:.4f}")
        
        return baseline_metrics
    
    @torch.no_grad()
    def evaluate_hierarchical(self, threshold: float) -> Dict[str, float]:
        """Evaluate intelligent hierarchical system performance using hierarchical accuracy.
        
        Args:
            threshold: confidence threshold
        """
        print(f"Evaluating intelligent hierarchical system (threshold={threshold})...")
        self.hierarchical_classifier.eval()
        
        # Use the intelligent classifier's evaluation method
        test_loader = create_test_dataloader(batch_size=64)
        results = self.hierarchical_classifier.evaluate_threshold_hierarchical(test_loader, [threshold])
        
        # Get the metrics for the specified threshold
        metrics = results[threshold]
        
        # Calculate severe error rate (1 - hierarchical accuracy)
        severe_error_rate = 1 - metrics['hierarchical_accuracy']
        
        hierarchical_metrics = {
            'accuracy': metrics['hierarchical_accuracy'],  # This is hierarchical accuracy
            'severe_error_rate': severe_error_rate,
            'subclass_usage': metrics['subclass_usage'],
            'subclass_accuracy': metrics['exact_accuracy'],
            'superclass_accuracy': metrics['partial_accuracy'],
            'avg_decision_confidence': metrics.get('avg_confidence', 0.0),
            'threshold': threshold,
            'traditional_accuracy': metrics['traditional_accuracy'],
            'predictions': [],  # Placeholder for compatibility
            'targets': [],      # Placeholder for compatibility
            'use_subclass': []  # Placeholder for compatibility
        }
        
        print(f"Intelligent hierarchical performance:")
        print(f"  Hierarchical accuracy: {metrics['hierarchical_accuracy']:.4f}")
        print(f"  Traditional accuracy: {metrics['traditional_accuracy']:.4f}")
        print(f"  Exact accuracy: {metrics['exact_accuracy']:.4f}")
        print(f"  Partial accuracy: {metrics['partial_accuracy']:.4f}")
        print(f"  Subclass usage: {metrics['subclass_usage']:.4f}")
        print(f"  Intelligent degradation rate: {metrics['intelligent_degradation_rate']:.4f}")
        
        return hierarchical_metrics
    
    def _calculate_severe_error_rate(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Calculate severe error rate (cross-superclass errors)
        
        Args:
            true_labels: ground-truth labels
            pred_labels: predicted labels
            
        Returns:
            severe error rate
        """
        # Use mapping from hierarchical classifier
        fine_to_coarse = self.hierarchical_classifier.fine_to_coarse
        
        if len(fine_to_coarse) == 0:
            return 0.0
        
        total_errors = 0
        severe_errors = 0
        
        for true_label, pred_label in zip(true_labels, pred_labels):
            if true_label != pred_label:
                total_errors += 1
                # Check cross-superclass error
                true_super = fine_to_coarse.get(true_label, -1)
                pred_super = fine_to_coarse.get(pred_label, -1)
                
                if true_super != pred_super:
                    severe_errors += 1
        
        return severe_errors / len(true_labels) if len(true_labels) > 0 else 0.0
    
    def compare_systems(self, baseline_metrics: Dict, hierarchical_metrics: Dict) -> Dict:
        """
        Compare performance between baseline and hierarchical systems
        
        Args:
            baseline_metrics: baseline metrics
            hierarchical_metrics: hierarchical metrics
            
        Returns:
            comparison results
        """
        print("\nComparing system performance...")
        
        # Calculate improvements
        accuracy_improvement = hierarchical_metrics['accuracy'] - baseline_metrics['accuracy']
        severe_error_improvement = baseline_metrics['severe_error_rate'] - hierarchical_metrics['severe_error_rate']
        
        # Calculate relative improvements
        accuracy_rel_improvement = (accuracy_improvement / baseline_metrics['accuracy']) * 100
        severe_error_rel_improvement = (severe_error_improvement / baseline_metrics['severe_error_rate']) * 100 if baseline_metrics['severe_error_rate'] > 0 else 0
        
        # Statistical significance test
        # Use McNemar test to compare classifier performance
        baseline_correct = (baseline_metrics['predictions'] == baseline_metrics['targets'])
        hierarchical_correct = (hierarchical_metrics['predictions'] == hierarchical_metrics['targets'])
        
        # Create 2x2 contingency table
        both_correct = np.sum(baseline_correct & hierarchical_correct)
        baseline_only = np.sum(baseline_correct & ~hierarchical_correct)
        hierarchical_only = np.sum(~baseline_correct & hierarchical_correct)
        both_wrong = np.sum(~baseline_correct & ~hierarchical_correct)
        
        # McNemar test
        if baseline_only + hierarchical_only > 0:
            mcnemar_stat = (abs(baseline_only - hierarchical_only) - 1) ** 2 / (baseline_only + hierarchical_only)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_value = 1.0
        
        comparison_results = {
            'baseline_accuracy': baseline_metrics['accuracy'],
            'hierarchical_accuracy': hierarchical_metrics['accuracy'],
            'accuracy_improvement': accuracy_improvement,
            'accuracy_rel_improvement': accuracy_rel_improvement,
            'baseline_severe_error_rate': baseline_metrics['severe_error_rate'],
            'hierarchical_severe_error_rate': hierarchical_metrics['severe_error_rate'],
            'severe_error_improvement': severe_error_improvement,
            'severe_error_rel_improvement': severe_error_rel_improvement,
            'mcnemar_statistic': mcnemar_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'confusion_matrix': {
                'both_correct': both_correct,
                'baseline_only': baseline_only,
                'hierarchical_only': hierarchical_only,
                'both_wrong': both_wrong
            },
            'threshold': hierarchical_metrics['threshold'],
            'subclass_usage': hierarchical_metrics['subclass_usage']
        }
        
        # Print comparison results
        print(f"Performance comparison:")
        print(f"{'='*50}")
        print(f"Accuracy:")
        print(f"  Baseline: {baseline_metrics['accuracy']:.4f}")
        print(f"  Hierarchical: {hierarchical_metrics['accuracy']:.4f}")
        print(f"  Delta: {accuracy_improvement:+.4f} ({accuracy_rel_improvement:+.2f}%)")
        print(f"")
        print(f"Severe error rate:")
        print(f"  Baseline: {baseline_metrics['severe_error_rate']:.4f}")
        print(f"  Hierarchical: {hierarchical_metrics['severe_error_rate']:.4f}")
        print(f"  Delta: {severe_error_improvement:+.4f} ({severe_error_rel_improvement:+.2f}%)")
        print(f"")
        print(f"Statistical significance:")
        print(f"  McNemar statistic: {mcnemar_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant (p<0.05): {'Yes' if comparison_results['significant'] else 'No'}")
        print(f"")
        print(f"Decision distribution:")
        print(f"  Subclass usage: {hierarchical_metrics['subclass_usage']:.4f}")
        print(f"  Optimal threshold: {hierarchical_metrics['threshold']}")
        
        return comparison_results
    
    def analyze_error_patterns(self, baseline_metrics: Dict, hierarchical_metrics: Dict, save_dir: str = 'results'):
        """
        Analyze error pattern changes
        
        Args:
            baseline_metrics: baseline metrics
            hierarchical_metrics: hierarchical metrics
            save_dir: output directory
        """
        print("\nAnalyzing error patterns...")
        
        # Baseline error analysis
        baseline_errors = baseline_metrics['predictions'] != baseline_metrics['targets']
        baseline_error_indices = np.where(baseline_errors)[0]
        
        # Hierarchical system error analysis
        hierarchical_errors = hierarchical_metrics['predictions'] != hierarchical_metrics['targets']
        hierarchical_error_indices = np.where(hierarchical_errors)[0]
        
        # Error change analysis
        fixed_errors = set(baseline_error_indices) - set(hierarchical_error_indices)  # fixed by hierarchical
        new_errors = set(hierarchical_error_indices) - set(baseline_error_indices)    # new errors introduced
        persistent_errors = set(baseline_error_indices) & set(hierarchical_error_indices)  # remain errors
        
        error_analysis = {
            'total_samples': len(baseline_metrics['targets']),
            'baseline_errors': len(baseline_error_indices),
            'hierarchical_errors': len(hierarchical_error_indices),
            'fixed_errors': len(fixed_errors),
            'new_errors': len(new_errors),
            'persistent_errors': len(persistent_errors),
            'error_reduction': len(baseline_error_indices) - len(hierarchical_error_indices)
        }
        
        print(f"Error pattern analysis:")
        print(f"  Total samples: {error_analysis['total_samples']}")
        print(f"  Baseline errors: {error_analysis['baseline_errors']}")
        print(f"  Hierarchical errors: {error_analysis['hierarchical_errors']}")
        print(f"  Fixed errors: {error_analysis['fixed_errors']}")
        print(f"  New errors: {error_analysis['new_errors']}")
        print(f"  Persistent errors: {error_analysis['persistent_errors']}")
        print(f"  Net error reduction: {error_analysis['error_reduction']}")
        
        # Save error analysis results
        save_dir = Path(save_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(save_dir / f'error_pattern_analysis_{timestamp}.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_analysis = {}
            for k, v in error_analysis.items():
                if isinstance(v, (np.integer, np.floating)):
                    serializable_analysis[k] = v.item()
                else:
                    serializable_analysis[k] = v
            json.dump(serializable_analysis, f, indent=4)
        
        return error_analysis
    
    def generate_comprehensive_report(self, baseline_metrics: Dict, hierarchical_metrics: Dict, 
                                    comparison_results: Dict, save_dir: str = 'results'):
        """
        Generate a comprehensive evaluation report
        
        Args:
            baseline_metrics: baseline metrics
            hierarchical_metrics: hierarchical metrics
            comparison_results: comparison results
            save_dir: output directory
        """
        print("\nGenerating comprehensive evaluation report...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create comprehensive report
        comprehensive_report = {
            'evaluation_timestamp': timestamp,
            'test_dataset_size': len(self.test_loader.dataset),
            'baseline_performance': {
                'accuracy': baseline_metrics['accuracy'],
                'severe_error_rate': baseline_metrics['severe_error_rate'],
                'avg_confidence': baseline_metrics['avg_confidence'],
                'confidence_std': baseline_metrics['confidence_std']
            },
            'hierarchical_performance': {
                'accuracy': hierarchical_metrics['accuracy'],
                'severe_error_rate': hierarchical_metrics['severe_error_rate'],
                'subclass_usage': hierarchical_metrics['subclass_usage'],
                'superclass_usage': hierarchical_metrics['superclass_usage'],
                'sub_accuracy': hierarchical_metrics['sub_accuracy'],
                'super_accuracy': hierarchical_metrics['super_accuracy'],
                'avg_decision_confidence': hierarchical_metrics['avg_decision_confidence'],
                'optimal_threshold': hierarchical_metrics['threshold']
            },
            'performance_improvement': {
                'accuracy_absolute': comparison_results['accuracy_improvement'],
                'accuracy_relative_percent': comparison_results['accuracy_rel_improvement'],
                'severe_error_absolute': comparison_results['severe_error_improvement'],
                'severe_error_relative_percent': comparison_results['severe_error_rel_improvement']
            },
            'statistical_significance': {
                'mcnemar_statistic': comparison_results['mcnemar_statistic'],
                'p_value': comparison_results['p_value'],
                'is_significant': comparison_results['significant']
            },
            'goal_achievement': {
                'severe_error_reduction_target': 0.1,  # 10%
                'severe_error_reduction_achieved': float(comparison_results['severe_error_rel_improvement'] / 100),
                'severe_error_goal_met': bool(comparison_results['severe_error_rel_improvement'] >= 10),
                'accuracy_change_target': 0.005,  # ±0.5%
                'accuracy_change_achieved': float(abs(comparison_results['accuracy_improvement'])),
                'accuracy_goal_met': bool(abs(comparison_results['accuracy_improvement']) <= 0.005)
            }
        }
        
        # Save report
        report_path = save_dir / 'reports' / f'system_evaluation_report_{timestamp}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=4)
        
        # Generate visualizations
        self.plot_system_comparison(baseline_metrics, hierarchical_metrics, comparison_results, save_dir, timestamp)
        
        print(f"Comprehensive report saved to: {report_path}")
        
        # Print goal achievement
        print(f"\nGoal achievement:")
        print(f"{'='*50}")
        severe_reduction = comprehensive_report['goal_achievement']['severe_error_reduction_achieved']
        accuracy_change = comprehensive_report['goal_achievement']['accuracy_change_achieved']
        
        print(f"Severe error reduction:")
        print(f"  Target: ≥10%")
        print(f"  Actual: {severe_reduction*100:.2f}%")
        print(f"  Met: {'✓' if comprehensive_report['goal_achievement']['severe_error_goal_met'] else '✗'}")
        
        print(f"\nTop-1 accuracy change:")
        print(f"  Target: ±0.5%")
        print(f"  Actual: {accuracy_change:.4f} ({accuracy_change*100:.2f}%)")
        print(f"  Met: {'✓' if comprehensive_report['goal_achievement']['accuracy_goal_met'] else '✗'}")
        
        return comprehensive_report
    
    def plot_system_comparison(self, baseline_metrics: Dict, hierarchical_metrics: Dict,
                             comparison_results: Dict, save_dir: Path, timestamp: str):
        """
        Create system comparison visualizations
        
        Args:
            baseline_metrics: baseline metrics
            hierarchical_metrics: hierarchical metrics
            comparison_results: comparison results
            save_dir: output directory
            timestamp: timestamp
        """
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy and severe error rate comparison
        metrics = ['Accuracy', 'Severe Error Rate']
        baseline_values = [baseline_metrics['accuracy'], baseline_metrics['severe_error_rate']]
        hierarchical_values = [hierarchical_metrics['accuracy'], hierarchical_metrics['severe_error_rate']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        ax1.bar(x + width/2, hierarchical_values, width, label='Hierarchical', alpha=0.8)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Value')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (b_val, h_val) in enumerate(zip(baseline_values, hierarchical_values)):
            ax1.text(i - width/2, b_val + 0.01, f'{b_val:.3f}', ha='center', va='bottom')
            ax1.text(i + width/2, h_val + 0.01, f'{h_val:.3f}', ha='center', va='bottom')
        
        # 2. Improvements
        improvements = ['Accuracy\nImprovement', 'Severe Error\nReduction']
        improvement_values = [
            comparison_results['accuracy_rel_improvement'],
            comparison_results['severe_error_rel_improvement']
        ]
        colors = ['green' if x > 0 else 'red' for x in improvement_values]
        
        bars = ax2.bar(improvements, improvement_values, color=colors, alpha=0.7)
        ax2.set_ylabel('Relative Improvement (%)')
        ax2.set_title('Performance Improvements')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, val in zip(bars, improvement_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{val:+.2f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. Decision distribution
        decision_labels = ['Subclass\nDecisions', 'Superclass\nDecisions']
        decision_values = [hierarchical_metrics['subclass_usage'], hierarchical_metrics['superclass_usage']]
        
        ax3.pie(decision_values, labels=decision_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Decision Distribution\n(Threshold = {hierarchical_metrics["threshold"]})')
        
        # 4. System comparison matrix
        confusion_data = comparison_results['confusion_matrix']
        confusion_matrix = np.array([
            [confusion_data['both_correct'], confusion_data['baseline_only']],
            [confusion_data['hierarchical_only'], confusion_data['both_wrong']]
        ])
        
        im = ax4.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        ax4.set_title('System Comparison Matrix')
        ax4.set_xlabel('Baseline System')
        ax4.set_ylabel('Hierarchical System')
        
        # Set labels
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Correct', 'Wrong'])
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Correct', 'Wrong'])
        
        # Add value annotations
        for i in range(2):
            for j in range(2):
                text = ax4.text(j, i, confusion_matrix[i, j], ha="center", va="center", 
                              color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")
        
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = save_dir / 'visualizations' / f'system_comparison_{timestamp}.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"System comparison visualization saved to: {plot_path}")

def create_test_dataloader(data_dir: str = '../../data', batch_size: int = 128) -> data.DataLoader:
    """
    Create CIFAR-100 test dataloader
    
    Args:
        data_dir: data directory
        batch_size: batch size
        
    Returns:
        test dataloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # avoid multiprocessing issues
        pin_memory=True
    )
    
    return test_loader


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 hierarchical system evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--save_dir', type=str, default='results', help='output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--threshold', type=float, default=0.8, help='hierarchical system confidence threshold')
    
    args = parser.parse_args()
    
    try:
        # Create test dataloader
        print("Preparing test data...")
        test_loader = create_test_dataloader(args.data_dir, args.batch_size)
        
        # Create evaluator
        evaluator = SystemEvaluator(test_loader)
        
        # Evaluate baseline
        baseline_metrics = evaluator.evaluate_baseline()
        
        # Evaluate hierarchical
        hierarchical_metrics = evaluator.evaluate_hierarchical(args.threshold)
        
        # Compare systems
        comparison_results = evaluator.compare_systems(baseline_metrics, hierarchical_metrics)
        
        # Analyze error patterns
        error_analysis = evaluator.analyze_error_patterns(baseline_metrics, hierarchical_metrics, args.save_dir)
        
        # Generate report
        comprehensive_report = evaluator.generate_comprehensive_report(
            baseline_metrics, hierarchical_metrics, comparison_results, args.save_dir
        )
        
        print("\nSystem evaluation complete!")
        
    except Exception as e:
        print(f"System evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 