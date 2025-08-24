#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Tuple, List, Optional
import os

# Import project modules
from models.resnet import ModifiedResNet50

class IntelligentHierarchicalClassifier(nn.Module):
    """
    Intelligent hierarchical classifier - supports true hierarchical predictions
    without forcing superclass-to-subclass mapping.
    
    Core idea:
    - High confidence: return subclass prediction
    - Low confidence: return superclass prediction (acknowledge uncertainty)
    - Metrics: hierarchical accuracy (exact + partial)
    """
    
    def __init__(self, subclass_checkpoint_path: str, superclass_checkpoint_path: str):
        """Initialize classifier.
        
        Args:
            subclass_checkpoint_path: Subclass classifier checkpoint path
            superclass_checkpoint_path: Superclass classifier checkpoint path
        """
        super().__init__()
        
        # Device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load subclass model
        self._load_subclass_model(subclass_checkpoint_path)
        
        # Load superclass model
        self._load_superclass_model(superclass_checkpoint_path)
        
        # Load class mappings
        self._load_class_mappings()
        
        print(f"Intelligent hierarchical classifier initialized. Device: {self.device}")
        print(f"Subclass count: 100, Superclass count: {len(self.coarse_names)}")
        print("Key feature: intelligent hierarchical prediction without forced mapping")
    
    def _load_subclass_model(self, checkpoint_path: str):
        """Load subclass classifier model."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Subclass checkpoint not found: {checkpoint_path}")
        
        print(f"Loading subclass model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create subclass model
        self.subclass_model = ModifiedResNet50(mode='subclass_only', dropout_rate=0.4)
        self.subclass_model.load_state_dict(checkpoint['model_state_dict'])
        self.subclass_model.to(self.device)
        self.subclass_model.eval()
        
        # Extract backbone, dropout, and subclass head
        self.backbone = self.subclass_model.features
        self.dropout = self.subclass_model.dropout
        self.subclass_head = self.subclass_model.subclass_head
        
        print(f"Subclass model loaded. Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    def _load_superclass_model(self, checkpoint_path: str):
        """Load superclass classifier model."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Superclass checkpoint not found: {checkpoint_path}")
        
        print(f"Loading superclass model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create superclass model
        superclass_model = ModifiedResNet50(mode='superclass_only', dropout_rate=0.4)
        superclass_model.load_state_dict(checkpoint['model_state_dict'])
        superclass_model.to(self.device)
        superclass_model.eval()
        
        # Extract superclass head and dropout
        self.superclass_dropout = superclass_model.dropout
        self.superclass_head = superclass_model.superclass_head
        
        # Save mapping info
        self.fine_to_coarse = checkpoint.get('fine_to_coarse_mapping', {})
        self.coarse_names = checkpoint.get('coarse_names', [])
        
        print(f"Superclass model loaded. Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    def _load_class_mappings(self):
        """Load class mapping information."""
        # If missing in checkpoint, load from CIFAR-100 dataset
        if not self.fine_to_coarse or not self.coarse_names:
            try:
                meta_file = './data/cifar-100-python/meta'
                with open(meta_file, 'rb') as f:
                    meta = pickle.load(f)
                    fine_names = meta['fine_label_names']
                    self.coarse_names = meta['coarse_label_names']
                
                train_file = './data/cifar-100-python/train'
                with open(train_file, 'rb') as f:
                    train_data = pickle.load(f, encoding='bytes')
                    fine_labels = train_data[b'fine_labels']
                    coarse_labels = train_data[b'coarse_labels']
                
                # Create mapping
                self.fine_to_coarse = {}
                for fine_idx, coarse_idx in zip(fine_labels, coarse_labels):
                    if fine_idx not in self.fine_to_coarse:
                        self.fine_to_coarse[fine_idx] = coarse_idx
                
                print("Loaded class mapping from CIFAR-100 dataset")
                
            except Exception as e:
                print(f"Warning: Failed to load class mapping: {e}")
                self.fine_to_coarse = {}
                self.coarse_names = []
        
        # Create reverse mapping (superclass -> fines) for analysis only
        self.coarse_to_fine = {}
        for fine_idx, coarse_idx in self.fine_to_coarse.items():
            if coarse_idx not in self.coarse_to_fine:
                self.coarse_to_fine[coarse_idx] = []
            self.coarse_to_fine[coarse_idx].append(fine_idx)
    
    def intelligent_hierarchical_predict(self, x: torch.Tensor, threshold: float = 0.8) -> Dict[str, torch.Tensor]:
        """
        Intelligent hierarchical prediction (no forced mapping).
        
        Args:
            x: Input images [batch_size, 3, 32, 32]
            threshold: Confidence threshold
        
        Returns:
            Dictionary with:
            - features
            - sub_logits, super_logits
            - sub_probs, super_probs
            - sub_confidence, super_confidence
            - use_subclass (bool mask)
            - sub_predictions, super_predictions
            - decision_confidence
        """
        # Feature extraction
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # [batch_size, 2048]
        # Apply dropout (same as baseline)
        features = self.dropout(features)
        
        # Subclass prediction
        sub_logits = self.subclass_head(features)
        sub_probs = F.softmax(sub_logits, dim=1)
        sub_confidence = sub_probs.max(dim=1)[0]
        sub_predictions = sub_probs.argmax(dim=1)
        
        # Superclass prediction (with its dropout)
        super_features = self.superclass_dropout(features)
        super_logits = self.superclass_head(super_features)
        super_probs = F.softmax(super_logits, dim=1)
        super_confidence = super_probs.max(dim=1)[0]
        super_predictions = super_probs.argmax(dim=1)
        
        # Decision: based on subclass confidence
        use_subclass = sub_confidence > threshold
        
        # Decision confidence
        decision_confidence = torch.where(use_subclass, sub_confidence, super_confidence)
        
        return {
            'features': features,
            'sub_logits': sub_logits,
            'super_logits': super_logits,
            'sub_probs': sub_probs,
            'super_probs': super_probs,
            'sub_confidence': sub_confidence,
            'super_confidence': super_confidence,
            'use_subclass': use_subclass,
            'sub_predictions': sub_predictions,
            'super_predictions': super_predictions,
            'decision_confidence': decision_confidence
        }
    
    def evaluate_hierarchical(self, predictions: Dict[str, torch.Tensor], 
                            targets: torch.Tensor) -> Dict[str, float]:
        """Hierarchical evaluation (exact + partial correctness).
        
        Args:
            predictions: Prediction dict
            targets: Ground truth labels [batch_size]
        
        Returns:
            Metrics dict with exact_accuracy, partial_accuracy,
            hierarchical_accuracy, intelligent_degradation_rate,
            subclass_usage, superclass_usage
        """
        use_subclass = predictions['use_subclass']
        sub_predictions = predictions['sub_predictions']
        super_predictions = predictions['super_predictions']
        
        batch_size = targets.size(0)
        
        # Exact correct (subclass predictions correct)
        exact_correct = torch.zeros(batch_size, dtype=torch.bool)
        exact_correct[use_subclass] = (sub_predictions[use_subclass] == targets[use_subclass])
        
        # Partial correct (superclass prediction correct when used)
        partial_correct = torch.zeros(batch_size, dtype=torch.bool)
        superclass_mask = ~use_subclass
        
        if superclass_mask.sum() > 0:
            for i in range(batch_size):
                if not use_subclass[i]:
                    true_super = self.fine_to_coarse.get(targets[i].item(), -1)
                    pred_super = super_predictions[i].item()
                    if true_super == pred_super:
                        partial_correct[i] = True
        
        # Accuracies
        exact_accuracy = exact_correct.float().mean().item()
        partial_accuracy = partial_correct.float().mean().item()
        hierarchical_accuracy = (exact_correct | partial_correct).float().mean().item()
        
        # Intelligent degradation rate
        if superclass_mask.sum() > 0:
            intelligent_degradation_rate = partial_correct[superclass_mask].float().mean().item()
        else:
            intelligent_degradation_rate = 0.0
        
        # Usage stats
        subclass_usage = use_subclass.float().mean().item()
        superclass_usage = 1 - subclass_usage
        
        return {
            'exact_accuracy': exact_accuracy,
            'partial_accuracy': partial_accuracy,
            'hierarchical_accuracy': hierarchical_accuracy,
            'intelligent_degradation_rate': intelligent_degradation_rate,
            'subclass_usage': subclass_usage,
            'superclass_usage': superclass_usage
        }
    
    def evaluate_threshold_hierarchical(self, dataloader, thresholds: List[float]) -> Dict[float, Dict[str, float]]:
        """Evaluate hierarchical performance across thresholds."""
        self.eval()
        results = {}
        
        # Collect predictions once
        all_targets = []
        all_sub_predictions = []
        all_super_predictions = []
        all_sub_confidences = []
        all_super_confidences = []
        
        print("Collecting predictions...")
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.intelligent_hierarchical_predict(images, threshold=0.0)
                
                all_targets.append(targets.cpu())
                all_sub_predictions.append(outputs['sub_predictions'].cpu())
                all_super_predictions.append(outputs['super_predictions'].cpu())
                all_sub_confidences.append(outputs['sub_confidence'].cpu())
                all_super_confidences.append(outputs['super_confidence'].cpu())
        
        # Merge
        all_targets = torch.cat(all_targets)
        all_sub_predictions = torch.cat(all_sub_predictions)
        all_super_predictions = torch.cat(all_super_predictions)
        all_sub_confidences = torch.cat(all_sub_confidences)
        all_super_confidences = torch.cat(all_super_confidences)
        
        print(f"Evaluating {len(thresholds)} thresholds...")
        
        # Evaluate each threshold
        for threshold in thresholds:
            # Decision mask
            use_subclass = all_sub_confidences > threshold
            
            # Construct prediction dict
            predictions = {
                'use_subclass': use_subclass,
                'sub_predictions': all_sub_predictions,
                'super_predictions': all_super_predictions,
                'sub_confidence': all_sub_confidences,
                'super_confidence': all_super_confidences
            }
            
            # Hierarchical evaluation
            hierarchical_metrics = self.evaluate_hierarchical(predictions, all_targets)
            
            # Traditional accuracy for reference (subclass only)
            traditional_accuracy = (all_sub_predictions == all_targets).float().mean().item()
            
            # Average decision confidence
            decision_confidences = torch.where(use_subclass, all_sub_confidences, all_super_confidences)
            avg_confidence = decision_confidences.mean().item()
            
            # Aggregate
            results[threshold] = {
                **hierarchical_metrics,
                'traditional_accuracy': traditional_accuracy,
                'avg_confidence': avg_confidence
            }
            
            print(f"Threshold {threshold:.2f}: hierarchical_acc {hierarchical_metrics['hierarchical_accuracy']:.4f}, "
                  f"degradation {hierarchical_metrics['intelligent_degradation_rate']:.4f}, "
                  f"subclass_usage {hierarchical_metrics['subclass_usage']:.4f}")
        
        return results
    
    def analyze_confidence_calibration(self, dataloader) -> Dict[str, List[Dict]]:
        """Analyze confidence calibration."""
        self.eval()
        
        all_sub_confidences = []
        all_super_confidences = []
        all_sub_predictions = []
        all_super_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.intelligent_hierarchical_predict(images, threshold=0.0)
                
                all_sub_confidences.append(outputs['sub_confidence'].cpu())
                all_super_confidences.append(outputs['super_confidence'].cpu())
                all_sub_predictions.append(outputs['sub_predictions'].cpu())
                all_super_predictions.append(outputs['super_predictions'].cpu())
                all_targets.append(targets.cpu())
        
        all_sub_confidences = torch.cat(all_sub_confidences)
        all_super_confidences = torch.cat(all_super_confidences)
        all_sub_predictions = torch.cat(all_sub_predictions)
        all_super_predictions = torch.cat(all_super_predictions)
        all_targets = torch.cat(all_targets)
        
        # Subclass calibration
        sub_calibration = self._analyze_confidence_bins(
            all_sub_confidences, all_sub_predictions, all_targets, "subclass"
        )
        
        # Superclass calibration
        super_calibration = self._analyze_confidence_bins(
            all_super_confidences, all_super_predictions, all_targets, "superclass"
        )
        
        return {
            'subclass_calibration': sub_calibration,
            'superclass_calibration': super_calibration
        }
    
    def _analyze_confidence_bins(self, confidences: torch.Tensor, predictions: torch.Tensor, 
                               targets: torch.Tensor, prediction_type: str) -> List[Dict]:
        """Analyze calibration across confidence bins."""
        confidence_bins = np.linspace(0, 1, 11)
        calibration_results = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
            if bin_mask.sum() > 0:
                if prediction_type == "subclass":
                    bin_accuracy = (predictions[bin_mask] == targets[bin_mask]).float().mean().item()
                else:  # superclass
                    bin_accuracy = 0.0
                    correct_count = 0
                    total_count = bin_mask.sum().item()
                    
                    for j in range(len(targets)):
                        if bin_mask[j]:
                            true_super = self.fine_to_coarse.get(targets[j].item(), -1)
                            pred_super = predictions[j].item()
                            if true_super == pred_super:
                                correct_count += 1
                    
                    bin_accuracy = correct_count / total_count if total_count > 0 else 0.0
                
                bin_confidence = confidences[bin_mask].mean().item()
                calibration_results.append({
                    'bin_range': f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}",
                    'avg_confidence': bin_confidence,
                    'actual_accuracy': bin_accuracy,
                    'sample_count': bin_mask.sum().item(),
                    'calibration_error': abs(bin_confidence - bin_accuracy)
                })
        
        return calibration_results
    
    def forward(self, x: torch.Tensor, threshold: float = 0.8) -> Dict[str, torch.Tensor]:
        """Forward pass wrapper."""
        return self.intelligent_hierarchical_predict(x, threshold)
    
    def save_model(self, save_path: str):
        """Save complete intelligent hierarchical system."""
        checkpoint = {
            'backbone_state_dict': self.backbone.state_dict(),
            'subclass_head_state_dict': self.subclass_head.state_dict(),
            'superclass_head_state_dict': self.superclass_head.state_dict(),
            'fine_to_coarse_mapping': self.fine_to_coarse,
            'coarse_to_fine_mapping': self.coarse_to_fine,
            'coarse_names': self.coarse_names,
            'model_type': 'IntelligentHierarchicalClassifier'
        }
        
        torch.save(checkpoint, save_path)
        print(f"Saved intelligent hierarchical system to: {save_path}")


def create_intelligent_hierarchical_classifier(
    subclass_checkpoint: str = "checkpoints/best_models/subclass_best.pth",
    superclass_checkpoint: str = "checkpoints/best_models/superclass_best.pth"
) -> IntelligentHierarchicalClassifier:
    """Factory for intelligent hierarchical classifier."""
    return IntelligentHierarchicalClassifier(subclass_checkpoint, superclass_checkpoint)




if __name__ == "__main__":
    # Quick self-test
    try:
        classifier = create_intelligent_hierarchical_classifier()
        print("Classifier created successfully!")
        
        # Forward test
        dummy_input = torch.randn(2, 3, 32, 32).to(classifier.device)
        with torch.no_grad():
            outputs = classifier(dummy_input, threshold=0.8)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Subclass predictions: {outputs['sub_predictions']}")
            print(f"Superclass predictions: {outputs['super_predictions']}")
            print(f"Subclass confidence: {outputs['sub_confidence']}")
            print(f"Superclass confidence: {outputs['super_confidence']}")
            print(f"Use subclass: {outputs['use_subclass']}")
            print("Note: No forced mapping from superclass to subclass!")
        
    except Exception as e:
        print(f"Self-test failed: {e}")
        print("Ensure subclass and superclass checkpoints exist.")