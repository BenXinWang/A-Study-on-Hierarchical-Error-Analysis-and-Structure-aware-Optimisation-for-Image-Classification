#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
from pathlib import Path
import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import project modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.resnet import ModifiedResNet50
from models.transforms import Cutout, MixUp

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

class SuperclassTrainerPartialFreeze:
    """Superclass head trainer - conservative partial-freeze strategy (unfreeze only Layer4)."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Training run timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load superclass mapping
        self.fine_to_coarse, self.coarse_names = load_cifar100_superclass_mapping(config['data_dir'])
        print(f"Loaded {len(self.coarse_names)} superclasses")
        
        # Model setup
        self._setup_model()
        
        # Optimizer and scheduler
        self._setup_optimizer()
        
        # Dataloaders
        self._setup_dataloaders()
        
        # Loss function - same as subclass training
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
        
        # Save directories
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Dedicated directory for this run
        self.train_dir = self.save_dir / f'superclass_{self.timestamp}'
        self.train_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard log directory
        self.log_dir = self.train_dir / 'logs'
        self.writer = SummaryWriter(self.log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Save config
        self.save_config()
        
        # Early stopping attributes
        self.patience = config.get('early_stopping_patience', 12)
        self.best_val_acc = 0.0  # FIXED: Initialize best validation accuracy
        self.patience_counter = 0
        self.early_stopping = False
        
        print(f"Superclass trainer (partial-freeze) initialized, device: {self.device}")
    
    def _setup_model(self):
        """Setup model - partial-freeze strategy."""
        # Create model in superclass mode
        self.model = ModifiedResNet50(mode='superclass_only', 
                                    dropout_rate=self.config.get('dropout_rate', 0.4))
        
        # Load pretrained backbone weights
        checkpoint_path = self.config.get('pretrained_checkpoint', 'checkpoints/best_models/subclass_best.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained model: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load only backbone weights, ignore classifier heads
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                             if k in model_dict and 'subclass_head' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} pretrained parameters")
        
        # Partial freeze: unfreeze only Layer4
        print("🧊 Applying conservative partial-freeze strategy...")
        
        # Inspect ResNet50 features
        features_children = list(self.model.features.children())
        print(f"Backbone total blocks: {len(features_children)}")
        
        # Freeze first 7 blocks (0-6): Conv1 + BN + ReLU + MaxPool + Layer1 + Layer2 + Layer3
        for i in range(7):
            layer = features_children[i]
            for param in layer.parameters():
                param.requires_grad = False
            print(f"❄️  Freeze block {i}: {layer.__class__.__name__}")
        
        # Unfreeze Layer4 (index 7)
        layer4 = features_children[7]
        for param in layer4.parameters():
            param.requires_grad = True
        print(f"🔥 Unfreeze block 7: {layer4.__class__.__name__} (Layer4)")
        
        # Pooling block (index 8) remains frozen (no parameters)
        if len(features_children) > 8:
            pool_layer = features_children[8]
            print(f"❄️  Keep block 8 frozen: {pool_layer.__class__.__name__} (no parameters)")
        
        # Parameter stats
        frozen_params = sum(p.numel() for p in self.model.features.parameters() if not p.requires_grad)
        backbone_trainable_params = sum(p.numel() for p in self.model.features.parameters() if p.requires_grad)
        head_trainable_params = sum(p.numel() for p in self.model.superclass_head.parameters() if p.requires_grad)
        total_trainable_params = backbone_trainable_params + head_trainable_params
        
        print(f"📊 Parameter stats:")
        print(f"   Frozen params: {frozen_params:,}")
        print(f"   Backbone trainable params: {backbone_trainable_params:,}")
        print(f"   Head trainable params: {head_trainable_params:,}")
        print(f"   Total trainable params: {total_trainable_params:,}")
        
        self.model.to(self.device)
    
    def _setup_optimizer(self):
        """Setup optimizer and LR scheduler - mirror subclass training hyperparameters."""
        # Trainable backbone parameters (Layer4 only)
        backbone_params = [p for p in self.model.features.parameters() if p.requires_grad]
        head_params = list(self.model.superclass_head.parameters())
        
        # Optimizer config identical to subclass
        self.optimizer = optim.SGD([
            {'params': backbone_params, 'lr': self.config.get('backbone_lr', 0.005), 'initial_lr': self.config.get('backbone_lr', 0.005)},
            {'params': head_params, 'lr': self.config.get('learning_rate', 0.05), 'initial_lr': self.config.get('learning_rate', 0.05)}
        ], momentum=self.config.get('momentum', 0.9),
           nesterov=self.config.get('nesterov', True),
           weight_decay=self.config.get('weight_decay', 5e-4))
        
        print(f"📚 Optimizer configuration (mirrored from subclass training):")
        print(f"   Backbone LR: {self.config.get('backbone_lr', 0.005)}")
        print(f"   Head LR: {self.config.get('learning_rate', 0.05)}")
        print(f"   Momentum: {self.config.get('momentum', 0.9)}")
        print(f"   Nesterov: {self.config.get('nesterov', True)}")
        print(f"   Weight decay: {self.config.get('weight_decay', 5e-4)}")
        
        # LR scheduler - identical to subclass
        scheduler_type = self.config.get('lr_scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 10),
                gamma=self.config.get('lr_gamma', 0.1)
            )
        else:
            self.scheduler = None
        
        print(f"   LR scheduler: {scheduler_type}")
        print(f"   Min LR: {self.config.get('min_lr', 1e-6)}")
    
    def _setup_dataloaders(self):
        """Setup dataloaders - reuse subclass augmentation configuration."""
        print("🔧 Setting up dataloaders (mirroring subclass augmentation strategy)...")
        
        # Basic transforms
        basic_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        
        # Cutout transform (same as subclass)
        cutout_transform = [
            Cutout(
                n_holes=2,
                length=16,
                p=0.7
            )
        ] if self.config.get('use_cutout', True) else []
        
        # Compose transform
        train_transform = transforms.Compose(
            basic_transform + [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ] + cutout_transform
        )
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Load original CIFAR-100 training set
        original_train_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=False,
            transform=None
        )
        
        # Check if cross-validation is enabled
        use_cross_validation = self.config.get('use_cross_validation', False)
        cv_fold = self.config.get('cv_fold', 0)
        n_cv_folds = self.config.get('n_cv_folds', 5)
        
        if use_cross_validation and n_cv_folds > 1:
            # Cross-validation split using sklearn KFold
            from sklearn.model_selection import KFold
            
            total_size = len(original_train_dataset)
            kfold = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            
            # Get the specific fold indices
            all_splits = list(kfold.split(range(total_size)))
            train_indices, val_indices = all_splits[cv_fold]
            
            print(f"Cross-validation fold {cv_fold + 1}/{n_cv_folds}")
            print(f"Data split: train {len(train_indices)} samples, val {len(val_indices)} samples")
        else:
            # Standard train/val split (90%/10%)
            total_size = len(original_train_dataset)
            train_size = int(0.90 * total_size)
            val_size = total_size - train_size
            
            # Fixed seed for reproducibility
            generator = torch.Generator().manual_seed(42)
            train_indices, val_indices = random_split(
                range(total_size), [train_size, val_size], generator=generator
            )
            
            print(f"Data split: train {len(train_indices)} samples, val {len(val_indices)} samples")
        
        # Train dataset
        train_dataset_with_transform = datasets.CIFAR100(
            root='./data',
            train=True,
            download=False,
            transform=train_transform
        )
        train_superclass_dataset = SuperclassDataset(train_dataset_with_transform, self.fine_to_coarse)
        train_dataset = Subset(train_superclass_dataset, train_indices)
        
        # Validation dataset
        val_dataset_with_transform = datasets.CIFAR100(
            root='./data',
            train=True,
            download=False,
            transform=val_transform
        )
        val_superclass_dataset = SuperclassDataset(val_dataset_with_transform, self.fine_to_coarse)
        val_dataset = Subset(val_superclass_dataset, val_indices)
        
        # Disable pin_memory for MPS device
        use_pin_memory = self.device.type != 'mps'
        
        # Dataloaders - same batch/loader config
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=use_pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=use_pin_memory
        )
        
        print(f"Dataloaders ready:")
        print(f"  Train: {len(train_dataset)} samples, {len(self.train_loader)} batches")
        print(f"  Val:   {len(val_dataset)} samples, {len(self.val_loader)} batches")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # MixUp (same params as subclass)
        mixup = None
        if self.config.get('use_mixup', True):
            mixup = MixUp(
                alpha=self.config.get('mixup_alpha', 0.2),
                prob=self.config.get('mixup_prob', 0.8)
            )
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            if mixup is not None:
                # Apply MixUp
                mixed_images, targets_a, targets_b, lam = mixup(images, targets)
                
                # Forward
                outputs = self.model(mixed_images)
                
                # Mixed loss
                loss = lam * self.criterion(outputs['logits'], targets_a) + \
                       (1 - lam) * self.criterion(outputs['logits'], targets_b)
                
                # For accuracy, use the label with the larger mix weight
                targets_for_acc = torch.where(
                    torch.tensor(lam >= 0.5, device=self.device).expand_as(targets_a),
                    targets_a, targets_b
                )
            else:
                # Standard training
                outputs = self.model(images)
                loss = self.criterion(outputs['logits'], targets)
                targets_for_acc = targets
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Stats
            total_loss += loss.item()
            pred = outputs['logits'].argmax(dim=1)
            correct += pred.eq(targets_for_acc).sum().item()
            total += targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), correct / total
    
    @torch.no_grad()
    def validate(self):
        """Validate model performance."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc='Validating')
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs['logits'], targets)
            
            # Stats
            total_loss += loss.item()
            pred = outputs['logits'].argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.val_loader), correct / total
    
    @torch.no_grad()
    def test_on_real_test_set(self):
        """Evaluate on the real test set."""
        print("🧪 Evaluating on the real test set...")
        
        # Test dataset
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        test_dataset = datasets.CIFAR100(
            root=self.config['data_dir'],
            train=False,
            download=False,
            transform=test_transform
        )
        
        test_superclass_dataset = SuperclassDataset(test_dataset, self.fine_to_coarse)
        test_loader = DataLoader(
            test_superclass_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(test_loader, desc='Testing')
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs['logits'], targets)
            
            total_loss += loss.item()
            pred = outputs['logits'].argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        test_loss = total_loss / len(test_loader)
        test_acc = correct / total
        
        print(f"🎯 Test set performance: Loss={test_loss:.4f}, Accuracy={test_acc*100:.2f}%")
        return test_loss, test_acc
    
    def generate_detailed_test_results(self):
        """Generate detailed test results including CSV predictions and visualizations."""
        import pandas as pd
        import numpy as np
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        print("\n📊 Generating detailed test results...")
        
        # Set up test data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        test_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            transform=transform,
            download=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=False if self.device.type == 'mps' else True
        )
        
        # Load best model
        best_model_path = self.train_dir / 'best_superclass_model.pth'
        if not best_model_path.exists():
            print(f"⚠️  Best model not found at {best_model_path}")
            return
        
        checkpoint = torch.load(best_model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Collect predictions
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Generating detailed predictions"):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Convert to superclass labels
                superclass_targets = torch.tensor([self.fine_to_coarse[t.item()] for t in targets], 
                                                device=self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs['logits'], dim=1)
                predicted = probabilities.argmax(dim=1)
                
                # Collect results
                all_targets.extend(superclass_targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.max(dim=1)[0].cpu().numpy())
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate accuracy
        accuracy = 100. * (all_predictions == all_targets).mean()
        print(f"📈 Test accuracy: {accuracy:.2f}%")
        
        # Create results directory (same format as subclass)
        results_dir = Path('results/superclass_test_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions CSV (same format as subclass)
        predictions_df = pd.DataFrame({
            'true_label': all_targets,
            'predicted_label': all_predictions,
            'confidence': all_probabilities,
            'correct': all_targets == all_predictions
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = results_dir / 'predictions' / f'predictions_{timestamp}.csv'
        csv_path.parent.mkdir(exist_ok=True)
        predictions_df.to_csv(csv_path, index=False)
        print(f"💾 Predictions saved to: {csv_path}")
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Plot confusion matrix (same format as subclass)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            xticklabels=self.coarse_names,
            yticklabels=self.coarse_names,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Superclass Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # Save confusion matrix (same format as subclass)
        viz_path = results_dir / 'visualizations' / f'confusion_matrix_{timestamp}.png'
        viz_path.parent.mkdir(exist_ok=True)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Confusion matrix saved to: {viz_path}")
        
        print("✅ Detailed test results generation completed!")
    
    def save_checkpoint(self, epoch, val_acc, test_acc=None):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'history': self.history,
            'config': self.config,
            'fine_to_coarse_mapping': self.fine_to_coarse,
            'coarse_names': self.coarse_names,
            'training_strategy': 'partial_freeze_layer4'
        }
        
        # Save per-epoch checkpoint
        torch.save(checkpoint, self.train_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save best model - FIXED: Only save if this is truly the best validation accuracy
        if not hasattr(self, 'best_val_acc_saved') or val_acc > self.best_val_acc_saved:
            self.best_val_acc_saved = val_acc
            torch.save(checkpoint, self.train_dir / 'best_superclass_model.pth')
            
            # Save to best_models folder with standardized filename
            best_models_dir = self.save_dir / 'best_models'
            best_models_dir.mkdir(exist_ok=True)
            torch.save(checkpoint, best_models_dir / 'superclass_best.pth')
            
            # For cross-validation, also save to main best_models directory
            if 'main_save_dir' in self.config and self.config['main_save_dir'] is not None:
                main_best_models_dir = Path(self.config['main_save_dir']) / 'best_models'
                main_best_models_dir.mkdir(exist_ok=True)
                cv_fold = self.config.get("cv_fold", 0)
                cv_model_path = main_best_models_dir / f'superclass_cv_fold_{cv_fold}_best.pth'
                torch.save(checkpoint, cv_model_path)
                
                # Also update the main superclass_best.pth if this is the best across all folds
                main_superclass_path = main_best_models_dir / 'superclass_best.pth'
                if not main_superclass_path.exists() or val_acc > self.best_val_acc_saved:
                    torch.save(checkpoint, main_superclass_path)
                    print(f"💾 Updated main superclass_best.pth (val_acc: {val_acc:.4f})")
                
                print(f"💾 New best model saved to: {cv_model_path} (val_acc: {val_acc:.4f})")
            else:
                print(f"💾 New best model saved to: {best_models_dir / 'superclass_best.pth'} (val_acc: {val_acc:.4f})")
    
    def save_config(self):
        """Save training configuration."""
        config_with_meta = {
            **self.config,
            'timestamp': self.timestamp,
            'device': str(self.device),
            'num_superclasses': len(self.coarse_names),
            'coarse_names': self.coarse_names,
            'training_strategy': 'partial_freeze_layer4',
            'freeze_description': 'Freeze Conv1+Layer1+Layer2+Layer3; unfreeze Layer4 + classifier head'
        }
        
        with open(self.train_dir / 'config.json', 'w') as f:
            json.dump(config_with_meta, f, indent=4)
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.history['train_loss']:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, [acc*100 for acc in self.history['train_acc']], 'b-', label='Training Accuracy')
        ax2.plot(epochs, [acc*100 for acc in self.history['val_acc']], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate curves
        if self.history['learning_rates']:
            backbone_lrs = [lr_pair[0] for lr_pair in self.history['learning_rates']]
            head_lrs = [lr_pair[1] for lr_pair in self.history['learning_rates']]
            ax3.plot(epochs, backbone_lrs, 'g-', label='Backbone LR', alpha=0.7)
            ax3.plot(epochs, head_lrs, 'orange', label='Head LR', alpha=0.7)
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True)
        
        # Validation accuracy distribution
        if len(self.history['val_acc']) > 1:
            ax4.hist(self.history['val_acc'], bins=20, alpha=0.7, color='purple')
            ax4.set_title('Validation Accuracy Distribution')
            ax4.set_xlabel('Accuracy')
            ax4.set_ylabel('Frequency')
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Save training history plot (same format as subclass)
        plots_dir = self.train_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / f'training_history_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training history plot saved to: {plots_dir / f'training_history_{self.timestamp}.png'}")
    
    def train(self):
        """Full training procedure."""
        best_val_acc = 0
        
        print(f"🚀 Start superclass head training (partial-freeze), total {self.config['epochs']} epochs")
        print(f"🧊 Freeze strategy: Conservative partial-freeze (unfreeze Layer4 only)")
        print(f"Number of superclasses: {len(self.coarse_names)}")
        print(f"Optimizer: {self.config.get('optimizer_type', 'sgd')}")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 每5个epoch在测试集上评估一次
            test_acc = None
            if (epoch + 1) % 5 == 0:
                _, test_acc = self.test_on_real_test_set()
            
            # 更新历史记录
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 记录学习率
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.history['learning_rates'].append(current_lrs)
            
            # 更新TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate/Backbone', current_lrs[0], epoch)
            self.writer.add_scalar('Learning_Rate/Head', current_lrs[1], epoch)
            
            if test_acc is not None:
                self.writer.add_scalar('Accuracy/Test', test_acc, epoch)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 保存检查点
            self.save_checkpoint(epoch, val_acc, test_acc)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            if test_acc is not None:
                print(f"Test Acc: {test_acc*100:.2f}%")
            print(f"Learning Rates: Backbone={current_lrs[0]:.6f}, Head={current_lrs[1]:.6f}")
            
            # Track best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"🎉 New best validation accuracy: {best_val_acc*100:.2f}%")
            
            # Early stopping check - FIXED: Use validation accuracy instead of loss
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered! No improvement in validation accuracy for {self.patience} epochs.")
                    self.early_stopping = True
                    break
        
        # Final evaluation on test set
        print("\n🏁 Training finished, running final test evaluation...")
        final_test_loss, final_test_acc = self.test_on_real_test_set()
        
        # Generate detailed test results (CSV and visualizations)
        self.generate_detailed_test_results()
        
        # Close writer and save plots
        self.writer.close()
        self.plot_training_history()
        
        # Save training log (English)
        log_path = Path('results') / 'superclass_training_log_partial_freeze.txt'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as f:
            f.write(f"Superclass head training complete (partial-freeze)\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Freeze strategy: conservative partial-freeze (Layer4 unfrozen)\n")
            f.write(f"Best validation accuracy: {best_val_acc*100:.2f}%\n")
            f.write(f"Final test accuracy: {final_test_acc*100:.2f}%\n")
            f.write(f"Epochs trained: {len(self.history['val_acc'])}\n")
            f.write(f"Early stopping: {'Yes' if self.early_stopping else 'No'}\n")
            f.write(f"Model save path (standardized): {self.save_dir / 'superclass_best_checkpoint.pth'}\n")
            f.write(f"Model save path (legacy): {self.save_dir / 'superclass_head_partial_freeze.pth'}\n")
            f.write(f"Model save path (best_models): {self.save_dir / 'best_models' / 'superclass_best.pth'}\n")
        
        print(f"\n✅ Training complete!")
        print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
        print(f"Final test accuracy: {final_test_acc*100:.2f}%")
        print(f"Model saved to (standardized): {self.save_dir / 'superclass_best_checkpoint.pth'}")
        print(f"Model saved to (legacy): {self.save_dir / 'superclass_head_partial_freeze.pth'}")
        print(f"Model saved to (best_models): {self.save_dir / 'best_models' / 'superclass_best.pth'}")
        print(f"Training log saved to: {log_path}")
        
        return best_val_acc, final_test_acc

def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-100 superclass head (partial-freeze strategy)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.05, help='head learning rate (same as subclass training)')
    parser.add_argument('--backbone_lr', type=float, default=0.005, help='backbone learning rate (same as subclass training)')
    parser.add_argument('--patience', type=int, default=12, help='early stopping patience (same as subclass training)')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='output directory')
    parser.add_argument('--pretrained_checkpoint', type=str, default='checkpoints/best_models/subclass_best.pth',
                       help='pretrained subclass model checkpoint path (for backbone)')
    # Cross-validation parameters
    parser.add_argument('--use_cv', action='store_true', help='use cross-validation')
    parser.add_argument('--cv_fold', type=int, default=0, help='cross-validation fold (0-based)')
    parser.add_argument('--n_cv_folds', type=int, default=5, help='number of cross-validation folds')
    
    args = parser.parse_args()
    
    if args.use_cv:
        print(f"🚀 Starting cross-validation training (fold {args.cv_fold + 1}/{args.n_cv_folds})")
        # Modify save directory to include cross-validation information
        save_dir = f"{args.save_dir}/cv_fold_{args.cv_fold}"
        main_save_dir = args.save_dir
    else:
        save_dir = args.save_dir
        main_save_dir = None
    
    # Training config - mirrored from subclass training
    config = {
        'data_dir': args.data_dir,
        'save_dir': save_dir,
        'main_save_dir': main_save_dir,
        'pretrained_checkpoint': args.pretrained_checkpoint,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,            
        'backbone_lr': args.backbone_lr,     
        'epochs': args.epochs,
        'num_workers': 4,                    
        'optimizer_type': 'sgd',             
        'momentum': 0.9,                     
        'nesterov': True,                    
        'weight_decay': 5e-4,                
        'lr_scheduler': 'cosine',            
        'min_lr': 1e-6,                     
        'dropout_rate': 0.4,                 
        'label_smoothing': 0.1,              
        'use_cutout': True,                  
        'use_mixup': True,                   
        'mixup_alpha': 0.2,                  
        'mixup_prob': 0.8,                   
        'early_stopping_patience': args.patience,
        # Cross-validation configuration
        'use_cross_validation': args.use_cv,
        'cv_fold': args.cv_fold,
        'n_cv_folds': args.n_cv_folds
    }
    
    print("🔥 Training superclass head with partial-freeze strategy")
    print("Strategy details:")
    print("  - Freeze: Conv1 + Layer1 + Layer2 + Layer3")
    print("  - Unfreeze: Layer4 + superclass head")
    print(f"  - Backbone LR: {args.backbone_lr}")
    print(f"  - Head LR: {args.lr}")
    print("  - Goal: balance efficiency and performance")
    
    # Create trainer and run training
    trainer = SuperclassTrainerPartialFreeze(config)
    best_val_acc, final_test_acc = trainer.train()
    
    print(f"\n🎉 Partial-freeze training finished!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Final test accuracy: {final_test_acc*100:.2f}%")

if __name__ == '__main__':
    main() 