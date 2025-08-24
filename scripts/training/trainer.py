import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime

# Optional optimizer imports (kept for compatibility)
from torch.optim import RAdam

"""Training utilities for CIFAR-100 subclass head.

This module contains the BaselineTrainer used to train the subclass classifier head.
Functionality and hyperparameters remain unchanged; only logging messages and
documentation have been standardized to English, and console logs now match the
superclass trainer style (tqdm progress bars and epoch summaries).
"""

# Import Cutout transform
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.transforms import Cutout

from models.transforms import MixUp

class BaselineTrainer:
    def __init__(self, model, config):
        """Initialize trainer.

        Args:
            model: ModifiedResNet50 instance
            config: Training configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize MixUp
        self.mixup = MixUp(
            alpha=config.get('mixup_alpha', 1.0),
            prob=config.get('mixup_prob', 1.0)
        )
        self.use_mixup = config.get('use_mixup', True)
        
        # Generate timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure subclass mode
        self.model.set_mode('subclass_only')
        
        # Initialize optimizer
        optimizer_type = config.get('optimizer_type', 'sgd')
        if optimizer_type == 'sgd':
            self.optimizer = optim.SGD([
                {'params': self.model.features.parameters(), 'lr': config.get('backbone_lr', 0.01), 'initial_lr': config.get('backbone_lr', 0.01)},
                {'params': self.model.subclass_head.parameters(), 'lr': config.get('learning_rate', 0.1), 'initial_lr': config.get('learning_rate', 0.1)}
            ], momentum=config.get('momentum', 0.9),
               nesterov=config.get('nesterov', True),
               weight_decay=config.get('weight_decay', 5e-4))
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW([
                {'params': self.model.features.parameters(), 'lr': config['backbone_lr'], 'initial_lr': config['backbone_lr']},
                {'params': self.model.subclass_head.parameters(), 'lr': config['learning_rate'], 'initial_lr': config['learning_rate']}
            ], weight_decay=config['weight_decay'],
               betas=config.get('betas', (0.9, 0.999)),
               eps=config.get('eps', 1e-8))
        else:
            self.optimizer = optim.Adam([
                {'params': self.model.features.parameters(), 'lr': config['backbone_lr'], 'initial_lr': config['backbone_lr']},
                {'params': self.model.subclass_head.parameters(), 'lr': config['learning_rate'], 'initial_lr': config['learning_rate']}
            ], weight_decay=config['weight_decay'])
        
        # Initialize LR scheduler
        scheduler_type = config.get('lr_scheduler', 'step')
        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('lr_step_size', 10),
                gamma=config.get('lr_gamma', 0.8),
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config['epochs'] - config.get('warmup_epochs', 0),
                eta_min=config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'cosine_warmup':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config['epochs'] - config.get('warmup_epochs', 0),
                eta_min=config.get('min_lr', 1e-6)
            )
        
        # Loss function (label smoothing)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))
        
        # Create save directory
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dedicated directory for this run
        self.train_dir = self.save_dir / f'subclass_{self.timestamp}'
        self.train_dir.mkdir(parents=True, exist_ok=True)
        
        # Create TensorBoard log directory
        self.log_dir = self.train_dir / 'logs'
        self.writer = SummaryWriter(self.log_dir)
        
        # Initialize dataloaders
        self._setup_dataloaders()
        
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
        
        # Log configuration to TensorBoard
        self.writer.add_text('Configuration', str(self.config))
        
        # Early stopping attributes
        self.patience = config.get('early_stopping_patience', 10)
        self.best_val_acc = 0.0  # FIXED: Initialize best validation accuracy
        self.patience_counter = 0
        self.early_stopping = False
    
    def _setup_dataloaders(self):
        """Set up dataloaders with optional cross-validation support."""
        # Import Cutout
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from models.transforms import Cutout
        
        # Basic transforms
        basic_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        
        # Cutout transform (same parameters as before)
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
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Load CIFAR-100 datasets
        full_train_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )
        
        # Check if cross-validation is enabled
        use_cross_validation = self.config.get('use_cross_validation', False)
        cv_fold = self.config.get('cv_fold', 0)
        n_cv_folds = self.config.get('n_cv_folds', 5)
        
        if use_cross_validation and n_cv_folds > 1:
            # Cross-validation split using sklearn KFold
            from sklearn.model_selection import KFold
            
            total_size = len(full_train_dataset)
            kfold = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            
            # Get the specific fold indices
            all_splits = list(kfold.split(range(total_size)))
            train_indices, val_indices = all_splits[cv_fold]
            
            train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
            
            print(f"Cross-validation fold {cv_fold + 1}/{n_cv_folds}")
            print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        else:
            # Standard train/val split (90%/10%)
            train_size = int(0.9 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        
        # Load test set
        test_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )
        
        # Create dataloaders
        # Disable pin_memory for MPS device
        use_pin_memory = self.device.type != 'mps'
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=use_pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=use_pin_memory
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=use_pin_memory
        )
    
    def save_config(self):
        """Save training configuration."""
        config_path = self.train_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def train_epoch(self):
        """Train for one epoch (tqdm progress style)."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            if self.use_mixup:
                # Apply MixUp
                mixed_images, targets_a, targets_b, lam = self.mixup(images, targets)
                
                # Forward
                outputs = self.model(mixed_images)
                
                # Mixed loss
                loss = lam * self.criterion(outputs['logits'], targets_a) + \
                       (1 - lam) * self.criterion(outputs['logits'], targets_b)
                
                # For accuracy, use the label with the larger mix weight
                targets_for_acc = torch.where(torch.tensor(lam >= 0.5, device=self.device).expand_as(targets_a),
                                           targets_a, targets_b)
            else:
                # Standard training without MixUp
                outputs = self.model(images)
                loss = self.criterion(outputs['logits'], targets)
                targets_for_acc = targets
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Stats
            current_loss = loss.item()
            total_loss += current_loss
            pred = outputs['logits'].argmax(dim=1)
            correct += pred.eq(targets_for_acc).sum().item()
            total += targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
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
        
        pbar = tqdm(self.val_loader, desc=f'Validating')
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
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100.*correct/total:.2f}%'})
        
        return total_loss / len(self.val_loader), correct / total
    
    def save_checkpoint(self, epoch, val_acc):
        """Save checkpoint (with compatibility best model outputs)."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        # Save epoch checkpoint in run directory
        torch.save(checkpoint, self.train_dir / f'checkpoint_epoch_{epoch}.pth')
        
        # Save best model - FIXED: Only save if this is truly the best validation accuracy
        if not hasattr(self, 'best_val_acc_saved') or val_acc > self.best_val_acc_saved:
            self.best_val_acc_saved = val_acc
            torch.save(checkpoint, self.train_dir / 'best_model.pth')
            
            # Save to best_models folder with standardized filename
            best_models_dir = self.save_dir / 'best_models'
            best_models_dir.mkdir(exist_ok=True)
            torch.save(checkpoint, best_models_dir / 'subclass_best.pth')
            
            # For cross-validation, also save to main best_models directory
            if self.config.get('use_cross_validation', False):
                main_save_dir = self.config.get('main_save_dir', self.save_dir)
                main_best_models_dir = Path(main_save_dir) / 'best_models'
                main_best_models_dir.mkdir(exist_ok=True)
                cv_fold = self.config.get("cv_fold", 0)
                cv_model_path = main_best_models_dir / f'subclass_cv_fold_{cv_fold}_best.pth'
                torch.save(checkpoint, cv_model_path)
                print(f"💾 New best model saved to: {cv_model_path} (val_acc: {val_acc:.4f})")
            else:
                print(f"💾 New best model saved to: {best_models_dir / 'subclass_best.pth'} (val_acc: {val_acc:.4f})")
    
    def plot_training_history(self):
        """Plot training curves."""
        # Create plots directory
        plot_dir = self.train_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # Seaborn style
        sns.set_style("whitegrid")
        
        # Create 2x2 subplots to match superclass format
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. Loss curve
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Accuracy curve (percentage)
        ax2.plot(epochs, [acc*100 for acc in self.history['train_acc']], 'b-', label='Training Accuracy')
        ax2.plot(epochs, [acc*100 for acc in self.history['val_acc']], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Learning rate curves
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
        
        # 4. Validation accuracy distribution
        if len(self.history['val_acc']) > 1:
            ax4.hist(self.history['val_acc'], bins=20, alpha=0.7, color='purple')
            ax4.set_title('Validation Accuracy Distribution')
            ax4.set_xlabel('Accuracy')
            ax4.set_ylabel('Frequency')
            ax4.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(plot_dir / f'training_history_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def update_history(self, train_loss, train_acc, val_loss, val_acc):
        """Update training history."""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
        # Record current learning rates
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.history['learning_rates'].append(current_lrs)
    
    def update_tensorboard(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Update TensorBoard."""
        # Losses
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Accuracies
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Learning rates
        for i, lr in enumerate(self.history['learning_rates'][-1]):
            name = 'Backbone' if i == 0 else 'Classifier'
            self.writer.add_scalar(f'Learning_Rate/{name}', lr, epoch)
        
        # Parameter and gradient histograms
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    def train(self):
        """Full training loop (console output standardized)."""
        best_val_acc = 0
        warmup_epochs = self.config.get('warmup_epochs', 0)
        
        print(f"🚀 Start subclass head training for {self.config['epochs']} epochs")
        print(f"Optimizer: {self.config.get('optimizer_type', 'sgd')}")

        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # LR warmup
            if epoch < warmup_epochs:
                lr_scale = min(1., float(epoch + 1) / warmup_epochs)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr_scale * pg['initial_lr']
            
            # 1. Train one epoch
            train_loss, train_acc = self.train_epoch()
            
            # 2. Validate
            val_loss, val_acc = self.validate()
            
            # 3. Update history
            self.update_history(train_loss, train_acc, val_loss, val_acc)
            
            # 4. Update TensorBoard
            self.update_tensorboard(epoch, train_loss, train_acc, val_loss, val_acc)
            
            # 5. Step scheduler (after warmup)
            if epoch >= warmup_epochs and hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # 6. Save checkpoints and plots
            self.save_checkpoint(epoch, val_acc)
            self.plot_training_history()
            
            # Print epoch summary
            print(f"\nResults:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            current_lrs = self.history['learning_rates'][-1]
            print(f"Learning Rates: Backbone={current_lrs[0]:.6f}, Head={current_lrs[1]:.6f}")
            
            # Track best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc*100:.2f}%")
            
            # Early stopping check - FIXED: Use validation accuracy instead of loss
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered! No improvement in validation accuracy for {self.patience} epochs.")
                    self.early_stopping = True
                    break
        
        # Close writer
        self.writer.close()
        
        # Save training curves once more at the end
        self.plot_training_history()

        # Write standardized training log file
        log_path = Path('results') / 'subclass_training_log.txt'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            f.write("Subclass head training complete\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Best validation accuracy: {best_val_acc*100:.2f}%\n")
            f.write(f"Epochs trained: {len(self.history['val_acc'])}\n")
            f.write(f"Early stopping: {'Yes' if self.early_stopping else 'No'}\n")
            f.write(f"Model save path (standardized): {self.save_dir / 'best_models' / 'subclass_best.pth'}\n")
            f.write(f"Model save path (run directory): {self.train_dir / 'best_model.pth'}\n")
            f.write(f"Model save path (best_models): {self.save_dir / 'best_models' / 'subclass_best.pth'}\n")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc*100:.2f}%")
        print(f"Model saved to (standardized): {self.save_dir / 'best_models' / 'subclass_best.pth'}")
        print(f"Model saved to (run directory): {self.train_dir / 'best_model.pth'}")
        print(f"Model saved to (best_models): {self.save_dir / 'best_models' / 'subclass_best.pth'}")
        print(f"Training log saved to: {log_path}")
        
        # Run final test evaluation
        print(f"\n🏁 Training finished, running final test evaluation...")
        test_acc = self.evaluate_test_set()
        print(f"🎯 Test set performance: Accuracy={test_acc*100:.2f}%")
        
        return best_val_acc
    
    def evaluate_test_set(self):
        """Evaluate the model on the test set."""
        print(f"🧪 Evaluating on the real test set...")
        
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for batch_idx, (images, targets) in enumerate(pbar):
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs['logits'], targets)
                
                test_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        test_acc = correct / total
        test_loss = test_loss / len(self.test_loader)
        
        print(f"📈 Test accuracy: {test_acc*100:.2f}%")
        
        return test_acc 