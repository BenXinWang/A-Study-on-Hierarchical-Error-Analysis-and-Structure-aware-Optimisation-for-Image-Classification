import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer import BaselineTrainer
from models.resnet import ModifiedResNet50
import argparse

def main(args):
    # Training configuration
    config = {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,      # Use smaller initial learning rate
        'backbone_lr': args.backbone_lr,       # Backbone learning rate
        'weight_decay': 5e-4,      # SGD weight decay
        'epochs': args.epochs,
        'num_workers': 4,
        'optimizer_type': 'sgd',   # Use SGD optimizer
        'momentum': 0.9,          # Momentum parameter
        'nesterov': True,          # Use Nesterov momentum
        'lr_scheduler': 'cosine',  # Use cosine learning rate scheduler
        'min_lr': 1e-6,           # Minimum learning rate for cosine scheduler
        'dropout_rate': 0.4,
        'label_smoothing': 0.1,
        'use_cutout': True,        # Enable Cutout
        'use_mixup': True,         # Enable MixUp
        'mixup_alpha': 0.4,        # MixUp alpha parameter
        'mixup_prob': 0.8,         # MixUp application probability
        'early_stopping_patience': args.patience,  # Early stopping patience
        # Cross-validation configuration
        'use_cross_validation': args.use_cv,
        'cv_fold': args.cv_fold,
        'n_cv_folds': args.n_cv_folds
    }
    
    if args.use_cv:
        print(f"🚀 Starting cross-validation training (fold {args.cv_fold + 1}/{args.n_cv_folds})")
        # For cross-validation, use separate directories but still save best model to main best_models directory
        config['save_dir'] = f"{args.save_dir}/cv_fold_{args.cv_fold}"
        config['main_save_dir'] = args.save_dir  # Keep reference to main save directory
    
    # Create model
    model = ModifiedResNet50(
        mode='subclass_only',
        dropout_rate=config['dropout_rate']  # Get dropout rate from config
    )
    
    # Create trainer
    trainer = BaselineTrainer(model, config)
    
    # Start training
    best_acc = trainer.train()
    print(f"\nTraining completed! Best validation accuracy: {best_acc*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR-100 subclass classifier')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--backbone_lr', type=float, default=0.005, help='backbone learning rate')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='output directory')
    # 交叉验证参数
    parser.add_argument('--use_cv', action='store_true', help='use cross-validation')
    parser.add_argument('--cv_fold', type=int, default=0, help='cross-validation fold (0-based)')
    parser.add_argument('--n_cv_folds', type=int, default=5, help='number of cross-validation folds')
    
    args = parser.parse_args()
    main(args) 