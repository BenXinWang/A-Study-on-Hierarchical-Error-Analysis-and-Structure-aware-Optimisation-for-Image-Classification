#!/usr/bin/env python3
"""
Cross-validation training script for CIFAR-100 subclass classifier.
This script runs training across multiple folds and aggregates results.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
import argparse
from datetime import datetime
import time

def run_cv_training(n_folds=3, **kwargs):
    """Run cross-validation training across multiple folds."""
    
    # Record start time for total training duration
    start_time = time.time()
    
    print(f"🚀 Starting cross-validation training with {n_folds} folds")
    print(f"Configuration: {kwargs}")
    
    cv_results = []
    best_fold = 0
    best_acc = 0.0
    
    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Training fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Build command for this fold
        cmd = [
            sys.executable, 'scripts/training/train_subclass.py',
            '--use_cv',
            '--cv_fold', str(fold),
            '--n_cv_folds', str(n_folds)
        ]
        
        # Add other arguments
        for key, value in kwargs.items():
            if key in ['epochs', 'batch_size', 'lr', 'backbone_lr', 'patience']:
                cmd.extend([f'--{key}', str(value)])
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run training for this fold
        try:
            print(f"Starting training for fold {fold + 1}...")
            
            # Create log file for this fold
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path('checkpoints/logs') / f'subclass_cv_fold_{fold}_training_log_{timestamp}.txt'
            log_file.parent.mkdir(exist_ok=True)
            
            # Run training and capture output to log file
            with open(log_file, 'w') as f:
                f.write(f"Subclass cross-validation fold {fold + 1}/{n_folds} training log\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write("="*80 + "\n\n")
                
                # Run the training process and capture output
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output to both console and log file
                for line in process.stdout:
                    print(line, end='')  # Print to console
                    f.write(line)        # Write to log file
                    f.flush()            # Ensure immediate write
                
                # Wait for process to complete
                return_code = process.wait()
                
                f.write(f"\n\nProcess completed with return code: {return_code}\n")
            
            if return_code == 0:
                print(f"Training completed successfully for fold {fold + 1}")
                print(f"Training log saved to: {log_file}")
                
                # Extract validation accuracy from log file
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    for line in log_content.split('\n'):
                        if "Best validation accuracy:" in line:
                            acc_str = line.split(":")[-1].strip().replace('%', '')
                            acc = float(acc_str)
                            cv_results.append({
                                'fold': fold,
                                'validation_accuracy': acc,
                                'status': 'success',
                                'log_file': str(log_file)
                            })
                            
                            if acc > best_acc:
                                best_acc = acc
                                best_fold = fold
                            break
                    else:
                        cv_results.append({
                            'fold': fold,
                            'validation_accuracy': None,
                            'status': 'accuracy_not_found',
                            'log_file': str(log_file)
                        })
            else:
                raise subprocess.CalledProcessError(return_code, cmd)
                
        except subprocess.CalledProcessError as e:
            print(f"Training failed for fold {fold}: {e}")
            cv_results.append({
                'fold': fold,
                'validation_accuracy': None,
                'status': 'failed',
                'error': str(e),
                'log_file': str(log_file) if 'log_file' in locals() else None
            })
    
    # Calculate statistics
    successful_results = [r for r in cv_results if r['status'] == 'success' and r['validation_accuracy'] is not None]
    
    if successful_results:
        accuracies = [r['validation_accuracy'] for r in successful_results]
        mean_acc = sum(accuracies) / len(accuracies)
        std_acc = (sum((acc - mean_acc) ** 2 for acc in accuracies) / len(accuracies)) ** 0.5
        
        cv_summary = {
            'n_folds': n_folds,
            'successful_folds': len(successful_results),
            'mean_validation_accuracy': mean_acc,
            'std_validation_accuracy': std_acc,
            'best_validation_accuracy': best_acc,
            'best_fold': best_fold,
            'fold_results': cv_results,
            'timestamp': datetime.now().isoformat(),
            'training_logs': [r.get('log_file') for r in cv_results if r.get('log_file')]
        }
    else:
        cv_summary = {
            'n_folds': n_folds,
            'successful_folds': 0,
            'error': 'No successful training runs',
            'fold_results': cv_results,
            'timestamp': datetime.now().isoformat()
        }
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f'subclass_cv_training_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    # Save training log with CV summary
    training_log_file = results_dir / 'subclass_training_log.txt'
    with open(training_log_file, 'w') as f:
        f.write("Subclass Cross-Validation Training Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total folds: {n_folds}\n")
        f.write(f"Successful folds: {len(successful_results)}\n")
        
        if successful_results:
            f.write(f"Mean validation accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
            f.write(f"Best validation accuracy: {best_acc:.2f}% (fold {best_fold + 1})\n")
            f.write(f"Best model saved to: checkpoints/cv_fold_{best_fold}/best_models/subclass_best.pth\n")
            f.write(f"Results file: {results_file}\n")
        else:
            f.write("No successful training runs\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUBCLASS CROSS-VALIDATION TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total folds: {n_folds}")
    print(f"Successful folds: {len(successful_results)}")
    
    if successful_results:
        print(f"Mean validation accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"Best validation accuracy: {best_acc:.2f}% (fold {best_fold + 1})")
        print(f"Best model saved to: checkpoints/cv_fold_{best_fold}/best_models/subclass_best.pth")
        
        # Check if CV model is better than current best model and replace if needed
        print(f"\n🔍 Checking if CV model is better than current best model...")
        current_best_path = 'checkpoints/best_models/subclass_best.pth'
        
        if Path(current_best_path).exists():
            try:
                import torch
                current_checkpoint = torch.load(current_best_path, map_location='cpu')
                current_acc = current_checkpoint.get('val_acc', 0.0)
                
                print(f"Current best model validation accuracy: {current_acc:.4f}")
                print(f"CV best model validation accuracy: {best_acc/100:.4f}")
                
                if best_acc/100 > current_acc:
                    print(f"✅ CV model is better! Replacing current best model...")
                    import shutil
                    cv_best_path = f'checkpoints/cv_fold_{best_fold}/best_models/subclass_best.pth'
                    shutil.copy2(cv_best_path, current_best_path)
                    print(f"✅ Successfully replaced {current_best_path} with CV best model")
                else:
                    print(f"ℹ️  Current best model is still better, keeping original model")
            except Exception as e:
                print(f"⚠️  Error checking model comparison: {e}")
        else:
            print(f"ℹ️  No current best model found, CV model becomes the new best")
            import shutil
            cv_best_path = f'checkpoints/cv_fold_{best_fold}/best_models/subclass_best.pth'
            shutil.copy2(cv_best_path, current_best_path)
            print(f"✅ Successfully copied CV best model to {current_best_path}")
        
        # Generate detailed test results for the best model
        print(f"\n🔍 Generating detailed test results for best model (fold {best_fold + 1})...")
        best_model_cmd = [
            sys.executable, 'scripts/evaluation/evaluate.py',
            '--checkpoint_path', f'checkpoints/cv_fold_{best_fold}/best_models/subclass_best.pth',
            '--save_dir', 'results/subclass_test_results'
        ]
        
        try:
            subprocess.run(best_model_cmd, check=True)
            print("✅ Detailed test results generated successfully!")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to generate detailed test results: {e}")
    
    # Calculate and display total training time
    end_time = time.time()
    total_training_time = end_time - start_time
    
    # Convert to hours, minutes, seconds
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print(f"\n⏱️  Total training time: {hours:02d}:{minutes:02d}:{seconds:02d} (HH:MM:SS)")
    print(f"Results saved to: {results_file}")
    print(f"Training log saved to: {training_log_file}")
    
    # Add training time to summary
    cv_summary['total_training_time_seconds'] = total_training_time
    cv_summary['total_training_time_formatted'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    return cv_summary

def main():
    parser = argparse.ArgumentParser(description='Cross-validation training for CIFAR-100 subclass classifier')
    parser.add_argument('--n_folds', type=int, default=3, help='Number of cross-validation folds')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--backbone_lr', type=float, default=0.005, help='Backbone learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Run cross-validation training
    cv_results = run_cv_training(
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        patience=args.patience
    )
    
    return cv_results

if __name__ == '__main__':
    main()