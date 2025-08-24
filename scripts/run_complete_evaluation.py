#!/usr/bin/env python3
"""
Complete Evaluation Pipeline for Intelligent Hierarchical Classification System
Excludes training processes, includes all evaluation, analysis, and visualization steps.
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

# Add scripts to path
sys.path.append('scripts')

def print_section(title, char="="):
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f"{title}")
    print(f"{char * 60}")

def print_step(step_num, title):
    """Print a formatted step header."""
    print(f"\n[Step {step_num}] {title}")
    print("-" * 50)

def step1_model_validation():
    """Step 1: Validate model loading and basic functionality."""
    print_step(1, "Model Validation")
    
    try:
        from hierarchy.intelligent_classifier import create_intelligent_hierarchical_classifier
        import torch
        
        classifier = create_intelligent_hierarchical_classifier()
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 32).to(classifier.device)
        with torch.no_grad():
            outputs = classifier(dummy_input, threshold=0.8)
        
        print("✅ Model validation completed:")
        print(f"  - Device: {classifier.device}")
        print(f"  - Subclass count: {len(classifier.fine_to_coarse)}")
        print(f"  - Superclass count: {len(classifier.coarse_names)}")
        
        return True, {"classifier": classifier}
    
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False, None

def find_best_cv_model(model_type):
    """Find the best CV model by checking CV results files."""
    import json
    from pathlib import Path
    import glob
    
    # Look for CV results files
    if model_type == 'subclass':
        pattern = 'results/subclass_cv_training_results_*.json'
    else:  # superclass
        pattern = 'results/superclass_cv_training_results_*.json'
    
    cv_files = glob.glob(pattern)
    if not cv_files:
        print(f"⚠️  No CV results found for {model_type} model")
        return None
    
    # Get the most recent CV results file
    latest_file = max(cv_files, key=lambda x: Path(x).stat().st_mtime)
    
    try:
        with open(latest_file, 'r') as f:
            cv_results = json.load(f)
        
        best_fold = cv_results.get('best_fold', 0)
        model_path = f'checkpoints/cv_fold_{best_fold}/best_models/{model_type}_best.pth'
        
        if Path(model_path).exists():
            print(f"✅ Found best {model_type} model: {model_path} (fold {best_fold})")
            return model_path
        else:
            print(f"⚠️  Best {model_type} model not found at: {model_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error reading CV results: {e}")
        return None

def step2_subclass_error_analysis():
    """Step 2: Analyze subclass baseline error patterns."""
    print_step(2, "Subclass Error Analysis")
    
    try:
        import torch
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        import torchvision.datasets as datasets
        from models.resnet import ModifiedResNet50
        import pickle
        from pathlib import Path
        
        # Load model
        model_path = find_best_cv_model('subclass')
        if model_path is None:
            model_path = 'checkpoints/best_models/subclass_best.pth'
        
        model = ModifiedResNet50(mode='subclass_only')
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Create test dataloader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                               std=[0.2675, 0.2565, 0.2761])
        ])
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        # Collect predictions
        true_labels = []
        pred_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                logits = outputs['logits']
                _, predicted = torch.max(logits.data, 1)
                
                true_labels.extend(target.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())
        
        # Load class mappings
        # Load CIFAR-100 class mappings
        meta_file = './data/cifar-100-python/meta'
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
            fine_names = meta['fine_label_names']
            coarse_names = meta['coarse_label_names']
        
        train_file = './data/cifar-100-python/train'
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f, encoding='bytes')
            fine_labels = train_data[b'fine_labels']
            coarse_labels = train_data[b'coarse_labels']
        
        # Create fine-to-coarse mapping
        fine_to_coarse = {}
        for fine_idx, coarse_idx in zip(fine_labels, coarse_labels):
            if fine_idx not in fine_to_coarse:
                fine_to_coarse[fine_idx] = coarse_idx
        
        # Analyze errors
        from evaluation.error_analyzer import analyze_errors, perform_statistical_test
        
        # Convert numeric labels to class names for error analysis
        true_class_names = [fine_names[label] for label in true_labels]
        pred_class_names = [fine_names[label] for label in pred_labels]
        
        # Create superclass mapping for error analysis
        superclass_mapping = {}
        for fine_idx, fine_name in enumerate(fine_names):
            coarse_idx = fine_to_coarse[fine_idx]
            superclass_mapping[fine_name] = coarse_names[coarse_idx]
        
        error_details = analyze_errors(true_class_names, pred_class_names, superclass_mapping)
        test_results = perform_statistical_test(error_details['error_rate'])
        
        print("✅ Subclass error analysis completed:")
        print(f"  - Error rate: {error_details['error_rate']:.4f}%")
        print(f"  - Reasonable error rate: {error_details['sensible_rate']:.4f}%")
        print(f"  - Severe error rate: {100 - error_details['sensible_rate']:.4f}%")
        print(f"  - Statistical significance: {'Yes' if test_results['is_significant'] else 'No'} (p={test_results['p_value']:.6f})")
        
        return True, {"error_details": error_details, "test_results": test_results}
    
    except Exception as e:
        print(f"❌ Subclass error analysis failed: {e}")
        return False, None

def step4_intelligent_system_test():
    """Step 4: Test intelligent hierarchical system."""
    print_step(4, "Intelligent System Test")
    
    try:
        from hierarchy.test_intelligent_system import evaluate_intelligent_system
        
        results, best_threshold, best_hierarchical_acc = evaluate_intelligent_system()
        
        print("✅ Intelligent system test completed:")
        print(f"  - Best hierarchical accuracy: {best_hierarchical_acc:.4f}")
        print(f"  - Best threshold: {best_threshold}")
        
        return True, {"results": results, "best_threshold": best_threshold,
                     "best_hierarchical_acc": best_hierarchical_acc}
    
    except Exception as e:
        print(f"❌ Intelligent system test failed: {e}")
        return False, None

def step5_performance_comparison():
    """Step 5: Compare baseline and hierarchical system performance."""
    print_step(5, "Performance Comparison")
    
    try:
        from hierarchy.system_evaluator import SystemEvaluator, create_test_dataloader
        
        test_loader = create_test_dataloader(batch_size=64)
        evaluator = SystemEvaluator(test_loader)
        
        baseline_metrics = evaluator.evaluate_baseline()
        hierarchical_metrics = evaluator.evaluate_hierarchical(0.95)  # Use optimal threshold
        comparison_results = evaluator.compare_systems(baseline_metrics, hierarchical_metrics)
        
        print("✅ Performance comparison completed:")
        print(f"  - Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"  - Hierarchical accuracy: {hierarchical_metrics['accuracy']:.4f}")
        print(f"  - Improvement: {hierarchical_metrics['accuracy'] - baseline_metrics['accuracy']:.4f}")
        
        return True, {"baseline_metrics": baseline_metrics,
                     "hierarchical_metrics": hierarchical_metrics,
                     "comparison_results": comparison_results}
    
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False, None

def step3_error_table_generation():
    """Step 3: Generate detailed error table with reasonable/severe error categorization."""
    print_step(3, "Error Table Generation")
    
    try:
        import glob
        from pathlib import Path
        
        # Find the most recent predictions file
        predictions_pattern = 'results/subclass_test_results/predictions/predictions_*.csv'
        prediction_files = glob.glob(predictions_pattern)
        
        if not prediction_files:
            print("❌ No prediction files found!")
            return False, None
        
        # Get the most recent file
        latest_prediction_file = max(prediction_files, key=lambda x: Path(x).stat().st_mtime)
        
        # Import and run the error table generation
        from evaluation.generate_error_table import load_predictions, categorize_errors, generate_error_summary, save_error_table
        
        # Load predictions
        df = load_predictions(latest_prediction_file)
        
        # Categorize errors
        error_df = categorize_errors(df)
        
        if len(error_df) == 0:
            print("No errors to analyze!")
            return True, {"error_count": 0}
        
        # Save error table
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'results/subclass_test_results/error_analysis'
        error_table_file, summary_file = save_error_table(error_df, output_dir, timestamp)
        
        # Generate summary
        summary = generate_error_summary(error_df)
        
        print("✅ Error table generation completed:")
        print(f"  - Total errors: {summary['total_errors']}")
        print(f"  - Reasonable errors: {summary['reasonable_errors']} ({summary['reasonable_error_rate']:.2f}%)")
        print(f"  - Severe errors: {summary['severe_errors']} ({summary['severe_error_rate']:.2f}%)")
        print(f"  - Error table: {error_table_file}")
        print(f"  - Summary: {summary_file}")
        
        # Display sample error table
        print("\n📋 Sample Error Table (First 10 errors):")
        print("-" * 110)
        print(f"{'Index':<8} {'True':<18} {'Predicted':<18} {'Superclass':<45} {'Type':<30}")
        print("-" * 110)
        
        for _, row in error_df.head(10).iterrows():
            superclass_info = f"{row['true_superclass']} → {row['predicted_superclass']}"
            print(f"{row['sample_index']:<8} {row['true_class_name']:<18} {row['predicted_class_name']:<18} {superclass_info:<45} {row['error_type']:<30}")
        
        if len(error_df) > 10:
            print(f"... and {len(error_df) - 10} more errors")
        print("-" * 110)
        
        return True, {
            "error_table_file": str(error_table_file),
            "summary_file": str(summary_file),
            "error_summary": summary,
            "error_count": len(error_df)
        }
    
    except Exception as e:
        print(f"❌ Error table generation failed: {e}")
        return False, None

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Complete Evaluation Pipeline")
    parser.add_argument("--skip-steps", nargs="+", type=int, default=[],
                       help="Steps to skip (1-4)")
    args = parser.parse_args()
    
    print_section("Intelligent Hierarchical Classification System - Complete Evaluation")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Skipping steps: {args.skip_steps if args.skip_steps else 'None'}")
    
    # Define all steps
    steps = [
        ("Model Validation", step1_model_validation),
        ("Subclass Error Analysis", step2_subclass_error_analysis),
        ("Error Table Generation", step3_error_table_generation),
        ("Intelligent System Test", step4_intelligent_system_test),
        ("Performance Comparison", step5_performance_comparison)
    ]
    
    results = {}
    successful_steps = 0
    failed_steps = 0
    
    # Execute steps
    for i, (step_name, step_func) in enumerate(steps, 1):
        if i in args.skip_steps:
            print(f"\n[Step {i}] {step_name} - SKIPPED")
            continue
            
        success, step_result = step_func()
        if success:
            successful_steps += 1
            results[f"step_{i}"] = step_result
        else:
            failed_steps += 1
            results[f"step_{i}"] = None
    
    # Final summary
    print_section("Evaluation Summary")
    print(f"Total steps: {len(steps)}")
    print(f"Successful: {successful_steps}")
    print(f"Failed: {failed_steps}")
    print(f"Skipped: {len(args.skip_steps)}")
    
    if failed_steps == 0:
        print("🎉 All steps completed successfully!")
    else:
        print(f"⚠️  {failed_steps} steps failed. Check the output above for details.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    main()
