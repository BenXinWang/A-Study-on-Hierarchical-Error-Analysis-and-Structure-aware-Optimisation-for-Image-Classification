#!/usr/bin/env python3
"""
Generate detailed error table for subclass predictions.
This script creates a comprehensive table of all prediction errors,
categorizing them as reasonable (same superclass) or severe (different superclass).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import json

# CIFAR-100类别名称
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# 超类映射（从CIFAR-100文档）
SUPERCLASS_MAPPING = {
    'apple': 'fruits_and_vegetables', 'aquarium_fish': 'aquatic_animals', 'baby': 'people',
    'bear': 'large_carnivores', 'beaver': 'small_mammals', 'bed': 'furniture',
    'bee': 'insects', 'beetle': 'insects', 'bicycle': 'vehicles_1',
    'bottle': 'household_items', 'bowl': 'household_items', 'boy': 'people',
    'bridge': 'large_man-made_structures', 'bus': 'vehicles_1', 'butterfly': 'insects',
    'camel': 'large_omnivores_and_herbivores', 'can': 'household_items',
    'castle': 'large_man-made_structures', 'caterpillar': 'insects', 'cattle': 'large_omnivores_and_herbivores',
    'chair': 'furniture', 'chimpanzee': 'primates', 'clock': 'household_items',
    'cloud': 'natural_scenes', 'cockroach': 'insects', 'couch': 'furniture',
    'crab': 'aquatic_animals', 'crocodile': 'reptiles', 'cup': 'household_items',
    'dinosaur': 'reptiles', 'dolphin': 'aquatic_animals', 'elephant': 'large_omnivores_and_herbivores',
    'flatfish': 'aquatic_animals', 'forest': 'natural_scenes', 'fox': 'small_mammals',
    'girl': 'people', 'hamster': 'small_mammals', 'house': 'large_man-made_structures',
    'kangaroo': 'large_omnivores_and_herbivores', 'keyboard': 'household_items',
    'lamp': 'household_items', 'lawn_mower': 'vehicles_2', 'leopard': 'large_carnivores',
    'lion': 'large_carnivores', 'lizard': 'reptiles', 'lobster': 'aquatic_animals',
    'man': 'people', 'maple_tree': 'trees', 'motorcycle': 'vehicles_2',
    'mountain': 'natural_scenes', 'mouse': 'small_mammals', 'mushroom': 'fungi',
    'oak_tree': 'trees', 'orange': 'fruits_and_vegetables', 'orchid': 'flowers',
    'otter': 'small_mammals', 'palm_tree': 'trees', 'pear': 'fruits_and_vegetables',
    'pickup_truck': 'vehicles_1', 'pine_tree': 'trees', 'plain': 'natural_scenes',
    'plate': 'household_items', 'poppy': 'flowers', 'porcupine': 'small_mammals',
    'possum': 'small_mammals', 'rabbit': 'small_mammals', 'raccoon': 'small_mammals',
    'ray': 'aquatic_animals', 'road': 'natural_scenes', 'rocket': 'vehicles_2',
    'rose': 'flowers', 'sea': 'natural_scenes', 'seal': 'aquatic_animals',
    'shark': 'aquatic_animals', 'shrew': 'small_mammals', 'skunk': 'small_mammals',
    'skyscraper': 'large_man-made_structures', 'snail': 'insects',
    'snake': 'reptiles', 'spider': 'insects', 'squirrel': 'small_mammals',
    'streetcar': 'vehicles_1', 'sunflower': 'flowers', 'sweet_pepper': 'fruits_and_vegetables',
    'table': 'furniture', 'tank': 'vehicles_2', 'telephone': 'household_items',
    'television': 'household_items', 'tiger': 'large_carnivores', 'tractor': 'vehicles_2',
    'train': 'vehicles_1', 'trout': 'aquatic_animals', 'tulip': 'flowers',
    'turtle': 'reptiles', 'wardrobe': 'furniture', 'whale': 'aquatic_animals',
    'willow_tree': 'trees', 'wolf': 'large_carnivores', 'woman': 'people',
    'worm': 'insects'
}

def load_predictions(predictions_file):
    """加载预测结果文件"""
    df = pd.read_csv(predictions_file)
    return df

def categorize_errors(df):
    """对错误进行分类，标注reasonable error和severe error"""
    
    # 只保留错误预测
    error_df = df[df['correct'] == False].copy()
    
    if len(error_df) == 0:
        print("No errors found in the predictions!")
        return pd.DataFrame()
    
    # 添加类别名称
    error_df['true_class_name'] = error_df['true_label'].apply(lambda x: CIFAR100_CLASSES[x])
    error_df['predicted_class_name'] = error_df['predicted_label'].apply(lambda x: CIFAR100_CLASSES[x])
    
    # 添加超类信息
    error_df['true_superclass'] = error_df['true_class_name'].apply(lambda x: SUPERCLASS_MAPPING[x])
    error_df['predicted_superclass'] = error_df['predicted_class_name'].apply(lambda x: SUPERCLASS_MAPPING[x])
    
    # 判断是否为reasonable error（同超类）
    error_df['same_superclass'] = error_df['true_superclass'] == error_df['predicted_superclass']
    error_df['error_type'] = error_df['same_superclass'].apply(lambda x: 'Reasonable Error' if x else 'Severe Error')
    
    # 添加样本索引（从0开始）
    error_df['sample_index'] = error_df.index
    
    # 重新排列列的顺序
    columns_order = [
        'sample_index', 'true_label', 'true_class_name', 'true_superclass',
        'predicted_label', 'predicted_class_name', 'predicted_superclass',
        'confidence', 'error_type', 'same_superclass'
    ]
    
    return error_df[columns_order]

def generate_error_summary(error_df):
    """生成错误统计摘要"""
    if len(error_df) == 0:
        return {}
    
    total_errors = len(error_df)
    reasonable_errors = len(error_df[error_df['error_type'] == 'Reasonable Error'])
    severe_errors = len(error_df[error_df['error_type'] == 'Severe Error'])
    
    summary = {
        'total_errors': total_errors,
        'reasonable_errors': reasonable_errors,
        'severe_errors': severe_errors,
        'reasonable_error_rate': (reasonable_errors / total_errors) * 100,
        'severe_error_rate': (severe_errors / total_errors) * 100
    }
    
    return summary

def save_error_table(error_df, output_dir, timestamp):
    """保存错误表格"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细错误表格
    error_table_file = output_dir / f'error_table_{timestamp}.csv'
    error_df.to_csv(error_table_file, index=False)
    print(f"✅ Error table saved to: {error_table_file}")
    
    # 保存错误摘要
    summary = generate_error_summary(error_df)
    summary_file = output_dir / f'error_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Error summary saved to: {summary_file}")
    
    return error_table_file, summary_file

def print_error_summary(summary):
    """打印错误摘要"""
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Errors: {summary['total_errors']}")
    print(f"Reasonable Errors: {summary['reasonable_errors']} ({summary['reasonable_error_rate']:.2f}%)")
    print(f"Severe Errors: {summary['severe_errors']} ({summary['severe_error_rate']:.2f}%)")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Generate detailed error table for subclass predictions')
    parser.add_argument('--predictions_file', type=str, 
                       default='results/subclass_test_results/predictions/predictions_20250824_043131.csv',
                       help='Path to predictions CSV file')
    parser.add_argument('--output_dir', type=str, 
                       default='results/subclass_test_results/error_analysis',
                       help='Output directory for error analysis')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.predictions_file).exists():
        print(f"❌ Predictions file not found: {args.predictions_file}")
        return
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载预测结果
    df = load_predictions(args.predictions_file)
    
    # 分类错误
    error_df = categorize_errors(df)
    
    if len(error_df) == 0:
        print("No errors to analyze!")
        return
    
    # 保存错误表格
    error_table_file, summary_file = save_error_table(error_df, args.output_dir, timestamp)
    
    # 打印摘要
    summary = generate_error_summary(error_df)
    print_error_summary(summary)
    
    # 显示前10个错误样本
    print("\n📋 First 10 Error Samples:")
    print("-" * 80)
    print(f"{'Index':<6} {'True':<15} {'Predicted':<15} {'Superclass':<25} {'Type':<15} {'Confidence':<10}")
    print("-" * 80)
    
    for _, row in error_df.head(10).iterrows():
        superclass_info = f"{row['true_superclass']} → {row['predicted_superclass']}"
        print(f"{row['sample_index']:<6} {row['true_class_name']:<15} {row['predicted_class_name']:<15} {superclass_info:<25} {row['error_type']:<15} {row['confidence']:.3f}")
    
    if len(error_df) > 10:
        print(f"... and {len(error_df) - 10} more errors")
    
    print(f"\n✅ Error analysis completed!")
    print(f"📊 Detailed table: {error_table_file}")
    print(f"📈 Summary: {summary_file}")

if __name__ == '__main__':
    main()
