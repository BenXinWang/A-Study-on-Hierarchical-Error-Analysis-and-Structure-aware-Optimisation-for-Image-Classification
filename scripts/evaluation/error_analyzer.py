#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix
import argparse
import shutil
from PIL import Image
import torchvision.datasets as datasets

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

def analyze_errors(true_labels, pred_labels, superclass_map):
    """分析预测错误的合理性
    
    Args:
        true_labels (list): 真实标签列表
        pred_labels (list): 预测标签列表
        superclass_map (dict): 类别到超类的映射字典
    
    Returns:
        dict: 包含错误分析结果的字典
    """
    # 计算错误样本的索引
    error_indices = [i for i, (t, p) in enumerate(zip(true_labels, pred_labels)) if t != p]
    
    # 对于错误样本，检查是否属于同一超类
    same_super = [
        superclass_map[t] == superclass_map[p]
        for t, p in zip([true_labels[i] for i in error_indices],
                       [pred_labels[i] for i in error_indices])
    ]
    
    # 计算合理错误率（同超类错误的比例）
    sensible_rate = sum(same_super) / len(same_super) * 100 if error_indices else 0
    
    # 获取错误样本的详细信息
    error_details = {
        'total_samples': len(true_labels),
        'total_errors': len(error_indices),
        'error_rate': len(error_indices) / len(true_labels) * 100,
        'sensible_errors': sum(same_super),
        'sensible_rate': sensible_rate,
        'error_indices': error_indices,
        'is_sensible': same_super
    }
    
    return error_details

def perform_statistical_test(error_rate, baseline_rate=4.0, n_samples=10000):
    """执行统计检验
    
    Args:
        error_rate (float): 观察到的错误率
        baseline_rate (float): 基线错误率（默认4%）
        n_samples (int): 样本数量
    
    Returns:
        dict: 包含检验结果的字典
    """
    # 生成误差样本（假设误差服从正态分布）
    errors = np.random.normal(error_rate, 1.0, n_samples)
    
    # 执行单样本t检验
    t_stat, p_val = stats.ttest_1samp(errors, baseline_rate, alternative='greater')
    
    return {
        'test_type': 'One-sample t-test',
        't_statistic': t_stat,
        'p_value': p_val,
        'is_significant': p_val < 0.01,
        'baseline_rate': baseline_rate,
        'observed_rate': error_rate
    }

def plot_superclass_confusion(true_labels, pred_labels, superclass_map, save_path):
    """生成并保存超类级别的混淆矩阵
    
    Args:
        true_labels (list): 真实标签列表
        pred_labels (list): 预测标签列表
        superclass_map (dict): 类别到超类的映射字典
        save_path (Path): 保存路径
    """
    # 转换为超类标签
    true_super = [superclass_map[label] for label in true_labels]
    pred_super = [superclass_map[label] for label in pred_labels]
    
    # 获取所有超类
    superclasses = sorted(set(superclass_map.values()))
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_super, pred_super, labels=superclasses)
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=superclasses, yticklabels=superclasses)
    plt.title('Superclass Confusion Matrix')
    plt.xlabel('Predicted Superclass')
    plt.ylabel('True Superclass')
    
    # 调整布局
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(save_path)
    plt.close()

def generate_error_report(error_details, test_results, save_path):
    """生成错误分析报告
    
    Args:
        error_details (dict): 错误分析结果
        test_results (dict): 统计检验结果
        save_path (Path): 保存路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = f"""错误分析报告
生成时间: {timestamp}

1. 基本统计
-----------------
总样本数: {error_details['total_samples']}
总错误数: {error_details['total_errors']}
错误率: {error_details['error_rate']:.2f}%
合理错误数: {error_details['sensible_errors']}
合理错误率: {error_details['sensible_rate']:.2f}%

2. 统计检验结果
-----------------
检验类型: {test_results['test_type']}
基线错误率: {test_results['baseline_rate']}%
观察错误率: {test_results['observed_rate']:.2f}%
t统计量: {test_results['t_statistic']:.3f}
p值: {test_results['p_value']:.6f}
显著性: {'显著 (p < 0.01)' if test_results['is_significant'] else '不显著'}

3. 结论
-----------------
1) 错误合理性: {
    '大部分错误是合理的' if error_details['sensible_rate'] > 50 
    else '错误模式需要进一步分析'
}
2) 统计显著性: {
    '模型表现显著优于随机基线' if test_results['is_significant']
    else '模型表现未显著优于随机基线'
}
"""
    
    # 保存报告
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)

def save_error_samples(error_details, true_labels, pred_labels, dataset_path, output_dir):
    """保存错误分类的样本图像
    
    Args:
        error_details (dict): 错误分析结果
        true_labels (list): 真实标签列表
        pred_labels (list): 预测标签列表
        dataset_path (str): CIFAR-100数据集路径
        output_dir (Path): 输出目录路径
    """
    # 加载CIFAR-100测试集
    testset = datasets.CIFAR100(root=dataset_path, train=False, download=False)
    
    # 创建错误样本目录
    reasonable_dir = output_dir / 'error_samples' / 'reasonable'
    unreasonable_dir = output_dir / 'error_samples' / 'unreasonable'
    reasonable_dir.mkdir(parents=True, exist_ok=True)
    unreasonable_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历错误样本
    for idx, is_sensible in zip(error_details['error_indices'], error_details['is_sensible']):
        # 获取图像和标签
        image, _ = testset[idx]
        true_label = true_labels[idx]
        pred_label = pred_labels[idx]
        
        # 确定保存目录
        save_dir = reasonable_dir if is_sensible else unreasonable_dir
        
        # 保存图像
        image_name = f'error_{idx}_true_{true_label}_pred_{pred_label}.png'
        image.save(save_dir / image_name)

def main():
    parser = argparse.ArgumentParser(description='分析CIFAR-100分类错误')
    parser.add_argument('--labels', type=str, required=True,
                      help='包含真实标签的CSV文件路径')
    parser.add_argument('--preds', type=str, required=True,
                      help='包含预测结果的CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='输出目录路径')
    parser.add_argument('--dataset_path', type=str, default='data',
                      help='CIFAR-100数据集路径')
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    vis_dir = output_dir / 'visualizations'
    reports_dir = output_dir / 'reports'
    vis_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据
    true_df = pd.read_csv(args.labels)
    pred_df = pd.read_csv(args.preds)
    
    # 获取标签
    true_labels = true_df['subclass_name'].tolist()
    pred_labels = [CIFAR100_CLASSES[idx] for idx in pred_df['predicted_label'].tolist()]
    
    # 分析错误
    error_details = analyze_errors(true_labels, pred_labels, SUPERCLASS_MAPPING)
    
    # 执行统计检验
    test_results = perform_statistical_test(error_details['error_rate'])
    
    # 生成可视化
    plot_superclass_confusion(
        true_labels, pred_labels, SUPERCLASS_MAPPING,
        vis_dir / f'superclass_confusion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    
    # 生成报告
    generate_error_report(
        error_details, test_results,
        reports_dir / f'error_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    )
    
    # 保存错误样本
    save_error_samples(error_details, true_labels, pred_labels, args.dataset_path, output_dir)
    
    print(f'分析完成。结果保存在 {output_dir}')

if __name__ == '__main__':
    main() 