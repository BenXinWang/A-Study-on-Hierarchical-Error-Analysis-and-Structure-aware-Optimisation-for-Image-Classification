#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import json
from pathlib import Path
import os

# 导入智能分层分类器
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from intelligent_classifier import create_intelligent_hierarchical_classifier

class HierarchicalThresholdOptimizer:
    """
    分层阈值优化器 - 专注于优化分层准确率
    
    核心目标：
    - 最大化分层准确率（完全正确 + 部分正确）
    - 优化智能降级成功率
    - 平衡子类精确性与超类实用性
    """
    
    def __init__(self, classifier, test_loader, device):
        """
        初始化分层阈值优化器
        
        Args:
            classifier: 智能分层分类器
            test_loader: 测试数据加载器
            device: 计算设备
        """
        self.classifier = classifier
        self.test_loader = test_loader
        self.device = device
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("分层阈值优化器初始化完成")
        print(f"测试集大小: {len(test_loader.dataset)}")
        print("优化目标: 最大化分层准确率")
    
    def optimize_hierarchical_threshold(self, threshold_min: float = 0.3, 
                                      threshold_max: float = 0.9, 
                                      step_size: float = 0.05) -> dict:
        """
        优化分层阈值以最大化分层准确率
        
        Args:
            threshold_min: 最小阈值
            threshold_max: 最大阈值
            step_size: 步长
            
        Returns:
            优化结果字典
        """
        print("开始分层阈值优化...")
        print(f"搜索范围: {threshold_min} - {threshold_max}")
        print(f"步长: {step_size}")
        print("目标指标: hierarchical_accuracy (最大化)")
        
        # 生成阈值列表
        thresholds = np.arange(threshold_min, threshold_max + step_size, step_size)
        print(f"总共测试 {len(thresholds)} 个阈值")
        
        # 评估所有阈值
        results = self.classifier.evaluate_threshold_hierarchical(self.test_loader, thresholds.tolist())
        
        # 找到最优阈值
        best_threshold = None
        best_hierarchical_accuracy = 0.0
        
        for threshold, metrics in results.items():
            if metrics['hierarchical_accuracy'] > best_hierarchical_accuracy:
                best_hierarchical_accuracy = metrics['hierarchical_accuracy']
                best_threshold = threshold
        
        print("=" * 50)
        print("分层阈值优化完成!")
        print("=" * 50)
        print(f"最优阈值: {best_threshold}")
        print(f"最佳分层准确率: {best_hierarchical_accuracy:.4f}")
        
        # 显示最优阈值的详细指标
        best_metrics = results[best_threshold]
        print("\n最优阈值详细指标:")
        print(f"  完全正确率: {best_metrics['exact_accuracy']:.4f}")
        print(f"  部分正确率: {best_metrics['partial_accuracy']:.4f}")
        print(f"  分层准确率: {best_metrics['hierarchical_accuracy']:.4f}")
        print(f"  智能降级率: {best_metrics['intelligent_degradation_rate']:.4f}")
        print(f"  子类使用率: {best_metrics['subclass_usage']:.4f}")
        print(f"  超类使用率: {best_metrics['superclass_usage']:.4f}")
        print(f"  传统准确率: {best_metrics['traditional_accuracy']:.4f}")
        
        # 保存结果
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_file = f"results/hierarchical_threshold_optimization_{self.timestamp}.csv"
        results_df.to_csv(results_file, index_label='threshold')
        print(f"\n优化结果已保存到: {results_file}")
        
        # 生成可视化
        self._create_hierarchical_visualizations(results, best_threshold)
        
        return {
            'best_threshold': best_threshold,
            'best_hierarchical_accuracy': best_hierarchical_accuracy,
            'best_metrics': best_metrics,
            'all_results': results,
            'results_file': results_file
        }
    
    def analyze_hierarchical_sensitivity(self, best_threshold: float, 
                                       sensitivity_range: float = 0.1) -> dict:
        """
        分析最优阈值的敏感性
        
        Args:
            best_threshold: 最优阈值
            sensitivity_range: 敏感性分析范围
            
        Returns:
            敏感性分析结果
        """
        print(f"\n进行分层阈值敏感性分析...")
        print(f"最优阈值: {best_threshold}")
        print(f"分析范围: ±{sensitivity_range}")
        
        # 生成敏感性分析阈值
        sensitivity_thresholds = np.arange(
            max(0.0, best_threshold - sensitivity_range),
            min(1.0, best_threshold + sensitivity_range) + 0.01,
            0.01
        )
        
        # 评估敏感性
        sensitivity_results = self.classifier.evaluate_threshold_hierarchical(
            self.test_loader, sensitivity_thresholds.tolist()
        )
        
        # 计算敏感性统计
        hierarchical_accuracies = [metrics['hierarchical_accuracy'] for metrics in sensitivity_results.values()]
        degradation_rates = [metrics['intelligent_degradation_rate'] for metrics in sensitivity_results.values()]
        
        sensitivity_stats = {
            'hierarchical_accuracy_std': np.std(hierarchical_accuracies),
            'hierarchical_accuracy_range': max(hierarchical_accuracies) - min(hierarchical_accuracies),
            'degradation_rate_std': np.std(degradation_rates),
            'degradation_rate_range': max(degradation_rates) - min(degradation_rates),
            'threshold_range': sensitivity_range * 2,
            'num_thresholds': len(sensitivity_thresholds)
        }
        
        print("敏感性分析完成:")
        print(f"分层准确率标准差: {sensitivity_stats['hierarchical_accuracy_std']:.6f}")
        print(f"智能降级率标准差: {sensitivity_stats['degradation_rate_std']:.6f}")
        
        return {
            'sensitivity_results': sensitivity_results,
            'sensitivity_stats': sensitivity_stats
        }
    
    def _create_hierarchical_visualizations(self, results: dict, best_threshold: float):
        """创建分层性能可视化图表"""
        # 准备数据
        thresholds = list(results.keys())
        hierarchical_accuracies = [results[t]['hierarchical_accuracy'] for t in thresholds]
        exact_accuracies = [results[t]['exact_accuracy'] for t in thresholds]
        partial_accuracies = [results[t]['partial_accuracy'] for t in thresholds]
        degradation_rates = [results[t]['intelligent_degradation_rate'] for t in thresholds]
        subclass_usages = [results[t]['subclass_usage'] for t in thresholds]
        traditional_accuracies = [results[t]['traditional_accuracy'] for t in thresholds]
        
        # 创建综合分析图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('智能分层分类系统性能分析', fontsize=16, fontweight='bold')
        
        # 1. 分层准确率对比
        axes[0, 0].plot(thresholds, hierarchical_accuracies, 'b-', linewidth=2, label='分层准确率')
        axes[0, 0].plot(thresholds, traditional_accuracies, 'r--', linewidth=2, label='传统准确率')
        axes[0, 0].axvline(x=best_threshold, color='green', linestyle=':', alpha=0.7, label=f'最优阈值 ({best_threshold:.2f})')
        axes[0, 0].set_xlabel('置信度阈值')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].set_title('分层准确率 vs 传统准确率')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 准确率分解
        axes[0, 1].plot(thresholds, exact_accuracies, 'g-', linewidth=2, label='完全正确率')
        axes[0, 1].plot(thresholds, partial_accuracies, 'orange', linewidth=2, label='部分正确率')
        axes[0, 1].axvline(x=best_threshold, color='green', linestyle=':', alpha=0.7, label=f'最优阈值')
        axes[0, 1].set_xlabel('置信度阈值')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].set_title('分层准确率分解')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 智能降级分析
        axes[1, 0].plot(thresholds, degradation_rates, 'purple', linewidth=2, label='智能降级成功率')
        axes[1, 0].plot(thresholds, subclass_usages, 'brown', linewidth=2, label='子类使用率')
        axes[1, 0].axvline(x=best_threshold, color='green', linestyle=':', alpha=0.7, label=f'最优阈值')
        axes[1, 0].set_xlabel('置信度阈值')
        axes[1, 0].set_ylabel('比率')
        axes[1, 0].set_title('智能降级与使用率分析')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 性能提升分析
        improvements = [h_acc - t_acc for h_acc, t_acc in zip(hierarchical_accuracies, traditional_accuracies)]
        axes[1, 1].plot(thresholds, improvements, 'red', linewidth=2, label='分层准确率提升')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].axvline(x=best_threshold, color='green', linestyle=':', alpha=0.7, label=f'最优阈值')
        axes[1, 1].set_xlabel('置信度阈值')
        axes[1, 1].set_ylabel('准确率提升')
        axes[1, 1].set_title('分层准确率相对传统准确率的提升')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = f"results/visualizations/hierarchical_performance_analysis_{self.timestamp}.png"
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"分层性能分析图已保存到: {plot_file}")
        
        # 创建阈值分布分析图
        self._create_threshold_distribution_plot(results, best_threshold)
    
    def _create_threshold_distribution_plot(self, results: dict, best_threshold: float):
        """创建阈值分布分析图"""
        thresholds = list(results.keys())
        hierarchical_accuracies = [results[t]['hierarchical_accuracy'] for t in thresholds]
        subclass_usages = [results[t]['subclass_usage'] for t in thresholds]
        degradation_rates = [results[t]['intelligent_degradation_rate'] for t in thresholds]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('智能分层系统阈值分布分析', fontsize=14, fontweight='bold')
        
        # 分层准确率热力图风格
        ax1.scatter(thresholds, hierarchical_accuracies, c=degradation_rates, 
                   s=100, cmap='viridis', alpha=0.7)
        ax1.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.8, 
                   label=f'最优阈值 ({best_threshold:.2f})')
        ax1.set_xlabel('置信度阈值')
        ax1.set_ylabel('分层准确率')
        ax1.set_title('分层准确率 (颜色表示智能降级率)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 使用率与性能关系
        ax2.scatter(subclass_usages, hierarchical_accuracies, c=thresholds, 
                   s=100, cmap='plasma', alpha=0.7)
        ax2.set_xlabel('子类使用率')
        ax2.set_ylabel('分层准确率')
        ax2.set_title('子类使用率 vs 分层准确率 (颜色表示阈值)')
        ax2.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label('智能降级率')
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('置信度阈值')
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = f"results/visualizations/hierarchical_threshold_distribution_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"阈值分布分析图已保存到: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='分层阈值优化器')
    parser.add_argument('--threshold_min', type=float, default=0.3, help='最小阈值')
    parser.add_argument('--threshold_max', type=float, default=0.9, help='最大阈值')
    parser.add_argument('--step_size', type=float, default=0.05, help='步长')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--sensitivity_range', type=float, default=0.1, help='敏感性分析范围')
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    
    print("创建智能分层分类器...")
    # 创建智能分层分类器
    try:
        classifier = create_intelligent_hierarchical_classifier()
    except Exception as e:
        print(f"创建分层分类器失败: {e}")
        print("请确保子类和超类模型检查点存在")
        return
    
    # 准备测试数据
    print("准备测试数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                           std=[0.2675, 0.2565, 0.2761])
    ])
    
    test_dataset = datasets.CIFAR100(root='./data', train=False,
                                     download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    # 创建优化器
    optimizer = HierarchicalThresholdOptimizer(classifier, test_loader, device)
    
    # 执行分层阈值优化
    optimization_results = optimizer.optimize_hierarchical_threshold(
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        step_size=args.step_size
    )
    
    # 敏感性分析
    sensitivity_results = optimizer.analyze_hierarchical_sensitivity(
        optimization_results['best_threshold'],
        args.sensitivity_range
    )
    
    # 保存完整报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        'optimization_results': optimization_results,
        'sensitivity_results': sensitivity_results,
        'parameters': vars(args),
        'timestamp': timestamp
    }
    
    report_file = f"results/reports/hierarchical_threshold_optimization_report_{timestamp}.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # 自定义JSON编码器处理numpy类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    print(f"完整分层优化报告已保存到: {report_file}")
    print("分层阈值优化完成!")


if __name__ == "__main__":
    main() 