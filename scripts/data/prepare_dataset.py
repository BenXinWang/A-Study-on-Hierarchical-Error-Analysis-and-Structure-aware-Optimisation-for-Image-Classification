#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

def prepare_cifar100_data(data_dir='../data'):
    """准备CIFAR-100数据集
    
    Args:
        data_dir: 数据集保存目录
        
    Returns:
        tuple: (trainset, testset) 包含训练集和测试集
    """
    # 确保数据目录存在
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),  # CIFAR-100 均值
            (0.2675, 0.2565, 0.2761)   # CIFAR-100 标准差
        )
    ])
    
    # 下载并准备训练集
    print("正在下载和准备CIFAR-100训练集...")
    trainset = torchvision.datasets.CIFAR100(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )
    
    # 下载并准备测试集
    print("正在下载和准备CIFAR-100测试集...")
    testset = torchvision.datasets.CIFAR100(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )
    
    print("数据集准备完成！")
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    
    return trainset, testset

if __name__ == "__main__":
    prepare_cifar100_data() 