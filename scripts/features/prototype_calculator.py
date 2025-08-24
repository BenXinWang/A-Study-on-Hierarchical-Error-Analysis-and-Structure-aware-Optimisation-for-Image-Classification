import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加F以使用normalize函数
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from collections import defaultdict

# CIFAR-100超类映射
SUPERCLASS_MAPPING = {
    'aquatic_mammals': [4, 30, 55, 72, 95],  # beaver, dolphin, otter, seal, whale
    'fish': [1, 32, 67, 73, 91],             # aquarium fish, flatfish, ray, shark, trout
    'flowers': [54, 62, 70, 82, 92],         # orchid, poppy, rose, sunflower, tulip
    'food_containers': [9, 10, 16, 28, 61],  # bottle, bowl, can, cup, plate
    'fruit_and_vegetables': [0, 51, 53, 57, 83],  # apple, pear, pepper, pineapple, sweet pepper
    'household_electrical_devices': [22, 39, 40, 86, 87],  # clock, keyboard, lamp, telephone, television
    'household_furniture': [5, 20, 25, 84, 94],  # bed, chair, couch, table, wardrobe
    'insects': [6, 7, 14, 18, 24],          # bee, beetle, butterfly, caterpillar, cockroach
    'large_carnivores': [3, 42, 43, 88, 97], # bear, leopard, lion, tiger, wolf
    'large_man-made_outdoor_things': [12, 17, 37, 68, 76], # bridge, castle, house, road, skyscraper
    'large_natural_outdoor_scenes': [23, 33, 49, 60, 71],  # cloud, forest, mountain, plain, sea
    'large_omnivores_and_herbivores': [15, 19, 21, 31, 38], # camel, cattle, chimpanzee, elephant, kangaroo
    'medium_mammals': [34, 63, 64, 66, 75],  # fox, porcupine, possum, raccoon, skunk
    'non-insect_invertebrates': [26, 45, 77, 79, 99], # crab, lobster, snail, spider, worm
    'people': [2, 11, 35, 46, 98],          # baby, boy, girl, man, woman
    'reptiles': [27, 29, 44, 78, 93],       # crocodile, dinosaur, lizard, snake, turtle
    'small_mammals': [36, 50, 65, 74, 80],  # hamster, mouse, rabbit, shrew, squirrel
    'trees': [47, 52, 56, 59, 96],          # maple tree, oak tree, palm tree, pine tree, willow tree
    'vehicles_1': [8, 13, 48, 58, 90],      # bicycle, bus, motorcycle, pickup truck, train
    'vehicles_2': [41, 69, 81, 85, 89],     # lawn mower, rocket, tank, tractor, streetcar
}

class ModifiedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练模型
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 修改第一个卷积层以适应32x32输入
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 分离特征提取器和分类器
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Linear(2048, 100)  # CIFAR-100有100个类
        
    def forward(self, x):
        # 提取特征
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # 分类
        logits = self.classifier(x)
        return x, logits  # 返回特征和logits

def load_data(features_path, metadata_path):
    """加载特征和元数据"""
    features = torch.load(features_path)
    metadata = pd.read_csv(metadata_path)
    return features, metadata

def calculate_confidences(model, dataloader, device):
    """计算每个样本的预测置信度"""
    model.eval()
    confidences = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            # 只使用logits，忽略features
            _, logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confidences.append(probs.max(dim=1)[0].cpu())
    return torch.cat(confidences)

def calculate_class_prototypes(features, labels, confidences, confidence_threshold=0.8):
    """计算类原型（加权平均）并进行L2标准化"""
    num_classes = 100
    feature_dim = features.shape[1]
    prototypes = torch.zeros(num_classes, feature_dim)
    
    # 转换为NumPy进行快速计算
    features_np = features.numpy()
    confidences_np = confidences.numpy()
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    
    print("\n特征输入统计信息:")
    print(f"特征形状: {features.shape}")
    print(f"特征范数: {torch.norm(features, p=2, dim=1).mean().item():.4f}")
    print(f"特征最大值: {features.max().item():.4f}")
    print(f"特征最小值: {features.min().item():.4f}")
    
    for class_idx in range(num_classes):
        # 获取该类别的样本
        class_mask = (labels_np == class_idx)
        if not np.any(class_mask):
            print(f"警告：类别 {class_idx} 没有样本")
            continue
            
        class_features = features_np[class_mask]
        class_confidences = confidences_np[class_mask]
        
        # 应用置信度阈值
        conf_mask = class_confidences >= confidence_threshold
        if np.any(conf_mask):
            valid_features = class_features[conf_mask]
            valid_confidences = class_confidences[conf_mask]
            
            # 计算加权平均
            weights = valid_confidences / valid_confidences.sum()
            weights = weights.reshape(-1, 1)  # 调整形状以进行广播
            prototype = np.sum(valid_features * weights, axis=0)
        else:
            # 如果没有高置信度样本，使用简单平均
            prototype = class_features.mean(axis=0)
        
        # 转换为PyTorch张量
        prototype_tensor = torch.from_numpy(prototype)
        
        # 检查是否有NaN或无穷大的值
        if torch.isnan(prototype_tensor).any() or torch.isinf(prototype_tensor).any():
            print(f"警告：类别 {class_idx} 的原型包含NaN或无穷大值")
            continue
        
        # 标准化前检查范数
        norm = torch.norm(prototype_tensor, p=2)
        if norm > 0:
            prototype_tensor = prototype_tensor / norm
        else:
            print(f"警告：类别 {class_idx} 的原型范数为0")
            continue
        
        prototypes[class_idx] = prototype_tensor
    
    print("\n类原型统计信息:")
    print(f"原型形状: {prototypes.shape}")
    print(f"原型范数: {torch.norm(prototypes, p=2, dim=1).mean().item():.4f}")
    print(f"原型最大值: {prototypes.max().item():.4f}")
    print(f"原型最小值: {prototypes.min().item():.4f}")
    
    return prototypes

def calculate_superclass_prototypes(class_prototypes):
    """计算超类原型并保持L2标准化"""
    superclass_prototypes = {}
    
    # 打印输入统计信息
    print("\n类原型统计信息:")
    print(f"类原型形状: {class_prototypes.shape}")
    print(f"类原型范数: {torch.norm(class_prototypes, p=2, dim=1).mean().item():.4f}")
    
    for superclass, class_indices in SUPERCLASS_MAPPING.items():
        # 获取该超类下所有类别的原型
        super_features = class_prototypes[class_indices]
        
        # 计算平均值作为超类原型
        prototype = super_features.mean(dim=0)
        
        # 进行L2标准化
        prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1).squeeze(0)
        
        # 打印每个超类的统计信息
        print(f"\n超类 {superclass} 统计信息:")
        print(f"原型范数: {torch.norm(prototype, p=2).item():.4f}")
        print(f"原型最小值: {prototype.min().item():.4f}")
        print(f"原型最大值: {prototype.max().item():.4f}")
        
        superclass_prototypes[superclass] = prototype
    
    return superclass_prototypes

def main():
    # 设置设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 准备数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_dataset = CIFAR100(
        root='./data',
        train=False,
        transform=transform,
        download=True
    )
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 加载特征和元数据
    features, metadata = load_data(
        './features/features.pt',
        './features/metadata.csv'
    )
    
    # 计算置信度
    model = ModifiedResNet50().to(device)
    confidences = calculate_confidences(model, dataloader, device)
    
    # 计算类原型（带L2标准化）
    class_prototypes = calculate_class_prototypes(
        features,
        metadata['label'].values,
        confidences
    )
    
    # 计算超类原型（保持L2标准化）
    superclass_prototypes = calculate_superclass_prototypes(class_prototypes)
    
    # 保存结果
    output = {
        'class': class_prototypes,
        'super': superclass_prototypes,
        'mapping': SUPERCLASS_MAPPING
    }
    torch.save(output, './features/prototypes.pt')
    
    # 验证
    print("类原型形状:", class_prototypes.shape)
    print("超类数量:", len(superclass_prototypes))
    print("示例超类 'aquatic_mammals' 特征维度:", 
          superclass_prototypes['aquatic_mammals'].shape)
    
    # 验证标准化效果
    print("\n=== 标准化验证 ===")
    print("类原型范数:", torch.norm(class_prototypes[0], p=2).item())
    print("超类原型范数:", torch.norm(superclass_prototypes['aquatic_mammals'], p=2).item())

if __name__ == "__main__":
    main() 