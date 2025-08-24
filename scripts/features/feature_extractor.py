import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

def get_device():
    """配置设备，优先使用MPS"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 MPS 后端进行加速")
    else:
        device = torch.device("cpu")
        print("MPS 不可用，使用 CPU")
    return device

def modify_resnet50():
    """加载并修改ResNet50模型以适应32x32输入"""
    # 加载预训练模型
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # 修改第一个卷积层以适应32x32输入
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 移除最后的全连接层
    model = nn.Sequential(*list(model.children())[:-1])
    
    return model

def extract_features(model, dataloader, device):
    """提取特征"""
    features = []
    labels = []
    filenames = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="提取特征")):
            images = images.to(device)
            batch_features = model(images)
            # 展平特征 (N, 2048, 1, 1) -> (N, 2048)
            batch_features = batch_features.squeeze(-1).squeeze(-1)
            
            features.append(batch_features.cpu())
            labels.extend(targets.tolist())
            # 生成文件名
            batch_filenames = [f"img_{batch_idx * dataloader.batch_size + i}" for i in range(len(targets))]
            filenames.extend(batch_filenames)
    
    features = torch.cat(features, dim=0)
    return features, labels, filenames

def save_data(features, labels, filenames, output_dir):
    """保存特征和元数据"""
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存特征张量
    feature_path = os.path.join(output_dir, "features.pt")
    torch.save(features, feature_path)
    print(f"特征已保存至: {feature_path}")
    
    # 保存元数据CSV
    metadata = pd.DataFrame({
        "filename": filenames,
        "label": labels
    })
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata.to_csv(metadata_path, index=False)
    print(f"元数据已保存至: {metadata_path}")

def main(args):
    # 获取设备
    device = get_device()
    
    # 准备数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_dataset = CIFAR100(
        root=args.data_dir,
        train=False,
        transform=transform,
        download=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 准备模型
    model = modify_resnet50()
    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        print(f"加载预训练模型: {args.model_path}")
    model = model.to(device)
    
    # 提取特征
    features, labels, filenames = extract_features(model, test_loader, device)
    
    # 保存数据
    save_data(features, labels, filenames, args.output_dir)
    
    print(f"特征维度: {features.shape}")
    print("特征提取完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-100特征提取")
    parser.add_argument("--model_path", type=str, default=None,
                        help="预训练模型路径（可选）")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="./features",
                        help="输出目录")
    
    args = parser.parse_args()
    main(args) 