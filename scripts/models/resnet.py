import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNet50(nn.Module):
    def __init__(self, mode='subclass_only', dropout_rate=0.5):
        """
        Args:
            mode (str): 运行模式
                - 'subclass_only': 仅使用子类分类头（基线训练）
                - 'superclass_only': 仅使用超类分类头（预留）
                - 'dual': 双分支模式（预留）
            dropout_rate (float): Dropout概率
        """
        super().__init__()
        # 1. 加载预训练的ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 2. 修改第一个卷积层以适应32x32输入
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 3. 提取特征提取器（去掉原始分类头）
        self.features = nn.Sequential(*list(model.children())[:-1])
        
        # 4. 添加dropout层
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 5. 构建双分支分类头
        self.subclass_head = nn.Linear(2048, 100)    # 子类分类器（100类）
        if mode != 'subclass_only':
            self.superclass_head = nn.Linear(2048, 20)   # 超类分类器（20类）
        
        # 6. 设置运行模式
        self.mode = mode
        
    def forward(self, x):
        # 1. 特征提取
        features = self.features(x)
        features = features.view(features.size(0), -1)  # 展平为 2048-D 向量
        
        # 2. 应用dropout
        features = self.dropout(features)
        
        # 3. 根据mode返回相应输出
        if self.mode == 'subclass_only':
            return {
                'features': features,
                'logits': self.subclass_head(features)
            }
        elif self.mode == 'superclass_only':
            return {
                'features': features,
                'logits': self.superclass_head(features)
            }
        else:  # mode == 'dual'
            return {
                'features': features,
                'subclass_logits': self.subclass_head(features),
                'superclass_logits': self.superclass_head(features)
            }
    
    def set_mode(self, mode):
        """切换模型运行模式"""
        assert mode in ['subclass_only', 'superclass_only', 'dual']
        self.mode = mode 