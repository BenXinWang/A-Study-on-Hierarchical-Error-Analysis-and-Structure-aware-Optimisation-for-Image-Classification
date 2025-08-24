import torch
import numpy as np
from PIL import Image
import random

class Cutout:
    """Randomly mask out one or more patches from an image.
    
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        p (float): Probability of applying the transform. Default: 0.5
    """
    def __init__(self, n_holes=1, length=16, p=0.5):
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to apply cutout to.
        
        Returns:
            PIL Image or Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if random.random() > self.p:
            return img
        
        # 如果输入是PIL图像，转换为tensor
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        
        h = img.size(1)
        w = img.size(2)
        
        mask = torch.ones((h, w), dtype=torch.float32)
        
        for n in range(self.n_holes):
            # 随机选择掩码的中心点
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            # 计算掩码的边界
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            # 应用掩码
            mask[y1:y2, x1:x2] = 0
        
        # 扩展掩码维度以匹配图像通道
        mask = torch.as_tensor(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

    def __repr__(self):
        return f"Cutout(n_holes={self.n_holes}, length={self.length}, p={self.p})"

class MixUp:
    """MixUp 数据增强
    
    将两个样本及其标签按 beta 分布采样的权重进行混合
    """
    def __init__(self, alpha=1.0, prob=1.0):
        """
        Args:
            alpha (float): Beta 分布的参数。较大的值（如1）产生更均匀的混合，
                         较小的值（如0.2）产生更接近原始样本的混合。
            prob (float): 应用 MixUp 的概率，范围[0,1]
        """
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, batch, targets):
        """
        Args:
            batch: 形状为 (batch_size, channels, height, width) 的图像批次
            targets: 形状为 (batch_size,) 的标签批次
            
        Returns:
            mixed_batch: 混合后的图像批次
            targets_a: 第一组标签
            targets_b: 第二组标签
            lam: 混合权重
        """
        # 根据概率决定是否应用 MixUp
        if np.random.random() > self.prob:
            return batch, targets, targets, 1.0
            
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = batch.size(0)
        index = torch.randperm(batch_size).to(batch.device)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index]
        targets_a = targets
        targets_b = targets[index]
        
        return mixed_batch, targets_a, targets_b, lam 