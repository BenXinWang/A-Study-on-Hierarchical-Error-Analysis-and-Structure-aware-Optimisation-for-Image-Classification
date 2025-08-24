import torch
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import random
import os
import argparse
from typing import Dict, List, Tuple, Union
from sklearn.utils import check_random_state
import sklearn
from pathlib import Path
import torch.nn.functional as F
from sklearn.decomposition import PCA

# 设置随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)  # GPU随机种子
torch.cuda.manual_seed_all(RANDOM_SEED)  # 多GPU随机种子
torch.backends.cudnn.deterministic = True  # 确保每次返回相同的值
torch.backends.cudnn.benchmark = False  # 关闭自动优化，确保结果可重复
sklearn.utils.check_random_state(RANDOM_SEED)

# 专业配色方案
COLORS = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 黄绿
    '#17becf',  # 青色
    '#aec7e8',  # 浅蓝
    '#ffbb78',  # 浅橙
    '#98df8a',  # 浅绿
    '#ff9896',  # 浅红
    '#c5b0d5',  # 浅紫
    '#c49c94',  # 浅棕
    '#f7b6d2',  # 浅粉
    '#c7c7c7',  # 浅灰
    '#dbdb8d',  # 浅黄绿
    '#9edae5',  # 浅青
]

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Visualize CIFAR-100 prototypes using t-SNE')
    parser.add_argument('--input-file', type=str, default='./features/prototypes.pt',
                      help='Path to the prototypes file')
    parser.add_argument('--output-dir', type=str, default='results',
                      help='Directory to save output files')
    parser.add_argument('--perplexity', type=float, default=5,
                      help='Perplexity parameter for t-SNE')
    parser.add_argument('--n-iter', type=int, default=5000,
                      help='Number of iterations for t-SNE')
    parser.add_argument('--dpi', type=int, default=300,
                      help='DPI for output images')
    parser.add_argument('--fig-width', type=int, default=1200,
                      help='Figure width in pixels')
    parser.add_argument('--fig-height', type=int, default=800,
                      help='Figure height in pixels')
    return parser.parse_args()

def load_prototypes(file_path: str) -> Dict:
    """加载原型文件
    
    Args:
        file_path: 原型文件路径
        
    Returns:
        包含原型和映射信息的字典
    """
    prototypes = torch.load(file_path)
    return prototypes

def get_superclass_features(prototypes: Dict) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    """获取超类特征和映射信息
    
    Args:
        prototypes: 包含原型和映射信息的字典
        
    Returns:
        超类特征数组、超类名称列表和子类信息字典的元组
    """
    superclass_features = []
    superclass_names = []
    subclass_info = {}
    
    for name, feature in prototypes['super'].items():
        superclass_features.append(feature.numpy())
        superclass_names.append(name)
        subclass_indices = prototypes['mapping'].get(name, [])
        if subclass_indices is None:
            subclass_indices = []
        subclass_info[name] = [f"Class {idx}" for idx in subclass_indices]
    
    return np.array(superclass_features), superclass_names, subclass_info

def apply_tsne(features: np.ndarray, perplexity: float = 15, n_iter: int = 3000) -> np.ndarray:
    """应用t-SNE降维
    
    Args:
        features: 输入特征数组
        perplexity: t-SNE perplexity参数
        n_iter: t-SNE迭代次数
        
    Returns:
        降维后的2D坐标数组
    """
    try:
        # 确保随机状态一致
        random_state = check_random_state(RANDOM_SEED)
        
        # 首先进行PCA降维到20维，以减少噪声
        pca = PCA(n_components=20, random_state=RANDOM_SEED)
        features_pca = pca.fit_transform(features)
        
        # 然后应用t-SNE，使用更稳定的参数
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(features) - 1),  # 确保perplexity不超过样本数
            n_iter=n_iter,
            random_state=random_state,
            init='pca',  # 使用PCA初始化，更稳定
            learning_rate='auto',  # 自动学习率
            metric='euclidean',  # 使用欧氏距离
            early_exaggeration=12.0,  # 标准早期夸大因子
            n_iter_without_progress=300,  # 增加收敛检查的迭代次数
            method='barnes_hut' if len(features) > 50 else 'exact'  # 大数据集使用barnes_hut
        )
        return tsne.fit_transform(features_pca)
    except Exception as e:
        print(f"t-SNE failed, using PCA instead: {e}")
        # 如果t-SNE失败，使用PCA作为备选
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        return pca.fit_transform(features)

def create_scatter_plot(coords: np.ndarray, 
                       names: List[str], 
                       subclass_info: Dict[str, List[str]], 
                       fig_width: int = 1200, 
                       fig_height: int = 800) -> go.Figure:
    """创建交互式散点图
    
    Args:
        coords: 2D坐标数组
        names: 超类名称列表
        subclass_info: 子类信息字典
        fig_width: 图像宽度
        fig_height: 图像高度
        
    Returns:
        Plotly图像对象
    """
    # 使用专业配色方案
    colors = COLORS[:len(names)]
    
    # 创建悬停文本
    hover_texts = []
    for i, name in enumerate(names):
        subclasses = "<br>".join(subclass_info[name])
        hover_texts.append(
            f"Superclass: {name}<br>"
            f"Subclasses:<br>{subclasses}"
        )
    
    # 创建散点图
    fig = go.Figure()
    
    # 添加散点
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers+text',
        marker=dict(
            size=15,
            color=colors,
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=names,
        textposition="top center",
        hovertext=hover_texts,
        hoverinfo='text',
        name=''
    ))
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text="CIFAR-100 Superclass Prototypes Visualization (t-SNE)",
            font=dict(size=24)
        ),
        xaxis_title=dict(
            text="t-SNE dimension 1",
            font=dict(size=18)
        ),
        yaxis_title=dict(
            text="t-SNE dimension 2",
            font=dict(size=18)
        ),
        showlegend=False,
        hovermode='closest',
        template='plotly_white',
        width=fig_width,
        height=fig_height,
        font=dict(size=14)
    )
    
    # 添加比例尺
    max_range = max(
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min()
    )
    scale_text = f"Scale: {max_range:.2f} units"
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text=scale_text,
        showarrow=False,
        font=dict(size=14)
    )
    
    return fig

def create_distance_heatmap(features: torch.Tensor, 
                          names: List[str], 
                          fig_width: int = 1200, 
                          fig_height: int = 800) -> go.Figure:
    """创建超类内类原型距离热力图
    
    Args:
        features: 特征张量
        names: 超类名称列表
        fig_width: 图像宽度
        fig_height: 图像高度
        
    Returns:
        Plotly图像对象
    """
    # 打印特征统计信息
    print("特征统计信息:")
    print(f"特征形状: {features.shape}")
    print(f"特征最小值: {features.min().item():.4f}")
    print(f"特征最大值: {features.max().item():.4f}")
    print(f"特征均值: {features.mean().item():.4f}")
    print(f"特征标准差: {features.std().item():.4f}")
    
    # 计算成对欧氏距离
    features_np = features.numpy()
    n = len(features_np)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dist = np.sqrt(np.sum((features_np[i] - features_np[j]) ** 2))
            distances[i, j] = dist
    
    # 归一化距离矩阵到 [0, 1] 范围
    if distances.max() > distances.min():
        distances = (distances - distances.min()) / (distances.max() - distances.min())
    
    print("\n距离矩阵统计信息:")
    print(f"距离最小值: {distances.min():.4f}")
    print(f"距离最大值: {distances.max():.4f}")
    print(f"距离均值: {distances.mean():.4f}")
    print(f"距离标准差: {distances.std():.4f}")
    
    # 将距离值格式化为两位小数
    distances_text = [[f"{x:.2f}" for x in row] for row in distances]
    
    # 创建热力图
    fig = ff.create_annotated_heatmap(
        z=distances,
        x=names,
        y=names,
        annotation_text=distances_text,
        colorscale='Viridis',
        showscale=True,
        font_colors=['black', 'white'],  # 根据背景色自动选择文字颜色
        reversescale=False
    )
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text="Superclass Prototypes Distance Matrix",
            font=dict(size=24)
        ),
        xaxis_title=dict(
            text="Superclass",
            font=dict(size=18)
        ),
        yaxis_title=dict(
            text="Superclass",
            font=dict(size=18)
        ),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10),
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(size=10)
        ),
        width=fig_width,
        height=fig_height
    )
    
    # 调整注释文本的大小和格式
    for annotation in fig.layout.annotations:
        annotation.font.size = 8  # 减小数字大小
    
    return fig

def visualize_prototypes(features_dir, results_dir, perplexity=5, n_iter=5000,
                        fig_width=1200, fig_height=800):
    """主要的可视化函数
    
    Args:
        features_dir: 特征目录路径
        results_dir: 结果保存目录
        perplexity: t-SNE perplexity参数
        n_iter: t-SNE迭代次数
        fig_width: 图像宽度
        fig_height: 图像高度
    """
    features_dir = Path(features_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载原型
    prototypes = load_prototypes(features_dir / 'prototypes.pt')
    
    # 获取超类特征和信息
    features, names, subclass_info = get_superclass_features(prototypes)
    
    # 应用t-SNE
    coords = apply_tsne(features, perplexity=perplexity, n_iter=n_iter)
    
    # 创建散点图
    scatter_fig = create_scatter_plot(
        coords, names, subclass_info,
        fig_width=fig_width, fig_height=fig_height
    )
    
    # 创建距离热图
    heatmap_fig = create_distance_heatmap(
        torch.tensor(features), names,
        fig_width=fig_width, fig_height=fig_height
    )
    
    # 保存图像
    scatter_fig.write_html(results_dir / 'prototype_scatter.html')
    heatmap_fig.write_html(results_dir / 'prototype_heatmap.html')
    
    print(f"可视化结果已保存到 {results_dir}")
    
    return {
        'scatter_plot': scatter_fig,
        'heatmap': heatmap_fig
    }

if __name__ == "__main__":
    args = parse_args()
    visualize_prototypes(
        os.path.dirname(args.input_file),
        args.output_dir,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        fig_width=args.fig_width,
        fig_height=args.fig_height
    ) 