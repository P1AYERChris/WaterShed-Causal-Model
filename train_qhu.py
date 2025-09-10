import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from config import *
from data_loader import WatershedDataset, get_dataloaders
from model import GraphUNetDualTransformer, count_parameters, init_weights,HydrologicalLoss

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 内存优化配置
def setup_memory_optimization():
    """设置内存优化"""
    if torch.cuda.is_available():
        # 启用内存缓存分配器
        torch.backends.cudnn.benchmark = True
        # 设置内存分配策略
        torch.cuda.empty_cache()
        logger.info("GPU内存优化已启用")

def filter_network_data(edge_index, edge_attr, valid_node_indices, total_nodes):
    """
    根据有效节点索引过滤网络数据
    
    参数:
        edge_index: [2, num_edges] 边索引
        edge_attr: [num_edges, attr_dim] 边属性  
        valid_node_indices: 有效节点索引列表
        total_nodes: 原始总节点数
        
    返回:
        filtered_edge_index: 过滤后的边索引
        filtered_edge_attr: 过滤后的边属性
        node_mapping: 新旧索引映射字典
    """
    if edge_index.size(1) == 0:
        return edge_index, edge_attr, {}
    
    # 创建旧索引到新索引的映射
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_node_indices)}
    
    # 过滤边：只保留两端都是有效节点的边
    valid_edges = []
    valid_attrs = []
    
    for i in range(edge_index.size(1)):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        
        if source in node_mapping and target in node_mapping:
            new_source = node_mapping[source]
            new_target = node_mapping[target]
            valid_edges.append([new_source, new_target])
            if edge_attr.size(0) > 0:
                valid_attrs.append(edge_attr[i])
    
    # 转换为张量
    if valid_edges:
        filtered_edge_index = torch.tensor(valid_edges, dtype=torch.long).t()
        if valid_attrs:
            filtered_edge_attr = torch.stack(valid_attrs)
        else:
            filtered_edge_attr = torch.zeros((len(valid_edges), edge_attr.size(1)), dtype=torch.float)
    else:
        filtered_edge_index = torch.zeros((2, 0), dtype=torch.long)
        filtered_edge_attr = torch.zeros((0, edge_attr.size(1) if edge_attr.size(0) > 0 else 1), dtype=torch.float)
    
    logger.info(f"网络过滤: {edge_index.size(1)} -> {filtered_edge_index.size(1)} 条边")
    return filtered_edge_index, filtered_edge_attr, node_mapping

def filter_causal_network_data(edge_index, edge_weight, valid_basin_ids, all_basin_ids,variables):
    """
    根据有效流域ID过滤因果网络数据
    
    参数:
        edge_index: [2, num_edges] 因果边索引
        edge_weight: [num_edges] 因果边权重
        valid_basin_ids: 有效流域ID列表
        variables: 变量列表
        
    返回:
        filtered_edge_index: 过滤后的边索引
        filtered_edge_weight: 过滤后的边权重
    """
    if edge_index.size(1) == 0:
        return edge_index, edge_weight
    
    # 创建有效节点集合（使用原始索引位置）
    valid_nodes = set()
    for basin_id in valid_basin_ids:
        if basin_id in all_basin_ids:
            # 使用原始完整列表中的位置
            original_basin_idx = all_basin_ids.index(basin_id)
            for var_idx, var in enumerate(variables):
                node_idx = original_basin_idx * len(variables) + var_idx
                valid_nodes.add(node_idx)
    
    # 过滤边
    valid_edges = []
    valid_weights = []
    
    for i in range(edge_index.size(1)):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        
        if source in valid_nodes and target in valid_nodes:
            valid_edges.append([source, target])
            valid_weights.append(edge_weight[i].item())
    
    # 转换为张量
    if valid_edges:
        filtered_edge_index = torch.tensor(valid_edges, dtype=torch.long).t()
        filtered_edge_weight = torch.tensor(valid_weights, dtype=torch.float)
    else:
        filtered_edge_index = torch.zeros((2, 0), dtype=torch.long)
        filtered_edge_weight = torch.zeros((0,), dtype=torch.float)
    
    logger.info(f"因果网络过滤: {edge_index.size(1)} -> {filtered_edge_index.size(1)} 条边")
    return filtered_edge_index, filtered_edge_weight

def load_and_filter_data():
    """
    加载所有预处理数据并进行网络清洗
    
    返回:
        scale_data: 包含三个尺度清洗后数据的字典
        time_series_data: 时序数据字典
        variables: 变量列表
    """
    logger.info("开始加载和清洗数据...")
    
    scale_data = {}
    time_series_data = {}
    
    # 加载变量列表
    variables = VARIABLES
    
    for scale in SCALES:
        logger.info(f"处理{scale}尺度数据...")
        
        try:
            # 1. 加载有效流域ID
            valid_basin_ids_path = os.path.join(PROCESSED_DATA_DIR, f'{scale}_valid_basin_ids.json')
            with open(valid_basin_ids_path, 'r') as f:
                valid_basin_ids = json.load(f)
            
            # 2. 加载原始流域ID和索引映射
            basin_ids_path = os.path.join(PROCESSED_DATA_DIR, f'{scale}_basin_ids.json')
            with open(basin_ids_path, 'r') as f:
                all_basin_ids = json.load(f)
            
            # 3. 创建有效节点索引列表
            valid_node_indices = []
            for basin_id in valid_basin_ids:
                if basin_id in all_basin_ids:
                    valid_node_indices.append(all_basin_ids.index(basin_id))
            
            logger.info(f"{scale}尺度: {len(valid_basin_ids)}/{len(all_basin_ids)} 个有效流域")
            
            # 4. 加载和过滤河流网络
            river_edge_index = torch.load(os.path.join(RIVER_NETWORK_DIR, f'{scale}_river_edge_index.pt'),weights_only=True)
            river_edge_attr = torch.load(os.path.join(RIVER_NETWORK_DIR, f'{scale}_river_edge_attr.pt'),weights_only=True)
            
            filtered_river_edge_index, filtered_river_edge_attr, node_mapping = filter_network_data(
                river_edge_index, river_edge_attr, valid_node_indices, len(all_basin_ids)
            )
            
            # 5. 加载和过滤因果网络
            causal_edge_index_path = os.path.join(CAUSAL_NETWORK_DIR, f'{scale}_causal_edge_index.pt')
            causal_edge_weight_path = os.path.join(CAUSAL_NETWORK_DIR, f'{scale}_causal_edge_weight.pt')
            
            if os.path.exists(causal_edge_index_path) and os.path.exists(causal_edge_weight_path):
                causal_edge_index = torch.load(causal_edge_index_path,weights_only=True)
                causal_edge_weight = torch.load(causal_edge_weight_path,weights_only=True)
                
                filtered_causal_edge_index, filtered_causal_edge_weight = filter_causal_network_data(
                    causal_edge_index, causal_edge_weight, valid_basin_ids, all_basin_ids,variables
                )
            else:
                logger.warning(f"{scale}尺度因果网络文件不存在，创建空网络")
                filtered_causal_edge_index = torch.zeros((2, 0), dtype=torch.long)
                filtered_causal_edge_weight = torch.zeros((0,), dtype=torch.float)
            
            # 6. 加载时序数据
            time_series_path = os.path.join(PROCESSED_DATA_DIR, f'{scale}_time_series.csv')
            time_series_df = pd.read_csv(time_series_path)
            time_series_data[scale] = time_series_df
            
            # 7. 保存清洗后的数据
            scale_data[scale] = {
                'valid_basin_ids': valid_basin_ids,
                'river_edge_index': filtered_river_edge_index,
                'river_edge_attr': filtered_river_edge_attr,
                'causal_edge_index': filtered_causal_edge_index,
                'causal_edge_weight': filtered_causal_edge_weight,
                'num_nodes': len(valid_basin_ids)
            }
            
            logger.info(f"{scale}尺度数据加载完成: {len(valid_basin_ids)}个节点, "
                       f"{filtered_river_edge_index.size(1)}条河流边, "
                       f"{filtered_causal_edge_index.size(1)}条因果边")
                       
        except Exception as e:
            logger.error(f"处理{scale}尺度数据时出错: {str(e)}")
            raise
    
    return scale_data, time_series_data, variables

def create_multi_scale_dataloaders(time_series_data, scale_data, variables):
    """
    为三个尺度创建数据加载器
    
    返回:
        dataloaders: 包含三个尺度数据加载器的字典
    """
    logger.info("创建多尺度数据加载器...")
    
    dataloaders = {}
    
    for scale in SCALES:
        logger.info(f"创建{scale}尺度数据加载器...")
        
        basin_ids = scale_data[scale]['valid_basin_ids']
        df = time_series_data[scale]
        
        train_loader, val_loader, test_loader = get_dataloaders(
            df, basin_ids, variables, 
            INPUT_TIME_STEPS, FORECAST_HORIZON,
            batch_size=BATCH_SIZE,
            num_workers=4,
            stride=1
        )
        
        dataloaders[scale] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    return dataloaders

def filter_watershed_hierarchy(watershed_hierarchy, scale_data):
    """
    根据有效流域过滤流域层次结构
    
    参数:
        watershed_hierarchy: 原始层次结构映射列表
        scale_data: 包含各尺度有效流域数据的字典
    
    返回:
        filtered_hierarchy: 过滤后的层次结构映射列表
    """
    filtered_hierarchy = []
    
    # 为每个层次映射创建索引转换
    for level_idx, hierarchy_mapping in enumerate(watershed_hierarchy):
        # 确定当前层级的尺度名称
        if level_idx == 0:  # fine -> medium
            fine_scale, coarse_scale = 'fine', 'medium'
        else:  # medium -> coarse  
            fine_scale, coarse_scale = 'medium', 'coarse'
        
        # 获取有效流域数量
        num_fine_nodes = scale_data[fine_scale]['num_nodes']
        num_coarse_nodes = scale_data[coarse_scale]['num_nodes']
        
        # 创建过滤后的映射
        filtered_mapping = {}
        
        for coarse_idx, fine_indices in hierarchy_mapping.items():
            # 只保留在有效范围内的索引
            if coarse_idx < num_coarse_nodes:
                valid_fine_indices = [idx for idx in fine_indices if idx < num_fine_nodes]
                if valid_fine_indices:  # 只保留非空映射
                    filtered_mapping[coarse_idx] = valid_fine_indices
        
        filtered_hierarchy.append(filtered_mapping)
        logger.info(f"层次{level_idx}: 过滤前{len(hierarchy_mapping)}个映射, 过滤后{len(filtered_mapping)}个映射")
    
    return filtered_hierarchy

def create_scalers_and_normalize_data(time_series_data, variables):
    """
    创建数据MinMax归一化器并归一化数据
    
    参数:
        time_series_data: 时序数据字典
        variables: 变量列表
        
    返回:
        normalized_data: 归一化后的数据字典
        scalers: 归一化器字典
    """
    logger.info("开始数据MinMax归一化...")
    
    normalized_data = {}
    scalers = {}
    
    for scale in SCALES:
        logger.info(f"MinMax归一化{scale}尺度数据...")
        
        df = time_series_data[scale]
        scalers[scale] = {}
        normalized_df = df.copy()
        
        # 对每个变量进行MinMax归一化
        for var in variables:
            if var in df.columns:
                # 计算训练集的统计信息（使用前70%的数据）
                train_size = int(len(df) * TRAIN_RATIO)
                train_data = df[var].iloc[:train_size].dropna()
                
                if len(train_data) > 0:
                    min_val = train_data.min()
                    max_val = train_data.max()
                    
                    # 检查数据范围
                    if max_val > min_val:
                        # MinMax归一化到[0,1]范围
                        normalized_df[var] = (df[var] - min_val) / (max_val - min_val)
                        scalers[scale][var] = {'min': min_val, 'max': max_val}
                        logger.info(f"{scale}尺度 {var}: min={min_val:.4f}, max={max_val:.4f}")
                    else:
                        logger.warning(f"{scale}尺度 {var}: 最大值等于最小值，跳过归一化")
                        scalers[scale][var] = {'min': min_val, 'max': min_val + 1.0}  # 避免除零
                else:
                    logger.warning(f"{scale}尺度 {var}: 训练数据为空")
                    scalers[scale][var] = {'min': 0.0, 'max': 1.0}
        
        normalized_data[scale] = normalized_df
    
    # 保存归一化器
    scalers_path = os.path.join(PROCESSED_DATA_DIR, 'data_scalers.json')
    with open(scalers_path, 'w') as f:
        json.dump(scalers, f, indent=2, default=str)
    
    logger.info("数据MinMax归一化完成")
    return normalized_data, scalers

def inverse_transform_predictions(predictions, scalers, variables, scale='fine'):
    """
    将预测结果反归一化到原始尺度（MinMax）
    
    参数:
        predictions: 预测结果 [num_samples, num_basins, num_features, forecast_horizon]
        scalers: 归一化器字典
        variables: 变量列表
        scale: 尺度名称
        
    返回:
        denormalized_predictions: 反归一化后的预测结果
    """
    # 检查输入类型并转换为numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # 检查输入是否包含无效值
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        logger.warning("预测结果包含NaN或Inf值，进行清理...")
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 限制预测值范围，确保在[0,1]内
    predictions = np.clip(predictions, 0.0, 1.0)

    # 使用numpy的copy方法而不是clone
    denormalized = predictions.copy()
    
    for i, var in enumerate(variables):
        if var in scalers[scale]:
            min_val = scalers[scale][var]['min']
            max_val = scalers[scale][var]['max']
            
            # 检查数值范围，避免溢出
            range_val = max_val - min_val
            if range_val > 1e6:  # 如果范围太大，进行安全处理
                logger.warning(f"变量 {var} 的范围过大 ({range_val:.2e})，进行安全反归一化")
                # 使用更安全的方法
                denormalized[:, :, i, :] = predictions[:, :, i, :] * range_val + min_val
                # 检查结果并处理溢出
                denormalized[:, :, i, :] = np.clip(denormalized[:, :, i, :], 
                                                  min_val - abs(min_val), 
                                                  max_val + abs(max_val))
            else:
                # 正常反归一化
                denormalized[:, :, i, :] = predictions[:, :, i, :] * range_val + min_val
    
    # 最终检查并清理
    denormalized = np.nan_to_num(denormalized, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return denormalized

def calculate_metrics_all_features_denormalized(predictions, targets, scalers, variables, scale='fine'):
    """
    计算反归一化后的评估指标
    
    参数:
        predictions: 归一化的预测结果 [num_samples, num_basins, num_features, forecast_horizon]
        targets: 归一化的目标结果 [num_samples, num_basins, num_features, forecast_horizon]
        scalers: 标准化器字典
        variables: 变量列表
        scale: 尺度名称
        
    返回:
        metrics: 包含整体和分特征指标的字典
    """
    # 反归一化预测和目标
    denorm_predictions = inverse_transform_predictions(predictions, scalers, variables, scale)
    denorm_targets = inverse_transform_predictions(targets, scalers, variables, scale)
    
    num_features = denorm_predictions.shape[2]
    feature_metrics = {}
    
    # 计算整体指标（反归一化后）
    y_true_flat = denorm_targets.flatten()
    y_pred_flat = denorm_predictions.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        overall_metrics = {'mse': float('inf'), 'mae': float('inf'), 'r2': 0.0}
    else:
        overall_metrics = {
            'mse': mean_squared_error(y_true_clean, y_pred_clean),
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'r2': r2_score(y_true_clean, y_pred_clean)
        }
    
    # 计算每个特征的指标（反归一化后）
    for i in range(num_features):
        feature_pred = denorm_predictions[:, :, i, :].flatten()
        feature_target = denorm_targets[:, :, i, :].flatten()
        
        # 移除NaN值
        mask = ~(np.isnan(feature_target) | np.isnan(feature_pred))
        feature_target_clean = feature_target[mask]
        feature_pred_clean = feature_pred[mask]
        
        if len(feature_target_clean) == 0:
            feature_metrics[f'feature_{i}'] = {'mse': float('inf'), 'mae': float('inf'), 'r2': 0.0}
        else:
            feature_metrics[f'feature_{i}'] = {
                'mse': mean_squared_error(feature_target_clean, feature_pred_clean),
                'mae': mean_absolute_error(feature_target_clean, feature_pred_clean),
                'r2': r2_score(feature_target_clean, feature_pred_clean)
            }
    
    return {
        'overall': overall_metrics,
        'features': feature_metrics
    }

def calculate_metrics_normalized(predictions, targets):
    """
    在归一化空间计算评估指标
    
    参数:
        predictions: 归一化的预测结果 [num_samples, num_basins, num_features, forecast_horizon]
        targets: 归一化的目标结果 [num_samples, num_basins, num_features, forecast_horizon]
        
    返回:
        metrics: 包含整体和分特征指标的字典
    """
    num_features = predictions.shape[2]
    feature_metrics = {}
    
    # 计算整体指标（归一化空间）
    y_true_flat = targets.flatten()
    y_pred_flat = predictions.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        overall_metrics = {'mse': float('inf'), 'mae': float('inf'), 'r2': 0.0}
    else:
        overall_metrics = {
            'mse': mean_squared_error(y_true_clean, y_pred_clean),
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'r2': r2_score(y_true_clean, y_pred_clean)
        }
    
    # 计算每个特征的指标（归一化空间）
    for i in range(num_features):
        feature_pred = predictions[:, :, i, :].flatten()
        feature_target = targets[:, :, i, :].flatten()
        
        # 移除NaN值
        mask = ~(np.isnan(feature_target) | np.isnan(feature_pred))
        feature_target_clean = feature_target[mask]
        feature_pred_clean = feature_pred[mask]
        
        if len(feature_target_clean) == 0:
            feature_metrics[f'feature_{i}'] = {'mse': float('inf'), 'mae': float('inf'), 'r2': 0.0}
        else:
            feature_metrics[f'feature_{i}'] = {
                'mse': mean_squared_error(feature_target_clean, feature_pred_clean),
                'mae': mean_absolute_error(feature_target_clean, feature_pred_clean),
                'r2': r2_score(feature_target_clean, feature_pred_clean)
            }
    
    return {
        'overall': overall_metrics,
        'features': feature_metrics
    }

def check_and_fix_nan_in_model(model, device):
    """
    检查模型参数中是否有NaN值并尝试修复
    """
    logger.info("检查模型参数中的NaN值...")
    nan_found = False
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"参数 {name} 包含NaN值")
            nan_found = True
            # 尝试修复：将NaN替换为0
            param.data[torch.isnan(param.data)] = 0.0
            logger.info(f"已将参数 {name} 中的NaN值替换为0")
    
    if not nan_found:
        logger.info("模型参数中没有发现NaN值")
    
    return nan_found

def check_gradients_detailed(model, batch_idx, stage="training"):
    """
    详细检查模型梯度，定位NaN/Inf的来源
    
    参数:
        model: 模型
        batch_idx: 批次索引
        stage: 检查阶段（"training" 或 "validation"）
    """
    logger.info(f"=== {stage}阶段 - Batch {batch_idx} 梯度检查 ===")
    
    total_params = 0
    nan_grads = 0
    inf_grads = 0
    large_grads = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        
        if param.grad is not None:
            grad = param.grad.data
            
            # 检查NaN
            if torch.isnan(grad).any():
                nan_grads += 1
                nan_indices = torch.where(torch.isnan(grad))
                #logger.error(f"参数 {name} 的梯度包含NaN，位置: {nan_indices}")
                logger.error(f"参数形状: {grad.shape}, 参数范数: {param.data.norm().item():.6f}")
                
                # 检查对应的参数值
                param_data = param.data
                if torch.isnan(param_data).any():
                    logger.error(f"参数 {name} 本身也包含NaN值!")
                
            # 检查Inf
            elif torch.isinf(grad).any():
                inf_grads += 1
                inf_indices = torch.where(torch.isinf(grad))
                #logger.error(f"参数 {name} 的梯度包含Inf，位置: {inf_indices}")
                logger.error(f"参数形状: {grad.shape}, 参数范数: {param.data.norm().item():.6f}")
                
            # 检查过大的梯度
            elif grad.norm().item() > 100.0:
                large_grads += 1
                #logger.warning(f"参数 {name} 的梯度范数过大: {grad.norm().item():.6f}")
                
            # 记录正常梯度的统计信息
            else:
                grad_norm = grad.norm().item()
                if grad_norm > 10.0:  # 记录较大的梯度
                    logger.info(f"参数 {name} 梯度范数: {grad_norm:.6f}")
        else:
            logger.warning(f"参数 {name} 没有梯度")
    
    logger.info(f"梯度检查结果: 总参数={total_params}, NaN梯度={nan_grads}, Inf梯度={inf_grads}, 过大梯度={large_grads}")
    
    return nan_grads > 0 or inf_grads > 0

def check_tensor_stability(tensor, name, stage="", batch_idx=0):
    """
    检查张量的数值稳定性
    
    参数:
        tensor: 要检查的张量
        name: 张量名称
        stage: 检查阶段
        batch_idx: 批次索引
    """
    if tensor is None:
        logger.warning(f"{stage} - {name}: 张量为None")
        return False
    
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        nan_ratio = nan_count / tensor.numel()
        logger.error(f"{stage} - {name}: 包含NaN值! 数量: {nan_count}/{tensor.numel()} ({nan_ratio:.2%})")
        return False
    
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        inf_ratio = inf_count / tensor.numel()
        logger.error(f"{stage} - {name}: 包含Inf值! 数量: {inf_count}/{tensor.numel()} ({inf_ratio:.2%})")
        return False
    
    # 检查数值范围
    tensor_min = tensor.min().item()
    tensor_max = tensor.max().item()
    tensor_mean = tensor.mean().item()
    tensor_std = tensor.std().item()
    
    if abs(tensor_max) > 1000 or abs(tensor_min) > 1000:
        logger.warning(f"{stage} - {name}: 数值范围过大 [{tensor_min:.6f}, {tensor_max:.6f}]")
    
    if tensor_std > 100:
        logger.warning(f"{stage} - {name}: 标准差过大 {tensor_std:.6f}")
    
    logger.info(f"{stage} - {name}: 形状={tensor.shape}, 范围=[{tensor_min:.6f}, {tensor_max:.6f}], 均值={tensor_mean:.6f}, 标准差={tensor_std:.6f}")
    
    return True

def debug_model_outputs(model, batch_x, multi_scale_graph_data, device):
    """
    调试模型各层的输出，找出NaN的来源
    """
    logger.info("开始调试模型输出...")
    
    # 检查输入数据
    logger.info(f"输入数据形状: {batch_x.shape}")
    logger.info(f"输入数据范围: [{batch_x.min().item():.6f}, {batch_x.max().item():.6f}]")
    
    # 检查图数据
    for scale_name, graph_data in multi_scale_graph_data.items():
        logger.info(f"{scale_name}尺度图数据:")
        for key, value in graph_data.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: 形状={value.shape}, 范围=[{value.min().item():.6f}, {value.max().item():.6f}]")
                if torch.isnan(value).any():
                    logger.error(f"  {key} 包含NaN值!")
                if torch.isinf(value).any():
                    logger.error(f"  {key} 包含Inf值!")
    
    # 尝试逐层前向传播来定位问题
    model.eval()
    with torch.no_grad():
        try:
            # 这里可以添加模型内部的调试代码
            # 如果模型有中间层输出，可以在这里检查
            logger.info("模型前向传播完成")
        except Exception as e:
            logger.error(f"模型前向传播出错: {str(e)}")
            raise

def check_input_projection(model, batch_x, device):
    """详细检查输入投影层的数值稳定性"""
    logger.info("=== 输入投影层检查 ===")
    
    # 检查输入
    logger.info(f"输入batch_x: 形状={batch_x.shape}, 范围=[{batch_x.min():.6f}, {batch_x.max():.6f}]")
    
    # 检查投影层结构
    input_projection = model.input_projection
    logger.info(f"输入投影层类型: {type(input_projection)}")
    
    if isinstance(input_projection, nn.Sequential):
        # 如果是Sequential，检查其中的Linear层
        logger.info(f"Sequential包含 {len(input_projection)} 个子模块")
        
        # 找到第一个Linear层
        first_linear = None
        for i, module in enumerate(input_projection):
            if isinstance(module, nn.Linear):
                first_linear = module
                logger.info(f"找到第{i+1}个Linear层: {module}")
                break
        
        if first_linear is not None:
            proj_weight = first_linear.weight
            proj_bias = first_linear.bias
        else:
            logger.error("在Sequential中未找到Linear层")
            return None
    else:
        # 如果是单个Linear层
        proj_weight = input_projection.weight
        proj_bias = input_projection.bias
    
    logger.info(f"投影权重: 形状={proj_weight.shape}, 范围=[{proj_weight.min():.6f}, {proj_weight.max():.6f}]")
    logger.info(f"投影偏置: 形状={proj_bias.shape}, 范围=[{proj_bias.min():.6f}, {proj_bias.max():.6f}]")
    
    # 手动计算投影结果 - 按照模型的实际处理方式
    with torch.no_grad():
        # 按照模型的处理方式：先permute，再投影
        batch_size, num_nodes, num_features, time_steps = batch_x.shape
        
        # 步骤1: permute到 [batch, nodes, time_steps, features]
        x_permuted = batch_x.permute(0, 1, 3, 2)  # [batch_size, num_nodes, time_steps, num_features]
        logger.info(f"permute后: 形状={x_permuted.shape}")
        
        # 步骤2: 使用输入投影层
        projected = input_projection(x_permuted)  # [batch_size, num_nodes, time_steps, hidden_dim]
        logger.info(f"投影后: 形状={projected.shape}")
        
        # 步骤3: 再permute回 [batch, nodes, hidden_dim, time_steps]
        projected_final = projected.permute(0, 1, 3, 2)  # [batch_size, num_nodes, hidden_dim, time_steps]
        logger.info(f"最终形状: 形状={projected_final.shape}")
        
        logger.info(f"投影输出: 范围=[{projected_final.min():.6f}, {projected_final.max():.6f}]")
        logger.info(f"投影输出统计: 均值={projected_final.mean():.6f}, 标准差={projected_final.std():.6f}")
        
        # 检查是否有异常值
        if torch.isnan(projected_final).any():
            logger.error("投影输出包含NaN!")
        if torch.isinf(projected_final).any():
            logger.error("投影输出包含Inf!")
        if projected_final.abs().max() > 100:
            logger.error(f"投影输出数值过大: {projected_final.abs().max():.6f}")
    
    return projected_final

def check_input_projection_gradients(model):
    """检查输入投影层的梯度"""
    input_projection = model.input_projection
    
    if isinstance(input_projection, nn.Sequential):
        # 找到第一个Linear层
        first_linear = None
        for module in input_projection:
            if isinstance(module, nn.Linear):
                first_linear = module
                break
        
        if first_linear is None:
            logger.error("在Sequential中未找到Linear层")
            return
        
        proj_weight = first_linear.weight
        proj_bias = first_linear.bias
    else:
        proj_weight = input_projection.weight
        proj_bias = input_projection.bias
    
    if proj_weight.grad is not None:
        weight_grad = proj_weight.grad
        logger.info(f"输入投影权重梯度: 范围=[{weight_grad.min():.6f}, {weight_grad.max():.6f}]")
        
        if torch.isnan(weight_grad).any():
            logger.error("输入投影权重梯度包含NaN!")
        if torch.isinf(weight_grad).any():
            logger.error("输入投影权重梯度包含Inf!")
    
    if proj_bias.grad is not None:
        bias_grad = proj_bias.grad
        logger.info(f"输入投影偏置梯度: 范围=[{bias_grad.min():.6f}, {bias_grad.max():.6f}]")
        
        if torch.isnan(bias_grad).any():
            logger.error("输入投影偏置梯度包含NaN!")
        if torch.isinf(bias_grad).any():
            logger.error("输入投影偏置梯度包含Inf!")

def check_model_forward_step_by_step(model, batch_x, multi_scale_graph_data, device):
    """逐步检查模型前向传播的每个步骤"""
    logger.info("=== 模型前向传播逐步检查 ===")
    
    # 步骤1: 检查输入投影
    logger.info("步骤1: 输入投影")
    projected = check_input_projection(model, batch_x, device)
    
    if torch.isnan(projected).any() or torch.isinf(projected).any():
        logger.error("输入投影层输出异常，停止检查")
        return False
    
    # 步骤2: 检查图卷积层
    logger.info("步骤2: 图卷积层")
    # 这里需要根据您的模型结构逐步检查
    # 由于模型结构复杂，建议先检查输入投影层
    
    return True
    
def train_epoch_optimized(model, dataloader, criterion, optimizer, device, all_scale_data, scaler):
    """优化后的训练epoch - 预测所有特征变量"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (batch_x, batch_y) in enumerate(tqdm(dataloader, desc="训练进度")):
        # 在 train_epoch_optimized 的第一批内加入
        if batch_idx == 0:
            with torch.no_grad():
                # x: [B, N, F, T]
                x0 = batch_x  # 已是归一化后的输入
                B, N, F, T = x0.shape
                x_flat = x0.reshape(-1, F)  # 合并 B,N,T 维
                mins = x_flat.min(dim=0).values
                maxs = x_flat.max(dim=0).values
                frac0 = (x_flat == 0).float().mean(dim=0)
                frac1 = (x_flat == 1).float().mean(dim=0)
                for i in range(F):
                    logger.info(f"feature[{i}] min={mins[i].item():.6f}, max={maxs[i].item():.6f}, "
                                f"p(x==0)={frac0[i].item():.4%}, p(x==1)={frac1[i].item():.4%}")
        # 将数据移到GPU
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)  # 标签保持float32

        """ # 检查每个 batch 样本的范围（只在第一个batch时输出详细信息）
        if batch_idx == 0:
            logger.info(f"=== 第一个batch数据统计 ===")
            # 使用新的检查函数
            check_tensor_stability(batch_x, "batch_x", "输入检查", batch_idx)
            check_tensor_stability(batch_y, "batch_y", "输入检查", batch_idx)
        
        # 检查输入
        if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
            logger.error(f"batch_x has NaN/Inf at batch {batch_idx}")
            continue
        if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
            logger.error(f"batch_y has NaN/Inf at batch {batch_idx}")
            continue """

        optimizer.zero_grad()
        
        # 准备多尺度图数据 - 转换为float16
        multi_scale_graph_data = {
            'fine': {
                'river_edge_index': all_scale_data['fine']['river_edge_index'].to(device),
                'river_edge_attr': all_scale_data['fine']['river_edge_attr'].to(device),
                'causal_edge_index': all_scale_data['fine']['causal_edge_index'].to(device),
                'causal_edge_weight': all_scale_data['fine']['causal_edge_weight'].to(device)
            },
            'medium': {
                'river_edge_index': all_scale_data['medium']['river_edge_index'].to(device),
                'river_edge_attr': all_scale_data['medium']['river_edge_attr'].to(device),
                'causal_edge_index': all_scale_data['medium']['causal_edge_index'].to(device),
                'causal_edge_weight': all_scale_data['medium']['causal_edge_weight'].to(device)
            },
            'coarse': {
                'river_edge_index': all_scale_data['coarse']['river_edge_index'].to(device),
                'river_edge_attr': all_scale_data['coarse']['river_edge_attr'].to(device),
                'causal_edge_index': all_scale_data['coarse']['causal_edge_index'].to(device),
                'causal_edge_weight': all_scale_data['coarse']['causal_edge_weight'].to(device)
            }
        }

        """ # 检查图数据（只在第一个batch时）
        if batch_idx == 0:
            logger.info("=== 图数据检查 ===")
            for scale_name, graph_data in multi_scale_graph_data.items():
                for key, value in graph_data.items():
                    if isinstance(value, torch.Tensor):
                        # 只检查浮点类型的张量
                        if value.dtype in [torch.float16, torch.float32, torch.float64]:
                            check_tensor_stability(value, f"{scale_name}_{key}", "图数据检查", batch_idx)
                        else:
                            # 对于整数类型，只记录基本信息
                            logger.info(f"{scale_name}_{key}: 整数类型张量，形状={value.shape}, 数据类型={value.dtype}")

        # 逐步检查模型前向传播
        forward_check_result = check_model_forward_step_by_step(model, batch_x, multi_scale_graph_data, device) """

        # 使用混合精度训练
        with autocast():
            # 前向传播 - 预测所有特征
            pred = model(batch_x, multi_scale_graph_data)  # [batch, nodes, features, forecast_horizon]
            
            # 注释掉模型输出检查
            # if batch_idx == 0:
            #     logger.info("=== 模型输出检查 ===")
            #     check_tensor_stability(pred, "pred", "模型输出", batch_idx)
            
            # 保留基本的NaN/Inf检查
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                logger.error(f"pred has NaN/Inf at batch {batch_idx}")
                logger.error(f"pred形状: {pred.shape}")
                
                # 调试模型输出
                if batch_idx == 0:
                    debug_model_outputs(model, batch_x, multi_scale_graph_data, device)
                
                continue

            # 计算所有特征的损失
            loss = criterion(pred, batch_y)
        
            # 保留基本的损失检查
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"loss is NaN/Inf at batch {batch_idx}")
                logger.error(f"pred统计: min={pred.min().item():.6f}, max={pred.max().item():.6f}, mean={pred.mean().item():.6f}")
                logger.error(f"batch_y统计: min={batch_y.min().item():.6f}, max={batch_y.max().item():.6f}, mean={batch_y.mean().item():.6f}")
                continue

        # 使用scaler进行反向传播
        scaler.scale(loss).backward()
        
        # 注释掉详细的梯度检查
        # if batch_idx == 0 or batch_idx % 5 == 0:  # 每5个batch检查一次
        #     logger.info(f"=== Batch {batch_idx} 反向传播后梯度检查 ===")
        #     has_gradient_problem = check_gradients_detailed(model, batch_idx, "训练")
            
        #     if has_gradient_problem:
        #         logger.error(f"Batch {batch_idx} 发现梯度问题，跳过此batch")
        #         optimizer.zero_grad()
        #         continue

        # 注释掉梯度范数检查
        # grad_norm = 0.0
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         param_norm = param.grad.data.norm(2)
        #         grad_norm += param_norm.item() ** 2
        # grad_norm = grad_norm ** (1. / 2)
        
        # if batch_idx % 10 == 0:  # 每10个batch输出一次梯度信息
        #     logger.info(f"Batch {batch_idx}: 梯度范数 = {grad_norm:.6f}")
            
        #     # 如果梯度范数过大，记录详细信息
        #     if grad_norm > 100.0:
        #         #logger.warning(f"Batch {batch_idx}: 梯度范数过大 ({grad_norm:.6f})，进行详细检查")
        #         check_gradients_detailed(model, batch_idx, "梯度过大检查")

        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 注释掉梯度裁剪后的检查
        # if batch_idx == 0 or batch_idx % 5 == 0:
        #     logger.info(f"=== Batch {batch_idx} 梯度裁剪后检查 ===")
        #     check_gradients_detailed(model, batch_idx, "梯度裁剪后")
            
        # 使用scaler更新参数
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 定期清理GPU内存
        if num_batches % 10 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def validate_epoch_optimized(model, dataloader, criterion, device, all_scale_data, scalers=None, variables=None, scale='fine'):
    """优化后的验证epoch - 在归一化空间评估"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="验证进度"):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # 准备多尺度图数据
            multi_scale_graph_data = {
                'fine': {
                    'river_edge_index': all_scale_data['fine']['river_edge_index'].to(device),
                    'river_edge_attr': all_scale_data['fine']['river_edge_attr'].to(device),
                    'causal_edge_index': all_scale_data['fine']['causal_edge_index'].to(device),
                    'causal_edge_weight': all_scale_data['fine']['causal_edge_weight'].to(device)
                },
                'medium': {
                    'river_edge_index': all_scale_data['medium']['river_edge_index'].to(device),
                    'river_edge_attr': all_scale_data['medium']['river_edge_attr'].to(device),
                    'causal_edge_index': all_scale_data['medium']['causal_edge_index'].to(device),
                    'causal_edge_weight': all_scale_data['medium']['causal_edge_weight'].to(device)
                },
                'coarse': {
                    'river_edge_index': all_scale_data['coarse']['river_edge_index'].to(device),
                    'river_edge_attr': all_scale_data['coarse']['river_edge_attr'].to(device),
                    'causal_edge_index': all_scale_data['coarse']['causal_edge_index'].to(device),
                    'causal_edge_weight': all_scale_data['coarse']['causal_edge_weight'].to(device)
                }
            }

            with autocast():
                # 前向传播 - 预测所有特征
                outputs = model(batch_x, multi_scale_graph_data)
                loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # 计算评估指标 - 在归一化空间
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # 使用归一化空间的指标计算
    metrics = calculate_metrics_normalized(predictions, targets)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return avg_loss, metrics

def normalize_edge_attributes(scale_data):
    """
    对边属性进行归一化
    
    参数:
        scale_data: 包含各尺度数据的字典
        
    返回:
        normalized_scale_data: 归一化后的数据字典
        edge_scalers: 边属性归一化器字典
    """
    logger.info("开始归一化边属性...")
    
    normalized_scale_data = {}
    edge_scalers = {}
    
    for scale in SCALES:
        logger.info(f"归一化{scale}尺度边属性...")
        
        # 获取原始数据
        river_edge_attr = scale_data[scale]['river_edge_attr']
        causal_edge_weight = scale_data[scale]['causal_edge_weight']
        
        # 归一化河流边属性
        if river_edge_attr.size(0) > 0:
            river_min = river_edge_attr.min(dim=0, keepdim=True)[0]
            river_max = river_edge_attr.max(dim=0, keepdim=True)[0]
            river_range = river_max - river_min
            
            # 避免除零
            river_range = torch.where(river_range == 0, torch.ones_like(river_range), river_range)
            
            normalized_river_attr = (river_edge_attr - river_min) / river_range
            
            edge_scalers[scale] = {
                'river_min': river_min,
                'river_max': river_max,
                'river_range': river_range
            }
            
            logger.info(f"{scale}尺度河流边属性归一化: 原始范围=[{river_edge_attr.min().item():.2f}, {river_edge_attr.max().item():.2f}], 归一化后范围=[{normalized_river_attr.min().item():.4f}, {normalized_river_attr.max().item():.4f}]")
        else:
            normalized_river_attr = river_edge_attr
            edge_scalers[scale] = {
                'river_min': torch.zeros(1, river_edge_attr.size(1)),
                'river_max': torch.ones(1, river_edge_attr.size(1)),
                'river_range': torch.ones(1, river_edge_attr.size(1))
            }
        
        # 创建归一化后的数据
        normalized_scale_data[scale] = {
            'valid_basin_ids': scale_data[scale]['valid_basin_ids'],
            'river_edge_index': scale_data[scale]['river_edge_index'],
            'river_edge_attr': normalized_river_attr,
            'causal_edge_index': scale_data[scale]['causal_edge_index'],
            'causal_edge_weight': causal_edge_weight,  # 因果边权重范围正常，不需要归一化
            'num_nodes': scale_data[scale]['num_nodes']
        }
    
    # 保存边属性归一化器
    edge_scalers_path = os.path.join(PROCESSED_DATA_DIR, 'edge_scalers.json')
    with open(edge_scalers_path, 'w') as f:
        # 转换为可序列化的格式
        serializable_scalers = {}
        for scale, scaler in edge_scalers.items():
            serializable_scalers[scale] = {
                'river_min': scaler['river_min'].tolist(),
                'river_max': scaler['river_max'].tolist(),
                'river_range': scaler['river_range'].tolist()
            }
        json.dump(serializable_scalers, f, indent=2)
    
    logger.info("边属性归一化完成")
    return normalized_scale_data, edge_scalers

def train_model():
    """主训练函数"""
    logger.info("开始模型训练...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置内存优化
    setup_memory_optimization()
    
    # 初始化混合精度训练
    scaler = GradScaler()
    logger.info("混合精度训练已启用")

    # 加载和清洗数据
    scale_data, time_series_data, variables = load_and_filter_data()
    
    # 归一化边属性
    normalized_scale_data, edge_scalers = normalize_edge_attributes(scale_data)

    # 添加数据归一化
    normalized_time_series_data, data_scalers = create_scalers_and_normalize_data(time_series_data, variables)

    # 归一化后，检查每个变量的统计信息
    for scale in SCALES:
        df = normalized_time_series_data[scale]
        logger.info(f"{scale}尺度数据统计 (MinMax归一化后):")
        for var in variables:
            if var in df.columns:
                min_val = df[var].min()
                max_val = df[var].max()
                mean_val = df[var].mean()
                std_val = df[var].std()
                nan_count = df[var].isna().sum()
                logger.info(f"{var}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}, NaN数={nan_count}")

    # 加载流域层次结构
    watershed_hierarchy = torch.load(WATERSHED_HIERARCHY_PATH,weights_only=True)
    
    # 创建数据加载器 - 使用fine尺度作为主要训练数据
    dataloaders = create_multi_scale_dataloaders(normalized_time_series_data, normalized_scale_data, variables)
    # 过滤流域层次结构以匹配有效流域
    watershed_hierarchy = filter_watershed_hierarchy(watershed_hierarchy, normalized_scale_data)
    
    # 使用fine尺度进行训练
    train_loader = dataloaders['fine']['train']
    val_loader = dataloaders['fine']['val']
    fine_scale_data = normalized_scale_data['fine']
    
    # 初始化模型
    model = GraphUNetDualTransformer(
        hidden_dim=HIDDEN_DIM,
        forecast_horizon=FORECAST_HORIZON,
        watershed_hierarchy=watershed_hierarchy,
        num_features=len(variables),
        basin_ids=fine_scale_data['valid_basin_ids'],
        time_steps=INPUT_TIME_STEPS,
        blocks_per_level=BLOCKS_PER_LEVEL,
        edge_attr_dim=fine_scale_data['river_edge_attr'].size(1) if fine_scale_data['river_edge_attr'].size(0) > 0 else 4,
        debug=True  # 关闭调试模式
    )
    
    # 初始化权重
    model.apply(init_weights)
    model.to(device)

     # 检查模型参数
    logger.info("=== 模型初始化检查 ===")
    check_and_fix_nan_in_model(model, device)
    
    # 注释掉模型结构检查
    # logger.info("=== 模型结构信息 ===")
    # total_params = count_parameters(model)
    # logger.info(f"模型参数量: {total_params:,}")

    # 注释掉详细的参数检查
    # for name, module in model.named_modules():
    #     if hasattr(module, 'weight') and module.weight is not None:
    #         weight_norm = module.weight.data.norm(2).item()
    #         logger.info(f"{name}.weight: 范数={weight_norm:.6f}, 形状={module.weight.shape}")
    #         if torch.isnan(module.weight).any():
    #             logger.error(f"{name}.weight 包含NaN值!")
    #     elif isinstance(module, nn.Sequential):
    #         for i, submodule in enumerate(module):
    #             if hasattr(submodule, 'weight') and submodule.weight is not None:
    #                 weight_norm = submodule.weight.data.norm(2).item()
    #                 logger.info(f"{name}[{i}].weight: 范数={weight_norm:.6f}, 形状={submodule.weight.shape}")
    #                 if torch.isnan(submodule.weight).any():
    #                     logger.error(f"{name}[{i}].weight 包含NaN值!")

    # 使用混合精度优化器
    criterion = HydrologicalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # 训练历史
    train_losses = []
    val_losses = []
    val_metrics_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("开始训练循环...")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch_optimized(model, train_loader, criterion, optimizer, device, normalized_scale_data,scaler)
        
        # 验证 - 使用反归一化的指标
        val_loss, val_metrics = validate_epoch_optimized(
            model, val_loader, criterion, device, normalized_scale_data, 
            # 移除scalers和variables参数，因为不再需要反归一化
            # scalers=data_scalers, variables=variables, scale='fine'
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)
        
        epoch_time = time.time() - start_time
        
        # 在训练循环中修改日志输出
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        logger.info(f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
        logger.info(f"整体验证指标 (归一化空间) - MSE: {val_metrics['overall']['mse']:.6f}, MAE: {val_metrics['overall']['mae']:.6f}, R²: {val_metrics['overall']['r2']:.6f}")

        # 输出每个特征的指标（归一化空间）
        for i, var_name in enumerate(VARIABLES):
            feature_metrics = val_metrics['features'][f'feature_{i}']
            logger.info(f"{var_name} (归一化空间) - MSE: {feature_metrics['mse']:.6f}, MAE: {feature_metrics['mae']:.6f}, R²: {feature_metrics['r2']:.6f}")
        logger.info(f"学习率: {optimizer.param_groups[0]['lr']:.2e}, 耗时: {epoch_time:.2f}秒")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存模型
            model_save_path = os.path.join(MODEL_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'data_scalers': data_scalers,  # 保存数据标准化器
                'edge_scalers': edge_scalers,  # 保存边属性标准化器
                'model_config': {
                    'hidden_dim': HIDDEN_DIM,
                    'forecast_horizon': FORECAST_HORIZON,
                    'num_features': len(variables),
                    'input_time_steps': INPUT_TIME_STEPS,
                    'blocks_per_level': BLOCKS_PER_LEVEL
                }
            }, model_save_path)
            
            logger.info(f"保存最佳模型 (验证损失: {val_loss:.6f})")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= PATIENCE:
            logger.info(f"早停触发 (耐心值: {PATIENCE})")
            break
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics_history
    }
    
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_metrics_history)
    
    logger.info("训练完成!")
    return model, history

def plot_training_curves(train_losses, val_losses, val_metrics_history):
    """绘制训练曲线"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(train_losses, label='训练损失')
        axes[0, 0].plot(val_losses, label='验证损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MSE曲线
        mse_values = [m['mse'] for m in val_metrics_history]
        axes[0, 1].plot(mse_values)
        axes[0, 1].set_title('验证MSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].grid(True)
        
        # MAE曲线
        mae_values = [m['mae'] for m in val_metrics_history]
        axes[1, 0].plot(mae_values)
        axes[1, 0].set_title('验证MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].grid(True)
        
        # R²曲线
        r2_values = [m['r2'] for m in val_metrics_history]
        axes[1, 1].plot(r2_values)
        axes[1, 1].set_title('验证R²')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("训练曲线已保存")
        
    except Exception as e:
        logger.warning(f"绘制训练曲线时出错: {str(e)}")

if __name__ == "__main__":
    try:
        model, history = train_model()
        logger.info("训练脚本执行完成!")
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}", exc_info=True)
        raise