import os
import logging
import numpy as np
import pandas as pd
import torch
import networkx as nx
from tqdm import tqdm
import time
from statsmodels.tsa.stattools import grangercausalitytests
from joblib import Parallel, delayed
from config import *

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'causal_analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def build_granger_network(data, maxlag=GRANGER_MAX_LAG, p_threshold=GRANGER_P_THRESHOLD, 
                         min_corr=MIN_CORRELATION, n_jobs=CAUSAL_N_JOBS):
    """
    构建格兰杰因果网络
    
    参数:
        data: 包含时间序列的DataFrame
        maxlag: 最大滞后值
        p_threshold: p值阈值，用于判断因果关系的显著性
        min_corr: 最小相关系数阈值
        n_jobs: 并行作业数
        
    返回:
        nx.DiGraph: 因果有向图
    """
    start_time = time.time()
    logger.info(f"开始构建格兰杰因果网络，数据形状: {data.shape}")
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 获取时间序列数据的列数
    columns = data.columns.tolist()
    num_series = len(columns)
    
    # 检查时间序列长度是否足够
    if len(data) < maxlag + 2:
        logger.warning(f"时间序列长度({len(data)})不足，无法进行Granger因果检验")
        return G
    
    # 计算相关矩阵以筛选潜在因果对
    corr_matrix = data.corr().abs().values
    
    # 筛选潜在因果对
    potential_pairs = []
    
    for i in range(num_series):
        for j in range(num_series):
            if i != j and corr_matrix[i, j] >= min_corr:
                potential_pairs.append((i, j, corr_matrix[i, j]))
    
    logger.info(f"筛选出 {len(potential_pairs)} 对潜在因果关系 (相关系数>={min_corr})")
    
    # 定义单个Granger测试函数
    def test_single_granger(source_idx, target_idx, correlation):
        source = columns[source_idx]
        target = columns[target_idx]
        
        try:
            # 提取两个序列
            pair_data = data[[target, source]]
            
            # 进行Granger因果检验
            test_result = grangercausalitytests(
                pair_data, maxlag=maxlag, verbose=False
            )
            
            # 查找所有滞后的最小p值
            min_p = min(test_result[i][0]['ssr_ftest'][1] for i in range(1, maxlag+1))
            
            # 找出具有最小p值的滞后值
            optimal_lag = min(
                range(1, maxlag+1),
                key=lambda i: test_result[i][0]['ssr_ftest'][1]
            )
            
            if min_p < p_threshold:
                # 返回因果边信息
                return (source, target, correlation, min_p, optimal_lag)
            return None
        except Exception as e:
            logger.debug(f"测试 {source} → {target} 时出错: {str(e)}")
            return None
    
    # 并行执行Granger因果检验
    logger.info(f"开始执行Granger因果检验，使用{n_jobs}个线程...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(test_single_granger)(i, j, corr) 
        for i, j, corr in tqdm(potential_pairs, desc="Granger检验进度")
    )
    
    # 添加显著的因果边到图中
    significant_edges = [edge for edge in results if edge is not None]
    
    for source, target, correlation, p_value, lag in significant_edges:
        # 添加边及其属性
        G.add_edge(
            source, 
            target, 
            weight=correlation,  # 使用相关系数作为边权重
            p_value=p_value,
            lag=lag
        )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Granger因果分析完成，用时:{elapsed_time:.1f}秒，发现{len(G.edges())}条显著因果边")
    
    return G

def build_multi_scale_causal_networks(time_series_data, basin_ids_dict, variables, n_jobs=CAUSAL_N_JOBS):
    """
    为不同尺度的流域构建因果网络
    
    参数:
        time_series_data: 包含时间序列的DataFrame
        basin_ids_dict: 不同尺度的流域ID字典 {'fine': [...], 'medium': [...], 'coarse': [...]}
        variables: 变量列表
        n_jobs: 并行作业数
        
    返回:
        causal_edges_dict: 不同尺度的因果边 {'fine': (edge_index, edge_weight), ...}
    """
    logger.info("开始构建多尺度因果网络...")
    
    causal_edges_dict = {}
    scales = ['fine', 'medium', 'coarse']
    
    # 为每个尺度构建因果网络
    for scale in scales:
        logger.info(f"处理{scale}尺度因果网络...")
        basin_ids = basin_ids_dict[scale]
        
        # 提取该尺度的时间序列数据
        scale_data = time_series_data[time_series_data['watershed_id'].isin(basin_ids)]
        
        # 重塑数据为适合因果分析的格式
        pivoted_data = []
        
        for var in variables:
            # 对每个流域的每个变量创建列
            for basin_id in basin_ids:
                basin_data = scale_data[scale_data['watershed_id'] == basin_id]
                if not basin_data.empty:
                    col_name = f"{var}_{basin_id}"
                    # 确保数据按时间排序
                    basin_var_series = basin_data.sort_values('time')[var]
                    
                    # 检查是否有足够的数据点
                    if len(basin_var_series) > GRANGER_MAX_LAG + 2:
                        pivoted_data.append((col_name, basin_var_series.values))
        
        # 创建DataFrame
        if pivoted_data:
            df = pd.DataFrame(dict(pivoted_data))
            
            # 构建Granger因果网络
            G = build_granger_network(
                df, maxlag=GRANGER_MAX_LAG, 
                p_threshold=GRANGER_P_THRESHOLD,
                min_corr=MIN_CORRELATION,
                n_jobs=n_jobs
            )
            
            # 将网络转换为边索引和权重
            edge_index, edge_weight = network_to_edge_index_weight(G, variables, basin_ids)
            
            causal_edges_dict[scale] = (edge_index, edge_weight)
            logger.info(f"{scale}尺度因果网络: {edge_index.shape[1]}条边")
            
            # 保存网络图以便可视化
            nx.write_gpickle(G, os.path.join(CAUSAL_NETWORK_DIR, f'{scale}_causal_network.gpickle'))
        else:
            logger.warning(f"{scale}尺度数据为空或不足，创建空因果网络")
            # 创建空网络
            causal_edges_dict[scale] = (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float)
            )
    
    return causal_edges_dict

def network_to_edge_index_weight(G, variables, basin_ids):
    """
    将NetworkX图转换为PyTorch Geometric兼容的边索引和权重
    
    参数:
        G: NetworkX有向图
        variables: 变量列表
        basin_ids: 流域ID列表
        
    返回:
        edge_index: [2, num_edges] 边索引
        edge_weight: [num_edges] 边权重
    """
    # 创建映射字典
    var_basin_to_idx = {}
    for basin_idx, basin_id in enumerate(basin_ids):
        for var_idx, var in enumerate(variables):
            node_idx = basin_idx * len(variables) + var_idx
            var_basin_to_idx[f"{var}_{basin_id}"] = node_idx
    
    # 提取边
    edge_index_list = []
    edge_weight_list = []
    
    for u, v, data in G.edges(data=True):
        # 检查节点名称是否在映射中
        if u in var_basin_to_idx and v in var_basin_to_idx:
            source_idx = var_basin_to_idx[u]
            target_idx = var_basin_to_idx[v]
            
            edge_index_list.append([source_idx, target_idx])
            edge_weight_list.append(data.get('weight', 0.5))  # 使用相关系数作为权重
    
    # 转换为PyTorch张量
    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)
    
    return edge_index, edge_weight

def visualize_causal_network(G, scale, output_dir=CAUSAL_NETWORK_DIR):
    """
    可视化因果网络
    
    参数:
        G: NetworkX有向图
        scale: 尺度名称
        output_dir: 输出目录
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        # 设置更大的图形尺寸
        plt.figure(figsize=(12, 10))
        
        # 节点按变量类型着色
        node_colors = []
        node_labels = {}
        
        for node in G.nodes():
            parts = node.split('_')
            var = parts[0]
            basin = '_'.join(parts[1:])
            
            # 根据变量类型决定颜色
            if 'streamflow' in var:
                node_colors.append('blue')
            elif 'precipitation' in var:
                node_colors.append('green')
            elif 'temperature' in var:
                node_colors.append('red')
            elif 'humidity' in var:
                node_colors.append('purple')
            else:
                node_colors.append('gray')
            
            # 简化节点标签
            node_labels[node] = f"{var[:4]}_{basin}"
        
        # 根据权重设置边宽度
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        
        # 绘制网络图
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, arrowsize=10)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        
        plt.title(f"{scale.capitalize()}尺度因果网络")
        plt.axis('off')
        
        # 保存图形
        output_path = os.path.join(output_dir, f'{scale}_causal_network.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"{scale}尺度因果网络可视化已保存至: {output_path}")
    
    except ImportError:
        logger.warning("缺少matplotlib库，无法生成因果网络可视化")
    except Exception as e:
        logger.error(f"生成因果网络可视化时出错: {str(e)}")