import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from config import *

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'watershed_network.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpatialEncoder(nn.Module):
    """空间位置和属性硬编码器 - 将经纬度和流域属性直接编码为固定特征"""
    def __init__(self, output_dim, input_dim):
        super(SpatialEncoder, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        # 创建固定的线性投影矩阵（不进行训练）
        projection = torch.zeros(input_dim, output_dim)
        # 使用确定性方法初始化投影矩阵
        for i in range(min(input_dim, output_dim)):
            projection[i, i] = 1.0
        
        # 如果输出维度更大，初始化额外的投影元素
        if output_dim > input_dim:
            for i in range(input_dim, output_dim):
                projection[i % input_dim, i] = 0.5
        
        self.projection = nn.Parameter(projection, requires_grad=False)
        
    def forward(self, features):
        """
        features: [num_nodes, feature_dim] - 包含坐标和属性的节点特征
        
        返回:
        spatial_code: [num_nodes, output_dim] - 硬编码的空间特征
        """
        # 使用固定的线性投影
        return torch.matmul(features, self.projection)

def process_watershed_network(shp_file_path, scale_name):
    """
    从HydroBASINS shapefile提取河网拓扑结构
    
    参数:
        shp_file_path: HydroBASINS shapefile路径
        scale_name: 尺度名称 ('fine', 'medium', 'coarse')
        
    返回:
        river_edge_index: 河网边索引 [2, num_edges]
        river_edge_attr: 河网边属性
        node_coords: 流域中心点坐标
        spatial_features: 空间特征
        spatial_encoding: 预计算的空间编码
        basin_ids: 流域ID列表
        basin_id_to_idx: 流域ID到索引的映射
    """
    start_time = time.time()
    logger.info(f"处理{scale_name}尺度流域网络: {shp_file_path}")
    
    try:
        # 读取流域数据
        watersheds = gpd.read_file(shp_file_path)
        logger.info(f"读取了 {len(watersheds)} 个流域")
        
        # 检查必要的属性
        required_attrs = ['HYBAS_ID', 'NEXT_DOWN']
        for attr in required_attrs:
            if attr not in watersheds.columns:
                logger.error(f"缺少必要的属性: {attr}")
                raise ValueError(f"缺少必要的属性: {attr}")
        
        # 创建流域ID到索引的映射
        basin_ids = watersheds['HYBAS_ID'].values.tolist()
        basin_id_to_idx = {int(basin_id): idx for idx, basin_id in enumerate(basin_ids)}
        
        # 提取流域中心点坐标
        logger.info("提取流域中心点坐标...")
        node_coords = []
        for _, ws in watersheds.iterrows():
            centroid = ws.geometry.centroid
            node_coords.append([centroid.x, centroid.y])
        node_coords = torch.tensor(node_coords, dtype=torch.float)
        
        # 提取河网连接关系
        logger.info("构建河网连接关系...")
        edge_index_list = []
        edge_attr_list = []
        
        for idx, row in tqdm(watersheds.iterrows(), total=len(watersheds)):
            # 获取当前流域和下游流域ID
            current_id = int(row['HYBAS_ID'])
            downstream_id = int(row['NEXT_DOWN'])
            
            # 跳过没有下游的流域
            if downstream_id == 0:
                continue
                
            # 确保下游流域在当前数据集中
            if downstream_id in basin_id_to_idx:
                from_idx = basin_id_to_idx[current_id]
                to_idx = basin_id_to_idx[downstream_id]
                
                edge_index_list.append([from_idx, to_idx])
                
                # 提取边属性 (如果存在)
                attrs = []
                for attr in SPATIAL_FEATURES:
                    if attr in row and pd.notnull(row[attr]):
                        attrs.append(float(row[attr]))
                    else:
                        attrs.append(0.0)
                        
                edge_attr_list.append(attrs)
        
        # 转换为PyTorch张量
        if edge_index_list:
            river_edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
            river_edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        else:
            # 创建空张量防止错误
            logger.warning(f"没有发现有效的河网连接关系，创建空的边索引和属性")
            river_edge_index = torch.zeros((2, 0), dtype=torch.long)
            river_edge_attr = torch.zeros((0, len(SPATIAL_FEATURES)), dtype=torch.float)
        
        # 提取空间特征用于硬编码
        logger.info("提取空间特征...")
        spatial_features = extract_spatial_features(watersheds)
        
        # 预计算空间编码 - 将空间特征转换为HIDDEN_DIM维的嵌入
        logger.info("预计算空间编码...")
        spatial_encoder = SpatialEncoder(
            output_dim=HIDDEN_DIM, 
            input_dim=spatial_features.size(1),
            embedding_dim=16
        )
        spatial_encoding = spatial_encoder(spatial_features)
        
        # 保存空间编码器模型以便以后使用
        encoder_save_path = os.path.join(FEATURES_DIR, f'{scale_name}_spatial_encoder.pt')
        torch.save(spatial_encoder.state_dict(), encoder_save_path)
        logger.info(f"空间编码器已保存到: {encoder_save_path}")
        
        logger.info(f"处理完成: 节点={len(watersheds)}个, 边={river_edge_index.size(1)}条")
        logger.info(f"处理耗时: {time.time() - start_time:.2f}秒")
        
        return (
            river_edge_index, 
            river_edge_attr, 
            node_coords, 
            spatial_features,
            spatial_encoding,
            basin_ids,
            basin_id_to_idx
        )
        
    except Exception as e:
        logger.error(f"处理流域网络时出错: {str(e)}", exc_info=True)
        raise

def extract_spatial_features(watersheds_gdf):
    """
    从HydroBASINS提取空间特征和流域属性
    
    参数:
        watersheds_gdf: 流域GeoDataFrame
    
    返回:
        combined_features: [num_nodes, feature_dim] 坐标和流域属性组合的特征张量
    """
    # 选择有用的属性作为特征
    attrs_to_use = [attr for attr in SPATIAL_FEATURES if attr in watersheds_gdf.columns]
    
    # 准备特征列表
    features = []
    for _, row in watersheds_gdf.iterrows():
        # 首先添加中心点坐标，确保空间位置信息一定被包含
        centroid = row.geometry.centroid
        feature = [centroid.x, centroid.y]
        
        # 然后添加流域属性
        for attr in attrs_to_use:
            if pd.notnull(row[attr]):
                feature.append(float(row[attr]))
            else:
                feature.append(0.0)
        
        features.append(feature)
    
    # 转换为张量并标准化
    features_tensor = torch.tensor(features, dtype=torch.float)
    
    # 标准化每个特征
    mean = features_tensor.mean(dim=0, keepdim=True)
    std = features_tensor.std(dim=0, keepdim=True) + 1e-6  # 避免除零
    features_tensor = (features_tensor - mean) / std
    
    logger.info(f"提取了形状为 {features_tensor.shape} 的组合特征 (坐标+流域属性)")
    return features_tensor

def process_watershed_hierarchy(fine_shp_path, medium_shp_path, coarse_shp_path):
    """
    基于HydroBASINS的PFAF_ID处理流域层次结构
    
    参数:
        fine_shp_path: 细尺度流域shapefile路径(如Level 9)
        medium_shp_path: 中尺度流域shapefile路径(如Level 6)
        coarse_shp_path: 粗尺度流域shapefile路径(如Level 3)
    
    返回:
        watershed_hierarchy: 流域层次映射字典列表
    """
    start_time = time.time()
    logger.info("处理流域层次结构...")
    
    try:
        # 读取不同尺度的流域数据
        fine_watersheds = gpd.read_file(fine_shp_path)
        medium_watersheds = gpd.read_file(medium_shp_path)
        coarse_watersheds = gpd.read_file(coarse_shp_path)
        
        logger.info(f"读取了 {len(fine_watersheds)} 个细尺度流域")
        logger.info(f"读取了 {len(medium_watersheds)} 个中尺度流域")
        logger.info(f"读取了 {len(coarse_watersheds)} 个粗尺度流域")
        
        # 检查PFAF_ID是否存在
        if 'PFAF_ID' not in fine_watersheds.columns or \
           'PFAF_ID' not in medium_watersheds.columns or \
           'PFAF_ID' not in coarse_watersheds.columns:
            logger.error("缺少PFAF_ID属性，无法构建层次结构")
            raise ValueError("缺少PFAF_ID属性")
        
        # 创建细尺度到中尺度流域的映射
        logger.info("构建细尺度到中尺度流域的映射...")
        fine_to_medium = {}
        
        # 获取索引到ID的映射
        fine_idx_to_id = {idx: int(row['HYBAS_ID']) for idx, row in fine_watersheds.iterrows()}
        medium_idx_to_id = {idx: int(row['HYBAS_ID']) for idx, row in medium_watersheds.iterrows()}
        coarse_idx_to_id = {idx: int(row['HYBAS_ID']) for idx, row in coarse_watersheds.iterrows()}
        
        # 获取ID到索引的映射
        fine_id_to_idx = {int(row['HYBAS_ID']): idx for idx, row in fine_watersheds.iterrows()}
        medium_id_to_idx = {int(row['HYBAS_ID']): idx for idx, row in medium_watersheds.iterrows()}
        coarse_id_to_idx = {int(row['HYBAS_ID']): idx for idx, row in coarse_watersheds.iterrows()}
        
        # 获取每个PFAF_ID的长度
        sample_fine_pfaf = str(fine_watersheds['PFAF_ID'].iloc[0])
        sample_medium_pfaf = str(medium_watersheds['PFAF_ID'].iloc[0])
        sample_coarse_pfaf = str(coarse_watersheds['PFAF_ID'].iloc[0])
        
        fine_pfaf_len = len(sample_fine_pfaf)
        medium_pfaf_len = len(sample_medium_pfaf)
        coarse_pfaf_len = len(sample_coarse_pfaf)
        
        logger.info(f"PFAF_ID长度: 细尺度={fine_pfaf_len}, 中尺度={medium_pfaf_len}, 粗尺度={coarse_pfaf_len}")
        
        # 创建映射：中尺度索引 -> [细尺度索引]
        for medium_idx, medium_row in medium_watersheds.iterrows():
            medium_pfaf = str(medium_row['PFAF_ID']).zfill(medium_pfaf_len)
            fine_to_medium[medium_idx] = []
            
            for fine_idx, fine_row in fine_watersheds.iterrows():
                fine_pfaf = str(fine_row['PFAF_ID']).zfill(fine_pfaf_len)
                
                # 检查前缀匹配 - 中尺度PFAF是细尺度PFAF的前缀
                if len(fine_pfaf) >= len(medium_pfaf) and fine_pfaf.startswith(medium_pfaf):
                    fine_to_medium[medium_idx].append(fine_idx)
        
        # 创建映射：粗尺度索引 -> [中尺度索引]
        logger.info("构建中尺度到粗尺度流域的映射...")
        medium_to_coarse = {}
        
        for coarse_idx, coarse_row in coarse_watersheds.iterrows():
            coarse_pfaf = str(coarse_row['PFAF_ID']).zfill(coarse_pfaf_len)
            medium_to_coarse[coarse_idx] = []
            
            for medium_idx, medium_row in medium_watersheds.iterrows():
                medium_pfaf = str(medium_row['PFAF_ID']).zfill(medium_pfaf_len)
                
                # 检查前缀匹配 - 粗尺度PFAF是中尺度PFAF的前缀
                if len(medium_pfaf) >= len(coarse_pfaf) and medium_pfaf.startswith(coarse_pfaf):
                    medium_to_coarse[coarse_idx].append(medium_idx)
        
        # 统计嵌套关系
        fine_mapped = sum(len(v) for v in fine_to_medium.values())
        medium_mapped = sum(len(v) for v in medium_to_coarse.values())
        fine_coverage = fine_mapped / len(fine_watersheds) * 100
        medium_coverage = medium_mapped / len(medium_watersheds) * 100
        
        logger.info(f"层次结构映射:")
        logger.info(f"- 细→中: {fine_mapped}/{len(fine_watersheds)} 细尺度流域已映射 ({fine_coverage:.1f}%)")
        logger.info(f"- 中→粗: {medium_mapped}/{len(medium_watersheds)} 中尺度流域已映射 ({medium_coverage:.1f}%)")
        
        # 构建层次结构
        watershed_hierarchy = [fine_to_medium, medium_to_coarse]
        
        # 保存流域索引关系
        hierarchy_info = {
            'fine_id_to_idx': fine_id_to_idx,
            'medium_id_to_idx': medium_id_to_idx,
            'coarse_id_to_idx': coarse_id_to_idx,
            'fine_idx_to_id': fine_idx_to_id,
            'medium_idx_to_id': medium_idx_to_id,
            'coarse_idx_to_id': coarse_idx_to_id
        }
        
        info_save_path = os.path.join(PROCESSED_DATA_DIR, 'hierarchy_info.pt')
        torch.save(hierarchy_info, info_save_path)
        logger.info(f"层次结构索引信息已保存到: {info_save_path}")
        
        logger.info(f"层次结构处理耗时: {time.time() - start_time:.2f}秒")
        return watershed_hierarchy
        
    except Exception as e:
        logger.error(f"处理流域层次结构时出错: {str(e)}", exc_info=True)
        raise