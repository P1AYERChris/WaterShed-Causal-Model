import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import time
import json
import xarray as xr
from tqdm import tqdm
from config import *
from data_loader import load_multi_var_nc_files, aggregate_grid_to_watershed, log_memory_usage
from watershed_network import process_watershed_network, process_watershed_hierarchy
from causal_analysis import build_multi_scale_causal_networks, visualize_causal_network

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'data_preprocessing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """创建必要的目录结构"""
    dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
        MODEL_DIR, OUTPUT_DIR, LOG_DIR,
        RIVER_NETWORK_DIR, CAUSAL_NETWORK_DIR, FEATURES_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("目录结构已创建")

def process_all_watersheds():
    """处理所有尺度的流域数据"""
    watershed_data = {}
    basin_ids_dict = {}
    
    # 处理不同尺度的流域数据
    for scale in SCALES:
        logger.info(f"处理{scale}尺度流域数据...")
        
        if scale == "fine":
            shp_path = FINE_WATERSHED_PATH
        elif scale == "medium":
            shp_path = MEDIUM_WATERSHED_PATH
        else:  # coarse
            shp_path = COARSE_WATERSHED_PATH
            
        # 处理流域网络
        (
            river_edge_index, 
            river_edge_attr, 
            node_coords, 
            spatial_features,
            spatial_encoding,
            basin_ids,
            basin_id_to_idx
        ) = process_watershed_network(shp_path, scale)
        
        # 保存数据
        basin_ids_dict[scale] = basin_ids
        watershed_data[scale] = {
            'river_edge_index': river_edge_index,
            'river_edge_attr': river_edge_attr,
            'node_coords': node_coords,
            'spatial_features': spatial_features,
            'spatial_encoding': spatial_encoding,
            'basin_id_to_idx': basin_id_to_idx
        }
        
        # 保存到文件
        torch.save(river_edge_index, os.path.join(RIVER_NETWORK_DIR, f'{scale}_river_edge_index.pt'))
        torch.save(river_edge_attr, os.path.join(RIVER_NETWORK_DIR, f'{scale}_river_edge_attr.pt'))
        torch.save(node_coords, os.path.join(FEATURES_DIR, f'{scale}_node_coords.pt'))
        torch.save(spatial_features, os.path.join(FEATURES_DIR, f'{scale}_spatial_features.pt'))
        torch.save(spatial_encoding, os.path.join(FEATURES_DIR, f'{scale}_spatial_encoding.pt'))
        
        # 保存流域ID和索引映射
        with open(os.path.join(PROCESSED_DATA_DIR, f'{scale}_basin_ids.json'), 'w') as f:
            json.dump(basin_ids, f)
        
        logger.info(f"{scale}尺度流域数据处理完成: {len(basin_ids)}个流域")
    
    return watershed_data, basin_ids_dict

def process_time_series_data():
    """处理时间序列数据"""
    try:
        # 从目录加载多个NC文件
        logger.info(f"开始从{NC_DATA_DIR}加载多个NC文件...")
        data_4d, coord_info, var_names = load_multi_var_nc_files(
            data_dir=NC_DATA_DIR,
            lon_range=LON_RANGE,
            lat_range=LAT_RANGE
        )
        
        # 检查变量名是否匹配配置
        available_vars = []
        for var in VARIABLES:
            if var in var_names:
                available_vars.append(var)
            else:
                # 查找近似匹配
                for loaded_var in var_names:
                    if var.lower() in loaded_var.lower() or loaded_var.lower() in var.lower():
                        logger.info(f"使用 '{loaded_var}' 代替 '{var}'")
                        available_vars.append(loaded_var)
                        break
                else:
                    logger.warning(f"找不到变量: {var}")
                    
        if not available_vars:
            logger.error(f"未找到指定的任何变量: {VARIABLES}，仅有: {var_names}")
            return None
            
        logger.info(f"成功加载数据，形状: {data_4d.shape}，变量: {available_vars}")
        
        # 创建xarray Dataset以便进行流域聚合
        ds = xr.Dataset()
        
        # 添加坐标
        ds.coords['lon'] = coord_info['lon']
        ds.coords['lat'] = coord_info['lat']
        ds.coords['time'] = coord_info['time']
        
        # 添加变量
        for i, var_name in enumerate(var_names):
            if var_name in available_vars:
                # 提取该变量的数据
                var_data = data_4d[:, :, :, i]
                # 添加到数据集
                ds[var_name] = xr.DataArray(
                    data=var_data,
                    dims=['time', 'lat', 'lon']
                )
        
        # 加载流域边界
        watersheds = {}
        for scale in SCALES:
            if scale == "fine":
                shp_path = FINE_WATERSHED_PATH
            elif scale == "medium":
                shp_path = MEDIUM_WATERSHED_PATH
            else:  # coarse
                shp_path = COARSE_WATERSHED_PATH
                
            watersheds[scale] = gpd.read_file(shp_path)
        
        # 聚合网格数据到流域尺度
        aggregated_data = {}
        for scale in SCALES:
            logger.info(f"将网格数据聚合到{scale}尺度流域...")
            scale_df = aggregate_grid_to_watershed(ds, watersheds[scale], available_vars)
            aggregated_data[scale] = scale_df
            
            # 保存聚合数据
            scale_df.to_csv(os.path.join(PROCESSED_DATA_DIR, f'{scale}_time_series.csv'), index=False)
            logger.info(f"{scale}尺度时间序列数据已保存，形状: {scale_df.shape}")
        
        return aggregated_data
        
    except Exception as e:
        logger.error(f"加载和处理时间序列数据时出错: {str(e)}", exc_info=True)
        return None

def main():
    """执行所有预处理步骤"""
    start_time = time.time()
    logger.info("开始数据预处理流程...")
    
    try:
        # 创建目录结构
        create_directories()
        
        # 处理流域数据
        logger.info("处理流域网络数据...")
        watershed_data, basin_ids_dict = process_all_watersheds()
        
        # 处理流域层次结构
        logger.info("处理流域层次结构...")
        watershed_hierarchy = process_watershed_hierarchy(
            FINE_WATERSHED_PATH, MEDIUM_WATERSHED_PATH, COARSE_WATERSHED_PATH
        )
        torch.save(watershed_hierarchy, WATERSHED_HIERARCHY_PATH)
        
        # 处理时间序列数据
        logger.info("处理时间序列数据...")
        time_series_data = process_time_series_data()
        
        if time_series_data:
            # 合并所有尺度的时间序列数据进行因果分析
            all_time_series = pd.concat(list(time_series_data.values()))
            
            # 构建多尺度因果网络
            logger.info("构建多尺度因果网络...")
            causal_edges_dict = build_multi_scale_causal_networks(
                all_time_series,
                basin_ids_dict,
                VARIABLES,
                n_jobs=CAUSAL_N_JOBS
            )
            
            # 保存因果网络数据
            for scale in SCALES:
                if scale in causal_edges_dict:
                    edge_index, edge_weight = causal_edges_dict[scale]
                    torch.save(edge_index, os.path.join(CAUSAL_NETWORK_DIR, f'{scale}_causal_edge_index.pt'))
                    torch.save(edge_weight, os.path.join(CAUSAL_NETWORK_DIR, f'{scale}_causal_edge_weight.pt'))
                    logger.info(f"{scale}尺度因果网络已保存: {edge_index.shape[1]}条边")
        
        # 保存配置信息
        config_info = {
            'scales': SCALES,
            'variables': VARIABLES,
            'input_time_steps': INPUT_TIME_STEPS,
            'forecast_horizon': FORECAST_HORIZON,
            'num_basins': {scale: len(basin_ids) for scale, basin_ids in basin_ids_dict.items()},
            'preprocessed_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(PROCESSED_DATA_DIR, 'config_info.json'), 'w') as f:
            json.dump(config_info, f, indent=2)
        
        # 输出处理结果摘要
        for scale in SCALES:
            logger.info(f"{scale}尺度统计:")
            logger.info(f"- 流域数: {len(basin_ids_dict[scale])}个")
            logger.info(f"- 河网边数: {watershed_data[scale]['river_edge_index'].shape[1]}条")
            if scale in causal_edges_dict:
                logger.info(f"- 因果边数: {causal_edges_dict[scale][0].shape[1]}条")
        
        elapsed_time = time.time() - start_time
        logger.info(f"数据预处理完成! 总耗时: {elapsed_time/60:.2f}分钟")
        
    except Exception as e:
        logger.error(f"预处理过程中出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()