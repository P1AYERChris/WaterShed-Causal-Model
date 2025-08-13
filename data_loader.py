import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Point
import time
import glob
from tqdm import tqdm
import gc
from config import *

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'data_loader.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_nc_files(data_dir):
    """检查目录中的NC文件并返回基本信息"""
    nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    
    if not nc_files:
        logger.warning(f"在 {data_dir} 中未找到NC文件")
        return []
    
    file_info = []
    
    for nc_file in nc_files:
        try:
            ds = xr.open_dataset(nc_file)
            info = {
                'file': os.path.basename(nc_file),
                'vars': list(ds.data_vars.keys()),
                'coords': list(ds.coords),
                'dims': {dim: ds.dims[dim] for dim in ds.dims}
            }
            file_info.append(info)
            ds.close()
        except Exception as e:
            logger.error(f"检查文件 {os.path.basename(nc_file)} 时出错: {str(e)}")
    
    return file_info

def load_multi_var_nc_files(data_dir, nc_files=None, 
                           lon_range=LON_RANGE, lat_range=LAT_RANGE,
                           time_range=None):
    """
    从包含多变量的NC文件中加载数据
    
    参数:
        data_dir: 包含NC文件的目录
        nc_files: NC文件列表，如果为None则加载目录中所有NC文件
        lon_range: 经度范围 (最小值, 最大值)
        lat_range: 纬度范围 (最小值, 最大值)
        time_range: 时间范围 (起始时间, 结束时间)，可选
        
    返回:
        四维数组: [时间, 纬度, 经度, 变量]
        坐标信息字典
        变量列表
    """
    if nc_files is None:
        nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    else:
        nc_files = [os.path.join(data_dir, f) for f in nc_files]
    
    if not nc_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到NC文件")
    
    logger.info(f"开始加载 {len(nc_files)} 个NC文件: {[os.path.basename(f) for f in nc_files]}")
    
    # 存储所有变量的数据
    all_var_data = []
    all_var_names = []
    coord_info = None
    
    # 依次处理每个NC文件
    for nc_file in nc_files:
        try:
            logger.info(f"打开文件: {os.path.basename(nc_file)}")
            ds = xr.open_dataset(nc_file)
            
            # 输出文件信息以便调试
            logger.info(f"文件中的变量: {list(ds.data_vars.keys())}")
            logger.info(f"文件中的坐标: {list(ds.coords)}")
            
            # 检测坐标名称
            lon_name = next((name for name in ds.coords if name.lower() in ['lon', 'longitude']), None)
            lat_name = next((name for name in ds.coords if name.lower() in ['lat', 'latitude']), None)
            time_name = next((name for name in ds.coords if name.lower() in ['time', 'valid_time']), None)
            
            if not lon_name or not lat_name:
                logger.error(f"无法识别经纬度坐标，跳过文件: {os.path.basename(nc_file)}")
                continue
                
            logger.info(f"使用坐标: 经度={lon_name}, 纬度={lat_name}, 时间={time_name}")
            
            # 应用空间筛选
            logger.info(f"应用空间范围筛选: 经度={lon_range}, 纬度={lat_range}")
            ds_subset = ds.sel(
                **{lon_name: slice(lon_range[0], lon_range[1]),
                   lat_name: slice(lat_range[1], lat_range[0])}  # 纬度可能是倒序
            )
            
            # 应用时间筛选（如果指定）
            if time_range and time_name:
                logger.info(f"应用时间范围筛选: {time_range}")
                ds_subset = ds_subset.sel(**{time_name: slice(time_range[0], time_range[1])})
            
            # 处理每个变量
            for var_name in ds_subset.data_vars:
                logger.info(f"提取变量: {var_name}")
                var_data = ds_subset[var_name].values
                
                # 确保数据是浮点型
                if not np.issubdtype(var_data.dtype, np.floating):
                    var_data = var_data.astype(np.float32)
                
                all_var_data.append(var_data)
                all_var_names.append(var_name)
                
                # 只保存一次坐标信息
                if coord_info is None:
                    coord_info = {
                        'lon': ds_subset[lon_name].values,
                        'lat': ds_subset[lat_name].values
                    }
                    if time_name:
                        coord_info['time'] = ds_subset[time_name].values
                        
                        # 输出时间范围信息
                        if len(coord_info['time']) > 0:
                            try:
                                time_start = pd.to_datetime(coord_info['time'][0])
                                time_end = pd.to_datetime(coord_info['time'][-1])
                                logger.info(f"数据时间范围: {time_start} 到 {time_end}")
                            except:
                                logger.info(f"时间数据不是标准格式，无法格式化显示")
            
            # 关闭数据集释放内存
            ds.close()
            
        except Exception as e:
            logger.error(f"处理文件 {os.path.basename(nc_file)} 时出错: {str(e)}")
            continue
    
    if not all_var_data:
        raise ValueError("未能成功加载任何变量，请检查NC文件格式")
    
    # 检查所有变量的形状是否一致
    shapes = [data.shape for data in all_var_data]
    if len(set(shapes)) > 1:
        logger.warning(f"变量形状不一致: {list(zip(all_var_names, shapes))}")
        
        # 找出共同的形状
        common_shape = max(shapes, key=shapes.count)
        logger.info(f"使用最常见形状: {common_shape}")
        
        # 筛选形状一致的变量
        filtered_data = []
        filtered_names = []
        for i, data in enumerate(all_var_data):
            if data.shape == common_shape:
                filtered_data.append(data)
                filtered_names.append(all_var_names[i])
            else:
                logger.warning(f"跳过形状不一致的变量: {all_var_names[i]}, 形状: {data.shape}")
        
        all_var_data = filtered_data
        all_var_names = filtered_names
    
    # 重塑数组以符合预期的四维格式 [time, lat, lon, variables]
    reshaped_data = []
    for i, data in enumerate(all_var_data):
        # 处理不同维度情况
        if len(data.shape) == 3:  # 时间、纬度、经度
            reshaped_data.append(data)
        elif len(data.shape) == 2:  # 可能只有纬度、经度
            # 添加时间维度
            reshaped_data.append(data.reshape(1, *data.shape))
            logger.info(f"为变量 {all_var_names[i]} 添加时间维度")
        else:
            logger.warning(f"变量 {all_var_names[i]} 形状异常: {data.shape}，尝试调整")
            # 尝试调整到三维
            if len(data.shape) == 4 and data.shape[0] == 1:  # 可能有多余的维度
                reshaped_data.append(data[0])
            else:
                logger.error(f"无法处理变量 {all_var_names[i]} 的形状，跳过")
                continue
    
    if not reshaped_data:
        raise ValueError("处理后没有可用的变量数据")
    
    # 堆叠所有变量到一个四维数组
    try:
        combined_data = np.stack(reshaped_data, axis=-1).astype(np.float32)
        logger.info(f"成功创建形状为 {combined_data.shape} 的数据数组，包含变量: {all_var_names}")
        
        # 添加形状描述
        if len(combined_data.shape) == 4:
            time_steps, lat_size, lon_size, var_count = combined_data.shape
            logger.info(f"数据维度: [时间步={time_steps}, 纬度点数={lat_size}, 经度点数={lon_size}, 变量数={var_count}]")
        
        return combined_data, coord_info, all_var_names
        
    except Exception as e:
        logger.error(f"创建组合数据数组时出错: {str(e)}")
        raise ValueError(f"无法组合变量数据: {str(e)}")

def aggregate_grid_to_watershed(nc_dataset, watershed_gdf, variables=None, aggregation_method='mean'):
    """
    将网格数据聚合到流域尺度
    
    参数:
        nc_dataset: xarray Dataset对象，包含网格数据
        watershed_gdf: GeoDataFrame对象，包含流域边界
        variables: 要处理的变量列表，如果为None则处理所有变量
        aggregation_method: 聚合方法，'mean', 'sum', 'max', 'min'之一
        
    返回:
        pandas DataFrame: 时间 x 流域 x 变量的聚合数据
    """
    start_time = time.time()
    logger.info(f"开始将网格数据聚合到{len(watershed_gdf)}个流域")
    
    # 确定坐标变量名
    lon_name = next((name for name in nc_dataset.coords if name.lower() in ['lon', 'longitude']), 'lon')
    lat_name = next((name for name in nc_dataset.coords if name.lower() in ['lat', 'latitude']), 'lat')
    time_name = next((name for name in nc_dataset.coords if name.lower() in ['time', 'valid_time']), 'time')
    
    # 获取经纬度网格
    lons = nc_dataset[lon_name].values
    lats = nc_dataset[lat_name].values
    times = nc_dataset[time_name].values
    
    # 确定要处理的变量
    if variables is None:
        variables = list(nc_dataset.data_vars.keys())
    else:
        variables = [var for var in variables if var in nc_dataset.data_vars]
        
    logger.info(f"处理变量: {variables}")
    
    # 创建点网格
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]
    
    # 为每个点找到对应的流域
    logger.info("将网格点分配到流域...")
    point_to_watershed = {}
    
    for i, point in enumerate(tqdm(points)):
        for idx, row in watershed_gdf.iterrows():
            if row.geometry.contains(point):
                point_to_watershed[i] = idx
                break
    
    logger.info(f"成功将{len(point_to_watershed)}个点分配到流域 (共{len(points)}个点)")
    
    # 初始化结果DataFrame
    result_dfs = []
    
    # 处理每个时间步
    for t, time_val in enumerate(times):
        # 为每个流域和变量创建一个条目
        rows = []
        
        for watershed_idx, row in watershed_gdf.iterrows():
            watershed_id = row['HYBAS_ID']
            
            # 找到该流域包含的所有点
            watershed_points = [i for i, ws_idx in point_to_watershed.items() if ws_idx == watershed_idx]
            
            if not watershed_points:
                continue
                
            # 为每个变量聚合数据
            values = {}
            for var in variables:
                var_data = nc_dataset[var].isel({time_name: t}).values.flatten()
                points_data = [var_data[i] for i in watershed_points if i < len(var_data)]
                
                if not points_data:
                    values[var] = np.nan
                    continue
                    
                # 应用聚合方法
                if aggregation_method == 'mean':
                    values[var] = np.nanmean(points_data)
                elif aggregation_method == 'sum':
                    values[var] = np.nansum(points_data)
                elif aggregation_method == 'max':
                    values[var] = np.nanmax(points_data)
                elif aggregation_method == 'min':
                    values[var] = np.nanmin(points_data)
                else:
                    values[var] = np.nanmean(points_data)
            
            # 添加时间和流域ID
            values['time'] = time_val
            values['watershed_id'] = watershed_id
            rows.append(values)
            
        # 创建当前时间步的DataFrame
        if rows:
            t_df = pd.DataFrame(rows)
            result_dfs.append(t_df)
            
    # 合并所有时间步的结果
    if result_dfs:
        result_df = pd.concat(result_dfs)
        logger.info(f"聚合完成，结果形状: {result_df.shape}，耗时: {time.time() - start_time:.2f}秒")
        return result_df
    else:
        logger.warning("聚合结果为空!")
        return pd.DataFrame()

def log_memory_usage(message=""):
    """记录当前进程的内存使用情况"""
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # 转换为GB
    rss_gb = mem_info.rss / (1024 * 1024 * 1024)
    vms_gb = mem_info.vms / (1024 * 1024 * 1024)
    
    logger.info(f"内存使用 ({message}): RSS={rss_gb:.1f}GB, VMS={vms_gb:.1f}GB")

class WatershedDataset(Dataset):
    """水文预测数据集"""
    
    def __init__(self, data_df, basin_ids, variables, input_time_steps, forecast_horizon, mode='train', 
                 train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, stride=1):
        """
        初始化数据集
        
        参数:
            data_df: 包含时间序列数据的DataFrame
            basin_ids: 流域ID列表
            variables: 变量名列表
            input_time_steps: 输入序列长度
            forecast_horizon: 预测序列长度
            mode: 'train', 'val', 或 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            stride: 滑动窗口步长
        """
        super(WatershedDataset, self).__init__()
        
        self.variables = variables
        self.input_time_steps = input_time_steps
        self.forecast_horizon = forecast_horizon
        self.basin_ids = basin_ids
        self.num_basins = len(basin_ids)
        self.basin_id_to_idx = {basin_id: i for i, basin_id in enumerate(basin_ids)}
        
        # 确保数据按时间排序
        data_df = data_df.sort_values('time')
        
        # 获取所有唯一时间点
        unique_times = data_df['time'].unique()
        total_times = len(unique_times)
        
        # 创建滑动窗口样本索引
        samples = []
        for t in range(0, total_times - input_time_steps - forecast_horizon + 1, stride):
            samples.append((t, t + input_time_steps, t + input_time_steps + forecast_horizon))
        
        # 划分数据集
        num_samples = len(samples)
        train_end = int(train_ratio * num_samples)
        val_end = int((train_ratio + val_ratio) * num_samples)
        
        if mode == 'train':
            self.samples = samples[:train_end]
        elif mode == 'val':
            self.samples = samples[train_end:val_end]
        else:  # test
            self.samples = samples[val_end:]
            
        self.data_df = data_df
        self.unique_times = unique_times
        
        logger.info(f"{mode}集: {len(self.samples)}个样本")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        start_idx, input_end_idx, target_end_idx = self.samples[idx]
        
        # 提取时间窗口
        input_times = self.unique_times[start_idx:input_end_idx]
        target_times = self.unique_times[input_end_idx:target_end_idx]
        
        # 初始化输入和目标张量
        x = np.zeros((self.num_basins, len(self.variables), self.input_time_steps))
        y = np.zeros((self.num_basins, self.forecast_horizon))
        
        # 填充输入数据
        for t, time_val in enumerate(input_times):
            time_data = self.data_df[self.data_df['time'] == time_val]
            
            for _, row in time_data.iterrows():
                basin_id = row['watershed_id']
                if basin_id in self.basin_id_to_idx:
                    basin_idx = self.basin_id_to_idx[basin_id]
                    for v, var in enumerate(self.variables):
                        if var in row:
                            x[basin_idx, v, t] = row[var]
        
        # 填充目标数据 (只使用第一个变量，通常是流量)
        target_var = self.variables[0]
        for t, time_val in enumerate(target_times):
            time_data = self.data_df[self.data_df['time'] == time_val]
            
            for _, row in time_data.iterrows():
                basin_id = row['watershed_id']
                if basin_id in self.basin_id_to_idx:
                    basin_idx = self.basin_id_to_idx[basin_id]
                    if target_var in row:
                        y[basin_idx, t] = row[target_var]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

def get_dataloaders(data_df, basin_ids, variables, input_time_steps, forecast_horizon, 
                   batch_size=BATCH_SIZE, num_workers=4, stride=1):
    """
    创建数据加载器
    
    参数:
        data_df: 包含时间序列数据的DataFrame
        basin_ids: 流域ID列表
        variables: 变量名列表
        input_time_steps: 输入序列长度
        forecast_horizon: 预测序列长度
        batch_size: 批次大小
        num_workers: 数据加载线程数
        stride: 滑动窗口步长
        
    返回:
        train_loader, val_loader, test_loader
    """
    train_dataset = WatershedDataset(
        data_df, basin_ids, variables, input_time_steps, forecast_horizon, 
        mode='train', stride=stride
    )
    
    val_dataset = WatershedDataset(
        data_df, basin_ids, variables, input_time_steps, forecast_horizon, 
        mode='val', stride=stride
    )
    
    test_dataset = WatershedDataset(
        data_df, basin_ids, variables, input_time_steps, forecast_horizon, 
        mode='test', stride=stride
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    logger.info(f"创建数据加载器: 训练集={len(train_dataset)}样本, 验证集={len(val_dataset)}样本, 测试集={len(test_dataset)}样本")
    logger.info(f"批次大小: {batch_size}, 线程数: {num_workers}")
    
    return train_loader, val_loader, test_loader