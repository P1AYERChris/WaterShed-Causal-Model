# V01.水文预测模型数据预处理完整逻辑与数据流

## 预处理总体流程

```
原始数据 → 流域网络处理 → 层次结构构建 → 时间序列处理 → 因果网络构建 → 预处理数据
```

## 各模块详细说明与数据形状

### 1. 读取原始数据 (主要处理函数)

#### 1.1 流域数据处理 (process_all_watersheds)
- **输入**: HydroBASINS shapefile文件 (3个尺度)
- **输出**:
  ```
  watershed_data: {
    'fine': {流域网络数据},
    'medium': {流域网络数据},
    'coarse': {流域网络数据}
  }
  basin_ids_dict: {
    'fine': [流域ID列表],
    'medium': [流域ID列表],
    'coarse': [流域ID列表]
  }
  ```

#### 1.2 单尺度流域网络处理 (process_watershed_network)
- **输入**: 单一尺度的shapefile路径
- **输出**:
  - `river_edge_index`: [2, num_edges] - 河网连接关系
  - `river_edge_attr`: [num_edges, num_features] - 河网边属性
  - `node_coords`: [num_nodes, 2] - 流域中心点坐标
  - `spatial_features`: [num_nodes, feature_dim] - 空间特征
  - `spatial_encoding`: [num_nodes, HIDDEN_DIM] - 预计算的空间编码
  - `basin_ids`: [num_nodes] - 流域ID列表
  - `basin_id_to_idx`: {basin_id: index} - 流域ID到索引映射

### 2. 特征与编码处理

#### 2.1 空间特征提取 (extract_spatial_features)
- **输入**: 流域GeoDataFrame
- **处理**: 提取坐标和流域属性
- **输出**: `features_tensor`: [num_nodes, 2+num_attrs] - 标准化的特征张量
  - 前2维: 经纬度坐标
  - 后续维度: UP_AREA, SUB_AREA, DIST_SINK等流域属性

#### 2.2 空间编码生成 (SpatialEncoder)
- **输入**: `spatial_features`: [num_nodes, feature_dim]
- **处理**: 固定线性投影
- **输出**: `spatial_encoding`: [num_nodes, HIDDEN_DIM]

### 3. 流域层次结构处理 (process_watershed_hierarchy)
- **输入**: 三个尺度的流域shapefile路径
- **处理**: 基于PFAF_ID创建层次映射
- **输出**: `watershed_hierarchy`: [fine_to_medium, medium_to_coarse]
  - `fine_to_medium`: {medium_idx: [fine_idx1, fine_idx2, ...]}
  - `medium_to_coarse`: {coarse_idx: [medium_idx1, medium_idx2, ...]}

### 4. 时间序列数据处理

#### 4.1 NC文件加载 (load_multi_var_nc_files)
- **输入**: NC文件目录
- **处理**: 读取并合并多个NC文件
- **输出**:
  - `data_4d`: [time, lat, lon, num_vars] - 四维气象/水文数据
  - `coord_info`: {lon: [...], lat: [...], time: [...]} - 坐标信息
  - `var_names`: [var1, var2, ...] - 变量名列表

#### 4.2 网格聚合到流域 (aggregate_grid_to_watershed)
- **输入**:
  - `nc_dataset`: xarray.Dataset - 网格数据
  - `watershed_gdf`: GeoDataFrame - 流域边界
- **处理**: 空间聚合(均值、最大值等)
- **输出**: `result_df`: DataFrame[time, watershed_id, var1, var2, ...]

### 5. 因果网络构建

#### 5.1 多尺度因果网络 (build_multi_scale_causal_networks)
- **输入**:
  - `time_series_data`: DataFrame - 时间序列数据
  - `basin_ids_dict`: Dict - 流域ID字典
- **处理**: Granger因果检验
- **输出**: `causal_edges_dict`: Dict - 不同尺度的因果边
  ```
  {
    'fine': (edge_index, edge_weight),
    'medium': (edge_index, edge_weight),
    'coarse': (edge_index, edge_weight)
  }
  ```
  - `edge_index`: [2, num_causal_edges] - 因果边索引
  - `edge_weight`: [num_causal_edges] - 因果边权重

#### 5.2 Granger因果网络构建 (build_granger_network)
- **输入**: 时间序列DataFrame
- **处理**: 计算相关性、Granger检验
- **输出**: NetworkX有向图 G

### 6. 保存的预处理数据 (按尺度)

对每个尺度('fine', 'medium', 'coarse')，保存：
1. `{scale}_river_edge_index.pt`: [2, num_edges]
2. `{scale}_river_edge_attr.pt`: [num_edges, num_features]
3. `{scale}_node_coords.pt`: [num_nodes, 2]
4. `{scale}_spatial_features.pt`: [num_nodes, feature_dim]
5. `{scale}_spatial_encoding.pt`: [num_nodes, HIDDEN_DIM]
6. `{scale}_basin_ids.json`: 流域ID列表
7. `{scale}_time_series.csv`: 时间 x 流域 x 变量
8. `{scale}_causal_edge_index.pt`: [2, num_causal_edges]
9. `{scale}_causal_edge_weight.pt`: [num_causal_edges]
10. `{scale}_causal_network.gpickle`: NetworkX因果网络
11. `{scale}_spatial_encoder.pt`: 空间编码器参数

另外还有：
1. `watershed_hierarchy.pt`: 流域层次结构
2. `hierarchy_info.pt`: ID-索引映射
3. `config_info.json`: 预处理配置信息

这些预处理数据共同构成模型训练所需的完整输入，通过统一的形状和格式便于模型高效加载和处理。
