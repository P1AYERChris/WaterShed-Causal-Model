import os

# 路径配置
DATA_DIR = "/data/yinguo_model/因果模型/watershed_causal_model/data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = "data/yinguo_model/因果模型/watershed_causal_model/models"
OUTPUT_DIR = "data/yinguo_model/因果模型/watershed_causal_model/outputs"
LOG_DIR = "data/yinguo_model/因果模型/watershed_causal_model/logs"

# 确保所有目录存在
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 时序数据路径
NC_DATA_DIR = os.path.join(RAW_DATA_DIR, "hydro_data")  # 存放多个NC文件的目录

# HydroBASINS数据路径
WATERSHEDS_DIR = os.path.join(RAW_DATA_DIR, "watersheds")
FINE_WATERSHED_PATH = os.path.join(WATERSHEDS_DIR, "hybas_as_lev09_v1c/hybas_as_lev09_v1c.shp")
MEDIUM_WATERSHED_PATH = os.path.join(WATERSHEDS_DIR, "hybas_as_lev06_v1c/hybas_as_lev06_v1c.shp")
COARSE_WATERSHED_PATH = os.path.join(WATERSHEDS_DIR, "hybas_as_lev03_v1c/hybas_as_lev03_v1c.shp")

# 预处理数据保存路径
RIVER_NETWORK_DIR = os.path.join(PROCESSED_DATA_DIR, "river_networks")
CAUSAL_NETWORK_DIR = os.path.join(PROCESSED_DATA_DIR, "causal_networks")
WATERSHED_HIERARCHY_PATH = os.path.join(PROCESSED_DATA_DIR, "watershed_hierarchy.pt")
FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, "features")

# 确保预处理数据目录存在
for dir_path in [RIVER_NETWORK_DIR, CAUSAL_NETWORK_DIR, FEATURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 数据参数
INPUT_TIME_STEPS = 12  # 输入时间步长度(历史数据)
FORECAST_HORIZON = 3   # 预测时间步长度
VARIABLES = ["streamflow", "precipitation", "temperature", "humidity", "pressure"]  # 模型使用的变量

# 空间范围参数(有shp数据就不需要先划分了)
LON_RANGE = (-180, 180)  # 中国地区经度范围
LAT_RANGE = (-90, 90)   # 中国地区纬度范围

# 多尺度参数
SCALES = ["fine", "medium", "coarse"]
SCALE_LEVELS = {"fine": 9, "medium": 6, "coarse": 3}  # HydroBASINS等级

# 可用于空间硬编码的属性列表
SPATIAL_FEATURES = [

    'UP_AREA',    # 上游面积
    'SUB_AREA',   # 流域面积
    'DIST_SINK',  # 到汇水点的距离
    'ENDO',    # 内陆流域标志
    'COAST',  # 海岸线标志
    'ORDER',  # 水系等级
    'SORT',  # 水系排序
    
]

# 因果分析参数
GRANGER_MAX_LAG = 8    # Granger因果检验最大滞后
GRANGER_P_THRESHOLD = 0.05  # 因果检验p值阈值
MIN_CORRELATION = 0.2  # 最小相关性阈值
CAUSAL_N_JOBS = 32      # 因果分析并行作业数

# 模型架构参数
HIDDEN_DIM = 64        # 隐藏层维度
BLOCKS_PER_LEVEL = 2   # 每个尺度层的时空块数量
NUM_HEADS = 4          # 自注意力头数
DROPOUT = 0.1          # Dropout比例

# 训练参数
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
PATIENCE = 5          # 早停耐心值