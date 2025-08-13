import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class RiverGraphConv(MessagePassing):
    """处理流域级别的河网物理连接"""
    def __init__(self, hidden_dim, edge_attr_dim=1):
        super(RiverGraphConv, self).__init__(aggr='add')
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(hidden_dim*2, hidden_dim)
        
        # 边属性编码器 - 用于多维边特征
        self.edge_encoder = nn.Linear(edge_attr_dim, 1) if edge_attr_dim > 1 else nn.Identity()
        
        # 可学习权重和门控
        self.gate = nn.Parameter(torch.ones(1) * 0.6)
        self.learnable_weights = None
    
    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, hidden_dim] - 流域节点特征
        edge_index: [2, num_edges] - 河网边索引
        edge_attr: [num_edges, edge_attr_dim] - 河网边属性
        """
        # 初始化可学习权重
        if self.learnable_weights is None or self.learnable_weights.size(0) != edge_index.size(1):
            self.learnable_weights = nn.Parameter(torch.zeros(edge_index.size(1), device=x.device))
        
        # 编码多维边属性
        if hasattr(edge_attr, 'size') and len(edge_attr.size()) > 1:
            edge_weight = self.edge_encoder(edge_attr).squeeze(-1)
        else:
            edge_weight = edge_attr
        
        # 混合预定义权重和可学习权重
        gate_value = torch.sigmoid(self.gate)
        mixed_weight = gate_value * edge_weight + (1-gate_value) * torch.sigmoid(self.learnable_weights)
        
        # 消息传递
        return self.propagate(edge_index, x=x, edge_weight=mixed_weight)
    
    def message(self, x_j, edge_weight):
        # 计算加权消息
        return edge_weight.view(-1, 1) * self.lin(x_j)
    
    def update(self, aggr_out, x):
        # 更新节点特征
        combined = torch.cat([aggr_out, x], dim=-1)
        return F.relu(self.update_lin(combined))


class CausalGraphConv(MessagePassing):
    """处理变量间的因果关系"""
    def __init__(self, hidden_dim):
        super(CausalGraphConv, self).__init__(aggr='add')
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(hidden_dim*2, hidden_dim)
        
        # 门控和可学习权重
        self.gate = nn.Parameter(torch.ones(1) * 0.6)
        self.learnable_weights = None
    
    def forward(self, x, edge_index, edge_weight):
        """
        x: [num_nodes*num_features, hidden_dim] - 变量节点特征
        edge_index: [2, num_causal_edges] - 因果边索引
        edge_weight: [num_causal_edges] - 因果强度
        """
        # 初始化可学习权重
        if self.learnable_weights is None or self.learnable_weights.size(0) != edge_index.size(1):
            self.learnable_weights = nn.Parameter(torch.zeros(edge_index.size(1), device=x.device))
        
        # 混合预定义权重和可学习权重
        gate_value = torch.sigmoid(self.gate)
        mixed_weight = gate_value * edge_weight + (1-gate_value) * torch.sigmoid(self.learnable_weights)
        
        # 消息传递
        return self.propagate(edge_index, x=x, edge_weight=mixed_weight)
    
    def message(self, x_j, edge_weight):
        # 计算加权消息
        return edge_weight.view(-1, 1) * self.lin(x_j)
    
    def update(self, aggr_out, x):
        # 更新节点特征
        combined = torch.cat([aggr_out, x], dim=-1)
        return F.relu(self.update_lin(combined))


class FusionLayer(nn.Module):
    """融合河网和因果信息"""
    def __init__(self, in_dim, out_dim):
        super(FusionLayer, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x):
        """
        x: [batch_size, num_nodes, num_features, hidden_dim*2] 或 [num_nodes, num_features, hidden_dim*2]
        """
        original_shape = x.shape
        # 处理张量维度
        if len(original_shape) == 4:  # 批次维度存在
            batch_size, num_nodes, num_features, feat_dim = original_shape
            x_flat = x.reshape(batch_size * num_nodes * num_features, feat_dim)
            out = self.fusion(x_flat)
            return out.reshape(batch_size, num_nodes, num_features, -1)
        else:  # 无批次维度
            num_nodes, num_features, feat_dim = original_shape
            x_flat = x.reshape(num_nodes * num_features, feat_dim)
            out = self.fusion(x_flat)
            return out.reshape(num_nodes, num_features, -1)


class TransformerLayer(nn.Module):
    """自注意力时间处理层"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        x: [seq_len, batch_size, hidden_dim]
        """
        # 自注意力
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # 前馈网络
        x = x + self.ffn(self.norm2(x))
        
        return x


class SpatioTemporalBlock(nn.Module):
    """先空间后时间的处理块"""
    def __init__(self, hidden_dim, time_steps):
        super(SpatioTemporalBlock, self).__init__()
        # 空间处理组件
        self.river_conv = RiverGraphConv(hidden_dim)
        self.causal_conv = CausalGraphConv(hidden_dim)
        self.fusion = FusionLayer(hidden_dim*2, hidden_dim)
        
        # 时间处理组件
        self.time_proj = nn.Linear(time_steps, hidden_dim)
        self.transformer = TransformerLayer(hidden_dim)
        
    def forward(self, x, river_edge_index, river_edge_attr, causal_edge_index, causal_edge_weight):
        """
        x: [batch_size, num_nodes, num_features, time_steps]
        """
        batch_size, num_nodes, num_features, time_steps = x.shape
        
        # 1. 空间处理 - 对每个时间步
        x_spatial = []
        
        for t in range(time_steps):
            x_t = x[:, :, :, t]  # [batch_size, num_nodes, num_features]
            
            # 处理每个批次
            batch_processed = []
            for b in range(batch_size):
                x_bt = x_t[b]  # [num_nodes, num_features]
                
                # 1.1 河网处理 - 流域级别
                # 提取流域表示
                watershed_repr = x_bt.mean(dim=1)  # [num_nodes, hidden_dim]
                # 应用河网图卷积
                watershed_updated = self.river_conv(watershed_repr, river_edge_index, river_edge_attr)  # [num_nodes, hidden_dim]
                
                # 1.2 因果处理 - 变量级别
                # 重塑为变量节点
                feature_nodes = x_bt.reshape(num_nodes * num_features, -1)  # [num_nodes*num_features, hidden_dim]
                # 应用因果图卷积
                feature_nodes_updated = self.causal_conv(feature_nodes, causal_edge_index, causal_edge_weight)  # [num_nodes*num_features, hidden_dim]
                feature_nodes_updated = feature_nodes_updated.reshape(num_nodes, num_features, -1)  # [num_nodes, num_features, hidden_dim]
                
                # 1.3 信息融合
                # 扩展流域表示以匹配特征维度
                watershed_expanded = watershed_updated.unsqueeze(1).expand(-1, num_features, -1)  # [num_nodes, num_features, hidden_dim]
                # 连接两种信息
                combined = torch.cat([feature_nodes_updated, watershed_expanded], dim=-1)  # [num_nodes, num_features, hidden_dim*2]
                # 融合
                fused = self.fusion(combined)  # [num_nodes, num_features, hidden_dim]
                
                batch_processed.append(fused)
                
            # 合并当前时间步的所有批次
            x_spatial.append(torch.stack(batch_processed))  # [batch_size, num_nodes, num_features, hidden_dim]
            
        # 合并所有时间步
        x_spatial = torch.stack(x_spatial, dim=3)  # [batch_size, num_nodes, num_features, time_steps, hidden_dim]
        
        # 2. 时间处理
        # 2.1 时间维度投影
        x_flat = x_spatial.reshape(batch_size * num_nodes * num_features, time_steps, -1)
        x_flat = x_flat.transpose(0, 1)  # [time_steps, batch*nodes*features, hidden_dim]
        
        # 2.2 应用Transformer层
        x_trans = self.transformer(x_flat)
        
        # 2.3 时间聚合
        x_trans = x_trans.transpose(0, 1)  # [batch*nodes*features, time_steps, hidden_dim]
        x_agg = x_trans.mean(dim=1)  # [batch*nodes*features, hidden_dim]
        
        # 重塑回原始格式
        x_out = x_agg.reshape(batch_size, num_nodes, num_features, -1)  # [batch_size, num_nodes, num_features, hidden_dim]
        
        return x_out


class GraphPooling(nn.Module):
    """基于层次结构的流域图池化"""
    def __init__(self, hidden_dim):
        super(GraphPooling, self).__init__()
        self.score_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, hierarchy_mapping):
        """
        x: [batch_size, num_fine_nodes, num_features, hidden_dim]
        hierarchy_mapping: {coarse_idx: [fine_idx1, fine_idx2, ...]}
        """
        batch_size, num_fine_nodes, num_features, hidden_dim = x.shape
        num_coarse_nodes = len(hierarchy_mapping)
        
        # 创建结果张量
        x_pooled = torch.zeros(batch_size, num_coarse_nodes, num_features, hidden_dim, device=x.device)
        
        for b in range(batch_size):
            # 计算每个细节点的重要性得分
            node_scores = self.score_layer(x[b].mean(dim=1))  # [num_fine_nodes, 1]
            
            # 执行加权池化
            for coarse_idx, fine_indices in hierarchy_mapping.items():
                if not fine_indices:  # 跳过空列表
                    continue
                    
                # 过滤有效索引
                valid_indices = [i for i in fine_indices if i < num_fine_nodes]
                if not valid_indices:
                    continue
                
                # 收集这些流域的特征和得分
                fine_features = x[b, valid_indices]  # [len(valid), num_features, hidden_dim]
                fine_scores = node_scores[valid_indices]  # [len(valid), 1]
                
                # 计算归一化权重
                weights = F.softmax(fine_scores, dim=0)  # [len(valid), 1]
                
                # 加权池化
                weighted_sum = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * fine_features, dim=0)  # [num_features, hidden_dim]
                x_pooled[b, coarse_idx] = weighted_sum
        
        return x_pooled


class GraphUnpooling(nn.Module):
    """基于层次结构的流域图上池化"""
    def __init__(self):
        super(GraphUnpooling, self).__init__()
        
    def forward(self, x, hierarchy_mapping, num_fine_nodes):
        """
        x: [batch_size, num_coarse_nodes, num_features, hidden_dim]
        hierarchy_mapping: {coarse_idx: [fine_idx1, fine_idx2, ...]}
        num_fine_nodes: 细粒度节点数量
        """
        batch_size, num_coarse_nodes, num_features, hidden_dim = x.shape
        
        # 创建结果张量
        x_unpooled = torch.zeros(batch_size, num_fine_nodes, num_features, hidden_dim, device=x.device)
        
        for b in range(batch_size):
            # 将粗粒度特征映射到细粒度
            for coarse_idx, fine_indices in hierarchy_mapping.items():
                if coarse_idx >= num_coarse_nodes:
                    continue
                    
                # 过滤有效索引
                valid_indices = [i for i in fine_indices if i < num_fine_nodes]
                if not valid_indices:
                    continue
                
                # 将粗粒度特征复制到所有对应的细粒度节点
                for fine_idx in valid_indices:
                    x_unpooled[b, fine_idx] = x[b, coarse_idx]
        
        return x_unpooled


class GraphUNetDualTransformer(nn.Module):
    """多尺度时空双图水文预测模型"""
    def __init__(self, hidden_dim, forecast_horizon, watershed_hierarchy, 
                 num_features, basin_ids, time_steps, blocks_per_level=2):
        super(GraphUNetDualTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.levels = len(watershed_hierarchy) + 1
        self.watershed_hierarchy = watershed_hierarchy
        self.num_features = num_features
        self.basin_ids = basin_ids
        self.time_steps = time_steps
        
        # 时空处理块
        self.st_blocks = nn.ModuleList()
        for i in range(self.levels):
            level_blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                level_blocks.append(SpatioTemporalBlock(hidden_dim, time_steps))
            self.st_blocks.append(level_blocks)
        
        # 图池化层
        self.graph_pooling = nn.ModuleList()
        for i in range(self.levels - 1):
            self.graph_pooling.append(GraphPooling(hidden_dim))
        
        # 图上池化层
        self.graph_unpooling = nn.ModuleList()
        for i in range(self.levels - 1):
            self.graph_unpooling.append(GraphUnpooling())
        
        # 预测层
        self.prediction_layers = nn.ModuleList()
        for i in range(self.levels):
            self.prediction_layers.append(nn.Linear(hidden_dim, forecast_horizon))
    
    def forward(self, x, river_edge_index, river_edge_attr, causal_edge_index, 
                causal_edge_weight, spatial_encoding=None):
        """
        x: [batch_size, num_nodes, num_features, time_steps]
        river_edge_index: [2, num_river_edges]
        river_edge_attr: [num_river_edges, edge_attr_dim]
        causal_edge_index: [2, num_causal_edges]
        causal_edge_weight: [num_causal_edges]
        spatial_encoding: [num_nodes, hidden_dim] - 预计算的空间编码
        """
        batch_size, num_nodes, num_features, time_steps = x.shape
        
        # 保存中间特征用于跳跃连接
        features = []
        node_counts = [num_nodes]
        
        # 1. 下采样路径（编码器）
        current_x = x
        
        for i in range(self.levels):
            # 应用当前尺度的时空处理块
            for block in self.st_blocks[i]:
                current_x = block(current_x, river_edge_index, river_edge_attr, 
                                 causal_edge_index, causal_edge_weight)
            
            # 应用预计算的空间编码（如果提供）
            if spatial_encoding is not None:
                # 扩展空间编码到所有特征和批次
                expanded_encoding = spatial_encoding.unsqueeze(1).expand(-1, num_features, -1)  # [num_nodes, num_features, hidden_dim]
                expanded_encoding = expanded_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, num_nodes, num_features, hidden_dim]
                current_x = current_x + expanded_encoding
            
            # 保存特征用于跳跃连接
            features.append(current_x)
            
            # 如果不是最粗尺度，进行图池化
            if i < self.levels - 1:
                # 使用当前层的层次映射
                hierarchy_map = self.watershed_hierarchy[i]
                current_x = self.graph_pooling[i](current_x, hierarchy_map)
                
                # 更新节点数量
                node_counts.append(current_x.size(1))
        
        # 2. 上采样路径（解码器）
        outputs = []
        
        # 最粗尺度的预测
        coarse_pred = self.prediction_layers[-1](current_x)  # [batch_size, num_coarse_nodes, num_features, forecast_horizon]
        outputs.append(coarse_pred)
        
        # 从粗到细逐步上采样
        for i in range(self.levels - 2, -1, -1):
            # 图上池化 - 将粗尺度特征映射到细尺度
            current_x = self.graph_unpooling[i](current_x, self.watershed_hierarchy[i], node_counts[i])
            
            # 跳跃连接 - 添加相应尺度的特征
            current_x = current_x + features[i]
            
            # 预测当前尺度
            current_pred = self.prediction_layers[i](current_x)
            outputs.insert(0, current_pred)
        
        return outputs


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    """初始化模型权重"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)