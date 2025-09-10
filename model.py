import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import math

try:
    # 工程里原先引用的 logger
    from data_loader import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# =========================
# 基础图算子
# =========================
class RiverGraphConv(MessagePassing):
    """处理流域级别河网连通"""
    def __init__(self, hidden_dim, edge_attr_dim=1):
        super().__init__(aggr='add')
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.edge_encoder = nn.Linear(edge_attr_dim, 1) if edge_attr_dim > 1 else nn.Identity()
        self.gate = nn.Parameter(torch.ones(1) * 0.3)

    def forward(self, x, edge_index, edge_attr):
        if hasattr(edge_attr, 'size') and len(edge_attr.size()) > 1:
            edge_weight = self.edge_encoder(edge_attr).squeeze(-1)
        else:
            edge_weight = edge_attr
        gate_value = torch.sigmoid(self.gate)
        mixed_weight = torch.clamp(gate_value * edge_weight, 0.0, 1.0)
        return self.propagate(edge_index, x=x, edge_weight=mixed_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * self.lin(x_j)

    def update(self, aggr_out, x):
        out = self.update_lin(torch.cat([aggr_out, x], dim=-1))
        return F.leaky_relu(out, 0.01)


class CausalGraphConv(MessagePassing):
    """变量间因果图卷积（节点 = (流域, 特征)）"""
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Parameter(torch.ones(1) * 0.3)

    def forward(self, x, edge_index, edge_weight):
        gate_value = torch.sigmoid(self.gate)
        mixed_weight = torch.clamp(gate_value * edge_weight, 0.0, 1.0)
        return self.propagate(edge_index, x=x, edge_weight=mixed_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * self.lin(x_j)

    def update(self, aggr_out, x):
        out = self.update_lin(torch.cat([aggr_out, x], dim=-1))
        return F.leaky_relu(out, 0.01)


# =========================
# 图层次池化 / 上采样
# =========================
class GraphPooling(nn.Module):
    """层次结构池化：将细粒度节点聚合到 coarse"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.score_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, hierarchy_mapping):
        """
        x: [B, NF, Fm, H]
        hierarchy_mapping: dict {coarse_idx: [fine_idx,...]}
        """
        B, NF, Fm, H = x.shape   # 注意：用 Fm 取代 F，避免遮蔽 functional F
        NC = len(hierarchy_mapping)
        device = x.device
        out = torch.zeros(B, NC, Fm, H, device=device)
        for b in range(B):
            # 对每个细节点的 (沿特征平均) 取 score
            scores = self.score_layer(x[b].mean(dim=1))  # [NF,1]
            for c_idx, fine_list in hierarchy_mapping.items():
                if not fine_list:
                    continue
                valid = [i for i in fine_list if i < NF]
                if not valid:
                    continue
                feats = x[b, valid]      # [k, Fm, H]
                sc = scores[valid]       # [k, 1]
                w = F.softmax(sc, dim=0) # 这里的 F 仍是 functional
                pooled = (w.unsqueeze(-1) * feats).sum(dim=0)  # [Fm,H]
                out[b, c_idx] = pooled
        return out


class GraphUnpooling(nn.Module):
    """层次结构上采样：将 coarse 特征广播回细粒度节点"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

    def forward(self, x, hierarchy_mapping, num_fine_nodes):
        """
        x: [B, NC, Fm, H]
        """
        B, NC, Fm, H = x.shape   # 用 Fm 取代 F
        device = x.device
        out = torch.zeros(B, num_fine_nodes, Fm, H, device=device)
        x_tf = self.feature_transform(x.reshape(-1, H)).reshape(B, NC, Fm, H)
        for b in range(B):
            for c_idx, fine_list in hierarchy_mapping.items():
                if c_idx >= NC:
                    continue
                valid = [i for i in fine_list if i < num_fine_nodes]
                for fi in valid:
                    out[b, fi] = x_tf[b, c_idx]
        return out


# =========================
# 时空块（方案A：保留 per-feature 表达）
# =========================
class SpatioTemporalBlock(nn.Module):
    """
    输入：
        x_global: [B, N, H, T]          （全局融合向量）
        per_feature_x: [B, N, F, H, T]  （保留每特征的隐藏表示）
    输出：
        x_out_global: 同形状 [B,N,H,T]
        per_feature_out: 同形状 [B,N,F,H,T] （更新后的每特征表示）
    """
    def __init__(self, hidden_dim, time_steps, num_features,
                 edge_attr_dim=4, debug=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.time_steps = time_steps
        self.debug = debug
        self.per_feature_norm = nn.LayerNorm(hidden_dim)

        # 时间位置编码 (固定)
        self.time_pos_enc = nn.Parameter(
            self._build_sin_cos(time_steps, hidden_dim),
            requires_grad=False
        )

        # 时间卷积：对 global 表达做 temporal conv
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            groups=1
        )

        # 图卷积
        self.river_conv = RiverGraphConv(hidden_dim, edge_attr_dim)
        self.causal_conv = CausalGraphConv(hidden_dim)

        # 空间融合（河网 + 因果池化）
        self.spatial_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU()
        )

        # 变量节点静态嵌入（补充可学习区分度）
        self.feature_emb = nn.Embedding(num_features, hidden_dim)
        nn.init.normal_(self.feature_emb.weight, mean=0.0, std=0.02)

        # 时空双分支融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def _build_sin_cos(self, T, H):
        pe = torch.zeros(1, 1, T, H)
        pos = torch.arange(0, T).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, H, 2).float() * -(math.log(10000.0) / H))
        H_even = (H // 2) * 2
        pe[..., 0:H_even:2] = torch.sin(pos * div[:H_even // 2])
        pe[..., 1:H_even:2] = torch.cos(pos * div[:H_even // 2])
        if H % 2 == 1:
            pe[..., -1] = torch.sin(pos * div[-1])
        return pe  # [1,1,T,H]

    def forward(self, x_global, per_feature_x,
                river_edge_index, river_edge_attr,
                causal_edge_index, causal_edge_weight):
        """
        x_global: [B,N,H,T]
        per_feature_x: [B,N,F,H,T]
        """
        B, N, H, T = x_global.shape
        _, _, Fm, _, _ = per_feature_x.shape
        device = x_global.device

        if self.debug:
            logger.info(f"[STBlock] 输入 global {x_global.shape} per_feature {per_feature_x.shape}")

        # ---------- 时间分支 ----------
        # x_global -> [B,N,T,H] 加位置编码
        x_t = x_global.permute(0, 1, 3, 2)  # [B,N,T,H]
        x_t = x_t + self.time_pos_enc  # 广播

        # 合并 (B*N) 做 Conv1d
        x_bn_th = x_t.reshape(B * N, T, H).transpose(1, 2)  # [B*N,H,T]
        x_temporal = self.temporal_conv(x_bn_th)            # [B*N,H,T]
        x_temporal = x_temporal.transpose(1, 2).reshape(B, N, T, H)  # [B,N,T,H]

        # ---------- 空间 + 因果分支 ----------
        x_spatial = torch.zeros(B, N, H, T, device=device)
        per_feature_updated = torch.zeros_like(per_feature_x)

        # 逐时间步处理（可后续向量化）
        for b in range(B):
            for t in range(T):
                # 当前时间的全局节点表示
                x_bt = x_global[b, :, :, t]          # [N,H]
                # 每特征节点表示
                feat_bt = per_feature_x[b, :, :, :, t]  # [N,F,H]

                # 河网更新（流域节点级）
                watershed_upd = self.river_conv(x_bt, river_edge_index, river_edge_attr)  # [N,H]

                # 因果图：展平 (N*Fm)
                feat_nodes = (feat_bt + self.feature_emb.weight.unsqueeze(0))  # [N,F,H]
                feat_nodes_flat = feat_nodes.reshape(N * Fm, H)
                feat_nodes_upd = self.causal_conv(
                    feat_nodes_flat,
                    causal_edge_index,
                    causal_edge_weight
                ).reshape(N, Fm, H)

                # 保存更新后的 per-feature
                per_feature_updated[b, :, :, :, t] = feat_nodes_upd

                # 池化回流域级（简单平均，可换注意力）
                causal_pooled = feat_nodes_upd.mean(dim=1)  # [N,H]

                fused = self.spatial_fuse(torch.cat([watershed_upd, causal_pooled], dim=-1))  # [N,H]
                x_spatial[b, :, :, t] = fused

        # === Per-feature normalization (新增) ===
        # per_feature_updated: [B,N,Fm,H,T]
        pf = per_feature_updated.permute(0, 1, 4, 2, 3).contiguous()  # [B,N,T,Fm,H]
        pf_flat = pf.reshape(-1, Fm, H)                               # [(B*N*T), Fm, H]
        # LayerNorm 作用在最后一维 H，不改变前两维结构
        pf_normed = self.per_feature_norm(pf_flat)                    # [(B*N*T), Fm, H]
        per_feature_updated = pf_normed.reshape(B, N, T, Fm, H).permute(0, 1, 3, 4, 2).contiguous()

        # ---------- 融合 ----------
        # x_temporal: [B,N,T,H] -> flatten
        x_temp_flat = x_temporal.reshape(B * N * T, H)
        x_spatial_flat = x_spatial.permute(0, 1, 3, 2).reshape(B * N * T, H)
        fused = self.fusion(torch.cat([x_temp_flat, x_spatial_flat], dim=-1))  # [B*N*T,H]
        x_out = fused.reshape(B, N, T, H).permute(0, 1, 3, 2)  # [B,N,H,T]

        return x_out, per_feature_updated


# =========================
# 预测头
# =========================
class PhysicallyInformedPrediction(nn.Module):
    def __init__(self, hidden_dim, forecast_horizon, debug=False):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.5)  # 可学习缩放
        self.projection = nn.Linear(hidden_dim, forecast_horizon)
        self.debug = debug

    def forward(self, x):
        # x: [..., H]
        x = self.pre_norm(x)
        x = x * self.scale
        logits = self.projection(x)
        out = torch.sigmoid(logits)
        if self.debug:
            with torch.no_grad():
                logger.info(f"[Head] logits std={logits.std():.4f} out_range=({out.min():.4f},{out.max():.4f})")
        return out


# =========================
# 损失（当前仅 MSE）
# =========================
class HydrologicalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)


# =========================
# 主模型（方案 A 集成）
# =========================
class GraphUNetDualTransformer(nn.Module):
    """
    方案A：保留 per-feature 表达：
        - 输入阶段：每个特征独立投影 -> [B,N,F,T,H]
        - 注意力聚合 -> global 表达 [B,N,T,H] -> 转成 [B,N,H,T] 兼容原逻辑
        - 时空块同时接收 global + per_feature
        - 下采样/上采样同时处理 per_feature latent
        - 预测：直接用 per-feature latent 逐特征预测（不再复制同一向量）
    """
    def __init__(self, hidden_dim, forecast_horizon, watershed_hierarchy,
                 num_features, basin_ids, time_steps,
                 blocks_per_level=2, edge_attr_dim=4,
                 dropout=0.2, debug=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.num_features = num_features
        self.time_steps = time_steps
        self.basin_ids = basin_ids
        self.debug = debug

        # 层次数（最细 + 每个层次的 coarse）
        self.levels = len(watershed_hierarchy) + 1
        self.watershed_hierarchy = watershed_hierarchy

        self.dropout = nn.Dropout(dropout)

        # 方案A 输入模块：per-feature 独立投影 + feature attention
        self.feature_proj = nn.Linear(1, hidden_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.global_fuse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # 时空块
        self.st_blocks = nn.ModuleList()
        for _level in range(self.levels):
            level_list = nn.ModuleList()
            for _ in range(blocks_per_level):
                level_list.append(
                    SpatioTemporalBlock(hidden_dim, time_steps, num_features,
                                        edge_attr_dim=edge_attr_dim, debug=debug)
                )
            self.st_blocks.append(level_list)

        # 每层归一化（作用于 global 表达的 hidden 维度）
        self.level_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(self.levels)]
        )

        # 层次池化/上采样（针对 per-feature latent）
        self.graph_pooling = nn.ModuleList(
            [GraphPooling(hidden_dim) for _ in range(self.levels - 1)]
        )
        self.graph_unpooling = nn.ModuleList(
            [GraphUnpooling(hidden_dim) for _ in range(self.levels - 1)]
        )

        # 预测层：每层一个（可按需要只用最细层）
        self.prediction_layers = nn.ModuleList([
            PhysicallyInformedPrediction(hidden_dim, forecast_horizon, debug=debug)
            for _ in range(self.levels)
        ])

    # --------- 工具函数 ----------
    def _build_feature_attention_global(self, per_feature_x):
        """
        per_feature_x: [B,N,F,H,T]
        返回 global 表达: [B,N,H,T]
        """
        B, N, Fm, H, T = per_feature_x.shape
        # 对时间逐步做注意力（可向量化）
        global_list = []
        for t in range(T):
            slice_t = per_feature_x[:, :, :, :, t]  # [B,N,F,H]
            gate = self.feature_gate(slice_t)       # [B,N,F,1]
            attn = torch.softmax(gate, dim=2)
            fused = (attn * slice_t).sum(dim=2)     # [B,N,H]
            global_list.append(fused)
        global_x = torch.stack(global_list, dim=-1)  # [B,N,H,T]
        global_x = self.global_fuse(global_x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # LayerNorm over H
        return global_x

    # --------- 前向 ----------
    def forward(self, x, multi_scale_graph_data, spatial_encoding=None):
        """
        x: [B,N,F,T]  归一化输入
        multi_scale_graph_data: {
            'fine':   {'river_edge_index', 'river_edge_attr', 'causal_edge_index', 'causal_edge_weight'},
            'medium': {...},
            'coarse': {...}
        }
        """
        B, N, Fm, T = x.shape
        if self.debug:
            logger.info(f"=== 模型前向传播调试 ===")
            logger.info(f"输入 x: shape={x.shape}, range=({x.min():.6f},{x.max():.6f})")

        # ---------- 输入阶段：per-feature 独立投影 ----------
        # x.unsqueeze(-1): [B,N,F,T,1] -> feature_proj -> [B,N,F,T,H]
        per_feature_latent = self.feature_proj(x.unsqueeze(-1))  # [B,N,F,T,H]
        # 调整为 [B,N,F,H,T]
        per_feature_latent = per_feature_latent.permute(0, 1, 2, 4, 3).contiguous()

        # 构建 global 表达（初始）
        global_x = self._build_feature_attention_global(per_feature_latent)  # [B,N,H,T]

        if self.debug:
            logger.info(f"输入投影后 global_x: {global_x.shape}, range=({global_x.min():.6f},{global_x.max():.6f})")

        # 保存编码阶段特征（global 和 per-feature）用于 skip
        global_skips = []
        per_feature_skips = []
        node_counts = [N]

        current_global = global_x
        current_per_feature = per_feature_latent  # [B,N,F,H,T]

        # ---------- 编码（下采样路径） ----------
        for lvl in range(self.levels):
            if self.debug:
                logger.info(f"\n--- Level {lvl} 开始 ---")
                logger.info(f"Level {lvl} 输入 global={current_global.shape} per_feature={current_per_feature.shape}")

            # 图数据选择
            if lvl == 0:
                scale_name = 'fine'
            elif lvl == 1:
                scale_name = 'medium'
            else:
                scale_name = 'coarse'
            gdata = multi_scale_graph_data[scale_name]
            river_edge_index = gdata['river_edge_index']
            river_edge_attr = gdata['river_edge_attr']
            causal_edge_index = gdata['causal_edge_index']
            causal_edge_weight = gdata['causal_edge_weight']
            res_scale = 0.5  # 可以先设 0.5；若仍偏大再降到 0.25
            # 多个 block 叠加（残差）
            for block in self.st_blocks[lvl]:
                block_out_global, block_out_per_feat = block(
                    current_global, current_per_feature,
                    river_edge_index, river_edge_attr,
                    causal_edge_index, causal_edge_weight
                )
                # Global 残差（先缩放，再加，再 dropout）
                current_global = current_global + res_scale * block_out_global
                current_global = self.dropout(current_global)

                # Per-feature 残差（之前是直接赋值，改为残差形式）
                current_per_feature = current_per_feature + res_scale * block_out_per_feat

            # 层级归一化 (global)
            Bc, Nc, Hc, Tc = current_global.shape
            cg_flat = current_global.permute(0, 1, 3, 2).reshape(-1, Hc)
            cg_norm = self.level_norms[lvl](cg_flat)
            current_global = cg_norm.reshape(Bc, Nc, Tc, Hc).permute(0, 1, 3, 2)

            if spatial_encoding is not None:
                # spatial_encoding: [N,H] -> broadcast
                se = spatial_encoding.unsqueeze(0).unsqueeze(-1)  # [1,N,H,1]
                if se.shape[1] == Nc:
                    current_global = current_global + se

            global_skips.append(current_global)
            per_feature_skips.append(current_per_feature)

            # 池化（除最后一层）
            if lvl < self.levels - 1:
                hierarchy_map = self.watershed_hierarchy[lvl]
                _, n_nodes, f_nodes, h_dim, t_steps = current_per_feature.shape

                pooled_time = []
                pooled_time_per_feature = []
                for tt in range(t_steps):
                    # per-feature at time tt: [B,N,F,H]
                    pf_t = current_per_feature[..., tt]  # [B,N,F,H]
                    pooled_pf_t = self.graph_pooling[lvl](pf_t, hierarchy_map)  # [B, Nc, F, H]
                    pooled_time_per_feature.append(pooled_pf_t)
                    # 构建 global（注意力）
                    gate = self.feature_gate(pooled_pf_t)            # [B,Nc,F,1]
                    attn = torch.softmax(gate, dim=2)
                    global_t = (attn * pooled_pf_t).sum(dim=2)       # [B,Nc,H]
                    pooled_time.append(global_t)

                # 重建带时间维
                current_global = torch.stack(pooled_time, dim=-1)  # [B,Nc,H,T]
                # per-feature latent
                current_per_feature = torch.stack(pooled_time_per_feature, dim=-1)  # [B,Nc,F,H,T]

                node_counts.append(current_global.size(1))

        # ---------- 解码（上采样路径） ----------
        outputs = []

        # 粗尺度预测：对时间做 avg+max 融合
        if current_global.dim() == 4:
            avg = current_global.mean(dim=-1)
            mx, _ = current_global.max(dim=-1)
            coarse_global_vec = 0.5 * (avg + mx)  # [B,Nc,H]
        else:
            coarse_global_vec = current_global

        # 使用 per-feature latent（同样时间聚合）
        if current_per_feature.dim() == 5:
            pf_avg = current_per_feature.mean(dim=-1)  # [B,Nc,F,H]
        else:
            pf_avg = current_per_feature  # [B,Nc,F,H]

        # 逐特征预测
        coarse_pred_in = pf_avg.reshape(-1, self.hidden_dim)
        coarse_pred = self.prediction_layers[-1](coarse_pred_in).reshape(
            B, pf_avg.size(1), self.num_features, self.forecast_horizon
        )
        outputs.append(coarse_pred)

        if self.debug:
            logger.info("Decoder 粗尺度预测后")
            logger.info(f"coarse_pred: shape={coarse_pred.shape}, range=({coarse_pred.min():.6f},{coarse_pred.max():.6f})")

        # 准备解码循环
        current_pf = pf_avg  # [B,Nc,F,H]
        # 回到多尺度：自顶向下
        for lvl in range(self.levels - 2, -1, -1):
            if self.debug:
                logger.info(f"\n--- Decoder Level {lvl} 开始 ---")

            # 上采样 per-feature latent
            unpooled_pf = self.graph_unpooling[lvl](
                current_pf, self.watershed_hierarchy[lvl], node_counts[lvl]
            )  # [B, fine_nodes, F, H]

            # 跳跃融合（使用编码阶段的 per-feature latent，时间维已有；取时间均值）
            skip_pf = per_feature_skips[lvl]  # [B,fine,F,H,T]
            skip_pf_mean = skip_pf.mean(dim=-1)  # [B,fine,F,H]
            current_pf = unpooled_pf + skip_pf_mean

            if self.debug:
                rng_min = current_pf.min().item()
                rng_max = current_pf.max().item()
                logger.info(f"跳跃连接后 per-feature latent: shape={current_pf.shape}, range=({rng_min:.6f},{rng_max:.6f})")

            # 预测（直接用 per-feature latent）
            pred_in = current_pf.reshape(-1, self.hidden_dim)
            pred = self.prediction_layers[lvl](pred_in).reshape(
                B, current_pf.size(1), self.num_features, self.forecast_horizon
            )
            # 插入到前面（多尺度可选：这里只保留最细作为输出，也可保留全部）
            outputs.insert(0, pred)

        final_out = outputs[0]
        if self.debug:
            logger.info("\n--- 最终预测 ---")
            logger.info(f"最终输出: shape={final_out.shape}, range=({final_out.min():.6f},{final_out.max():.6f})")

        return final_out  # [B, fine_nodes, F, horizon]


# =========================
# 实用函数
# =========================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    """初始化模型权重"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)