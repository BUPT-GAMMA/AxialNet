import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from axialnet.utils.registry import MODELS

# ==========================================
# 基础模块定义 (与之前类似，保持不变)
# ==========================================
# class HGNN_GAT_Layer(nn.Module):
#     def __init__(self, hidden_dim, heads=4):
#         super().__init__()
#         # 使用 GATConv，保持输入输出维度一致 (concat=False)
#         self.conv = HeteroConv({
#             ('net', 'to', 'grid_l1'): GATConv((hidden_dim, hidden_dim), hidden_dim, heads=heads, concat=False, add_self_loops=False),
#             ('grid_l1', 'to', 'net'): GATConv((hidden_dim, hidden_dim), hidden_dim, heads=heads, concat=False, add_self_loops=False),
#         }, aggr='mean')

#     def forward(self, x_dict, edge_index_dict):
#         out_dict = self.conv(x_dict, edge_index_dict)
#         return {key: F.elu(x) for key, x in out_dict.items()}

class HGNN_GAT_Layer(nn.Module):
    def __init__(self, hidden_dim, heads=4):
        super().__init__()
        # 使用 GATConv，保持输入输出维度一致 (concat=False)
        self.conv = HeteroConv({
            ('net', 'to', 'grid_l1'): GATConv((hidden_dim, hidden_dim), hidden_dim, heads=heads, concat=False, add_self_loops=False),
            ('grid_l1', 'to', 'net'): GATConv((hidden_dim, hidden_dim), hidden_dim, heads=heads, concat=False, add_self_loops=False),
        }, aggr='mean')

        # ==========================================
        # 新增：为异构图的每种节点定义残差连接所需的线性层
        # ==========================================
        self.res_proj = nn.ModuleDict({
            'net': nn.Linear(hidden_dim, hidden_dim),
            'grid_l1': nn.Linear(hidden_dim, hidden_dim)
        })

    def forward(self, x_dict, edge_index_dict):
        # 1. 经过 GAT 卷积层聚合邻居信息
        out_dict = self.conv(x_dict, edge_index_dict)
        
        res_dict = {}
        for key, x in out_dict.items():
            # 2. 残差连接：原始特征过线性层后，与卷积层的输出相加
            # 注意：只有被作为目标节点更新了的节点类型（在 out_dict 中）才会参与计算
            residual = self.res_proj[key](x_dict[key])
            out = x + residual
            
            # 3. 加上激活函数
            res_dict[key] = F.elu(out)
            
        return res_dict

class MultiReceptiveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = out_channels // 4
        self.branch1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=4, dilation=4)
        self.branch5 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=8, dilation=8)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(inter_channels * 4, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        out = torch.cat([x1, x2, x3, x5], dim=1)
        return self.fusion(out)

class StripwiseAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, direction='horizontal'):
        B, C, H, W = x.shape
        if direction == 'horizontal':
            x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        elif direction == 'vertical':
            x_reshaped = x.permute(0, 3, 2, 1).contiguous().view(B * W, H, C)
            
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        attn_out = self.norm(x_reshaped + attn_out)
        
        if direction == 'horizontal':
            out = attn_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            out = attn_out.view(B, W, H, C).permute(0, 3, 2, 1).contiguous()
        return out

# ==========================================
# 复合模块：包含(全局卷积 + 双向条状注意力)
# ==========================================
class SpatialFusionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cnn = MultiReceptiveConv2d(hidden_dim, hidden_dim)
        self.h_attn = StripwiseAttention(hidden_dim)
        self.v_attn = StripwiseAttention(hidden_dim)
        
        # 如果还在循环中，需要将融合后的特征还给图卷积使用
        self.merge = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1),
            nn.InstanceNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, grid_2d, is_final=False):
        # 1. 多感受野提取局部/全局空间特征
        cnn_feat = self.cnn(grid_2d)
        
        # 2. 横向和纵向条状注意力
        h_feat = self.h_attn(cnn_feat, direction='horizontal')
        v_feat = self.v_attn(cnn_feat, direction='vertical')
        
        if is_final:
            # 最后一层：不融合，保持维度的解耦，输出高维表示给Decoder
            # 拼接 CNN特征和注意力特征 (通道数翻倍)
            out_h = torch.cat([cnn_feat, h_feat], dim=1)
            out_v = torch.cat([cnn_feat, v_feat], dim=1)
            return out_h, out_v, cnn_feat
        else:
            # 循环中间：合并特征以便后续输入到 Graph 中
            merged = torch.cat([cnn_feat, h_feat, v_feat], dim=1)
            return self.merge(merged)

# ==========================================
# 主编码器模型：实现你要求的循环控制
# ==========================================
@MODELS.register
class DualCongestionEncoder(nn.Module):
    def __init__(self, grid_in_dim=4, net_in_dim=4, hidden_dim=64, num_loops=1,
                 apply_graph_updates=False):
        super().__init__()
        self.num_loops = num_loops
        self.hidden_dim = hidden_dim
        # False preserves the paper configuration. True feeds GAT outputs back
        # into grid_1d/net_1d and therefore requires retraining.
        self.apply_graph_updates = bool(apply_graph_updates)

        # 初始特征投影
        self.grid_proj = nn.Linear(grid_in_dim, hidden_dim)
        self.net_proj = nn.Linear(net_in_dim, hidden_dim)

        # 循环模块定义
        self.spatial_blocks = nn.ModuleList([SpatialFusionBlock(hidden_dim) for _ in range(num_loops)])
        self.graph_blocks = nn.ModuleList([HGNN_GAT_Layer(hidden_dim) for _ in range(num_loops)])
        self.graph_blocks_2 = nn.ModuleList([HGNN_GAT_Layer(hidden_dim) for _ in range(num_loops)])  # 第二层图卷积，增加模型深度
        
        # 最后一次的全局卷积和条状注意力
        self.final_spatial = SpatialFusionBlock(hidden_dim)

    # ---------------- 数据转换工具 ----------------
    def _graph_to_image(self, grid_x, ptr, sizes):
        """支持将不同大小的 Graph 网格 Padding 对齐并打包为 2D Image Batch"""
        batch_size = ptr.numel() - 1
        max_h = int(sizes[:, 0].max().item())
        max_w = int(sizes[:, 1].max().item())
        channels = grid_x.size(-1)

        img_list = []
        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            h, w = int(sizes[i][0].item()), int(sizes[i][1].item())
            feat_1d = grid_x[start:end]

            feat_2d = feat_1d.view(h, w, channels).permute(2, 0, 1).contiguous()
            
            # 补齐到 Batch 内最大尺寸，保证 torch.stack 成功
            pad_h, pad_w = max_h - h, max_w - w
            if pad_h > 0 or pad_w > 0:
                feat_2d = F.pad(feat_2d, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            img_list.append(feat_2d)

        return torch.stack(img_list, dim=0)

    def _image_to_graph(self, grid_2d, ptr, sizes):
        """将 2D Image 裁剪掉 Padding 部分，重新展平还原为 Graph 节点"""
        batch_size = ptr.numel() - 1
        feat_1d_list = []
        for i in range(batch_size):
            h, w = int(sizes[i][0].item()), int(sizes[i][1].item())
            feat_valid = grid_2d[i, :, :h, :w] # 去除 Padding
            feat_1d = feat_valid.permute(1, 2, 0).reshape(-1, grid_2d.size(1)).contiguous()
            feat_1d_list.append(feat_1d)

        return torch.cat(feat_1d_list, dim=0)

    # ---------------- 前向传播 ----------------
    def forward(self, batch):
        # 初始化节点特征
        grid_1d = self.grid_proj(batch['grid_l1'].x.float())
        net_1d = self.net_proj(batch['net'].x.float())
        edge_index_dict = batch.edge_index_dict
        
        ptr = batch['grid_l1'].ptr
        sizes = batch['grid_l1'].size.view(-1, 2)

        # 核心逻辑：执行 N 次大循环 (Spatial -> Graph)
        for i in range(self.num_loops):
            # == 步骤 A：全局卷积 + 条状注意力 ==
            grid_2d = self._graph_to_image(grid_1d, ptr, sizes)
            grid_2d = self.spatial_blocks[i](grid_2d, is_final=False) # 内部进行H和V的融合
            grid_1d = self._image_to_graph(grid_2d, ptr, sizes)

            # == 步骤 B：图卷积 (HGNN) ==
            x_dict = {'grid_l1': grid_1d, 'net': net_1d}
            x_dict_updated = self.graph_blocks[i](x_dict, edge_index_dict)
            # x_dict_updated = self.graph_blocks_2[i](x_dict_updated, edge_index_dict)  # 第二层图卷积，增加模型深度
            
            if self.apply_graph_updates:
                grid_1d = x_dict_updated['grid_l1']
                net_1d = x_dict_updated['net']

            # 使用残差连接 (Residual) 保护原始特征， GAT内部已经实现
            # grid_1d = grid_1d + x_dict_updated['grid_l1']
            # net_1d = net_1d + x_dict_updated['net']

        # == 循环结束后：再执行一次 全局卷积 + 条状注意力 ==
        grid_2d = self._graph_to_image(grid_1d, ptr, sizes)
        # is_final=True 时，返回解耦的横纵向特征 (此时通道数翻倍，例如从64变为128)
        out_h_2d, out_v_2d, cnn_feat = self.final_spatial(grid_2d, is_final=True)

        # 重新展平回 PyG 的 1D 格式
        out_h_1d = self._image_to_graph(out_h_2d, ptr, sizes)
        out_v_1d = self._image_to_graph(out_v_2d, ptr, sizes)
        cnn_1d = self._image_to_graph(cnn_feat, ptr, sizes)

        # 返回高维特征字典，供后续Decoder解码
        return {
            'grid_h_feat': out_h_1d, # shape: [N_nodes, hidden_dim * 2]
            'grid_v_feat': out_v_1d, # shape: [N_nodes, hidden_dim * 2]
            'grid_cnn_feat': cnn_1d,  # shape: [N_nodes, hidden_dim]
            'net_feat': net_1d       # 如果 downstream 任务需要线网级特征，可以一同返回
        }
