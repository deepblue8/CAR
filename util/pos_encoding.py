import math
import torch
from torch import nn

def positionalencoding1d(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def bboxencoding2d(d_model, positions):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    num_pos = positions.shape[0]
    pe = positions.new_zeros((num_pos, d_model))

    assert d_model % 2 == 0
    d_model = d_model // 2
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=positions.device) *
                          -(math.log(10000.0) / d_model)))
    pos_x = ((positions[:, 0] + positions[:, 2]) / 2).unsqueeze(1)
    pos_y = ((positions[:, 1] + positions[:, 3]) / 2).unsqueeze(1)
    pe[:, 0:d_model:2] = torch.sin(pos_x.float() * div_term)
    pe[:, 1:d_model:2] = torch.cos(pos_x.float() * div_term)
    pe[:, d_model::2] = torch.sin(pos_y.float() * div_term)
    pe[:, d_model + 1::2] = torch.cos(pos_y.float() * div_term)

    return pe


def spatialencoding2d(d_model, height, width, device):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width, device=device)

    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=1000, device='cuda:0'):
        super(PositionalEncoding2D, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        pos_encoding = self.encoding[:, :seq_len, :]
        pos_encoding = pos_encoding.repeat(batch_size, 1, 1)
        return x + pos_encoding.detach()


def postion_attn_mask(x, xyxy=4, k=8):
    B, TN, _ = x.shape
    device = x.device
    x = x.reshape(-1, xyxy)
    x[:, 0] = (x[:, 0] + x[:, 2]) / 2
    x[:, 1] = (x[:, 1] + x[:, 3]) / 2
    x = x[:, :2]
    x = x.reshape(B, TN, -1)
    y = x
    _, M, _ = y.shape
    inner = -2 * torch.matmul(y, x.permute(0, 2, 1).contiguous())
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    pairwise_distance = - inner - yy - xx.permute(0, 2, 1).contiguous()
    _, idx = pairwise_distance.topk(k=k, dim=-1)

    idx_knn = idx.reshape(B, M, k)
    idx = idx_knn
    idx = idx.reshape(B * M, -1)
    attn_mask = torch.zeros_like(pairwise_distance, device=device).view(B * M, -1)

    for i in range(B * M):
        attn_mask[i, idx[i]] = 1

    attn_mask = attn_mask.reshape(B, M, TN)

    return attn_mask
