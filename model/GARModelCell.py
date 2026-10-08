import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from .MyMultiHeadAttention import MyMultiheadAttention
from .Transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder


class GARModelCell(nn.Module):
    def __init__(self, cfg, embed_features):
        super(GARModelCell, self).__init__()
        self.cfg = cfg
        self.embed_features = embed_features
        encoderLayer = TransformerEncoderLayer(self.embed_features, self.cfg.headNum, dropout=self.cfg.dropoutNum,
                                               normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.garEncoder = TransformerEncoder(encoderLayer, num_layers=self.cfg.encoderLayersNum, norm=encoderNorm)

        decoderLayer = TransformerDecoderLayer(self.embed_features, self.cfg.headNum, dropout=self.cfg.dropoutNum,
                                               normalize_before=True)
        decoderNorm = nn.LayerNorm(self.embed_features)
        self.garDecoder = TransformerDecoder(decoderLayer, num_layers=self.cfg.decoderLayersNum, norm=decoderNorm)
        self.selfAttn = MyMultiheadAttention(self.embed_features, 8, dropout=0.1)

    def forward(self, bImgFLN):
        spatialFeaturesAll = self.spatialEncoder(bImgFLN)
        bImgFLN = bImgFLN.permute(1, 0, 2).contiguous()
        spatialFeaturesAll = spatialFeaturesAll.permute(1, 0, 2).contiguous()
        attn_output, attn_output_weights = self.selfAttnCon(bImgFLN, bImgFLN, value=bImgFLN)
        topk_values, topk_indices = torch.topk(attn_output_weights.sum(dim=1), self.KEYT, dim=1)
        sorted_topk_indices, indices = torch.sort(topk_indices, dim=1)
        sorted_topk_indices_ex = sorted_topk_indices.unsqueeze(2).expand(-1, -1, bImgFLN.size(2))
        batchImgsFeaturesNew = torch.gather(bImgFLN, 1, sorted_topk_indices_ex)

        sorted_topk_indices_ex1 = sorted_topk_indices.unsqueeze(2).expand(-1, -1, spatialFeaturesAll.size(2))
        spatialFeatures = torch.gather(spatialFeaturesAll, 1, sorted_topk_indices_ex1)

        N, TKey, F = batchImgsFeaturesNew.shape
        pos_encoding = PositionalEncoding2D(F, T)
        batchImgsFeaturesNew = pos_encoding(batchImgsFeaturesNew)
        cls_temporal_tokens = self.tokenCLSInit.repeat_interleave(N, dim=0)
        temporalMatrix = torch.cat((cls_temporal_tokens, batchImgsFeaturesNew), dim=1)
        temporalFeatureEncoder = self.temporalEncoder(temporalMatrix)
        temporalFeatureEncoderT = temporalFeatureEncoder.permute(1, 0, 2).contiguous().reshape(TKey + 1, N, -1)
        features = temporalFeatureEncoderT[0] + spatialFeatures.mean(dim=1).reshape(N, F)
