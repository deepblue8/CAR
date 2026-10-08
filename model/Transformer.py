import copy
from typing import Optional

from torch import nn, Tensor
from torch.nn.modules.transformer import _get_clones

from model.MyMultiHeadAttention import MyMultiheadAttention
from model.MyMultiHeadAttentionCon import MyMultiheadAttentionTanhRelu


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, spatialFeatureEncoder, temporalFeatureDecoder):
        output = spatialFeatureEncoder
        for layer in self.layers:
            output = layer(output, temporalFeatureDecoder)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayerTanhRelu(nn.Module):
    def __init__(self, d_model, headNum, feedforwardDim=2048, dropout=0.1, normalize_before=True):
        super().__init__()
        self.layerNorm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.normalize_before = normalize_before

        self.selfAttn = MyMultiheadAttentionTanhRelu(d_model, headNum, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, headNum, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, feedforwardDim)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforwardDim, d_model)
        self.relu = nn.ReLU()

    def forward(self, features):
        lnFeatures = self.layerNorm(features)
        q = k = lnFeatures
        attnFeatures = self.selfAttn(q, k, value=lnFeatures)[0]
        r1Features = lnFeatures + self.dropout1(attnFeatures)
        r1Features = self.layerNorm(r1Features)
        feedFeatures = self.linear2(self.dropout2(self.relu(self.linear1(r1Features))))
        outFeatures = r1Features + self.dropout1(feedFeatures)
        outFeatures = self.layerNorm(outFeatures)

        return outFeatures


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, headNum, feedforwardDim=2048, dropout=0.1, normalize_before=True):
        super().__init__()
        self.layerNorm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.normalize_before = normalize_before

        self.selfAttn = MyMultiheadAttention(d_model, headNum, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, headNum, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, feedforwardDim)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforwardDim, d_model)
        self.relu = nn.ReLU()

    def forward(self, features):
        lnFeatures = self.layerNorm(features)
        q = k = lnFeatures
        attnFeatures = self.selfAttn(q, k, value=lnFeatures)[0]
        r1Features = lnFeatures + self.dropout1(attnFeatures)
        r1Features = self.layerNorm(r1Features)
        feedFeatures = self.linear2(self.dropout2(self.relu(self.linear1(r1Features))))
        outFeatures = r1Features + self.dropout1(feedFeatures)
        outFeatures = self.layerNorm(outFeatures)

        return outFeatures


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, headNum, feedforwardDim=2048, dropout=0.1, normalize_before=True):
        super().__init__()
        self.layerNorm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self.selfAttn = MyMultiheadAttention(d_model, headNum, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, headNum, dropout=dropout)
        self.linear1 = nn.Linear(d_model, feedforwardDim)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforwardDim, d_model)
        self.relu = nn.ReLU()

    def forward(self, spatialFeatureEncoder, temporalFeatureDecoder):
        spatialFeatureEncoder = self.layerNorm(spatialFeatureEncoder)
        q = spatialFeatureEncoder
        k = temporalFeatureDecoder
        feature = self.self_attn(q, k, value=temporalFeatureDecoder)[0]
        feature = spatialFeatureEncoder + self.dropout1(feature)
        feature2 = self.layerNorm(feature)
        feature2 = self.linear2(self.dropout2(self.relu(self.linear1(feature2))))
        feature = feature + self.dropout1(feature2)

        return feature

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
