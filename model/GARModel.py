import numpy as np
import torch
import torch.nn as nn

from util.pytorch_i3d import InceptionI3d
from util.roi_align import RoIAlign
from torchvision.models import resnet50

from util.utils import prep_images
from util.pos_encoding import spatialencoding2d, PositionalEncoding2D
from .GARModelCell import GARModelCell
from .MyMultiHeadAttentionCon import MyMultiheadAttentionCon, ECABlock
from .ResNet50 import MyResNet50
from .Transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, \
    TransformerEncoderLayerTanhRelu


class GARModel(nn.Module):
    def __init__(self, config):
        super(GARModel, self).__init__()

        self.config = config
        self.embed_features = config.embed_features
        self.spatial_encoding = config.spatial_encoding
        self.numLayers = config.numLayers
        self.KEYT = self.config.KEYT

        self.resnet = resnet50(pretrained=True)
        self.myresnet = MyResNet50()
        self.conv1x1 = torch.nn.Conv2d(2048, self.embed_features, kernel_size=1)
        self.bn2d = nn.BatchNorm2d(self.embed_features, eps=1e-5, momentum=0.1, affine=True)
        self.ecaBlock = ECABlock(self.embed_features)
        self.roi_align = RoIAlign(config.crop_size[0], config.crop_size[0])
        self.ln1 = nn.LayerNorm([2048, self.config.crop_size[0], self.config.crop_size[0]])
        self.fc = nn.Linear(self.config.crop_size[0] * self.config.crop_size[0] * self.embed_features,
                            self.embed_features)
        self.relu = nn.ReLU()

        self.ln2 = nn.LayerNorm(self.embed_features)
        self.bbox_fc = nn.Sequential(nn.Linear(51200, 1024), nn.ReLU(), nn.Linear(1024, self.embed_features))
        self.selfAttnCon = MyMultiheadAttentionCon(self.embed_features, self.config.headNum,
                                                   dropout=self.config.dropoutNum)
        self.tokenCLSInit = nn.Parameter(torch.randn(1, 1, self.embed_features))

        self.GARModelLayer = nn.ModuleList()
        for i in range(self.numLayers):
            self.GARModelLayer.append(GARModelCell(config, self.embed_features))

        self.dropout = nn.Dropout(p=0.5)
        self.actions_fc = nn.Linear(self.embed_features, self.config.actionsMumClasses)
        self.activities_fc = nn.Linear(self.embed_features, self.config.activitiesNumClasses)

        self.actions_classifier = nn.Sequential(
            nn.Linear(self.embed_features, self.embed_features * 2),
            nn.ReLU(),
            nn.Linear(self.embed_features * 2, self.config.actionsMumClasses),
            nn.Dropout(p=config.dropoutNum),
            nn.Softmax(dim=1)
        )
        self.groupClassifier = nn.Sequential(
            nn.Linear(self.embed_features, self.embed_features * 2),
            nn.ReLU(),
            nn.Linear(self.embed_features * 2, self.config.groupNumClasses),
            nn.Dropout(p=config.dropoutNum),
            nn.Softmax(dim=1)
        )
        self.activities_classifier = nn.Sequential(
            nn.Linear(self.embed_features, self.embed_features * 2),
            nn.ReLU(),
            nn.Linear(self.embed_features * 2, self.config.activitiesNumClasses),
            # nn.Linear(NFG,self.cfg.num_activities),
            nn.Dropout(p=config.dropoutNum),
            nn.Softmax(dim=1)
        )
        encoderLayer = TransformerEncoderLayer(self.embed_features, self.config.headNum, dropout=self.config.dropoutNum,
                                               normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.spatialEncoder = TransformerEncoder(encoderLayer, num_layers=self.config.encoderLayersNum,
                                                 norm=encoderNorm)

        encoderLayer = TransformerEncoderLayer(self.embed_features, self.config.headNum, dropout=self.config.dropoutNum,
                                               normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.temporalEncoder = TransformerEncoder(encoderLayer, num_layers=self.config.encoderLayersNum,
                                                  norm=encoderNorm)

        encoderLayerG = TransformerEncoderLayerTanhRelu(self.embed_features, self.config.headNum,
                                                        dropout=self.config.dropoutNum,
                                                        normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.spatialGroupEncoder = TransformerEncoder(encoderLayerG, num_layers=self.config.encoderLayersNum,
                                                      norm=encoderNorm)

        encoderLayerG = TransformerEncoderLayer(self.embed_features, self.config.headNum,
                                                dropout=self.config.dropoutNum,
                                                normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.temporalGroupEncoder = TransformerEncoder(encoderLayerG, num_layers=self.config.encoderLayersNum,
                                                       norm=encoderNorm)

        decoderLayer = TransformerDecoderLayer(self.embed_features, self.config.headNum, dropout=self.config.dropoutNum,
                                               normalize_before=True)
        decoderNorm = nn.LayerNorm(self.embed_features)
        self.garDecoder = TransformerDecoder(decoderLayer, num_layers=self.config.decoderLayersNum, norm=decoderNorm)

    def backboneFunc(self, images_in):
        B, T, C, H, W = images_in.shape
        imgsResNet50 = torch.reshape(images_in, (B * T, C, H, W))
        imgsResNet50 = prep_images(imgsResNet50)
        imgsFeatures = self.myresnet(imgsResNet50)
        OC, OH, OW = imgsFeatures.shape[-3:]
        device = imgsFeatures.device
        if self.spatial_encoding is True:
            sp = spatialencoding2d(OC, OH, OW, device).detach()
            imgsFeatures += sp
        return imgsFeatures

    def buildFeatFunc(self, imgsFeatures, boxes_in):
        B, T, N, n = boxes_in.shape
        MaxN = self.config.numMaxN
        OC, OH, OW = imgsFeatures.shape[-3:]

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MaxN, 4))
        boxes_scale = torch.zeros_like(boxes_in_flat)
        boxes_scale[:, 0], boxes_scale[:, 1], boxes_scale[:, 2], boxes_scale[:, 3] = OW, OH, OW, OH
        boxes_in_flat *= boxes_scale
        boxes = boxes_in_flat.reshape(B, T * MaxN, 4)
        boxes.requires_grad = False
        boxes_idx = torch.arange(B * T, device=boxes_in_flat.device, dtype=torch.int).reshape(-1, 1).repeat(1,MaxN)
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MaxN,))

        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(imgsFeatures, boxes_in_flat, boxes_idx_flat)
        boxes_features = boxes_features.reshape(B * T * MaxN, -1)
        boxes_features = self.bbox_fc(boxes_features).reshape(B, T, MaxN, -1)

        return boxes_features

    def fill_with_zeros(self, input_tensor):
        original_size = input_tensor.size()
        target_size = (original_size[1], original_size[1])
        filled_tensor = torch.zeros(target_size, dtype=input_tensor.dtype, device=input_tensor.device)
        filled_tensor = torch.full(target_size, -1, dtype=input_tensor.dtype, device=input_tensor.device)
        filled_tensor[:original_size[0], :original_size[1]] = input_tensor

        return filled_tensor

    def reGNI(self, groupInfo, groupLabels):
        B, MaxN = groupInfo.shape
        groupInfoLists = []
        glb = []
        for b in range(B):
            non_zero_tensor = groupInfo[b, :][groupInfo[b, :].nonzero()].squeeze()
            non_zero_tensor = groupInfo[b, :][groupInfo[b, :] != -1]
            unique_values, inverse_indices = torch.unique(non_zero_tensor, return_inverse=True)
            groupInfoList = []
            gl = []
            for index in unique_values:
                tmp = torch.full((MaxN,), -1)
                tmplabel = torch.full((MaxN,), -1)
                for j in range(MaxN):
                    if index == groupInfo[b, j]:
                        tmp[j] = int(index)
                        tmplabel[j] = groupLabels[b, j]
                gl.append(tmplabel)
                groupInfoList.append(tmp)

            gl = torch.stack(gl)
            glb.append(gl)
            groupInfoList = torch.stack(groupInfoList)
            groupInfoList = self.fill_with_zeros(groupInfoList)
            groupInfoLists.append(groupInfoList)

        groupInfoLists = torch.stack(groupInfoLists)
        glb = torch.cat(glb, dim=0)
        return groupInfoLists, glb

    def selectKeyTF(self, boxesFeatures):
        B, T, MaxN, F = boxesFeatures.shape
        boxesFeaturesAll = []
        for b in range(B):
            batchImgsFeatures = boxesFeatures[b, :, :, :]
            bImgFLN = batchImgsFeatures
            bImgFLN = bImgFLN.permute(1, 0, 2).contiguous()

            attn_output, attn_output_weights = self.selfAttnCon(bImgFLN, bImgFLN, value=bImgFLN)
            topk_values, topk_indices = torch.topk(attn_output_weights.sum(dim=1), self.KEYT, dim=1)
            sorted_topk_indices, indices = torch.sort(topk_indices, dim=1)

            sorted_topk_indices_ex = sorted_topk_indices.unsqueeze(2).expand(-1, -1, bImgFLN.size(2))
            batchImgsFeaturesNew = torch.gather(bImgFLN, 1, sorted_topk_indices_ex)
            N, TKey, F = batchImgsFeaturesNew.shape
            boxesFeaturesAll.append(batchImgsFeaturesNew)
        boxesFeaturesAll = torch.stack(boxesFeaturesAll)
        return boxesFeaturesAll

    def reGFI(self, boxesFeaturesKeyT, groupNumInfo):
        B, MaxN, T, F = boxesFeaturesKeyT.shape
        BFGLS = []
        for b in range(B):
            groupLabel = []
            BFGL = []
            for i in range(groupNumInfo[b, :, :].size(0)):
                BFG = []
                for j in range(groupNumInfo[b, :, :].size(1)):
                    if groupNumInfo[b, i, j].item() != -1:
                        BFG.append(boxesFeaturesKeyT[b, j, :, :])
                if len(BFG) == 0:
                    break
                BFG = torch.stack(BFG)
                padding_length = MaxN - BFG.size(0)
                BFG = torch.cat([BFG, torch.zeros(padding_length, BFG.size(1), BFG.size(2)).cuda()], dim=0)
                BFGL.append(BFG)

            BFGL = torch.stack(BFGL)
            padding_length = MaxN - BFGL.size(0)
            BFGL = torch.cat([BFGL, torch.zeros(padding_length, BFGL.size(1), BFGL.size(2), BFGL.size(3)).cuda()],
                             dim=0)
            BFGLS.append(BFGL)
        BFGLS = torch.stack(BFGLS)
        return BFGLS

    def buildGroupFeatFunc(self, boxesFeatures, groupId, groups):
        B, T, MaxN = groupId.shape
        groupId = torch.mean(groupId.float(), dim=1)
        groups = torch.mean(groups.float(), dim=1)
        groupNumInfo, glb = self.reGNI(groupId, groups)
        boxesFeaturesKeyT = self.selectKeyTF(boxesFeatures)
        groupFeat = self.reGFI(boxesFeaturesKeyT, groupNumInfo)
        return groupFeat, groupNumInfo, glb

    def conGroupInFunc(self, groupFeat, groupNumInfo):
        B, GN, MaxN, T, F = groupFeat.shape
        groupFeaturesList = []
        groupFeaturesListZero = []
        groupNum = []
        for b in range(B):
            groupFeaturesListB = []
            groupNumB = 0
            for gn in range(groupFeat[b, :, :, :, :].size(0)):
                temp = []
                for gnn in range(groupFeat[b, gn, :, :, :].size(0)):
                    if torch.all(groupNumInfo[b, gn, :] == -1).item() == True:
                        break
                    if int(groupNumInfo[b, gn, gnn]) != 0:
                        temp.append(groupFeat[b, gn, gnn, :, :])
                if torch.all(groupNumInfo[b, gn, :] == -1).item() == False:
                    groupNumB = groupNumB + 1
                    GFII = torch.stack(temp)
                    NNZ, KeyT, F = GFII.shape
                    tfg = self.temporalEncoder(GFII)
                    sfgTemp = tfg.reshape(NNZ, KeyT, F)  # T N F -> N T F
                    sfg = self.spatialEncoder(sfgTemp)
                    groupFeaturesListB.append(sfg.mean(dim=0))

            groupNum.append(groupNumB)
            groupFeaturesListB = torch.stack(groupFeaturesListB, dim=0)

            if groupFeaturesListB.size(0) != 13:
                a = torch.zeros((13 - int(groupFeaturesListB.size(0)), int(groupFeaturesListB.size(1)),
                                 int(groupFeaturesListB.size(2)))).cuda()
                groupFeaturesListBZ = torch.cat((groupFeaturesListB, a), dim=0)

            groupFeaturesList.append(groupFeaturesListB)
            groupFeaturesListZero.append(groupFeaturesListBZ)
        groupNum = torch.tensor(groupNum)
        groupFeaturesListZero = torch.stack(groupFeaturesListZero)
        groupFeaturesList = torch.cat(groupFeaturesList, dim=0)
        return groupFeaturesList, groupFeaturesListZero, groupNum

    def conGroupOutFunc(self, groupFeaturesListZ, groupNum):
        activies = []
        B, GN, KeyT, F = groupFeaturesListZ.shape
        for b in range(B):
            gn = groupNum[b]
            temporalFeatures = self.temporalGroupEncoder(
                groupFeaturesListZ[b, :, :, :][:int(gn)].permute(1, 0, 2).contiguous().reshape(KeyT, gn, F))

            SFT = temporalFeatures.mean(dim=0).reshape(gn, F).sum(dim=0)
            activies.append(SFT)
        activies = torch.stack(activies, dim=0)
        activities_scoresg = self.activities_classifier(activies)
        return activities_scoresg

    def garModelCellFunc(self, boxes_features, bboxes_num_in):
        B, T, MaxN, F = boxes_features.shape

        actions_scores = []
        activities_scores = []

        for b in range(B):
            n = bboxes_num_in[b][0]

            batchImgsFeatures = boxes_features[b, :, :int(n), :]
            bImgFLN = batchImgsFeatures

            spatialFeaturesAll = self.spatialEncoder(bImgFLN)
            bImgFLN = bImgFLN.permute(1, 0, 2).contiguous()
            spatialFeaturesAll = spatialFeaturesAll.permute(1, 0, 2).contiguous()

            attn_output, attn_output_weights = self.selfAttnCon(bImgFLN, bImgFLN, value=bImgFLN)
            topk_values, topk_indices = torch.topk(attn_output_weights.sum(dim=1), self.KEYT, dim=1)
            sorted_topk_indices, indices = torch.sort(topk_indices, dim=1)

            sorted_topk_indices_ex = sorted_topk_indices.unsqueeze(2).expand(-1, -1, bImgFLN.size(2))
            batchImgsFeaturesNew = torch.gather(bImgFLN, 1, sorted_topk_indices_ex)
            sorted_topk_indices_ex1 = sorted_topk_indices.unsqueeze(2).expand(-1, -1, spatialFeaturesAll.size(2))
            spatialFeaturesT = torch.gather(spatialFeaturesAll, 1, sorted_topk_indices_ex1)
            N, TKey, F = batchImgsFeaturesNew.shape
            spatialFeatures = spatialFeaturesT.permute(1, 0, 2).contiguous().reshape(TKey, N, -1)

            pos_encoding = PositionalEncoding2D(F, T)
            spatialFeaturesT = pos_encoding(spatialFeaturesT)
            cls_temporal_tokens = self.tokenCLSInit.repeat_interleave(N, dim=0)
            temporalMatrix = torch.cat((cls_temporal_tokens, spatialFeaturesT), dim=1)
            temporalMatrixT = temporalMatrix.permute(1, 0, 2).contiguous().reshape(TKey + 1, N, -1)
            temporalFeatureEncoder = self.temporalEncoder(temporalMatrix)
            temporalFeatureEncoderT = temporalFeatureEncoder.permute(1, 0, 2).contiguous().reshape(TKey + 1, N, -1)

            f2 = temporalFeatureEncoderT[0].squeeze(0) + spatialFeatures.mean(dim=0).contiguous().reshape(N, -1)
            features = self.dropout(f2)

            b_actions_scores = self.actions_classifier(features)
            b_activities_scores = self.activities_classifier(features)
            b_activities_scores = b_activities_scores.sum(dim=0)

            actions_scores.append(b_actions_scores)
            activities_scores.append(b_activities_scores)

        actions_scores = torch.cat(actions_scores, dim=0)
        activities_scores = torch.stack(activities_scores, dim=0)
        return actions_scores, activities_scores

    def forward(self, images_in, boxes_in, bboxes_num_in, groupInfo, groupLabels):
        imgsFeatures = self.backboneFunc(images_in)
        boxes_features = self.buildFeatFunc(imgsFeatures, boxes_in)
        groupFeat, groupNumInfo, glb = self.buildGroupFeatFunc(boxes_features, groupInfo, groupLabels)
        groupFeatList, groupFeatListZero, groupNum = self.conGroupInFunc(groupFeat, groupNumInfo)
        groupScores = self.groupClassifier(groupFeatList)
        GN, KeyT, cls = groupScores.shape
        groupScores = torch.mean(groupScores, dim=1).reshape(GN, -1)
        activitieScores = self.conGroupOutFunc(groupFeatListZero, groupNum)
        actions_scores, activities_scores = self.garModelCellFunc(boxes_features, bboxes_num_in)
        actionsScores = actions_scores
        activitiesScores = activitieScores + activities_scores
        return actionsScores, groupScores, activitiesScores, glb
