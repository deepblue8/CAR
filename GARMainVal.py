import shutil
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from util.dataset import returnDataset
from model.GARModel import GARModel
from util.early_stopping import EarlyStopping
from util.showPicture import showAccuracy, showLoss, showTrainPerAccuracy, showTestPerAccuracy
from util.utils import AverageMeter, show_epoch_info, print_log
from torch.utils import data


class GARMain:
    def __init__(self, config):
        self.config = EasyDict(config)
        self.build()

    def loadModel(self, filepath):
        state = torch.load(filepath)
        self.epoch = state['epoch']
        self.model.roi_align.load_state_dict(state['roi_align_state_dict'])
        self.model.bbox_fc.load_state_dict(state['bbox_fc_state_dict'])
        self.model.spatialEncoder.load_state_dict(state['spatialEncoder_state_dict'])
        self.model.selfAttnCon.load_state_dict(state['selfAttnCon_state_dict'])
        self.model.temporalEncoder.load_state_dict(state['temporalEncoder_state_dict'])
        self.model.dropout.load_state_dict(state['dropout_state_dict'])
        print('Load model states from: ', filepath)
        return self.epoch

    def forward(self, batchData):
        batchSize = batchData[0].shape[0]
        numFrames = batchData[0].shape[1]

        actionsScores, activitiesScores = self.model(batchData[0], batchData[1], batchData[4])

        actionsIn = batchData[2].reshape((batchSize, numFrames, self.config.numMaxN))
        activitiesIn = batchData[3].reshape((batchSize, numFrames))
        bboxesNum = batchData[4].reshape(batchSize, numFrames)

        actionsInNopad = []
        for b in range(batchSize):
            N = bboxesNum[b][0]
            actionsInNopad.append(actionsIn[b][0][:N])
        actionsIn = torch.cat(actionsInNopad, dim=0).reshape(-1, )
        activitiesIn = activitiesIn[:, 0].reshape(batchSize, )

        self.calculateLoss(batchSize, actionsScores, activitiesScores, actionsIn, activitiesIn)
        self.calculateAccuracy(actionsScores, activitiesScores, actionsIn, activitiesIn)

    def calculateLoss(self, batchSize, actionsScores, groupScores, activitiesScores, actionsIn, activitiesIn, groupIn):
        actionsLoss = self.actionsCriterion(actionsScores, actionsIn)
        groupLoss = self.groupCriterion(groupScores, groupIn.long())
        activitiesLoss = self.activitiesCriterion(activitiesScores, activitiesIn)

        actionsInOnehot = F.one_hot(actionsIn, self.config.actionsMumClasses)
        actionsMseLoss = F.mse_loss(actionsScores.float(), actionsInOnehot.float())

        groupInOnehot = F.one_hot(groupIn.long(), self.config.groupNumClasses)
        groupMseLoss = F.mse_loss(groupScores.float(), groupInOnehot.float())
        activitiesInOnehot = F.one_hot(activitiesIn, self.config.activitiesNumClasses)
        activitiesMseLoss = F.mse_loss(activitiesScores.float(), activitiesInOnehot.float())

        actionsLossTotal = self.config.actionsLossWeight * (actionsLoss + actionsMseLoss)
        groupLossTotal = self.config.actionsLossWeight * (groupMseLoss + groupLoss)
        activitiesLossTotal = activitiesLoss + activitiesMseLoss
        self.totalLoss = actionsLossTotal + activitiesLossTotal + groupLossTotal

        self.losses.update(self.totalLoss.item(), batchSize)

    def calculateAccuracy(self, actionsScores, activitiesScores, actionsIn, activitiesIn):
        actionsLabels = torch.argmax(actionsScores, dim=1)
        actionsCorrect = torch.sum(torch.eq(actionsLabels.int(), actionsIn.int()).float())
        actionsAccuracy = actionsCorrect.item() / actionsScores.shape[0]
        self.actionsMeter.update(actionsAccuracy, actionsScores.shape[0])

        activitiesLabels = torch.argmax(activitiesScores, dim=1)
        activitiesCorrect = torch.sum(torch.eq(activitiesLabels.int(), activitiesIn.int()).float())
        activitiesAccuracy = activitiesCorrect.item() / activitiesScores.shape[0]
        self.activitiesMeter.update(activitiesAccuracy, activitiesScores.shape[0])

        activitiesNum = torch.zeros(self.config.activitiesNumClasses).int()
        activitiesPerclassNum = torch.zeros(self.config.activitiesNumClasses).int()
        for i in range(activitiesLabels.shape[0]):
            activitiesPerclassNum[activitiesIn[i].int()] += 1
            if torch.eq(activitiesLabels[i].int(), activitiesIn[i].int()):
                activitiesNum[activitiesLabels[i].int()] += 1

        for i in range(self.config.activitiesNumClasses):
            if activitiesPerclassNum[i] != 0:
                activitiesAccuracy = activitiesNum[i] / activitiesPerclassNum[i]
                self.activitiesPerclassMeter[i].update(activitiesAccuracy, activitiesPerclassNum[i])

    @torch.no_grad()
    def val(self):
        model = self.model
        if self.config.trainOrVal is 'val':
            epoch = self.loadModel(self.config.checkPointPath)
            print("epoch:", epoch)
        model.eval()
        self.losses = AverageMeter()
        self.actionsMeter = AverageMeter()
        self.activitiesMeter = AverageMeter()
        self.activitiesPerclassMeter = [AverageMeter() for i in range(self.config.activitiesNumClasses)]
        for batchData in self.valLoader:
            batchData = [b.to(device=self.config.device) for b in batchData]
            batchSize = batchData[0].shape[0]
            numFrames = batchData[0].shape[1]

            actionsScores, groupScores, activitiesScores, glb = self.model(batchData[0], batchData[1], batchData[4],
                                                                           batchData[6], batchData[7])
            groupIn = self.bulidGroupLabel(glb).cuda()
            actionsIn = batchData[2].reshape((batchSize, numFrames, self.config.numMaxN))
            activitiesIn = batchData[3].reshape((batchSize, numFrames))
            bboxesNum = batchData[4].reshape(batchSize, numFrames)

            actionsInNopad = []
            for b in range(batchSize):
                N = bboxesNum[b][0]
                actionsInNopad.append(actionsIn[b][0][:N])
            actionsIn = torch.cat(actionsInNopad, dim=0).reshape(-1, )
            activitiesIn = activitiesIn[:, 0].reshape(batchSize, )
            self.calculateLoss(batchSize, actionsScores, groupScores, activitiesScores, actionsIn, activitiesIn,
                               groupIn)
            self.calculateAccuracy(actionsScores, activitiesScores, actionsIn, activitiesIn)
        activitiesPerclassList = []
        for i in range(self.config.activitiesNumClasses):
            activitiesPerclassList.append(self.activitiesPerclassMeter[i].avg * 100)

        valResult = {
            'epoch': self.epoch,
            'loss': self.losses.avg,
            'activitiesAcc': self.activitiesMeter.avg * 100,
            'actionsAcc': self.actionsMeter.avg * 100,
            'activitiesPerclassAcc': activitiesPerclassList
        }
        print('Epoch: [%d] '
              'Val Activity Accuracy %.3f%%\t'
              % (self.epoch, self.activitiesMeter.avg * 100), "activitiesPerclassAcc:",
              activitiesPerclassList)
        return valResult

    def build(self):
        self.buildSeed()
        self.buildModel()
        self.buildDatasetLoader()
        self.buildCriterion()

    def buildSeed(self):
        seed = self.config.train.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def buildModel(self):
        model = GARModel(self.config)
        self.model = model.to(self.config.device)

    def buildDatasetLoader(self):
        config = self.config
        trainingSet, validationSet = returnDataset(self.config)
        params = {
            'batch_size': config.dataset.train.batch_size,  # 16
            'shuffle': True,
            'num_workers': 4
        }
        training_loader = data.DataLoader(trainingSet, **params)
        params['batch_size'] = config.dataset.val.batch_size
        validation_loader = data.DataLoader(validationSet, **params)
        self.trainLoader = training_loader
        self.valLoader = validation_loader

    def buildCriterion(self):
        config = self.config
        if config.train.criterion == 'ce_loss':
            actions_weight = torch.tensor(config.train.actions_weight).cuda().detach()
            activities_weight = torch.tensor(config.train.activities_weight).cuda().detach()
            self.actionsCriterion = nn.CrossEntropyLoss(weight=actions_weight)
            self.activitiesCriterion = nn.CrossEntropyLoss(weight=activities_weight)
        else:
            raise NotImplementedError('not implemented criterion ' + config.criterion)
