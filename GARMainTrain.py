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

    def saveModel(self, epoch, filepath):
        state = {
            'epoch': epoch,
            'roi_align_state_dict': self.model.roi_align.state_dict(),
            'bbox_fc_state_dict': self.model.bbox_fc.state_dict(),
            'spatialEncoder_state_dict': self.model.spatialEncoder.state_dict(),
            'selfAttnCon_state_dict': self.model.selfAttnCon.state_dict(),
            'temporalEncoder_state_dict': self.model.temporalEncoder.state_dict(),
            'dropout_state_dict': self.model.dropout.state_dict(),
        }
        torch.save(state, filepath)
        print('model saved to:', filepath)

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

    # =======================Train Begin================== #
    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def adjust_lr(self, optimizer, new_lr):
        print('change learning rate:', new_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self):
        self.model.train()
        self.model.apply(self.set_bn_eval)
        self.epoch = 1

        train_accuracy = []
        test_accuracy = []
        train_loss = []
        test_loss = []
        train_per_accuracy_total = torch.zeros((self.config.activitiesNumClasses, self.config.MAXEpoch))
        test_per_accuracy_total = torch.zeros((self.config.activitiesNumClasses, self.config.MAXEpoch))

        save_path = self.config.resultPath
        early_stopping = EarlyStopping(save_path)

        bestResult = {'epoch': 0, 'activitiesAcc': 50, 'loss': 50}
        for epoch in range(self.config.train.startEpoch, self.config.train.totalEpochs):
            self.lr = self.lr_scheduler.get_lr()[0]
            self.epoch = epoch

            trainInfo = self.trainEpoch()

            show_epoch_info('Train', self.config.logPath, trainInfo)
            train_accuracy.append(trainInfo['activitiesAcc'])
            train_loss.append(trainInfo['loss'])
            if trainInfo['loss'] < bestResult['loss']:
                bestResult = trainInfo
                filepath = self.config.resultPath + '/epoch%d_%.2f%%.pth' % (
                    epoch, trainInfo['activitiesAcc'])
                self.saveModel(self.epoch, filepath)

            early_stopping(trainInfo['loss'], self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        showAccuracy(train_accuracy, test_accuracy)
        showLoss(train_loss, test_loss)

    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def trainEpoch(self):
        model = self.model
        self.losses = AverageMeter()
        self.actionsMeter = AverageMeter()
        self.activitiesMeter = AverageMeter()
        self.activitiesPerclassMeter = [AverageMeter() for i in range(self.config.activitiesNumClasses)]

        for batchData in self.trainLoader:
            batchData = [b.to(device=self.config.device) for b in batchData]
            model.train()
            model.apply(self.set_bn_eval)
            self.forward(batchData)
            self.optimizer.zero_grad()
            self.totalLoss.backward()
            self.optimizer.step()

        activitiesPerclassList = []
        for i in range(self.config.activitiesNumClasses):
            activitiesPerclassList.append(self.activitiesPerclassMeter[i].avg * 100)

        self.lr_scheduler.step()

        train_info = {
            'epoch': self.epoch,
            'loss': self.losses.avg,
            'activitiesAcc': self.activitiesMeter.avg * 100,
            'actionsAcc': self.actionsMeter.avg * 100,
            'activitiesPerclassAcc': activitiesPerclassList
        }
        return train_info

    def bulidGroupLabel(self,tensor_2d):
        selected_elements = torch.zeros(tensor_2d.size(0), dtype=tensor_2d.dtype, device=tensor_2d.device)

        for i in range(tensor_2d.size(0)):
            non_minus_one_indices = (tensor_2d[i] != -1).nonzero()
            if non_minus_one_indices.size(0) > 0:
                first_non_minus_one_index = non_minus_one_indices[0].item()
                selected_elements[i] = tensor_2d[i, first_non_minus_one_index]
        return selected_elements

    def forward(self, batchData):
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

        self.calculateLoss(batchSize, actionsScores, groupScores, activitiesScores, actionsIn, activitiesIn, groupIn)
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

    def update(self):
        self.lr_scheduler.optimizer.step()

    def build(self):
        self.buildSeed()
        self.buildModel()
        self.buildOptimizer()
        self.buildDatasetLoader()
        self.buildScheduler()
        self.buildCriterion()

    def buildSeed(self):
        seed = self.config.train.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def buildModel(self):
        model = GARModel(self.config)
        self.model = model.to(self.config.device)

    def buildOptimizer(self):
        config = self.config
        model = self.model
        try:
            optim = getattr(torch.optim, config.train.optimizer.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' +
                                      config.train.optimizer.type)
        optimizer = optim(filter(lambda p: p.requires_grad, model.parameters()), **config.train.optimizer.kwargs)
        self.optimizer = optimizer

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

    def buildScheduler(self):
        config = self.config
        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.train.scheduler.milestones,
            gamma=config.train.scheduler.gamma,
            last_epoch=self.config.train.startEpoch - 1)

    def buildCriterion(self):
        config = self.config
        if config.train.criterion == 'ce_loss':
            actions_weight = torch.tensor(config.train.actions_weight).cuda().detach()
            activities_weight = torch.tensor(config.train.activities_weight).cuda().detach()
            self.actionsCriterion = nn.CrossEntropyLoss(weight=actions_weight)
            self.activitiesCriterion = nn.CrossEntropyLoss(weight=activities_weight)
        else:
            raise NotImplementedError('not implemented criterion ' + config.criterion)
