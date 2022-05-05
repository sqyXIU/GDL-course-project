'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import lib.custom_transforms as custom_transforms

#modified
from sklearn.cluster import KMeans, SpectralClustering
import pandas as pd

import os
import argparse
import time

import models
import datasets
import math

from lib.NCEAverage import NCEAverage
from lib.LinearAverage import LinearAverage
from lib.NCECriterion import NCECriterion
from lib.utils import AverageMeter
from test import NN, kNN

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')

args = parser.parse_args()

if __name__ == '__main__':
    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10Instance(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

    testset = datasets.CIFAR10Instance(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    # trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True,
    #                                            transform=torchvision.transforms.Compose([
    #                                            torchvision.transforms.ToTensor(),
    #                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
    #                                            batch_size=batch_size_train, shuffle=True)
    
    # testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, download=True,
    #                                           transform=torchvision.transforms.Compose([
    #                                           torchvision.transforms.ToTensor(),
    #                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
    #                                           batch_size=batch_size_test, shuffle=True)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    ndata = trainset.__len__()

    print('==> Building model..')

    # modified: classifier
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.fc2(F.relu(self.fc1(x)))

            return x

    net = models.__dict__['ResNet18'](low_dim=args.low_dim)
    net2 = Classifier()
    # define leminiscate
    if args.nce_k > 0:
        lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
    else:
        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #     cudnn.benchmark = True

    # Model
    if args.test_only or len(args.resume)>0:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+args.resume)
        net.load_state_dict(checkpoint['net'])
        lemniscate = checkpoint['lemniscate']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # define loss function
    if hasattr(lemniscate, 'K'):
        criterion = NCECriterion(ndata)
    else:
        criterion = nn.CrossEntropyLoss()

    criterion2 = nn.CrossEntropyLoss()

    net.to(device)
    net2.to(device)
    lemniscate.to(device)
    criterion.to(device)
    criterion2.to(device)

    if args.test_only:
        acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
        sys.exit(0)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.Adam(net2.parameters(), weight_decay=5e-4)


    temploader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, drop_last=False, num_workers=8)
    alpha = 0.5

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr
        if epoch >= 80:
            lr = args.lr * (0.1 ** ((epoch-80) // 40))
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training
    def train(epoch, pseudoLabel):
        print('\nEpoch: %d' % epoch)
        adjust_learning_rate(optimizer, epoch)
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0

        # switch to train mode
        net.train()
        net2.train()

        end = time.time()
        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            data_time.update(time.time() - end)

            pseudolabels = torch.Tensor(pseudoLabel[indexes]).type(torch.LongTensor)
            inputs, targets, indexes, pseudolabels = inputs.to(device), targets.to(device), indexes.to(device), pseudolabels.to(device)
            optimizer.zero_grad()
            optimizer2.zero_grad()

            features = net(inputs)

            outputs = lemniscate(features, indexes)
            outputs2 = net2(features)

            loss1 = criterion(outputs, indexes)
            loss2 = criterion2(outputs2, pseudolabels) # each epoch different pseudo labels
            loss = alpha * loss1 + (1 - alpha) * loss2
            
            loss.backward()
            optimizer.step()
            optimizer2.step()

            train_loss.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            print('loss2', loss2.item())
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                  epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))

    acc_log = []
    for epoch in range(start_epoch, start_epoch+200):
        # get all features
        print('Getting Features')
        trainFeatures = torch.empty(0, args.low_dim).to(device)  
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            targets = targets.to(device)
            batchSize = inputs.size(0)
            features = net(inputs.to(device))
            trainFeatures = torch.cat([trainFeatures, features.data])

        print('Clustering the Features')
        # deepcluster = KMeans(n_clusters=10, algorithm="full", n_init=20)
        deepcluster = SpectralClustering(n_clusters=10, n_init=20)
        cluster = deepcluster.fit(trainFeatures.to('cpu'))
        pseudoLabel = cluster.labels_
        # clusterLoss = cluster.inertia_

        print('Training network')
        train(epoch, pseudoLabel)

        acc = kNN(epoch, net, lemniscate, trainloader, testloader, 200, args.nce_t, 0)
        acc_log.append(acc)

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'lemniscate': lemniscate,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc

        print('best accuracy: {:.2f}'.format(best_acc*100))

    pd.DataFrame(acc_log).to_csv('acc_log.csv')
    # acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, 1)
    print('last accuracy: {:.2f}'.format(acc*100))
