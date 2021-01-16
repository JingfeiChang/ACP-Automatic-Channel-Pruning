# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:14:23 2020

@author: ASUS
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from utils.options import args
import utils.common as utils

import os
import copy
import time
import math
import numpy as np
import heapq
import random
from importlib import import_module

conv_num_cfg = {
    'vgg16': 13,
    'resnet18': 8,
    'resnet34': 16,
    'resnet50': 16,
    'resnet101': 33,
    'resnet152': 50
}

max_channels_cfg = {
    'resnet18': [64, 64, 128, 128, 256, 256, 512, 512],
    'resnet34': [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512],
    'resnet50': [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512],
    'resnet101': [],
    'resnet152': [],
}

original_food_cfg = {
    'resnet18': [64, 64, 128, 128, 256, 256, 512, 512],
    'resnet34': [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512],
    'resnet50': [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512],
    'resnet101': [],
    'resnet152': [],
}

food_dimension = conv_num_cfg[args.cfg]
max_channels = max_channels_cfg[args.cfg]
original_food = original_food_cfg[args.cfg]

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
use_cuda = torch.cuda.is_available()
checkpoint = utils.checkpoint(args)
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
args.distributed = args.world_size > 1
traindir = os.path.join(args.data_path, 'ILSVRC2012_img_train')
valdir = os.path.join(args.data_path, 'ILSVRC2012_img_val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

trainLoader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.train_batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

testLoader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.eval_batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

if args.from_scratch == False:

    # Model
    print('==> Loading Model..')
    if args.arch == 'vgg':
        origin_model = import_module(f'model.{args.arch}').VGG(num_classes=1000)
    elif args.arch == 'resnet':
        origin_model = import_module(f'model.{args.arch}').resnet(args.cfg, food=original_food)
    elif args.arch == 'googlenet':
        pass
    elif args.arch == 'densenet':
        pass

    if args.base_food_model is None or not os.path.exists(args.base_food_model):
        raise ('base_Food_model path should be exist!')

    ckpt = torch.load(args.base_food_model, map_location=device)
    '''
    print("model's state_dict:")
    for param_tensor in ckpt:
        print(param_tensor,'\t',ckpt[param_tensor].size())

    print("origin_model's state_dict:")
    for param_tensor in origin_model.state_dict():
        print(param_tensor,'\t',origin_model.state_dict()[param_tensor].size())
    '''
    if use_cuda:
        if not args.distributed:
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                origin_model.features = torch.nn.DataParallel(origin_model.features)
                origin_model.cuda()
            else:
                origin_model = torch.nn.DataParallel(origin_model).cuda()
        else:
            origin_model.cuda()
            origin_model = torch.nn.parallel.DistributedDataParallel(origin_model)
    origin_model.load_state_dict(ckpt['state_dict'])
    oristate_dict = origin_model.state_dict()
    # print(oristate_dict)


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 5 and args.warm_up:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    # print('epoch{}\tlr{}'.format(epoch,lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Our artificial bee colony code is based on the framework at https://www.cnblogs.com/ybl20000418/p/11366576.html
# Define PSOGroup
class PSOGroup():
    """docstring for PSOGroup"""

    def __init__(self):
        super(PSOGroup, self).__init__()
        self.code = []
        self.fitness = 0


# Initialize global element
gbest = PSOGroup()
fmap_block = []
netchannels = []
FoodSource = []
VelSource = []
Velud = []
Particle = []
pbest = []
gbest_state = {}


def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)


def DBSCAN_clustering():
    global netchannels
    # register hook
    #reg_hook = [2, 4, 6, 9, 11, 14, 16, 19]   # resnet18
    reg_hook = [2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34]  # resnet34
    i = 0
    for m in origin_model.modules():
        if isinstance(m, nn.Conv2d):
            i = i + 1
            if i in reg_hook:
                handle = m.register_forward_hook(forward_hook)
            # handle.remove()
    scale_size = 224
    valdir = os.path.join(args.data_path, 'ILSVRC2012_img_val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    testset = datasets.ImageFolder(valdir,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.Resize(scale_size),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    image = []
    for images, labels in testset:
        images = Variable(torch.unsqueeze(images, dim=0).float(), requires_grad=False)
        image.append(images)
        # image = image.cuda()
    print(np.array(image[3]).shape)

    for i in random.sample(range(50000), 200):
        imagetest = image[i].cuda()
        with torch.no_grad():
            origin_model(imagetest)

    # get feature_map with size of (batchsize, channels, W, H)
    feature_map = []
    channels = conv_num_cfg[args.cfg]

    for k in range(channels):
        feature_map.append(fmap_block[k])

    for c in range(channels):
        for j in np.arange(c + channels, len(fmap_block), channels):
            feature_map[c] = torch.cat((feature_map[c], fmap_block[j]), dim=0)

    netchannels = torch.zeros(channels)
    for s in range(channels):
        # print(feature_map[s].shape)
        # change the size of feature_map from (batchsize, channels, W, H) to (batchsize, channels, W*H)
        a, b, c, d = feature_map[s].size()
        feature_map[s] = feature_map[s].view(a, b, -1)
        # print(feature_map[s].shape)

        feature_map[s] = torch.sum(feature_map[s], dim=0) / a
        # print(feature_map[s].shape)

        # clustering
        X = np.array(feature_map[s].cpu())
        clustering = DBSCAN(eps=0.010, min_samples=5, metric='cosine').fit(X)

        # defult: eps=0.5, min_samples=5
        # ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’
        labels = clustering.labels_

        # print(labels)

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        netchannels[s] = netchannels[s] + n_clusters_ + n_noise_

        # print('Estimated number of clusters: %d' % n_clusters_)
        # print('Estimated number of noise points: %d' % n_noise_)
    netchannels = np.array(netchannels)
    print(netchannels)
    return netchannels


# load pre-train params
def load_vgg_particle_model(model, random_rule):
    # print(ckpt['state_dict'])
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None  # Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (
                    random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num - 1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)


def load_resnet_particle_model(model, random_rule):
    cfg = {'resnet18': [2, 2, 2, 2],
           'resnet34': [3, 4, 6, 3],
           'resnet50': [3, 4, 6, 3],
           'resnet101': [3, 4, 23, 3],
           'resnet152': [3, 8, 36, 3]}

    global oristate_dict
    state_dict = model.state_dict()

    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_food_conv_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            if args.cfg == 'resnet18' or args.cfg == 'resnet34':
                iter = 2  # the number of convolution layers in a block, except for shortcut
            else:
                iter = 3
            for l in range(iter):
                conv_name = 'module.' + layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_food_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                # logger.info('weight_num {}'.format(conv_weight_name))
                # logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                # logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (
                        random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index
                    # logger.info('last_select_index{}'.format(last_select_index))

                elif last_select_index != None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_food_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    # for param_tensor in state_dict:
    # logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    # for param_tensor in model.state_dict():
    # logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))

    model.load_state_dict(state_dict)

# Training
def train(model, optimizer, trainLoader, args, epoch, topk=(1,)):
    model.train()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()
    trainLoader_size = 1167
    print_freq = trainLoader_size // args.train_batch_size // 10
    start_time = time.time()
    # trainLoader = get_data_set('train')
    # i = 0
    for batch, (input, target) in enumerate(trainLoader):
        # measure data loading time
        input, target = input.cuda(), target.cuda()

        inputs = torch.autograd.Variable(input)
        targets = torch.autograd.Variable(target)

        train_loader_len = int(math.ceil(1281167 / args.train_batch_size))

        adjust_learning_rate(optimizer, epoch, batch, train_loader_len, args)

        output = model(inputs)
        loss = loss_func(output, targets)
        optimizer.zero_grad()
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets, topk=topk)
        accuracy.update(prec1[0], inputs.size(0))
        top5_accuracy.update(prec1[1], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Top1 {:.2f}%\t'
                'Top5 {:.2f}%\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, trainLoader_size,
                    float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), cost_time
                )
            )
            start_time = current_time


# Testing
def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    top5_accuracy = utils.AverageMeter()

    start_time = time.time()
    # testLoader = get_data_set('test')
    # i = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(testLoader):
            # i+=1
            # if i > 5:
            # break
            inputs = input.cuda()
            targets = target.squeeze().long().cuda()
            targets = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets, topk=topk)
            accuracy.update(predicted[0], inputs.size(0))
            top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s\n'
                .format(float(losses.avg), float(accuracy.avg), float(top5_accuracy.avg), (current_time - start_time))
        )

    return top5_accuracy.avg, accuracy.avg


# Calculate fitness of a food source
def calculationFitness(food, args):
    global gbest
    global gbest_state

    if args.arch == 'vgg':
        model = import_module(f'model.{args.arch}').BeeVGG(foodsource=food, num_classes=1000).cuda()
        load_vgg_particle_model(model, args.random_rule)
    elif args.arch == 'resnet':
        model = import_module(f'model.{args.arch}').resnet(args.cfg, food=food)
        model = torch.nn.DataParallel(model).cuda()
        load_resnet_particle_model(model, args.random_rule)
    elif args.arch == 'googlenet':
        pass
    elif args.arch == 'densenet':
        pass

    # start_time = time.time()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # test(model, testLoader)

    model.train()

    # trainLoader = get_data_set('train')
    # i = 0
    for epoch in range(args.calfitness_epoch):
        # print(epoch)
        for batch, (input, target) in enumerate(trainLoader):
            # measure data loading time
            input = input.cuda()
            target = target.cuda()
            inputs = torch.autograd.Variable(input)
            targets = torch.autograd.Variable(target)

            train_loader_len = int(math.ceil(1281167 / args.train_batch_size))

            adjust_learning_rate(optimizer, epoch, batch, train_loader_len, args)

            # print('epoch{}\tlr{}'.format(epoch,lr))

            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()

        # trainLoader.reset()

    # test(model, loader.testLoader)

    fit_accurary = utils.AverageMeter()
    model.eval()
    # testLoader = get_data_set('test')
    # i = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(testLoader):
            # i+=1
            # if i > 5:
            # break
            inputs = input.cuda()
            targets = target.squeeze().long().cuda()
            targets = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            predicted = utils.accuracy(outputs, targets, topk=(1, 5))
            fit_accurary.update(predicted[1], inputs.size(0))
    # testLoader.reset()

    # current_time = time.time()
    '''
    logger.info(
            'Honey Source fintness {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(accurary.avg), (current_time - start_time))
        )
    '''
    if fit_accurary.avg == 0:
        fit_accurary.avg = 0.01

    if fit_accurary.avg > gbest.fitness:
        gbest_state = copy.deepcopy(model.module.state_dict() if len(args.gpus) > 1 else model.state_dict())
        gbest.code = copy.deepcopy(food)
        gbest.fitness = fit_accurary.avg

    return fit_accurary.avg


# Initialize PSOPruning
def initialize():
    print('==> Initilizing PSO_model..')
    global pbest, gbest, FoodSource, VelSource, Particle, Velud, netchannels
    gbest.fitness = 0

    for i in range(args.food_number):
        # food_number:Number of pruned structures
        FoodSource.append(copy.deepcopy(PSOGroup()))
        VelSource.append(copy.deepcopy(PSOGroup()))
        pbest.append(copy.deepcopy(PSOGroup()))
        Particle.append(copy.deepcopy(PSOGroup()))
        Velud.append(copy.deepcopy(PSOGroup()))

        # FoodSource[i].code.append(netchannels)

        for j in range(food_dimension):
            # Food dimension: num of conv layers. default: vgg16->13 conv layer to be pruned
            list = [-1, 0, 1]
            s = int(np.array(random.sample(list, 1)))
            FoodSource[i].code.append(int(netchannels[j] + i * s))
            if FoodSource[i].code[j] < 1:
                FoodSource[i].code[j] = 1
            if FoodSource[i].code[j] > max_channels[j]:
                FoodSource[i].code[j] = max_channels[j]

            # FoodSource[i].code.append(copy.deepcopy(random.randint(1,args.max_preserve)))
            VelSource[i].code.append(copy.deepcopy(random.uniform(-args.max_vel, args.max_vel)))

        print(FoodSource[i].code)
        # initialize food souce
        FoodSource[i].fitness = calculationFitness(FoodSource[i].code, args)

        # initialize particle
        Particle[i].code = copy.deepcopy(FoodSource[i].code)
        Particle[i].fitness = FoodSource[i].fitness

        # initialize velud
        Velud[i].code = copy.deepcopy(VelSource[i].code)

        # initialize employed bee
        # EmployedBee[i].code = copy.deepcopy(FoodSource[i].code)
        # EmployedBee[i].fitness=FoodSource[i].fitness

        # initialize pbest food
        pbest[i].code = copy.deepcopy(FoodSource[i].code)
        pbest[i].fitness = FoodSource[i].fitness

        # initialize gbest food
        if pbest[i].fitness > gbest.fitness:
            gbest.code = copy.deepcopy(pbest[i].code)
            gbest.fitness = pbest[i].fitness


# Send employed bees to find better food source
def PSOPruning():
    print('==> Start PSOPruning..')
    global FoodSource, VelSource, Particle, Velud, pbest, gbest
    # w = 1
    c1 = 2
    c2 = 2
    r = 2

    for i in range(args.food_number):

        Particle[i].code = copy.deepcopy(FoodSource[i].code)
        Velud[i].code = copy.deepcopy(VelSource[i].code)

        param2change = np.random.randint(0, food_dimension - 1, args.channelchange_num)
        # numpy.random.randint(low, high=None, size=None, dtype='l'):返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)

        R = random.uniform(0, 1)

        # update particle
        for j in range(args.channelchange_num):
            Particle[i].code[param2change[j]] = int(
                FoodSource[i].code[param2change[j]] + r * Velud[i].code[param2change[j]])
            if Particle[i].code[param2change[j]] < 1:
                Particle[i].code[param2change[j]] = 1
            if Particle[i].code[param2change[j]] > max_channels[param2change[j]]:
                Particle[i].code[param2change[j]] = max_channels[param2change[j]] - 1

        # update velocity
        for j in range(args.channelchange_num):
            Velud[i].code[j] = w * VelSource[i].code[j] + c1 * R * (
                        pbest[i].code[j] - FoodSource[i].code[j]) + c2 * R * (gbest.code[j] - FoodSource[i].code[j])
            Velud[i].code[j] = 0.729 * Velud[i].code[j]
            if Velud[i].code[j] > args.max_vel:
                Velud[i].code[j] = args.max_vel
            elif Velud[i].code[j] < -args.max_vel:
                Velud[i].code[j] = -args.max_vel

        Particle[i].fitness = calculationFitness(Particle[i].code, args)

        if Particle[i].fitness > pbest[i].fitness:
            pbest[i].code = copy.deepcopy(Particle[i].code)
            # FoodSource[i].trail = 0
            pbest[i].fitness = Particle[i].fitness

        # else:
        # FoodSource[i].trail = FoodSource[i].trail + 1
        if Particle[i].fitness > gbest.fitness:
            gbest.code = copy.deepcopy(Particle[i].code)
            # FoodSource[i].trail = 0
            gbest.fitness = Particle[i].fitness

        FoodSource[i].code = copy.deepcopy(Particle[i].code)
        VelSource[i].code = copy.deepcopy(Velud[i].code)



def main():
    global w
    start_epoch = 0
    best_acc = 0.0
    best_acc_top1 = 0.0
    code = []

    if args.from_scratch:

        print('==> Building Model..')
        if args.arch == 'vgg':
            model = import_module(f'model.{args.arch}').VGG(num_classes=1000).cuda()
        elif args.arch == 'resnet':
            model = import_module(f'model.{args.arch}').resnet(args.cfg)
            model = torch.nn.DataParallel(model).cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

        if args.resume:
            print('=> Resuming from ckpt {}'.format(args.resume))
            ckpt = torch.load(args.resume, map_location=device)
            best_acc = ckpt['best_acc']
            start_epoch = ckpt['epoch']

            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            # scheduler.load_state_dict(ckpt['scheduler'])
            print('=> Continue from epoch {}...'.format(start_epoch))


    else:

        if args.resume == None:

            test(origin_model, testLoader, topk=(1, 5))
            # testLoader.reset()

            if args.best_food_s == None:

                start_time = time.time()

                bee_start_time = time.time()

                print('==> Start PSOPruning..')

                DBSCAN_clustering()

                initialize()

                # memorizeBestSource()

                for cycle in range(args.max_cycle):
                    w = 0.5 * (8 - cycle) / 20 + 0.4

                    current_time = time.time()
                    logger.info(
                        'Search Cycle [{}]\t Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime {:.2f}s\n'
                            .format(cycle, gbest.code, float(gbest.fitness), (current_time - start_time))
                    )
                    start_time = time.time()

                    PSOPruning()

                    # memorizeBestSource()

                print('==> PSOPruning Complete!')

                bee_end_time = time.time()
                logger.info(
                    'Best Food Source {}\tBest Food Source fitness {:.2f}%\tTime Used{:.2f}s\n'
                        .format(gbest.code, float(gbest.fitness), (bee_end_time - bee_start_time))
                )
                # checkpoint.save_food_model(state)
            else:
                gbest.code = args.gbest
                # gbest_state = torch.load(args.best_food_s)

            # Modelmodel = import_module(f'model.{args.arch}').BeeVGG(foodsource=food, num_classes=1000).to(device)
            print('==> Building model..')
            if args.arch == 'vgg':
                model = import_module(f'model.{args.arch}').BeeVGG(foodsource=gbest.code, num_classes=1000).cuda()
            elif args.arch == 'resnet':
                model = import_module(f'model.{args.arch}').resnet(args.cfg, food=gbest.code)
                model = torch.nn.DataParallel(model).cuda()
            elif args.arch == 'googlenet':
                pass
            elif args.arch == 'densenet':
                pass
            code = gbest.code

            if args.best_food_s:
                bestckpt = torch.load(args.best_food_s)
                model.load_state_dict(bestckpt['state_dict'])
            else:
                model.load_state_dict(gbest_state)

            checkpoint.save_food_model(model.state_dict())

            print(args.random_rule + ' Done!')

            #model = torch.nn.DataParallel(model).cuda()

            if args.best_food == None:
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
                # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
                code = gbest.code
                start_epoch = args.calfitness_epoch

        else:
            # Model
            resumeckpt = torch.load(args.resume)
            state_dict = resumeckpt['state_dict']
            if args.best_food_past == None:
                code = resumeckpt['food_code']
            else:
                code = args.best_food_past
            print('==> Building model..')
            if args.arch == 'vgg':
                model = import_module(f'model.{args.arch}').BeeVGG(foodsource=code, num_classes=1000).cuda()
            elif args.arch == 'resnet':
                model = import_module(f'model.{args.arch}').resnet(args.cfg, food=code).cuda()
            elif args.arch == 'googlenet':
                pass
            elif args.arch == 'densenet':
                pass

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

            model.load_state_dict(state_dict)
            optimizer.load_state_dict(resumeckpt['optimizer'])
            # scheduler.load_state_dict(resumeckpt['scheduler'])
            start_epoch = resumeckpt['epoch']

            model = torch.nn.DataParallel(model).cuda()

    if args.test_only:
        test(model, testLoader, topk=(1, 5))

    else:
        for epoch in range(start_epoch, args.num_epochs):
            train(model, optimizer, trainLoader, args, epoch, topk=(1, 5))
            test_acc, test_acc_top1 = test(model, testLoader, topk=(1, 5))

            is_best = best_acc < test_acc
            best_acc_top1 = max(best_acc_top1, test_acc_top1)
            best_acc = max(best_acc, test_acc)

            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

            state = {
                'state_dict': model_state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'food_code': code
            }
            checkpoint.save_model(state, epoch + 1, is_best)
            # trainLoader.reset()
            # testLoader.reset()

        logger.info('Best accurary(top5): {:.3f} (top1): {:.3f}'.format(float(best_acc), float(best_acc_top1)))


if __name__ == '__main__':
    main()
