import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.googlenet import Inception
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from utils.options import args
import utils.common as utils

import os
import time
import copy
import sys
import random
import numpy as np
import heapq
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from data import cifar10, cifar100, imagenet
from importlib import import_module

checkpoint = utils.checkpoint(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

conv_num_cfg = {
    'vgg9': 6,
    'vgg16': 13,
    'vgg19': 16,
    'resnet56' : 27,
    'resnet110' : 54,
    'googlenet' : 27,
    'densenet':36,
    }

max_channels_cfg = {
    'vgg9': [128, 128, 256, 256, 512, 512],
    'vgg11': [64, 128, 256, 256, 512, 512, 512, 512],
    'vgg13': [64, 64, 128, 128, 256, 256, 512, 512, 512, 512],
    'vgg16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'vgg19': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
    'resnet56': [16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
                 64, 64, 64, 64, 64, 64, 64, 64, 64],
    'resnet110': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                  32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],  
    'googlenet': [96, 16, 32, 128, 32, 96, 96, 16, 48, 112, 24, 64, 128, 24, 64, 144, 32, 64, 
                  160, 32, 128, 160, 32, 128, 192, 48, 128]
    }

original_food_cfg = {
    'vgg9': [128, 128, 256, 256, 512, 512],
    'vgg16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'vgg19': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
    'resnet56': [16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
                 64, 64, 64, 64, 64, 64, 64, 64, 64],
    'resnet110': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                  32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    'googlenet': [96, 16, 32, 128, 32, 96, 96, 16, 48, 112, 24, 64, 128, 24, 64, 144, 32, 64, 
                  160, 32, 128, 160, 32, 128, 192, 48, 128]
    }

reg_hook = [3, 5, 6, 10, 12, 13, 17, 19, 20, 24, 26, 27, 31, 33, 34, 38, 40, 41, 45, 47, 48, 52, 54, 55, 59, 61, 62] 

food_dimension = conv_num_cfg[args.cfg]
max_channels = max_channels_cfg[args.cfg]
original_food = original_food_cfg[args.cfg]

# Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
    data_sets = CIFAR10
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)
    data_sets = CIFAR100
else:
    loader = imagenet.Data(args)
    data_sets = ImageFolder


# Model
print('==> Loading Model..')
if args.arch == 'vgg_cifar':
    origin_model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
elif args.arch == 'resnet_cifar':
    origin_model = import_module(f'model.{args.arch}').resnet(args.cfg, food=original_food).to(device)
elif args.arch == 'googlenet':
    origin_model = import_module(f'model.{args.arch}').googlenet().to(device)
elif args.arch == 'densenet':
    origin_model = import_module(f'model.{args.arch}').densenet().to(device)

if args.base_food_model is None or not os.path.exists(args.base_food_model):
    raise Exception('Food_model path should be exist!')


ckpt = torch.load(args.base_food_model, map_location=device)
origin_model.load_state_dict(ckpt['state_dict'])
oristate_dict = origin_model.state_dict()


#Define PSOGroup 
class PSOGroup():
    """docstring for PSOGroup"""
    def __init__(self):
        super(PSOGroup, self).__init__() 
        self.code = [] 
        self.fitness = 0

#Initilize global element best_food
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
    i=0
    # register hook
    if args.arch == 'vgg_cifar':
        for m in origin_model.modules():
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(forward_hook)
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
    elif args.arch == 'resnet_cifar':
        for m in origin_model.modules():
            if isinstance(m, nn.Conv2d):
                i=i+1
                if i%2==0:
                    m.register_forward_hook(forward_hook)
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
    elif args.arch == 'googlenet':
        for m in origin_model.modules():
            if isinstance(m, nn.Conv2d):
                i=i+1
                if i in reg_hook:
                    m.register_forward_hook(forward_hook)
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])    
    #handle.remove()            
    testset = data_sets(root=args.data_path, train=False, download=True, transform=transform_test)

    image = []
    for images, labels in testset:
        images = Variable(torch.unsqueeze(images, dim=0).float(), requires_grad=False)
        image.append(images)
        #image = image.cuda()
    print(np.array(image[3]).shape)

    for i in random.sample(range(10000),200):
        imagetest = image[i].cuda()
        with torch.no_grad():
            origin_model(imagetest)

    # get feature_map with size of (batchsize, channels, W, H)
    feature_map = []
    channels = conv_num_cfg[args.cfg]
    
    for k in range(channels):
        feature_map.append(fmap_block[k])

    for c in range(channels):
        for j in np.arange(c+channels, len(fmap_block), channels):
            feature_map[c] = torch.cat((feature_map[c], fmap_block[j]), dim=0)

    netchannels = torch.zeros(channels)
    for s in range(channels):
        # print(feature_map[s].shape)
        # change the size of feature_map from (batchsize, channels, W, H) to (batchsize, channels, W*H)
        a, b, c, d = feature_map[s].size()
        feature_map[s] = feature_map[s].view(a, b, -1)
        #print(feature_map[s].shape)
        
        feature_map[s] = torch.sum(feature_map[s], dim=0)/a
        #print(feature_map[s].shape)


    
        # clustering
        X = np.array(feature_map[s].cpu())
        clustering = DBSCAN(eps=0.007, min_samples=5, metric='cosine').fit(X)
        
        # defult: eps=0.5, min_samples=5
        # ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’
        labels = clustering.labels_

        #print(labels)

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
    
        netchannels[s] = netchannels[s]+n_clusters_+n_noise_

        #print('Estimated number of clusters: %d' % n_clusters_)
        #print('Estimated number of noise points: %d' % n_noise_)
    netchannels = np.array(netchannels)    
    print(netchannels)
    return netchannels

#load pre-train params
def load_vgg_particle_model(model, random_rule):
    #print(ckpt['state_dict'])
    global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)
            orifilter_num1 = oriweight.size(1)
            currentfilter_num1 = curweight.size(1)
            

            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    # heapq.nlargest(n, iterable[, key]) Return a list with the n largest elements from the dataset defined by iterable
                    # map(function, iterable, ...) 
                    # list.index(x[, start[, end]]) 
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        # enumerate() 
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                               
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                select_num = currentfilter_num1
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [0, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    # heapq.nlargest(n, iterable[, key]) Return a list with the n largest elements from the dataset defined by iterable
                    # map(function, iterable, ...) 
                    # list.index(x[, start[, end]]) 
                    select_index.sort()
                for index_i, i in enumerate(select_index):
                    # enumerate() 
                    state_dict[name + '.weight'][:, index_i, :, :] = \
                        oristate_dict[name + '.weight'][:,i, :, :]
                                         
                last_select_index = None

    model.load_state_dict(state_dict)

def load_dense_particle_model(model, random_rule):

    global oristate_dict
    

    state_dict = model.state_dict()

    conv_weight = []
    conv_trans_weight = []
    bn_weight = []
    bn_bias = []

    for i in range(3):
        for j in range(12):
            conv1_weight_name = 'dense%d.%d.conv1.weight' % (i + 1, j)
            conv_weight.append(conv1_weight_name)

            bn1_weight_name = 'dense%d.%d.bn1.weight' % (i + 1, j)
            bn_weight.append(bn1_weight_name)

            bn1_bias_name = 'dense%d.%d.bn1.bias' %(i+1,j)
            bn_bias.append(bn1_bias_name)

    for i in range(2):
        conv1_weight_name = 'trans%d.conv1.weight' % (i + 1)
        conv_weight.append(conv1_weight_name)
        conv_trans_weight.append(conv1_weight_name)

        bn_weight_name = 'trans%d.bn1.weight' % (i + 1)
        bn_weight.append(bn_weight_name)

        bn_bias_name = 'trans%d.bn1.bias' % (i + 1)
        bn_bias.append(bn_bias_name)
    
    bn_weight.append('bn.weight')
    bn_bias.append('bn.bias')


    
    for k in range(len(conv_weight)):
        conv_weight_name = conv_weight[k]
        oriweight = oristate_dict[conv_weight_name]
        curweight = state_dict[conv_weight_name]
        orifilter_num = oriweight.size(1)
        currentfilter_num = curweight.size(1)
        select_num = currentfilter_num
        #print(orifilter_num)
        #print(currentfilter_num)

        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                select_index = random.sample(range(0, orifilter_num-1), select_num)
                select_index.sort()
            else:
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            for i in range(curweight.size(0)):
                for index_j, j in enumerate(select_index):
                    state_dict[conv_weight_name][i][index_j] = \
                            oristate_dict[conv_weight_name][i][j]


    for k in range(len(bn_weight)):

        bn_weight_name = bn_weight[k]
        bn_bias_name = bn_bias[k]
        bn_bias.append(bn_bias_name)
        bn_weight.append(bn_weight_name)
        oriweight = oristate_dict[bn_weight_name]
        curweight = state_dict[bn_weight_name]

        orifilter_num = oriweight.size(0)
        currentfilter_num = curweight.size(0)
        select_num = currentfilter_num

        if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
            if random_rule == 'random_pretrain':
                select_index = random.sample(range(0, orifilter_num-1), select_num)
                select_index.sort()
            else:
                l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                select_index.sort()

            for index_j, j in enumerate(select_index):
                state_dict[bn_weight_name][index_j] = \
                        oristate_dict[bn_weight_name][j]
                state_dict[bn_bias_name][index_j] = \
                        oristate_dict[bn_bias_name][j]

    oriweight = oristate_dict['fc.weight']
    curweight = state_dict['fc.weight']
    orifilter_num = oriweight.size(1)
    currentfilter_num = curweight.size(1)
    select_num = currentfilter_num

    if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):
        if random_rule == 'random_pretrain':
            select_index = random.sample(range(0, orifilter_num-1), select_num)
            select_index.sort()
        else:
            l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
            select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
            select_index.sort()

        for i in range(curweight.size(0)): 
            for index_j, j in enumerate(select_index):
                state_dict['fc.weight'][i][index_j] = \
                        oristate_dict['fc.weight'][i][j]



    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.BatchNorm2d):
            bn_weight_name = name + '.weight'
            bn_bias_name = name + '.bias'
            if bn_weight_name not in bn_weight and bn_bias_name not in bn_bias:
                state_dict[bn_weight_name] = oristate_dict[bn_weight_name]
                state_dict[bn_bias_name] = oristate_dict[bn_bias_name]

    model.load_state_dict(state_dict)




def load_google_particle_model(model, random_rule):
    global oristate_dict
    state_dict = model.state_dict()
        
    last_select_index = None
    all_food_conv_name = []
    all_food_bn_name = []

    for name, module in model.named_modules():

        if isinstance(module, Inception):

            food_filter_channel_index = ['.branch5x5.3']  # the index of sketch filter and channel weight
            food_channel_index = ['.branch3x3.3', '.branch5x5.6']  # the index of sketch channel weight
            food_filter_index = ['.branch3x3.0', '.branch5x5.0']  # the index of sketch filter weight
            food_bn_index = ['.branch3x3.1', '.branch5x5.1', '.branch5x5.4'] #the index of sketch bn weight
            
            for bn_index in food_bn_index:
                all_food_bn_name.append(name + bn_index)

            for weight_index in food_filter_channel_index:
                last_select_index = None
                conv_name = name + weight_index + '.weight'
                all_food_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    #print(state_dict[conv_name].size())
                    #print(oristate_dict[conv_name].size())
                else:
                    select_index = range(orifilter_num)
         
            
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)


                select_index_1 = copy.deepcopy(select_index)


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                else:
                    select_index = range(orifilter_num)
                
                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                            state_dict[conv_name][index_i][index_j] = \
                                oristate_dict[conv_name][i][j]



            for weight_index in food_channel_index:

                conv_name = name + weight_index + '.weight'
                all_food_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]
                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                #print(state_dict[conv_name].size())
                #print(oristate_dict[conv_name].size())


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()


                    for i in range(state_dict[conv_name].size(0)):
                        for index_j, j in enumerate(select_index):
                            state_dict[conv_name][i][index_j] = \
                                oristate_dict[conv_name][i][j]


            for weight_index in food_filter_index:

                conv_name = name + weight_index + '.weight'
                all_food_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    for index_i, i in enumerate(select_index):
                            state_dict[conv_name][index_i] = \
                                oristate_dict[conv_name][i]


    for name, module in model.named_modules(): #Reassign non sketch weights to the new network

        if isinstance(module, nn.Conv2d):

            if name not in all_food_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_food_bn_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def load_resnet_particle_model(model, random_rule):

    cfg = { 
           'resnet56': [9,9,9],
           'resnet110': [18,18,18],
           }

    global oristate_dict
    state_dict = model.state_dict()
        
    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_honey_conv_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):
                conv_name = layer_name + str(k) + '.conv' + str(l+1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                #logger.info('weight_num {}'.format(conv_weight_name))
                #logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                #logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        logger.info('last_select_index'.format(last_select_index))
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]  

                    last_select_index = select_index
                    #logger.info('last_select_index{}'.format(last_select_index)) 

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
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    #for param_tensor in state_dict:
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    #for param_tensor in model.state_dict():
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))
 

    model.load_state_dict(state_dict)


# Training
def train(model, optimizer, trainLoader, args, epoch):

    model.train()
    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time

#Testinga
def test(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg

#Calculate fitness of a food source
def calculationFitness(food, train_loader, args):
    global gbest
    global gbest_state

    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').PSOVGG(args.cfg,foodsource=food).to(device)
        load_vgg_particle_model(model, args.random_rule)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg,food=food).to(device)
        load_resnet_particle_model(model, args.random_rule)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(food=food).to(device)
        load_google_particle_model(model, args.random_rule)
    elif args.arch == 'densenet':
        model = import_module(f'model.{args.arch}').densenet(food=food).to(device)
        load_dense_particle_model(model, args.random_rule)

    fit_accurary = utils.AverageMeter()
    train_accurary = utils.AverageMeter()



    #start_time = time.time()
    #if len(args.gpus) != 1:
        #model = nn.DataParallel(model, device_ids=args.gpus)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #test(model, loader.testLoader)

    model.train()
    for epoch in range(args.calfitness_epoch):
        for batch, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            #print("ok")
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()

            prec1 = utils.accuracy(output, targets)
            train_accurary.update(prec1[0], inputs.size(0))

    #test(model, loader.testLoader)

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader.testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = utils.accuracy(outputs, targets)
            fit_accurary.update(predicted[0], inputs.size(0))


    #current_time = time.time()
    '''
    logger.info(
            'Food Source fintness {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(accurary.avg), (current_time - start_time))
        )
    '''
    if fit_accurary.avg > gbest.fitness:
        gbest_state = copy.deepcopy(model.module.state_dict() if len(args.gpus) > 1 else model.state_dict())
        gbest.code = copy.deepcopy(food)
        gbest.fitness = fit_accurary.avg

    return fit_accurary.avg


#Initilize Bee-Pruning
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
        
        #FoodSource[i].code.append(netchannels)
        
        
        for j in range(food_dimension):
        # Food dimension: num of conv layers. default: vgg16->13 conv layer to be pruned
            list = [-1, 0, 1]  
            s = int(np.array(random.sample(list, 1)))
            FoodSource[i].code.append(int(netchannels[j]+i*s))
            if FoodSource[i].code[j] < 1:
                FoodSource[i].code[j] = 1
            if FoodSource[i].code[j] > max_channels[j]:
                FoodSource[i].code[j] = max_channels[j]
                
            #FoodSource[i].code.append(copy.deepcopy(random.randint(1,args.max_preserve)))
            VelSource[i].code.append(copy.deepcopy(random.uniform(-args.max_vel,args.max_vel)))
       
        print(FoodSource[i].code)
        #initialize food souce
        FoodSource[i].fitness = calculationFitness(FoodSource[i].code, loader.trainLoader, args)
        
        #initialize particle
        Particle[i].code = copy.deepcopy(FoodSource[i].code)
        Particle[i].fitness=FoodSource[i].fitness 
        
        #initialize velud
        Velud[i].code = copy.deepcopy(VelSource[i].code)
        
        #initialize employed bee  
        #EmployedBee[i].code = copy.deepcopy(FoodSource[i].code)
        #EmployedBee[i].fitness=FoodSource[i].fitness 

        #initialize pbest food 
        pbest[i].code = copy.deepcopy(FoodSource[i].code)
        pbest[i].fitness=FoodSource[i].fitness 

        #initialize gbest food
        if pbest[i].fitness > gbest.fitness:
            gbest.code = copy.deepcopy(pbest[i].code)
            gbest.fitness = pbest[i].fitness


#Send employed bees to find better food source
def PSOPruning():
    print('==> Start PSOPruning..')
    global FoodSource, VelSource, Particle, Velud, pbest, gbest
    #w = 1
    c1 = 2
    c2 = 2
    r = 2   
    
    for i in range(args.food_number):
                
        Particle[i].code = copy.deepcopy(FoodSource[i].code)
        Velud[i].code = copy.deepcopy(VelSource[i].code)
        
        param2change = np.random.randint(0, food_dimension-1, args.channelchange_num)
        
        
        R = random.uniform(0,1)
 
        # update particle        
        for j in range(args.channelchange_num):        
            Particle[i].code[param2change[j]] = int(FoodSource[i].code[param2change[j]]+ r*Velud[i].code[param2change[j]])
            if Particle[i].code[param2change[j]] < 1:
                Particle[i].code[param2change[j]] = 1
            if Particle[i].code[param2change[j]] > max_channels[param2change[j]]:
                Particle[i].code[param2change[j]] = max_channels[param2change[j]]-1
       
        # update velocity
        for j in range(args.channelchange_num):
            Velud[i].code[j] = w*VelSource[i].code[j]+c1*R*(pbest[i].code[j]-FoodSource[i].code[j])+c2*R*(gbest.code[j]-FoodSource[i].code[j])
            Velud[i].code[j] = 0.729*Velud[i].code[j]
            if Velud[i].code[j] > args.max_vel:
                Velud[i].code[j] = args.max_vel   
            elif Velud[i].code[j] < -args.max_vel:
                Velud[i].code[j] = -args.max_vel
        
        Particle[i].fitness = calculationFitness(Particle[i].code, loader.trainLoader, args)

        if Particle[i].fitness > pbest[i].fitness:                
            pbest[i].code = copy.deepcopy(Particle[i].code)              
            #FoodSource[i].trail = 0  
            pbest[i].fitness = Particle[i].fitness
            
        #else:          
            #FoodSource[i].trail = FoodSource[i].trail + 1
        if Particle[i].fitness > gbest.fitness:                
            gbest.code = copy.deepcopy(Particle[i].code)              
            #FoodSource[i].trail = 0  
            gbest.fitness = Particle[i].fitness
            
        FoodSource[i].code = copy.deepcopy(Particle[i].code)
        VelSource[i].code = copy.deepcopy(Velud[i].code)

def main():
    global w
    start_epoch = 0
    best_acc = 0.0
    code = []
    
    if args.resume == None:

        test(origin_model, loader.testLoader)

        if args.gbest == None:

            start_time = time.time()
            
            pso_start_time = time.time()
            
            print('==> Start Pruning..')
            
            print('==> Start clustering compressing..')
            
            DBSCAN_clustering()

            initialize()


            for cycle in range(args.max_cycle):
                
                w = 0.5*(8-cycle)/20+0.4

                current_time = time.time()
                logger.info(
                    'Search Cycle [{}]\t Best Food Source {}\tBest Food Source fitness {:.2f}%\tTime {:.2f}s\n'
                    .format(cycle, gbest.code, float(gbest.fitness), (current_time - start_time))
                )
                start_time = time.time()

                PSOPruning() 
                  

            print('==> PSOPruning Complete!')
            pso_end_time = time.time()
            logger.info(
                'Best Food Source {}\tBest Food Source fitness {:.2f}%\tTime Used{:.2f}s\n'
                .format(gbest.code, float(gbest.fitness), (pso_end_time - pso_start_time))
            )
            #checkpoint.save_pso_model(state)
        else:
            gbest.code = args.gbest

        # Model
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').PSOVGG(args.cfg, foodsource=gbest.code).to(device)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet(args.cfg,food=gbest.code).to(device)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(food=gbest.code).to(device)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(food=gbest.code).to(device)

        if args.best_food_s:
            bestckpt = torch.load(args.best_food_s)
            model.load_state_dict(bestckpt)
        else:
            model.load_state_dict(gbest_state)

        checkpoint.save_pso_model(model.state_dict())

        print(args.random_rule + ' Done!')

        #if len(args.gpus) != 1:
            #model = nn.DataParallel(model, device_ids=args.gpus)

        if args.gbest == None:

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        code = gbest.code

    else:
        # Model
        resumeckpt = torch.load(args.resume)
        state_dict = resumeckpt['state_dict']
        code = resumeckpt['honey_code']
        print('==> Building model..')
        if args.arch == 'vgg_cifar':
            model = import_module(f'model.{args.arch}').PSOVGG(args.cfg, foodsource=code).to(device)
        elif args.arch == 'resnet_cifar':
            model = import_module(f'model.{args.arch}').resnet(args.cfg,food=code).to(device)
        elif args.arch == 'googlenet':
            model = import_module(f'model.{args.arch}').googlenet(food=code).to(device)
        elif args.arch == 'densenet':
            model = import_module(f'model.{args.arch}').densenet(food=code).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(resumeckpt['optimizer'])
        scheduler.load_state_dict(resumeckpt['scheduler'])
        start_epoch = resumeckpt['epoch']

        #if len(args.gpus) != 1:
            #model = nn.DataParallel(model, device_ids=args.gpus)


    if args.test_only:
        test(model, loader.testLoader)

    else: 
        for epoch in range(start_epoch, args.num_epochs):
            train(model, optimizer, loader.trainLoader, args, epoch)
            scheduler.step()
            lr = scheduler.get_lr()
            print(lr)
            test_acc = test(model, loader.testLoader)

            is_best = best_acc < test_acc
            best_acc = max(best_acc, test_acc)

            #model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            model_state_dict = model.state_dict()
            state = {
                'state_dict': model_state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'honey_code': code
            }
            checkpoint.save_model(state, epoch + 1, is_best)

        logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

if __name__ == '__main__':
    main()
