import math
from datetime import datetime

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import copy
import numpy as np
import random
from tqdm import trange
from torchinfo import summary
from utils.distribute import uniform_distribute, train_dg_split, uniform_distribute1
from utils.sampling import iid, noniid
from utils.options import args_parser
from src.update import ModelUpdate
from src.nets import  CNN_v1, CNN_v2, CNN_v3, CNN_v4, ResNet,ResidualBlock
from src.aggregation_strategy import FedAvg
from src.test import tes_img,tes_img_2
import pandas as pd
import openpyxl
import csv
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
writer = SummaryWriter()


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

#-----------------------------------------------------------------------数据导入_START------------------------------------------------------------------
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset = torchvision.datasets.ImageFolder('.\\train',
                                                     transform=trans_mnist)
    dataset_test = {i: list() for i in range(0, 7)}
    for i in range(0, 7):
        dataset_test[i] = torchvision.datasets.ImageFolder(('.\\test\\{}'.format(i)),
                                                    transform=trans_mnist)

# -----------------------------------------------------------------------数据导入_END---------------------------------------------------------------------
#########################################################################################################################################################
# -----------------------------------------------------------------------数据分发_START-------------------------------------------------------------------
    dataset_all_len = len(dataset)   #数据集数量
    train_data = {i: list() for i in range(args.num_users)}
    for i in range(0,args.num_users):
        train_data[i] = copy.deepcopy(dataset)

    dataset_train = copy.deepcopy(dataset)

    if args.sampling == 'iid':
        local_dix = iid(dataset_train, args.num_users)
    elif args.sampling == 'noniid':
        labels = np.concatenate([np.array(dataset.targets)], axis=0)
        local_dix = noniid(dataset_train,labels,args)


    else:
        exit('Error: unrecognized sampling')

    idxs_users = list(range(args.num_users))
    train_data_len = []
    for idx in idxs_users:

        train_data[idx].imgs = []
        train_data[idx].targets = []
        train_data[idx].samples = []
        for i in local_dix[idx]:
            train_data[idx].imgs.append(dataset.imgs[i])
            train_data[idx].targets.append(dataset.targets[i])
            train_data[idx].samples.append(dataset.samples[i])

        train_data_len.append(len(train_data[idx]))
    max_train_data_len = max(train_data_len)
    min_train_data_len = min(train_data_len)
    print('max:',max_train_data_len)
    print('min:',min_train_data_len)
#------------------------------------------------------------------------数据分发_EDN------------------------------------------------------------------------
############################################################################################################################################################
#------------------------------------------------------------------------模型选择_START----------------------------------------------------------------------
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'ResNet' and args.dataset == 'mnist':
        print("ResNet")
        net_glob = ResNet(ResidualBlock).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = []
        createVar = locals()
        for i in range(args.num_users):
            createVar['net_glob' + str(i)] = CNN_v1(args=args).to(args.device)
            net_glob.append(createVar['net_glob' + str(i)])
        net_glob_centre = CNN_v1(args=args).to(args.device)

        node_w = [[] for _ in range(args.num_users)]
        for i in range(args.num_users):
            net_glob[i].train()
            node_w[i] = net_glob[i].state_dict()
        net_glob_centre.train()

    elif args.model == 'cnn3'and args.dataset == 'mnist':
        print("resnet18")
        net_glob = CNN_v3(args=args).to(args.device)
    elif args.model == 'cnn4' and args.dataset == 'mnist':
        print("vgg16")
        net_glob = CNN_v4(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    # print(net_glob)
#------------------------------------------------------------------------模型选择_EDN------------------------------------------------------------------------
############################################################################################################################################################
#------------------------------------------------------------------------模型训练_START----------------------------------------------------------------------
    for iter in trange(args.rounds):
        if args.lr_model == "Decay":
           args.lr = args.lr * 0.995

# ------------------------------------------------------------------1.Centralized_START-------------------------------------------------------------------
        if args.algorithm == 'Centralized':

            loss_train_avg = 0

            node = ModelUpdate(args=args, dataset=train_data[idx], idxs=local_dix[idx],
                               max_train_data_len=max_train_data_len, min_train_data_len=min_train_data_len,
                               dataset_all_len=dataset_all_len)

            w,  loss = node.train(local_net=copy.deepcopy(net_glob_centre).to(args.device),
                                          net=copy.deepcopy(net_glob_centre).to(args.device))
            loss_train_avg += 1 / args.num_users * loss

            net_glob_centre.load_state_dict(w)

            correct_every_fault = [[0 for i in range(7)] for j in range(7)]
            precision = 0
            recall = 0
            acc_test_avg, loss_test_avg,correct_every_fault,precision,recall,F1  = tes_img_2(net_glob_centre, dataset_test, args)

# ------------------------------------------------------------------Centralized_END--------------------------------------------------------------------
#####################################################################################################################################################################
#----------------------------------------------------------------------PS_DFL_MC所提方案-----------------------------------------------------------------------------
        elif args.algorithm == 'PS_DFL' or  args.algorithm == 'PS_DFL_MC':

            loss_train_avg = 0
            if iter == 0:
                MC_number = [0,0,0]
                node_number_group = 0
                node_list = list(range(args.num_users))
                group_node = [node_list[i:i + int(args.num_users/args.num_groups)] for i in range(0, args.num_users,int(args.num_users/args.num_groups))]


                group_now = 0

            for idx_groups in range(args.num_groups):
                print(group_now)
                group_w = [[] for _ in range(int(args.num_users/args.num_groups))]
                node_number_group = 0
                for idx in group_node[group_now]:
                # Local update

                    node = ModelUpdate(args=args, dataset=train_data[idx], idxs=local_dix[idx],
                                       max_train_data_len=max_train_data_len, min_train_data_len=min_train_data_len,
                                       dataset_all_len=dataset_all_len)

                    w, loss = node.train(local_net=copy.deepcopy(net_glob_centre).to(args.device),
                                                  net=copy.deepcopy(net_glob_centre).to(args.device))

                    group_w[node_number_group] = copy.deepcopy(w)
                    node_number_group += 1

                node_w_connect = FedAvg(group_w, args)
                net_glob_centre.load_state_dict(node_w_connect)

                if args.algorithm == 'PS_DFL':
                    group_now = (group_now + 1) % args.num_groups
                if args.algorithm == 'PS_DFL_MC':
                    MC = random.random()
                    p = 0.1
                    if MC > 1-p:
                        group_now = (group_now + 1) % args.num_groups
                        MC_number[0] += 1
                    elif MC <= p:
                        MC_number[1] += 1
                        group_now = (group_now - 1) % args.num_groups
                    else:
                        MC_number[2] += 1
                        group_now = group_now

            print(MC_number)
            correct_every_fault = [[0 for i in range(7)] for j in range(7)]
            precision = 0
            recall = 0
            F1 = 0
            acc_test_avg, loss_test_avg,correct_every_fault,precision,recall,F1  = tes_img_2(net_glob_centre, dataset_test, args)
#------------------------------------------------------------------------PS_DFL_END---------------------------------------------------------------------------------
#######################################################################################################################################################################
#------------------------------------------------------------------------数据记录_START---------------------------------------------------------------------------------
        if args.debug:
             dt = datetime.now()
             print(f"Round: {iter}")
             print(f"acc_test_avg: {acc_test_avg}")
             if args.title == 1:
                args.title = 0
                dt2 = datetime.now()
                header = {'Iteration':[], 'acc_test_avg':[], 'loss_train_avg':[], 'loss_test_avg':[], 'SNR':[],'hour':[], 'min':[],'precision':[],'recall':[],'F1':[]}

                data = pd.DataFrame(header)
                if args.distribution == "Dirichlet":
                    data.to_csv(f'..\\{dt2.date()}-{dt2.hour}_{args.algorithm}_{args.distribution}_a={args.alpha}_group={args.num_groups}_node={args.num_users}_bs={args.local_bs}.csv', mode='w', index=False)  # mod
                else:
                    data.to_csv(
                        f'..\\{dt2.date()}-{dt2.hour}_{args.algorithm}_{args.distribution}_group={args.num_groups}_node={args.num_users}_bs={args.local_bs}.csv',
                        mode='w', index=False)  # mod

             lists = [iter, round(acc_test_avg,3),round(loss_train_avg,4),round(loss_test_avg,4),0,dt.hour,dt,precision,recall,F1]

             if args.algorithm == 'PS_DFL' or  args.algorithm == 'CFL' or args.algorithm == 'DFL_gossip' or args.algorithm == 'RDFL' or args.algorithm == 'Centralized' or args.algorithm =="DFL_sequentia" or args.algorithm =="PS_DFL_MC":
                 for i in correct_every_fault:
                     lists.extend(i)
             data = pd.DataFrame([lists])
             if args.distribution == "Dirichlet":
                data.to_csv(f'..\\{dt2.date()}-{dt2.hour}_{args.algorithm}_{args.distribution}_a={args.alpha}_group={args.num_groups}_node={args.num_users}_bs={args.local_bs}.csv', mode='a', header=False,
                        index=False)
             else:
                 data.to_csv(f'..\\{dt2.date()}-{dt2.hour}_{args.algorithm}_{args.distribution}_group={args.num_groups}_node={args.num_users}_bs={args.local_bs}.csv',mode='a', header=False,index=False)  # mod 没有文件则创建，有则清空   a继续往下写
#------------------------------------------------------------------------数据记录_END------------------------------------------------------------------------------------



    writer.close()